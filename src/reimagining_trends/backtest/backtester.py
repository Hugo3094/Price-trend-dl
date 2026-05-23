"""
backtester.py
-------------
Backtester — orchestrates the full backtest pipeline:
  aux data → benchmark → betas → scores → portfolio → simulate → metrics → plots.
"""

import logging
import os
import traceback
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reimagining_trends.backtest.metrics import compute_backtest_metrics
from reimagining_trends.backtest.portfolio import construct_ls_portfolio, construct_lo_portfolio
from reimagining_trends.backtest.simulator import simulate_portfolio
from reimagining_trends.data.fetch_data import (
    _ensure_flat_columns, add_moving_average, cumret_scale, image_scale,
)
from reimagining_trends.imaging.ohlc_chart import generate_ohlc_image
from reimagining_trends.utils.cache import (
    benchmark_fp, backtest_fp, scores_fp,
    dict_hash, fingerprints_match, load_json, save_json, load_pickle, save_pickle,
)
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)

_MODEL_TYPE = {"MLP": "mlp", "GRU": "gru", "LSTM": "lstm", "CNN": "cnn"}


class Backtester:
    """
    Full backtest engine.

    Parameters
    ----------
    config    : Config
    raw_data  : {ticker: OHLCV DataFrame}  — full date range
    trainers  : {model_name: Trainer}
    """

    def __init__(self, config: Config, raw_data: dict, trainers: dict) -> None:
        self.cfg      = config
        self.raw_data = raw_data
        self.trainers = trainers
        os.makedirs(config.results_dir, exist_ok=True)
        self._cache_dir = os.path.join(config.results_dir, "backtest_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

        # populated in _load_aux_data / _build_panels
        self._rf: pd.Series           = pd.Series(dtype=float)
        self._mktcap: pd.DataFrame    = pd.DataFrame()  # date x permno
        self._sectors: pd.DataFrame   = pd.DataFrame()  # date x permno (point-in-time gsector)
        self._daily_ret: pd.DataFrame = pd.DataFrame()  # date x permno
        self._betas: pd.DataFrame     = pd.DataFrame()  # date x permno
        self._benchmark: pd.Series    = pd.Series(dtype=float)

        # benchmark signal panels (date x permno), populated in _precompute_signal_panels
        self._signal_mom:  pd.DataFrame = pd.DataFrame()
        self._signal_str:  pd.DataFrame = pd.DataFrame()
        self._signal_wstr: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run all backtests (models + benchmarks); return nested results dict."""
        logger.info("Backtest: loading auxiliary data …")
        self._load_aux_data()
        self._build_panels()
        self._build_benchmark()
        self._compute_betas()
        self._precompute_signal_panels()

        weightings = (
            ["equal", "proportional", "cap_weighted"]
            if self.cfg.bt_weighting == "all"
            else [self.cfg.bt_weighting]
        )
        port_types = (
            ["LS", "LO"]
            if self.cfg.bt_portfolio_type == "both"
            else [self.cfg.bt_portfolio_type]
        )

        all_results: dict = {}

        # ── Model strategies ──────────────────────────────────────────────
        for model_name, trainer in self.trainers.items():
            trainer.load_best()
            mtype = _MODEL_TYPE.get(model_name, "mlp")
            logger.info("Scoring %s on test dates …", model_name)
            scores_by_date, sc_fp = self._compute_all_scores(trainer, mtype)

            for weighting in weightings:
                for port_type in port_types:
                    key = f"{model_name}_{weighting}_{port_type}"
                    logger.info("Backtest: %s", key)
                    try:
                        bt_fp = backtest_fp(sc_fp, self.cfg, weighting, port_type)
                        result = self._single_backtest(
                            scores_by_date, weighting, port_type, key=key, bt_fp=bt_fp,
                        )
                        all_results[key] = result
                        m = result["metrics"]
                        logger.info(
                            "  net_sharpe=%.3f | net_ann_ret=%.3f | MDD=%.3f | IR=%.3f",
                            m.get("net_sharpe", np.nan),
                            m.get("net_ann_return", np.nan),
                            m.get("net_max_drawdown", np.nan),
                            m.get("net_IR", np.nan),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Backtest %s failed: %s\n%s",
                            key, exc, traceback.format_exc(),
                        )

        # ── Benchmark strategies (MOM, STR, WSTR) ─────────────────────────
        benchmarks = getattr(self.cfg, "bt_benchmarks", ["MOM", "STR", "WSTR"])
        for bench_name in benchmarks:
            bench_scores, bm_fp = self._benchmark_scores_for(bench_name)
            for weighting in weightings:
                for port_type in port_types:
                    key = f"{bench_name}_{weighting}_{port_type}"
                    logger.info("Benchmark backtest: %s", key)
                    try:
                        bt_fp = backtest_fp(bm_fp, self.cfg, weighting, port_type)
                        result = self._single_backtest(
                            bench_scores, weighting, port_type, key=key, bt_fp=bt_fp,
                        )
                        all_results[key] = result
                        m = result["metrics"]
                        logger.info(
                            "  net_sharpe=%.3f | net_ann_ret=%.3f | MDD=%.3f | IR=%.3f",
                            m.get("net_sharpe", np.nan),
                            m.get("net_ann_return", np.nan),
                            m.get("net_max_drawdown", np.nan),
                            m.get("net_IR", np.nan),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Benchmark backtest %s failed: %s\n%s",
                            key, exc, traceback.format_exc(),
                        )

        if all_results:
            self._plot_results(all_results)
            self._log_summary(all_results)

        return all_results

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _load_aux_data(self) -> None:
        """Load risk-free rate, market_cap and gsector from parquet."""
        rf_path = getattr(self.cfg, "bt_rf_path", None)
        if rf_path and os.path.exists(rf_path):
            rf_df = pd.read_parquet(rf_path)
            rf_df.columns = rf_df.columns.str.strip().str.lower()
            if rf_df.index.dtype != "datetime64[ns]":
                rf_df.index = pd.to_datetime(rf_df.index)
            col = "rf_returns" if "rf_returns" in rf_df.columns else rf_df.columns[0]
            rf_raw = rf_df[col].sort_index()
            if rf_raw.index.duplicated().any():
                rf_raw = rf_raw[~rf_raw.index.duplicated(keep="last")]
            self._rf = rf_raw
        else:
            logger.warning("RF path not found — using zero risk-free rate.")
            self._rf = pd.Series(dtype=float)

        parquet_path = getattr(self.cfg, "parquet_path", None)
        if not parquet_path or not os.path.exists(parquet_path):
            logger.warning("Parquet path unavailable — market_cap/sector set to defaults.")
            return

        df = pd.read_parquet(parquet_path)
        if df.index.name is not None and "date" not in df.columns:
            df = df.reset_index()
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"])

        id_col = "permno" if "permno" in df.columns else "ticker"

        if "market_cap" in df.columns:
            mktcap = (
                df.pivot_table(index="date", columns=id_col, values="market_cap")
                .sort_index()
            )
            mktcap.columns = mktcap.columns.astype(str)
            self._mktcap = mktcap

        if "gsector" in df.columns:
            sectors_pivot = (
                df.pivot_table(index="date", columns=id_col, values="gsector", aggfunc="last")
                .sort_index()
            )
            sectors_pivot.columns = sectors_pivot.columns.astype(str)
            # Convert string codes (e.g. "10") to numeric; keep NaN where missing
            self._sectors = sectors_pivot.apply(pd.to_numeric, errors="coerce")

    def _build_panels(self) -> None:
        """Build daily returns panel from raw_data."""
        frames = {}
        for ticker, df in self.raw_data.items():
            df = _ensure_flat_columns(df)
            frames[ticker] = df["Close"].pct_change()
        daily = pd.DataFrame(frames).sort_index()
        if daily.index.duplicated().any():
            daily = daily[~daily.index.duplicated(keep="last")]
        self._daily_ret = daily

    def _build_benchmark(self) -> None:
        """Market-cap weighted CRSP benchmark daily return."""
        if self._mktcap.empty:
            logger.warning("No market_cap data — benchmark = equal-weight universe.")
            self._benchmark = self._daily_ret.mean(axis=1)
            return

        # Lag market cap by 1 day to avoid look-ahead
        mktcap_lag = self._mktcap.shift(1).reindex(self._daily_ret.index)
        aligned = mktcap_lag.reindex(columns=self._daily_ret.columns)
        # Keep NaN weights — sum(skipna=True) excludes NaN contributions naturally
        w = aligned.div(aligned.sum(axis=1), axis=0)
        self._benchmark = (w * self._daily_ret).sum(axis=1)

    def _compute_betas(self) -> None:
        """Rolling OLS beta for each ticker vs benchmark."""
        window    = self.cfg.bt_beta_window
        min_obs   = self.cfg.bt_min_beta_obs
        bm        = self._benchmark

        betas = {}
        for ticker in self._daily_ret.columns:
            r = self._daily_ret[ticker]
            r, bm_aligned = r.align(bm, join="inner")
            cov = r.rolling(window, min_periods=min_obs).cov(bm_aligned)
            var = bm_aligned.rolling(window, min_periods=min_obs).var()
            betas[ticker] = (cov / var.replace(0, np.nan))

        self._betas = pd.DataFrame(betas).sort_index()

    def _precompute_signal_panels(self) -> None:
        """Compute MOM / STR / WSTR signal panels (date × permno) once."""
        # fillna(0.0) only here, at computation time — not stored in _daily_ret
        log_ret = np.log1p(self._daily_ret.fillna(0.0))

        r252 = log_ret.rolling(252, min_periods=126).sum()  # ~12 months
        r42  = log_ret.rolling(42,  min_periods=21).sum()   # ~2 months
        r21  = log_ret.rolling(21,  min_periods=15).sum()   # ~1 month
        r5   = log_ret.rolling(5,   min_periods=3).sum()    # ~1 week

        self._signal_mom  = r252 - r42  # 2-12 month: high = past winner → long
        self._signal_str  = -r21        # negated 1-month: high = past loser → long (reversal)
        self._signal_wstr = -r5         # negated 1-week:  high = past loser → long (reversal)

        logger.info(
            "Signal panels ready: MOM %s, STR %s, WSTR %s",
            self._signal_mom.shape, self._signal_str.shape, self._signal_wstr.shape,
        )

    def _benchmark_scores_for(self, signal_name: str) -> tuple[dict, dict]:
        """
        Return (scores_by_date, bm_fp) for a benchmark signal on the same
        rebalancing grid used by models (every h trading days in the test period).
        """
        bm_fp_dict = benchmark_fp(signal_name, self.cfg)

        cache_pkl  = os.path.join(self._cache_dir, f"bm_{signal_name}_scores.pkl")
        cache_json = os.path.join(self._cache_dir, f"bm_{signal_name}_scores_fingerprint.json")
        cached_fp  = load_json(cache_json)
        if fingerprints_match(bm_fp_dict, cached_fp):
            cached = load_pickle(cache_pkl)
            if cached is not None:
                logger.info("[%s] Benchmark scores cache hit.", signal_name)
                return cached, bm_fp_dict

        panel = {"MOM": self._signal_mom, "STR": self._signal_str, "WSTR": self._signal_wstr}[signal_name]

        test_dates = self._daily_ret.index[self._daily_ret.index > self.cfg.val_end]
        reb_dates  = test_dates[::self.cfg.horizon]

        scores_by_date: dict = {}
        for t in reb_dates:
            if t not in panel.index:
                continue
            row = panel.loc[t]
            scores = {str(col): float(v) for col, v in row.items() if not np.isnan(v)}
            if scores:
                scores_by_date[t] = scores

        save_pickle(scores_by_date, cache_pkl)
        save_json(bm_fp_dict, cache_json)
        return scores_by_date, bm_fp_dict

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _compute_all_scores(self, trainer, model_type: str) -> tuple[dict, dict]:
        """
        For each rebalancing date t (signal date), run model on all tickers.

        Returns
        -------
        (scores_by_date, sc_fp)
            scores_by_date : {date: {ticker: P(UP)}}
            sc_fp          : fingerprint dict for downstream backtest_fp()
        """
        sc_fp: dict = {}
        if trainer.fingerprint:
            ckpt_path = os.path.join(trainer.save_dir, "best_model.pt")
            sc_fp = scores_fp(trainer.fingerprint, ckpt_path)

            cache_pkl  = os.path.join(trainer.save_dir, "scores_cache.pkl")
            cache_json = os.path.join(trainer.save_dir, "scores_fingerprint.json")
            cached_fp  = load_json(cache_json)
            if fingerprints_match(sc_fp, cached_fp):
                cached = load_pickle(cache_pkl)
                if cached is not None:
                    logger.info("[%s] Scores cache hit.", trainer.model_name)
                    return cached, sc_fp

        test_dates   = self._daily_ret.index[self._daily_ret.index > self.cfg.val_end]
        reb_dates    = test_dates[::self.cfg.horizon]

        scores_by_date: dict = {}
        for t in reb_dates:
            scores = {}
            for ticker in self.raw_data:
                p = self._score_ticker(ticker, t, trainer, model_type)
                if p is not None:
                    scores[ticker] = p
            if scores:
                scores_by_date[t] = scores

        if sc_fp:
            save_pickle(scores_by_date, os.path.join(trainer.save_dir, "scores_cache.pkl"))
            save_json(sc_fp, os.path.join(trainer.save_dir, "scores_fingerprint.json"))

        return scores_by_date, sc_fp

    def _score_ticker(self, ticker: str, t, trainer, model_type: str):
        """Return P(UP) for one ticker at signal date t, or None if unavailable."""
        df = self.raw_data[ticker]
        df_t = _ensure_flat_columns(df).loc[:t]

        try:
            if model_type == "cnn":
                return self._score_cnn(df_t, trainer)
            else:
                return self._score_tabular(df_t, trainer)
        except Exception:
            return None

    def _score_tabular(self, df_t: pd.DataFrame, trainer) -> float | None:
        w = self.cfg.window
        if self.cfg.include_ma:
            df_t = add_moving_average(df_t, w)
        if self.cfg.scaling == "image":
            df_s = image_scale(df_t, w)
        else:
            df_s = cumret_scale(df_t, w)
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]
        if self.cfg.include_ma and "MA" in df_s.columns:
            feature_cols.append("MA")
        arr = df_s[feature_cols].values
        if len(arr) < w:
            return None
        x = arr[-w:][np.newaxis].astype(np.float32)   # (1, window, n_features)
        proba = trainer.predict(x)
        return float(proba[0, 1])

    def _score_cnn(self, df_t: pd.DataFrame, trainer) -> float | None:
        w = self.cfg.window
        df_t = df_t.copy()
        if self.cfg.include_ma:
            df_t["MA"] = df_t["Close"].rolling(w).mean()
            df_t = df_t.dropna(subset=["MA"])
        if len(df_t) < w:
            return None
        window_df = df_t.iloc[-w:]
        img = generate_ohlc_image(
            window_df, window=w,
            include_vol=self.cfg.include_vol,
            include_ma=self.cfg.include_ma,
        )
        x = img[np.newaxis, :, :, np.newaxis].astype(np.float32) / 255.0  # (1,H,W,1)
        proba = trainer.predict(x)
        return float(proba[0, 1])

    # ------------------------------------------------------------------
    # Single backtest run
    # ------------------------------------------------------------------

    def _single_backtest(
        self,
        scores_by_date: dict,
        weighting: str,
        port_type: str,
        key: str = "",
        bt_fp: Optional[dict] = None,
    ) -> dict:
        if bt_fp and key:
            cache_pkl  = os.path.join(self._cache_dir, f"{key}_result.pkl")
            cache_json = os.path.join(self._cache_dir, f"{key}_fingerprint.json")
            cached_fp  = load_json(cache_json)
            if fingerprints_match(bt_fp, cached_fp):
                cached = load_pickle(cache_pkl)
                if cached is not None:
                    logger.info("[%s] Backtest result cache hit.", key)
                    return cached

        schedule = self._build_schedule(scores_by_date, weighting, port_type)
        if not schedule:
            raise ValueError("Empty schedule — no rebalancing dates found.")

        borrow_daily = self.cfg.bt_borrow_cost_bps_per_year / 10_000.0 / 252.0

        sim = simulate_portfolio(
            schedule          = schedule,
            daily_returns     = self._daily_ret,
            rf_series         = self._rf,
            cost_bps          = self.cfg.bt_cost_bps,
            borrow_rate_daily = borrow_daily if port_type == "LS" else 0.0,
        )

        bm = self._rf if port_type == "LS" else self._benchmark
        metrics = compute_backtest_metrics(
            sim               = sim,
            rf_series         = self._rf,
            benchmark_returns = bm,
            horizon           = self.cfg.horizon,
            periods_per_year  = 252,
        )
        result = {"sim": sim, "metrics": metrics, "schedule": schedule}

        if bt_fp and key:
            save_pickle(result, os.path.join(self._cache_dir, f"{key}_result.pkl"))
            save_json(bt_fp, os.path.join(self._cache_dir, f"{key}_fingerprint.json"))

        return result

    def _build_schedule(self, scores_by_date: dict, weighting: str, port_type: str) -> list:
        """
        Returns [(entry_date, exit_date, weights_dict), …]
        entry = signal + 1 trading day
        exit  = signal + h trading days
        """
        all_dates = self._daily_ret.index
        schedule  = []

        for sig_t, scores in sorted(scores_by_date.items()):
            future = all_dates[all_dates > sig_t]
            if len(future) < self.cfg.horizon:
                continue
            entry_t = future[0]               # t+1
            exit_t  = future[self.cfg.horizon - 1]  # t+h

            n_decile = max(1, len(scores) // self.cfg.n_deciles)

            betas_at_t     = self._betas_at(sig_t)
            mktcap_at_t    = self._mktcap_at(sig_t)
            sectors_at_t   = self._sectors_at(sig_t)

            if port_type == "LS":
                weights = construct_ls_portfolio(
                    scores      = scores,
                    market_caps = mktcap_at_t,
                    sectors     = sectors_at_t,
                    betas       = betas_at_t,
                    n_decile    = n_decile,
                    weighting   = weighting,
                    neutrality  = self.cfg.bt_neutrality,
                )
            else:
                weights = construct_lo_portfolio(
                    scores      = scores,
                    market_caps = mktcap_at_t,
                    betas       = betas_at_t,
                    n_decile    = n_decile,
                    weighting   = weighting,
                    beta_neutral = "beta" in self.cfg.bt_neutrality,
                )

            schedule.append((entry_t, exit_t, weights))

        return schedule

    def _betas_at(self, t) -> dict:
        if self._betas.empty:
            return {}
        row = self._betas.loc[:t].iloc[-1] if (self._betas.index <= t).any() else pd.Series()
        return {ticker: float(v) for ticker, v in row.items() if not np.isnan(v)}

    def _mktcap_at(self, t) -> dict:
        if self._mktcap.empty:
            return {}
        # Use t-1 (previous day's market cap) to avoid look-ahead
        rows = self._mktcap.loc[:t]
        if len(rows) < 2:
            return {}
        row = rows.iloc[-2]  # one day before t
        return {ticker: float(v) for ticker, v in row.items() if not np.isnan(v)}

    def _sectors_at(self, t) -> dict:
        """Point-in-time sector: most recent gsector for each security as of date t."""
        if self._sectors.empty:
            return {}
        rows = self._sectors.loc[:t]
        if rows.empty:
            return {}
        # Forward-fill within the slice: sector changes are sparse
        row = rows.ffill().iloc[-1]
        result = {}
        for id_val, val in row.items():
            if not pd.isna(val):
                try:
                    result[str(id_val)] = int(val)
                except (ValueError, TypeError):
                    pass
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_results(self, results: dict) -> None:
        self._plot_cumulative(results)
        self._plot_metrics_table(results)

    def _plot_cumulative(self, results: dict) -> None:
        _BENCHMARK_NAMES = {"MOM", "STR", "WSTR"}
        fig, ax = plt.subplots(figsize=(12, 5))
        for key, res in results.items():
            sim = res["sim"]
            is_bench = key.split("_")[0] in _BENCHMARK_NAMES
            ax.plot(
                sim.index, sim["portfolio_value_net"],
                label=key,
                lw=1.5 if is_bench else 1.0,
                ls="--" if is_bench else "-",
                alpha=0.85,
            )
        ax.set_title("Cumulative portfolio value (net of costs)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio value (start = 1)")
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.cfg.results_dir, "backtest_cumulative.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure: %s", path)

    def _plot_metrics_table(self, results: dict) -> None:
        rows = []
        for key, res in results.items():
            m = res["metrics"]
            rows.append({
                "Strategy": key,
                "Net Ann. Ret.": f"{m.get('net_ann_return', np.nan):.3f}",
                "Net Sharpe":    f"{m.get('net_sharpe', np.nan):.3f}",
                "Net IR":        f"{m.get('net_IR', np.nan):.3f}",
                "MDD":           f"{m.get('net_max_drawdown', np.nan):.3f}",
                "Calmar":        f"{m.get('net_calmar', np.nan):.3f}",
                "Win Rate":      f"{m.get('net_win_rate', np.nan):.3f}",
                "Ann. Turnover": f"{m.get('ann_turnover', np.nan):.2f}",
            })
        df = pd.DataFrame(rows).set_index("Strategy")

        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), len(df) * 0.5 + 1.5))
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        plt.title("Backtest performance summary", fontsize=12, pad=10)
        plt.tight_layout()
        path = os.path.join(self.cfg.results_dir, "backtest_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure: %s", path)

    def _log_summary(self, results: dict) -> None:
        logger.info("=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        header = f"{'Strategy':<40} {'Net Sharpe':>10} {'Net Ann Ret':>12} {'MDD':>8} {'IR':>8}"
        logger.info(header)
        logger.info("-" * 70)
        for key, res in results.items():
            m = res["metrics"]
            logger.info(
                "%-40s %10.3f %12.3f %8.3f %8.3f",
                key,
                m.get("net_sharpe",     np.nan),
                m.get("net_ann_return", np.nan),
                m.get("net_max_drawdown", np.nan),
                m.get("net_IR",         np.nan),
            )
        logger.info("=" * 70)