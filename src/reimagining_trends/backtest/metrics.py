"""
metrics.py
----------
Backtest performance metrics — gross and net.
"""

import numpy as np
import pandas as pd


def _annualize(r: pd.Series, periods: int = 252) -> float:
    """Compound annualized return from daily return series."""
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0) if len(r) > 0 else np.nan


def _annvol(r: pd.Series, periods: int = 252) -> float:
    return float(r.std() * np.sqrt(periods)) if len(r) > 1 else np.nan


def _max_drawdown(pv: pd.Series) -> float:
    roll_max = pv.cummax()
    dd = (pv - roll_max) / roll_max
    return float(dd.min())


def _monthly_win_rate(r: pd.Series) -> float:
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    if len(monthly) == 0:
        return np.nan
    return float((monthly > 0).mean())


def compute_backtest_metrics(
    sim: pd.DataFrame,
    rf_series: pd.Series,
    benchmark_returns: pd.Series,
    horizon: int,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute gross and net performance metrics.

    Parameters
    ----------
    sim               : simulator output (date-indexed, gross_return / net_return columns)
    rf_series         : daily risk-free rate
    benchmark_returns : daily benchmark return (CRSP mktcap-weighted or rf for LS)
    horizon           : rebalancing frequency (for annualized turnover)
    periods_per_year  : 252 for daily

    Returns
    -------
    dict of metric_name → value (both gross_ and net_ prefixed)
    """
    rf  = rf_series.reindex(sim.index).fillna(0.0)
    bm  = benchmark_returns.reindex(sim.index).fillna(0.0)

    out = {}
    for tag, col in [("gross", "gross_return"), ("net", "net_return")]:
        r = sim[col]
        pv = sim[f"portfolio_value_{tag}"]

        ann_ret = _annualize(r, periods_per_year)
        ann_vol = _annvol(r, periods_per_year)
        excess  = r - rf

        sharpe  = float(excess.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan

        active  = r - bm
        ir      = float(active.mean() / active.std() * np.sqrt(periods_per_year)) if active.std() > 0 else np.nan

        mdd     = _max_drawdown(pv)
        calmar  = ann_ret / abs(mdd) if mdd != 0 else np.nan
        wr      = _monthly_win_rate(r)

        out[f"{tag}_ann_return"]   = ann_ret
        out[f"{tag}_ann_vol"]      = ann_vol
        out[f"{tag}_sharpe"]       = sharpe
        out[f"{tag}_max_drawdown"] = mdd
        out[f"{tag}_calmar"]       = calmar
        out[f"{tag}_IR"]           = ir
        out[f"{tag}_win_rate"]     = wr

    # Turnover (annualized)
    rebal_turns = sim.loc[sim["is_rebalance"], "turnover"]
    if len(rebal_turns) > 0:
        n_rebal_per_year = periods_per_year / horizon
        out["ann_turnover"] = float(rebal_turns.mean() * n_rebal_per_year)
    else:
        out["ann_turnover"] = np.nan

    return out