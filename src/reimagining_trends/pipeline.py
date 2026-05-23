"""
pipeline.py
-----------
Pipeline — single class that owns all state and orchestrates the full run:
  data → images → training → evaluation → interpretability → transfer → summary.

Usage
-----
    from reimagining_trends.pipeline import Pipeline
    from reimagining_trends.utils.config import Config

    Pipeline(Config()).run()
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from reimagining_trends.data.data_manager import DataManager
from reimagining_trends.data.fetch_data import (
    add_moving_average,
    cumret_scale,
    image_scale,
    make_labels,
)
from reimagining_trends.evaluation.evaluator import Evaluator
from reimagining_trends.evaluation.metrics import compute_ml_metrics
from reimagining_trends.imaging.ohlc_chart import IMAGE_SPECS, generate_ohlc_image, visualize_grid
from reimagining_trends.models.cnn import build_cnn
from reimagining_trends.models.lstm import build_attention_lstm, build_gru, build_lstm
from reimagining_trends.models.mlp import build_mlp
from reimagining_trends.training.train import Trainer, get_device, plot_history, set_seed
from reimagining_trends.utils.cache import model_fp as build_model_fp
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full price-trend prediction pipeline.

    Parameters
    ----------
    config : Config

    Attributes
    ----------
    raw_data  : dict[str, pd.DataFrame]   — downloaded OHLCV data
    tab_ds    : dict                       — tabular (MLP/GRU/LSTM) dataset
    img_ds    : dict                       — image (CNN) dataset
    trainers  : dict[str, Trainer]         — trained model wrappers
    evaluator : Evaluator                  — evaluation & plotting engine
    """

    def __init__(self, config: Config) -> None:
        self.cfg      = config
        self.device   = get_device()
        self.data_mgr = DataManager(config)
        self.evaluator: Evaluator | None = None

        self.raw_data: dict[str, pd.DataFrame] = {}
        self.tab_ds:   dict = {}
        self.img_ds:   dict = {}
        self.trainers: dict[str, Trainer] = {}

        os.makedirs(config.results_dir,    exist_ok=True)
        os.makedirs(config.checkpoints_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all pipeline stages in order."""
        set_seed(self.cfg.seed)
        logger.info("Config : %r", self.cfg)
        logger.info("Device : %s", self.device)
        logger.info("PyTorch: %s", torch.__version__)

        self._section_data()
        self._section_images()
        self._section_training()
        self._section_evaluation()

        if "CNN" in self.trainers:
            self._section_interpretability()

        self._section_transfer_learning()
        self._section_backtest()
        self._section_summary()

    # ------------------------------------------------------------------
    # Private section methods
    # ------------------------------------------------------------------

    def _section_data(self) -> None:
        logger.info("=" * 60)
        logger.info("2. Data and representations")
        logger.info("=" * 60)

        self.raw_data = self.data_mgr.download()

        ticker_sample = list(self.raw_data.keys())[0]
        df_ex = self.raw_data[ticker_sample].iloc[-60:].copy()
        df_img = image_scale(df_ex, self.cfg.window)
        df_cum = cumret_scale(df_ex, self.cfg.window)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(df_img["Close"], label="Image scale",  color="steelblue")
        axes[0].plot(df_cum["Close"], label="Cumret scale", color="tomato", linestyle="--")
        axes[0].set_title(f"{ticker_sample} — Normalised Close (last 60 days)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(df_img["Volume"], label="Image scale",  color="steelblue")
        axes[1].plot(df_cum["Volume"], label="Cumret scale", color="tomato", linestyle="--")
        axes[1].set_title("Normalised volume")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        self._save_fig("normalisations.png")

        logger.info("-> Image scale maps all stocks to a common [0,1] range.")
        logger.info("-> Figure: %s", self._path("normalisations.png"))

    def _section_images(self) -> None:
        logger.info("=" * 60)
        logger.info("3. OHLC image generation")
        logger.info("=" * 60)

        # Pick the first security that has enough data for the largest window (60d)
        ticker_sample = next(
            (k for k, v in self.raw_data.items() if len(v) >= 120),
            list(self.raw_data.keys())[0],
        )
        df_sample = self.raw_data[ticker_sample]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, w in zip(axes, [5, 20, 60]):
            df_w = add_moving_average(df_sample, window=w).dropna()
            img  = generate_ohlc_image(df_w.iloc[-w:], window=w, include_vol=True, include_ma=True)
            specs = IMAGE_SPECS[w]
            ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")
            ax.set_title(f"Window {w}d — {specs['height']}x{specs['width']} px", fontsize=11)
            ax.axis("off")
        plt.suptitle("Generated OHLC images (black background, white objects)", fontsize=13, y=1.02)
        plt.tight_layout()
        self._save_fig("ohlc_images.png")

        labels_5  = make_labels(df_sample, horizon=5)
        labels_20 = make_labels(df_sample, horizon=20)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for ax, lbl, h in zip(axes, [labels_5, labels_20], [5, 20]):
            counts = lbl.value_counts().sort_index()
            ax.bar(["DOWN (0)", "UP (1)"], counts.values,
                   color=["#d73027", "#1a9850"], edgecolor="white")
            ax.set_title(f"Labels — horizon {h}d ({ticker_sample})")
            pct = counts[1] / counts.sum() * 100
            ax.set_ylabel("Count")
            ax.text(0.5, 0.92, f"{pct:.1f}% positive", transform=ax.transAxes,
                    ha="center", fontsize=10, color="gray")
        plt.tight_layout()
        self._save_fig("label_distribution.png")

        preview_data = {k: self.raw_data[k] for k in list(self.raw_data.keys())[:3]}
        img_ds_preview = DataManager(self.cfg).build_image(raw_data=preview_data)
        visualize_grid(
            img_ds_preview["X_train"], img_ds_preview["y_train"],
            n=12, save_path=self._path("image_grid.png"),
        )
        logger.info("Image dataset preview: %d training images.", img_ds_preview["X_train"].shape[0])

    def _section_training(self) -> None:
        logger.info("=" * 60)
        logger.info("4. Model training")
        logger.info("=" * 60)

        self.tab_ds = self.data_mgr.build_tabular()
        self.img_ds = self.data_mgr.build_image()
        N_FEAT = self.tab_ds["X_train"].shape[2]
        logger.info("Features per time step: %d", N_FEAT)

        fit_kwargs = dict(
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            patience=self.cfg.patience,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if "MLP" in self.cfg.models_to_train:
            logger.info("--- MLP ---")
            model = build_mlp(window=self.cfg.window, n_features=N_FEAT,
                              hidden_dims=self.cfg.mlp_hidden_dims, dropout=self.cfg.mlp_dropout)
            self.trainers["MLP"] = self._build_trainer(model, "mlp", "MLP")
            fp = build_model_fp(self.cfg, "mlp", {
                "hidden_dims": self.cfg.mlp_hidden_dims, "dropout": self.cfg.mlp_dropout,
                "n_features": N_FEAT,
            })
            hist = self.trainers["MLP"].fit(
                self.tab_ds["X_train"], self.tab_ds["y_train"],
                self.tab_ds["X_val"],   self.tab_ds["y_val"], **fit_kwargs, fingerprint=fp)
            plot_history(hist, "MLP", save_path=self._path("history_mlp.png"))

        if "GRU" in self.cfg.models_to_train:
            logger.info("--- GRU ---")
            model = build_gru(n_features=N_FEAT, hidden_size=self.cfg.gru_hidden_size,
                              num_layers=self.cfg.gru_num_layers, dropout=self.cfg.gru_dropout)
            self.trainers["GRU"] = self._build_trainer(model, "gru", "GRU")
            fp = build_model_fp(self.cfg, "gru", {
                "hidden_size": self.cfg.gru_hidden_size, "num_layers": self.cfg.gru_num_layers,
                "dropout": self.cfg.gru_dropout, "n_features": N_FEAT,
            })
            hist = self.trainers["GRU"].fit(
                self.tab_ds["X_train"], self.tab_ds["y_train"],
                self.tab_ds["X_val"],   self.tab_ds["y_val"], **fit_kwargs, fingerprint=fp)
            plot_history(hist, "GRU", save_path=self._path("history_gru.png"))

        if "LSTM" in self.cfg.models_to_train:
            logger.info("--- LSTM ---")
            model = build_lstm(n_features=N_FEAT, hidden_size=self.cfg.lstm_hidden_size,
                               num_layers=self.cfg.lstm_num_layers, dropout=self.cfg.lstm_dropout,
                               bidirectional=self.cfg.lstm_bidirectional)
            self.trainers["LSTM"] = self._build_trainer(model, "lstm", "LSTM")
            fp = build_model_fp(self.cfg, "lstm", {
                "hidden_size": self.cfg.lstm_hidden_size, "num_layers": self.cfg.lstm_num_layers,
                "dropout": self.cfg.lstm_dropout, "bidirectional": self.cfg.lstm_bidirectional,
                "n_features": N_FEAT,
            })
            hist = self.trainers["LSTM"].fit(
                self.tab_ds["X_train"], self.tab_ds["y_train"],
                self.tab_ds["X_val"],   self.tab_ds["y_val"], **fit_kwargs, fingerprint=fp)
            plot_history(hist, "LSTM", save_path=self._path("history_lstm.png"))

        if "CNN" in self.cfg.models_to_train:
            logger.info("--- CNN ---")
            model = build_cnn(window=self.cfg.window, dropout=self.cfg.cnn_dropout)
            self.trainers["CNN"] = self._build_trainer(model, "cnn", "CNN")
            fp = build_model_fp(self.cfg, "cnn", {
                "dropout": self.cfg.cnn_dropout, "window": self.cfg.window,
            })
            hist = self.trainers["CNN"].fit(
                self.img_ds["X_train"], self.img_ds["y_train"],
                self.img_ds["X_val"],   self.img_ds["y_val"],
                **{**fit_kwargs, "batch_size": 32}, fingerprint=fp)
            plot_history(hist, "CNN", save_path=self._path("history_cnn.png"))

    def _section_evaluation(self) -> None:
        logger.info("=" * 60)
        logger.info("5. Evaluation and comparison")
        logger.info("=" * 60)

        self.evaluator = Evaluator(self.cfg)
        self.evaluator.evaluate(self.trainers, self.tab_ds, self.img_ds)
        self.evaluator.plot_all()

        cmp_df = self.evaluator.summary()
        logger.info("\n%s", cmp_df.to_string())

    def _section_interpretability(self) -> None:
        logger.info("=" * 60)
        logger.info("6. Interpretability")
        logger.info("=" * 60)

        self.evaluator.plot_gradcam(self.trainers["CNN"], self.img_ds, self.device)

        N_FEAT = self.tab_ds["X_train"].shape[2]
        logger.info("Training Attention-LSTM...")
        attn_lstm = build_attention_lstm(
            n_features=N_FEAT,
            hidden_size=self.cfg.attn_hidden_size,
            num_layers=self.cfg.attn_num_layers,
            dropout=self.cfg.attn_dropout,
        )
        trainer_attn = self._build_trainer(attn_lstm, "lstm", "attn_lstm")
        trainer_attn.fit(
            self.tab_ds["X_train"], self.tab_ds["y_train"],
            self.tab_ds["X_val"],   self.tab_ds["y_val"],
            epochs=self.cfg.epochs, batch_size=self.cfg.batch_size,
            patience=self.cfg.patience, verbose=False,
        )

        attn_lstm.eval()
        X_sample = torch.tensor(self.tab_ds["X_test"][:100], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, weights = attn_lstm(X_sample)
        mean_weights = weights.cpu().numpy().mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, self.cfg.window + 1), mean_weights,
               color="steelblue", edgecolor="white")
        ax.set_xlabel(f"Time step (1 = oldest, {self.cfg.window} = most recent)")
        ax.set_ylabel("Mean attention weight")
        ax.set_title("Attention LSTM — Mean temporal importance")
        ax.axvline(self.cfg.window, color="red", lw=1.5, ls="--", label="Most recent day")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_fig("attention_weights.png")
        logger.info("-> Attention weights: %s", self._path("attention_weights.png"))

    def _section_transfer_learning(self) -> None:
        logger.info("=" * 60)
        logger.info("7. Transfer learning")
        logger.info("=" * 60)

        keys = list(self.raw_data.keys())
        split_idx    = max(1, len(keys) - 5)
        us_tickers   = keys[:split_idx]
        intl_tickers = keys[split_idx:]

        if not intl_tickers:
            logger.warning("Not enough tickers for transfer learning split — skipping.")
            return

        logger.info("US tickers   : %s", us_tickers)
        logger.info("Intl tickers : %s", intl_tickers)

        us_data   = {k: self.raw_data[k] for k in us_tickers}
        intl_data = {k: self.raw_data[k] for k in intl_tickers}

        us_img_ds   = DataManager(self.cfg).build_image(raw_data=us_data)
        intl_img_ds = DataManager(self.cfg).build_image(raw_data=intl_data)

        cnn_us = build_cnn(window=self.cfg.window, dropout=self.cfg.cnn_dropout)
        tr_us  = self._build_trainer(cnn_us, "cnn", "cnn_us")
        tr_us.fit(us_img_ds["X_train"], us_img_ds["y_train"],
                  us_img_ds["X_val"],   us_img_ds["y_val"],
                  epochs=self.cfg.epochs, batch_size=32,
                  patience=self.cfg.patience, verbose=False)

        cnn_local = build_cnn(window=self.cfg.window, dropout=self.cfg.cnn_dropout)
        tr_local  = self._build_trainer(cnn_local, "cnn", "cnn_local")
        tr_local.fit(intl_img_ds["X_train"], intl_img_ds["y_train"],
                     intl_img_ds["X_val"],   intl_img_ds["y_val"],
                     epochs=self.cfg.epochs, batch_size=32,
                     patience=self.cfg.patience, verbose=False)

        tr_us.load_best()
        tr_local.load_best()
        X_test, y_test = intl_img_ds["X_test"], intl_img_ds["y_test"]

        prob_t = tr_us.predict(X_test)[:, 1]
        prob_l = tr_local.predict(X_test)[:, 1]
        met_t  = compute_ml_metrics(y_test, (prob_t > 0.5).astype(int), prob_t)
        met_l  = compute_ml_metrics(y_test, (prob_l > 0.5).astype(int), prob_l)

        df_cmp = pd.DataFrame({"Transfer (US→Intl)": met_t, "Local retrain": met_l}).T
        logger.info("\n%s", df_cmp.round(4).to_string())

        fig, ax = plt.subplots(figsize=(8, 3))
        df_cmp.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_title("Transfer learning vs local retrain")
        ax.set_xticklabels(df_cmp.index, rotation=0)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_fig("transfer_learning.png")
        logger.info("-> Figure: %s", self._path("transfer_learning.png"))

    def _section_backtest(self) -> None:
        logger.info("=" * 60)
        logger.info("9. Backtesting")
        logger.info("=" * 60)

        from reimagining_trends.backtest.backtester import Backtester

        bt = Backtester(self.cfg, self.raw_data, self.trainers)
        bt.run()

    def _section_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("8. Final summary")
        logger.info("=" * 60)

        df = self.evaluator.summary()
        df.index.name = "Model"
        logger.info("=" * 55)
        logger.info("  SUMMARY TABLE — TEST SET")
        logger.info("=" * 55)
        logger.info("\n%s", df.round(4).to_string())
        logger.info("=" * 55)
        logger.info("All figures saved to: %s/", self.cfg.results_dir)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self.cfg.results_dir, filename)

    def _save_fig(self, filename: str) -> None:
        plt.savefig(self._path(filename), dpi=150, bbox_inches="tight")
        plt.close()

    def _build_trainer(self, model, model_type: str, name: str) -> Trainer:
        save_dir = os.path.join(self.cfg.checkpoints_dir, f"{name.lower()}_w{self.cfg.window}")
        return Trainer(model, model_type, save_dir=save_dir, device=self.device, model_name=name)
