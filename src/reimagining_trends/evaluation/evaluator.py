"""
evaluator.py
------------
Evaluator — owns the full evaluation and reporting pipeline:
  predict → compute metrics → plot comparisons → summary table.
"""

import logging
import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from reimagining_trends.evaluation.metrics import (
    annualized_sharpe,
    compare_models,
    compute_ml_metrics,
    decile_portfolio_returns,
    plot_model_comparison,
)
from reimagining_trends.models.cnn import GradCAM
from reimagining_trends.training.train import Trainer
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs evaluation for all trained models and produces figures.

    Parameters
    ----------
    config : Config

    Attributes
    ----------
    results : dict
        Populated after :meth:`evaluate`.  Keys are model names; values
        contain y_true, y_pred, y_proba, accuracy, f1, auc, brier.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.results: dict = {}
        os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trainers: dict[str, Trainer],
        tab_ds: dict,
        img_ds: dict,
    ) -> dict:
        """
        Run inference on the test set for every trained model.

        Parameters
        ----------
        trainers : {model_name: Trainer}
        tab_ds   : tabular dataset (used by MLP, LSTM, GRU)
        img_ds   : image dataset   (used by CNN)

        Returns
        -------
        self.results
        """
        datasets = {
            "MLP":  tab_ds,
            "GRU":  tab_ds,
            "LSTM": tab_ds,
            "CNN":  img_ds,
        }
        self.results = {}

        for name, trainer in trainers.items():
            ds = datasets.get(name, tab_ds)
            trainer.load_best()
            proba  = trainer.predict(ds["X_test"])
            y_prob = proba[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            y_true = ds["y_test"]
            metrics = compute_ml_metrics(y_true, y_pred, y_prob)
            self.results[name] = {
                "y_true":   y_true,
                "y_pred":   y_pred,
                "y_proba":  y_prob,
                "returns":  ds.get("returns_test"),
                **metrics,
            }
            logger.info(
                "%s | acc=%.3f | auc=%.3f | f1=%.3f | brier=%.3f",
                name, metrics["accuracy"], metrics["auc"],
                metrics["f1"], metrics["brier"],
            )

        return self.results

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame comparing all models by ML metrics."""
        if not self.results:
            raise RuntimeError("Call evaluate() first.")
        ml_only = {
            k: {m: v for m, v in r.items() if m in ["accuracy", "f1", "auc", "brier"]}
            for k, r in self.results.items()
        }
        return compare_models(ml_only)

    def plot_all(self) -> None:
        """Generate and save all comparison figures."""
        if not self.results:
            raise RuntimeError("Call evaluate() first.")
        self._plot_comparison_heatmap()
        self._plot_confusion_matrices()
        self._plot_decile_returns()

    def plot_gradcam(
        self,
        cnn_trainer: Trainer,
        img_ds: dict,
        device: torch.device,
        n_samples: int = 3,
    ) -> None:
        """Grad-CAM grid for correct and incorrect CNN predictions."""
        cnn_model = cnn_trainer.model
        cnn_model.eval()
        gradcam = GradCAM(cnn_model)

        X_test = img_ds["X_test"]
        y_test = img_ds["y_test"]
        proba  = cnn_trainer.predict(X_test)[:, 1]
        y_pred = (proba > 0.5).astype(int)

        correct_idx   = np.where(y_pred == y_test)[0][:n_samples]
        incorrect_idx = np.where(y_pred != y_test)[0][:n_samples]
        all_idx = list(correct_idx) + list(incorrect_idx)

        fig, axes_grid = plt.subplots(len(all_idx), 3, figsize=(15, len(all_idx) * 3.5))
        if len(all_idx) == 1:
            axes_grid = axes_grid[np.newaxis, :]

        for row, idx in enumerate(all_idx):
            x_tensor = torch.tensor(
                X_test[idx:idx + 1].transpose(0, 3, 1, 2), dtype=torch.float32
            ).to(device)
            cam   = gradcam.generate(x_tensor)
            image = X_test[idx].squeeze()
            label = y_test[idx]
            pred  = y_pred[idx]

            axes_grid[row, 0].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
            axes_grid[row, 0].set_title(f"OHLC — {'UP' if label else 'DOWN'}", fontsize=9)
            axes_grid[row, 0].axis("off")

            axes_grid[row, 1].imshow(cam.cpu().numpy(), cmap="jet", aspect="auto")
            axes_grid[row, 1].set_title("Grad-CAM", fontsize=9)
            axes_grid[row, 1].axis("off")

            rgb = np.stack([image, image, image], axis=-1)
            cam_col = cm.jet(cam.cpu().numpy())[..., :3]
            overlay = 0.5 * rgb + 0.5 * cam_col
            color = "green" if pred == label else "red"
            axes_grid[row, 2].imshow(overlay, aspect="auto")
            axes_grid[row, 2].set_title(
                f"Pred: {'UP' if pred else 'DOWN'} {'✓' if pred == label else '✗'}",
                color=color, fontsize=9,
            )
            axes_grid[row, 2].axis("off")

        plt.suptitle("Grad-CAM — active regions in OHLC images", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save_fig("gradcam.png")
        logger.info("Grad-CAM saved.")

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------

    def _plot_comparison_heatmap(self) -> None:
        cmp_df = self.summary()
        plot_model_comparison(cmp_df, save_path=self._path("model_comparison.png"))
        logger.info("\n%s", cmp_df.to_string())

    def _plot_confusion_matrices(self) -> None:
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, res) in zip(axes, self.results.items()):
            cm_arr = confusion_matrix(res["y_true"], res["y_pred"])
            disp = ConfusionMatrixDisplay(cm_arr, display_labels=["DOWN", "UP"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name, fontsize=12)
        plt.suptitle("Confusion matrices — test set", fontsize=13, y=1.02)
        plt.tight_layout()
        self._save_fig("confusion_matrices.png")

    def _plot_decile_returns(self) -> None:
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
        if n == 1:
            axes = [axes]
        for ax, (name, res) in zip(axes, self.results.items()):
            y_true  = res["y_true"]
            returns = res.get("returns")

            if returns is None or np.all(np.isnan(returns)):
                logger.warning("%s: no real returns available — skipping decile plot", name)
                ax.set_title(f"{name}\n(no returns)", fontsize=11)
                ax.axis("off")
                continue

            dec = decile_portfolio_returns(
                y_true, res["y_proba"], returns, n_deciles=self.cfg.n_deciles
            )
            colors = ["#d73027" if v < 0 else "#1a9850" for v in dec["mean_return"]]
            ax.bar(dec.index, dec["mean_return"], color=colors, edgecolor="white")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            hl = dec.loc[10, "mean_return"] - dec.loc[1, "mean_return"]
            ax.set_title(f"{name}\nH-L = {hl:.4f}", fontsize=11)
            ax.set_xlabel("Decile")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Mean return")
        plt.suptitle("Mean return per decile — P(UP) sort strategy", fontsize=13, y=1.02)
        plt.tight_layout()
        self._save_fig("decile_returns.png")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self.cfg.results_dir, filename)

    def _save_fig(self, filename: str) -> None:
        path = self._path(filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure saved: %s", path)
