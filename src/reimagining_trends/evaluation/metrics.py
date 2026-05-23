"""
metrics.py
----------
Full model evaluation and comparison.
"""

import logging
import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_ml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Log a formatted sklearn classification report."""
    logger.info("=" * 40)
    logger.info("  %s", model_name)
    logger.info("=" * 40)
    logger.info(
        "\n%s",
        classification_report(y_true, y_pred, target_names=["DOWN", "UP"]),
    )


def annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 52,
    risk_free: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        return 0.0

    excess = returns - risk_free / periods_per_year
    mean = excess.mean()
    std = excess.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float(mean / std * np.sqrt(periods_per_year))


def decile_portfolio_returns(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    n_deciles: int = 10,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Sort samples into deciles by predicted up-probability.

    y_true is kept for backward compatibility with the existing tests.
    """
    _ = y_true

    df = pd.DataFrame(
        {
            "prob_up": y_proba,
            "return": returns,
        }
    )

    df["decile"] = pd.qcut(
        df["prob_up"],
        n_deciles,
        labels=False,
        duplicates="drop",
    ) + 1

    results = []
    for decile in sorted(df["decile"].dropna().unique()):
        subset = df[df["decile"] == decile]["return"].values
        results.append(
            {
                "decile": int(decile),
                "mean_return": subset.mean(),
                "std": subset.std(),
                "sharpe": annualized_sharpe(subset, periods_per_year),
                "n": len(subset),
            }
        )

    return pd.DataFrame(results).set_index("decile")


def hl_sharpe(
    decile_df: pd.DataFrame,
    y_proba: np.ndarray,
    returns: np.ndarray,
    n_deciles: int = 10,
    periods_per_year: int = 52,
) -> float:
    """
    Compute annualized Sharpe ratio of the H-L strategy.

    H-L means long the highest probability decile and short the lowest one.
    """
    _ = decile_df

    df = pd.DataFrame(
        {
            "prob_up": y_proba,
            "return": returns,
        }
    )

    df["decile"] = pd.qcut(
        df["prob_up"],
        n_deciles,
        labels=False,
        duplicates="drop",
    ) + 1

    if df["decile"].isna().all():
        return 0.0

    low_decile = int(df["decile"].min())
    high_decile = int(df["decile"].max())

    high_returns = df[df["decile"] == high_decile]["return"].values
    low_returns = df[df["decile"] == low_decile]["return"].values

    min_len = min(len(high_returns), len(low_returns))
    if min_len == 0:
        return 0.0

    hl_returns = high_returns[:min_len] - low_returns[:min_len]

    return annualized_sharpe(hl_returns, periods_per_year)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from per-model metric dictionaries."""
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("auc", ascending=False)


def plot_decile_returns(
    decile_df: pd.DataFrame,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot mean return and Sharpe ratio per decile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_ret = [
        "#d73027" if value < 0 else "#1a9850"
        for value in decile_df["mean_return"]
    ]
    axes[0].bar(
        decile_df.index,
        decile_df["mean_return"],
        color=colors_ret,
        edgecolor="white",
    )
    axes[0].set_title(f"{model_name} — Mean return per decile")
    axes[0].set_xlabel("Decile")
    axes[0].set_ylabel("Mean return")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].grid(axis="y", alpha=0.3)

    colors_sharpe = [
        "#d73027" if value < 0 else "#1a9850"
        for value in decile_df["sharpe"]
    ]
    axes[1].bar(
        decile_df.index,
        decile_df["sharpe"],
        color=colors_sharpe,
        edgecolor="white",
    )
    axes[1].set_title(f"{model_name} — Annualized Sharpe ratio per decile")
    axes[1].set_xlabel("Decile")
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap comparing ML metrics across models."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(comparison_df) * 0.8)))

    data = comparison_df.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(comparison_df.columns)))
    ax.set_xticklabels(comparison_df.columns, fontsize=10)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df.index, fontsize=10)

    for i in range(len(comparison_df)):
        for j in range(len(comparison_df.columns)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Model comparison", fontsize=13, pad=15)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_cumulative_returns(
    strategies: dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative returns for several strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, (name, returns) in enumerate(strategies.items()):
        cumulative = np.cumprod(1 + returns) - 1
        ax.plot(cumulative, label=name, color=colors[i % 10], linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative returns — H-L strategies", fontsize=13)
    ax.set_xlabel("Period")
    ax.set_ylabel("Cumulative return")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_gradcam_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    label: int,
    pred: int,
    save_path: Optional[str] = None,
) -> None:
    """Overlay a Grad-CAM heatmap on the original OHLC image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title("Original OHLC image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", aspect="auto", interpolation="nearest")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    rgb = np.stack([image, image, image], axis=-1)
    cam_colored = cm.jet(cam)[..., :3]
    overlay = 0.5 * rgb + 0.5 * cam_colored

    correct = "✓" if label == pred else "✗"
    color = "green" if label == pred else "red"

    axes[2].imshow(overlay, aspect="auto", interpolation="nearest")
    axes[2].set_title(
        f"Overlay — True: {'UP' if label == 1 else 'DOWN'} | "
        f"Pred: {'UP' if pred == 1 else 'DOWN'} {correct}",
        color=color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    cm_arr = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm_arr, display_labels=["DOWN", "UP"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {model_name}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def full_evaluation(
    model_results: dict[str, dict],
    save_dir: Optional[str] = None,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """Generate a complete evaluation report."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ml_metrics = {}
    hl_sharpes = {}

    for name, result in model_results.items():
        ml_metrics[name] = compute_ml_metrics(
            result["y_true"],
            result["y_pred"],
            result["y_proba"],
        )

        print_classification_report(
            result["y_true"],
            result["y_pred"],
            model_name=name,
        )

        plot_confusion_matrix(
            result["y_true"],
            result["y_pred"],
            model_name=name,
            save_path=(
                os.path.join(save_dir, f"{name}_confusion.png")
                if save_dir
                else None
            ),
        )

        if "returns" in result:
            deciles = decile_portfolio_returns(
                result["y_true"],
                result["y_proba"],
                result["returns"],
                periods_per_year=periods_per_year,
            )
            hl_sharpes[name] = hl_sharpe(
                deciles,
                result["y_proba"],
                result["returns"],
                periods_per_year=periods_per_year,
            )
            plot_decile_returns(
                deciles,
                model_name=name,
                save_path=(
                    os.path.join(save_dir, f"{name}_deciles.png")
                    if save_dir
                    else None
                ),
            )

    comparison_df = compare_models(ml_metrics)

    logger.info("Global model comparison")
    logger.info("\n%s", comparison_df.to_string())

    if hl_sharpes:
        logger.info("H-L annualized Sharpe ratios")
        for name, sharpe in hl_sharpes.items():
            logger.info("%s: %.4f", name, sharpe)

    plot_model_comparison(
        comparison_df,
        save_path=(
            os.path.join(save_dir, "model_comparison.png")
            if save_dir
            else None
        ),
    )

    return comparison_df