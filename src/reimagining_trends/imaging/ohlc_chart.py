"""
ohlc_chart.py
-------------
Generate black-and-white OHLC images from price data.
"""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from reimagining_trends.data.fetch_data import (
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
)

logger = logging.getLogger(__name__)

IMAGE_SPECS = {
    5: {"width": 32, "height": 15},
    20: {"width": 64, "height": 60},
    60: {"width": 96, "height": 180},
}

VOLUME_FRACTION = 0.20


def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns if needed."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _normalize_prices(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ma: Optional[np.ndarray] = None,
) -> tuple:
    """Normalize prices to [0, 1]."""
    all_prices = np.concatenate([highs, lows])

    if ma is not None:
        valid_ma = ma[~np.isnan(ma)]
        if len(valid_ma) > 0:
            all_prices = np.concatenate([all_prices, valid_ma])

    p_max = all_prices.max()
    p_min = all_prices.min()

    denom = p_max - p_min
    if denom == 0:
        denom = 1.0

    def norm(x):
        return (x - p_min) / denom

    return (
        norm(opens),
        norm(highs),
        norm(lows),
        norm(closes),
        norm(ma) if ma is not None else None,
    )


def _normalize_volume(volumes: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1]."""
    v_max = volumes.max()

    if v_max <= 1e-8:
        return np.zeros_like(volumes)

    return volumes / v_max


def _draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    h: int,
    w: int,
) -> None:
    """Draw a line with Bresenham algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = 255

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy


def generate_ohlc_image(
    df: pd.DataFrame,
    window: int = 20,
    include_vol: bool = True,
    include_ma: bool = True,
) -> np.ndarray:
    """
    Generate one OHLC image from a price window.
    """
    assert len(df) == window
    assert window in IMAGE_SPECS

    df = _ensure_flat_columns(df)

    specs = IMAGE_SPECS[window]

    width = specs["width"]
    height = specs["height"]

    volume_height = int(height * VOLUME_FRACTION)
    price_height = height - volume_height

    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    volumes = (
        df["Volume"].values.astype(float)
        if include_vol
        else None
    )

    ma = None
    if include_ma and "MA" in df.columns:
        ma = df["MA"].values.astype(float)

    opens_n, highs_n, lows_n, closes_n, ma_n = _normalize_prices(
        opens,
        highs,
        lows,
        closes,
        ma,
    )

    img = np.zeros((height, width), dtype=np.uint8)

    day_width = 3
    x_start = (width - window * day_width) // 2

    def to_px(value: float, h: int) -> int:
        return int((1.0 - np.clip(value, 0, 1)) * (h - 1))

    # Draw OHLC bars only inside price region
    for i in range(window):
        x_open = x_start + i * day_width
        x_bar = x_start + i * day_width + 1
        x_close = x_start + i * day_width + 2

        if x_close >= width:
            break

        y_high = to_px(highs_n[i], price_height)
        y_low = to_px(lows_n[i], price_height)
        y_open = to_px(opens_n[i], price_height)
        y_close = to_px(closes_n[i], price_height)

        y_high = np.clip(y_high, 0, price_height - 1)
        y_low = np.clip(y_low, 0, price_height - 1)
        y_open = np.clip(y_open, 0, price_height - 1)
        y_close = np.clip(y_close, 0, price_height - 1)

        img[y_high:y_low + 1, x_bar] = 255
        img[y_open, x_open] = 255
        img[y_close, x_close] = 255

    # Moving average only inside price region
    if ma_n is not None:
        prev_px = None

        for i in range(window):
            if np.isnan(ma_n[i]):
                prev_px = None
                continue

            x = x_start + i * day_width + 1
            y = to_px(ma_n[i], price_height)

            y = np.clip(y, 0, price_height - 1)

            if 0 <= x < width:
                img[y, x] = 255

                if prev_px is not None:
                    x0, y0 = prev_px
                    _draw_line(
                        img,
                        x0,
                        y0,
                        x,
                        y,
                        price_height,
                        width,
                    )

            prev_px = (x, y)

    # Volume strictly inside bottom region
    if include_vol and volumes is not None:
        vol_n = _normalize_volume(volumes)

        for i in range(window):
            x_bar = x_start + i * day_width + 1

            if x_bar >= width:
                break

            bar_h = int(vol_n[i] * volume_height)

            if bar_h > 0:
                y_top = height - bar_h
                img[y_top:height, x_bar] = 255

    return img


def _ratio_split_image_dataset(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    include_vol: bool,
    include_ma: bool,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    """Backward-compatible random split for tests."""
    rng = np.random.default_rng(seed)

    specs = IMAGE_SPECS[window]
    height = specs["height"]
    width = specs["width"]

    X_all, y_all, ret_all = [], [], []

    n_stocks = len(data)
    logger.info(
        "Building image dataset (ratio split): %d securities | window=%d | horizon=%d",
        n_stocks, window, horizon,
    )

    bar = tqdm(data.items(), desc="OHLC images", unit="stock", total=n_stocks, dynamic_ncols=True)
    for _, df in bar:
        df = _ensure_flat_columns(df).copy()

        if include_ma:
            df["MA"] = df["Close"].rolling(window).mean()
            df = df.dropna(subset=["MA"])

        closes = df["Close"].values

        for i in range(window, len(df) - horizon):
            window_df = df.iloc[i - window:i]

            current_close = closes[i]
            future_close = closes[i + horizon]

            label = int(future_close > current_close)
            ret = future_close / current_close - 1 if current_close > 0 else np.nan

            img = generate_ohlc_image(
                window_df,
                window=window,
                include_vol=include_vol,
                include_ma=include_ma,
            )

            X_all.append(img)
            y_all.append(label)
            ret_all.append(ret)

        bar.set_postfix(images=f"{len(X_all):,}")

    X = np.array(X_all, dtype=np.float32) / 255.0
    X = X[:, :, :, np.newaxis]

    y       = np.array(y_all,   dtype=np.int64)
    returns = np.array(ret_all, dtype=np.float64)

    idx = rng.permutation(len(X))
    X, y, returns = X[idx], y[idx], returns[idx]

    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val":   X[n_train:n_train + n_val],
        "y_val":   y[n_train:n_train + n_val],
        "X_test":  X[n_train + n_val:],
        "y_test":  y[n_train + n_val:],
        "returns_train": returns[:n_train],
        "returns_val":   returns[n_train:n_train + n_val],
        "returns_test":  returns[n_train + n_val:],
        "image_shape": (height, width, 1),
        "window": window,
        "horizon": horizon,
    }


def make_image_dataset(
    data: dict[str, pd.DataFrame],
    window: int = 20,
    horizon: int = 5,
    include_vol: bool = True,
    include_ma: bool = True,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    train_end: Optional[str] = DEFAULT_TRAIN_END,
    val_end: Optional[str] = DEFAULT_VAL_END,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Generate the full image dataset.
    """
    if val_ratio is None:
        val_ratio = 0.15

    if train_ratio is not None:
        return _ratio_split_image_dataset(
            data=data,
            window=window,
            horizon=horizon,
            include_vol=include_vol,
            include_ma=include_ma,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    all_dates = pd.concat([df.index.to_series() for df in data.values()])

    use_ratio_fallback = (
        not all(isinstance(df.index, pd.DatetimeIndex) for df in data.values())
        or train_end is None
        or val_end is None
        or all_dates.max() <= pd.Timestamp(train_end)
        or all_dates.min() > pd.Timestamp(train_end)
        or all_dates.min() > pd.Timestamp(val_end)
    )

    if use_ratio_fallback:
        return _ratio_split_image_dataset(
            data=data,
            window=window,
            horizon=horizon,
            include_vol=include_vol,
            include_ma=include_ma,
            train_ratio=0.70,
            val_ratio=val_ratio,
            seed=seed,
        )

    rng = np.random.default_rng(seed)

    specs = IMAGE_SPECS[window]
    height = specs["height"]
    width = specs["width"]

    buckets = {
        "train": ([], [], []),
        "val":   ([], [], []),
        "test":  ([], [], []),
    }

    skipped = 0
    total_images = 0

    n_stocks = len(data)
    logger.info(
        "Building image dataset: %d securities | window=%d | horizon=%d | "
        "vol=%s | ma=%s | train_end=%s | val_end=%s",
        n_stocks, window, horizon, include_vol, include_ma, train_end, val_end,
    )

    bar = tqdm(data.items(), desc="OHLC images", unit="stock", total=n_stocks, dynamic_ncols=True)
    for _, df in bar:
        df = _ensure_flat_columns(df).copy()

        if include_ma:
            df["MA"] = df["Close"].rolling(window).mean()
            df = df.dropna(subset=["MA"])

        splits = {
            "train": df[df.index <= train_end],
            "val": df[(df.index > train_end) & (df.index <= val_end)],
            "test": df[df.index > val_end],
        }

        for split_name, df_split in splits.items():
            min_len = window + horizon

            if len(df_split) < min_len:
                skipped += 1
                continue

            closes = df_split["Close"].values
            n = len(df_split)

            for i in range(window, n - horizon):
                window_df = df_split.iloc[i - window:i]

                current_close = closes[i]
                future_close = closes[i + horizon]

                label = int(future_close > current_close)
                ret = future_close / current_close - 1 if current_close > 0 else np.nan

                try:
                    img = generate_ohlc_image(
                        window_df,
                        window=window,
                        include_vol=include_vol,
                        include_ma=include_ma,
                    )

                    buckets[split_name][0].append(img)
                    buckets[split_name][1].append(label)
                    buckets[split_name][2].append(ret)
                    total_images += 1

                except Exception:
                    skipped += 1

        bar.set_postfix(images=f"{total_images:,}", skipped=skipped)

    logger.info(
        "Image generation complete: %d images | %d skipped",
        total_images, skipped,
    )
    if verbose and skipped > 0:
        logger.info("%d windows skipped", skipped)

    def concat_bucket(key: str):
        X_list, y_list, ret_list = buckets[key]

        if not X_list:
            raise ValueError(
                f"Split '{key}' is empty. "
                f"Check train_end='{train_end}', "
                f"val_end='{val_end}'."
            )

        X = np.array(X_list, dtype=np.float32) / 255.0
        X = X[:, :, :, np.newaxis]
        y       = np.array(y_list,   dtype=np.int64)
        returns = np.array(ret_list, dtype=np.float64)

        return X, y, returns

    X_train, y_train, ret_train = concat_bucket("train")
    X_val,   y_val,   ret_val   = concat_bucket("val")
    X_test,  y_test,  ret_test  = concat_bucket("test")

    idx = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    ret_train = ret_train[idx]

    if verbose:
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            logger.info(
                "%s: %s — %.1f%% positive labels",
                split_name,
                X.shape,
                y.mean() * 100,
            )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "returns_train": ret_train,
        "returns_val":   ret_val,
        "returns_test":  ret_test,
        "image_shape": (height, width, 1),
        "window": window,
        "horizon": horizon,
    }


def visualize_sample(
    img: np.ndarray,
    label: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display one OHLC image."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.imshow(
        img.squeeze(),
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
    )

    color = "green" if label == 1 else "red"
    lbl = "UP ↑" if label == 1 else "DOWN ↓"

    ax.set_title(title or f"Label: {lbl}", color=color)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def visualize_grid(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 16,
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of OHLC images."""
    n = min(n, len(X))

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3, rows * 2.5),
    )

    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]

        color = "green" if y[i] == 1 else "red"

        ax.imshow(
            X[i].squeeze(),
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            "UP ↑" if y[i] == 1 else "DOWN ↓",
            color=color,
            fontsize=9,
        )

        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("OHLC image samples", fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()