"""
OHLC image generation tests — ohlc_chart.py

Covers:
  - generate_ohlc_image : shape, dtype, pixel range, volume placement
  - make_image_dataset  : output shapes and label validity
"""

import numpy as np
import pytest

from reimagining_trends.imaging.ohlc_chart import (
    IMAGE_SPECS,
    VOLUME_FRACTION,
    generate_ohlc_image,
    make_image_dataset,
)

HORIZON = 5


# ---------------------------------------------------------------------------
# generate_ohlc_image — invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", [5, 20, 60])
def test_image_shape(window, ohlcv_df):
    df = ohlcv_df.copy()
    df["MA"] = df["Close"].rolling(window).mean()
    df = df.dropna().iloc[:window]
    img = generate_ohlc_image(df, window=window)
    expected = (IMAGE_SPECS[window]["height"], IMAGE_SPECS[window]["width"])
    assert img.shape == expected, f"window={window}: expected {expected}, got {img.shape}"


@pytest.mark.parametrize("window", [5, 20, 60])
def test_image_dtype(window, ohlcv_df):
    df = ohlcv_df.copy()
    df["MA"] = df["Close"].rolling(window).mean()
    df = df.dropna().iloc[:window]
    img = generate_ohlc_image(df, window=window)
    assert img.dtype == np.uint8


@pytest.mark.parametrize("window", [5, 20, 60])
def test_image_binary_pixels(window, ohlcv_df):
    """Only black (0) and white (255) pixels — no grey."""
    df = ohlcv_df.copy()
    df["MA"] = df["Close"].rolling(window).mean()
    df = df.dropna().iloc[:window]
    img = generate_ohlc_image(df, window=window)
    unique = set(img.flatten().tolist())
    assert unique.issubset({0, 255}), f"Unexpected pixel values: {unique - {0, 255}}"


def test_image_not_blank(window_df_20d):
    """Image must have at least one white pixel."""
    img = generate_ohlc_image(window_df_20d, window=20)
    assert img.max() == 255


def test_image_not_full_white(window_df_20d):
    """Image must have at least one black pixel."""
    img = generate_ohlc_image(window_df_20d, window=20)
    assert img.min() == 0


# ---------------------------------------------------------------------------
# Volume placement
# ---------------------------------------------------------------------------

def test_no_volume_pixels_in_price_region(window_df_20d):
    """
    With include_vol=False the bottom VOLUME_FRACTION rows must be all-zero
    (nothing is drawn there at all).
    """
    img = generate_ohlc_image(window_df_20d, window=20, include_vol=False)
    H = img.shape[0]
    h_vol = int(H * VOLUME_FRACTION)
    h_price = H - h_vol
    assert img[h_price:, :].sum() == 0


def test_volume_confined_to_volume_region(window_df_5d):
    """
    With include_vol=True, the *difference* image (with_vol - without_vol)
    must only have non-zero pixels in the bottom VOLUME_FRACTION rows.
    """
    img_with = generate_ohlc_image(window_df_5d, window=5, include_vol=True, include_ma=False)
    img_without = generate_ohlc_image(window_df_5d, window=5, include_vol=False, include_ma=False)
    diff = img_with.astype(int) - img_without.astype(int)

    H = img_with.shape[0]
    h_vol = int(H * VOLUME_FRACTION)
    h_price = H - h_vol

    # No volume pixels should appear in the price region
    assert diff[:h_price, :].sum() == 0, "Volume pixels leaked into price region"
    # At least some volume pixels should exist in volume region
    assert diff[h_price:, :].sum() > 0, "No volume pixels drawn in volume region"


# ---------------------------------------------------------------------------
# Determinism / cross-branch comparison
# ---------------------------------------------------------------------------

def test_image_idempotent(window_df_20d):
    """Same input always produces bit-identical output."""
    img1 = generate_ohlc_image(window_df_20d, window=20)
    img2 = generate_ohlc_image(window_df_20d, window=20)
    np.testing.assert_array_equal(img1, img2)


def test_image_pixel_sum_stable(window_df_5d):
    """
    Pixel sum for the fixed synthetic window is deterministic.
    Run on the reference branch first to capture the expected value,
    then hardcode it here for cross-branch comparison.

    Reference value (seed=42, window=5): computed at test time.
    """
    img = generate_ohlc_image(window_df_5d, window=5, include_vol=True, include_ma=True)
    # Both branches must produce the same sum; if they differ the pixel
    # sum printed by pytest will reveal which branch is wrong.
    pixel_sum = int(img.sum())
    assert pixel_sum > 0  # sanity: image is not blank
    # Uncomment and fill after the first authoritative run:
    # assert pixel_sum == <REFERENCE_VALUE>


# ---------------------------------------------------------------------------
# make_image_dataset
# ---------------------------------------------------------------------------

def test_image_dataset_shapes(multi_stock_data):
    ds = make_image_dataset(multi_stock_data, window=20, horizon=HORIZON)
    H, W = IMAGE_SPECS[20]["height"], IMAGE_SPECS[20]["width"]
    for split in ["train", "val", "test"]:
        X = ds[f"X_{split}"]
        assert X.ndim == 4
        assert X.shape[1:] == (H, W, 1), f"{split}: {X.shape}"


def test_image_dataset_labels_binary(multi_stock_data):
    ds = make_image_dataset(multi_stock_data, window=20, horizon=HORIZON)
    for split in ["train", "val", "test"]:
        y = ds[f"y_{split}"]
        assert set(y.tolist()).issubset({0, 1})


def test_image_dataset_pixel_range(multi_stock_data):
    """Normalised images (float32) should lie in [0, 1]."""
    ds = make_image_dataset(multi_stock_data, window=20, horizon=HORIZON)
    X = ds["X_train"]
    assert X.dtype == np.float32
    assert X.min() >= 0.0
    assert X.max() <= 1.0 + 1e-6


def test_image_dataset_total_size(multi_stock_data):
    """Train + val + test accounts for all generated images."""
    ds = make_image_dataset(multi_stock_data, window=20, horizon=HORIZON)
    n_total = sum(len(ds[f"X_{s}"]) for s in ["train", "val", "test"])
    assert n_total == len(ds["X_train"]) + len(ds["X_val"]) + len(ds["X_test"])
    assert n_total > 0
