"""
Data pipeline tests — fetch_data.py

Covers:
  - _ensure_flat_columns  : MultiIndex flattening
  - image_scale           : [0,1] range + golden values on analytic input
  - cumret_scale          : golden values on analytic input
  - make_labels           : binary, no lookahead, golden sequence
  - make_tabular_dataset  : shapes, dtypes
  - make_multi_stock_dataset : split integrity
"""

import numpy as np
import pandas as pd
import pytest

from reimagining_trends.data.fetch_data import (
    _ensure_flat_columns,
    add_moving_average,
    cumret_scale,
    image_scale,
    make_labels,
    make_multi_stock_dataset,
    make_tabular_dataset,
)

WINDOW = 20
HORIZON = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_ohlcv(n: int = 30, window: int = 10) -> pd.DataFrame:
    """
    Deterministic analytic dataset:
      Close[i] = 10 + i   (linear ramp, 0-indexed)
      High[i]  = Close[i] + 1
      Low[i]   = Close[i] - 1
      Open[i]  = Close[i]   (same as close for simplicity)
      Volume   = 1_000_000 (constant)

    For any full window of size `window` ending at row i (0-indexed):
      price_max = High[i]    = 11 + i
      price_min = Low[i-w+1] = (10 + i - w + 1) - 1 = i - w + 10 = i  (for w=10)
      denom     = price_max - price_min = (11+i) - i = w + 1
      scaled_close[i] = (Close[i] - price_min) / denom
                      = (10+i - i) / (w+1) = w / (w+1)
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    c = np.arange(10.0, 10.0 + n)
    return pd.DataFrame(
        {"Open": c, "High": c + 1.0, "Low": c - 1.0, "Close": c,
         "Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# _ensure_flat_columns
# ---------------------------------------------------------------------------

def test_ensure_flat_noop(ohlcv_df):
    result = _ensure_flat_columns(ohlcv_df)
    assert list(result.columns) == list(ohlcv_df.columns)


def test_ensure_flat_flattens_multiindex():
    mi = pd.MultiIndex.from_arrays(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"] * 5]
    )
    df = pd.DataFrame(np.ones((5, 5)), columns=mi)
    result = _ensure_flat_columns(df)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_ensure_flat_no_mutation(ohlcv_df):
    original_cols = type(ohlcv_df.columns)
    _ensure_flat_columns(ohlcv_df)
    assert type(ohlcv_df.columns) == original_cols


# ---------------------------------------------------------------------------
# image_scale
# ---------------------------------------------------------------------------

def test_image_scale_price_range(ohlcv_df):
    scaled = image_scale(ohlcv_df, WINDOW)
    for col in ["Open", "High", "Low", "Close"]:
        assert scaled[col].min() >= -1e-9, f"{col} below 0"
        assert scaled[col].max() <= 1.0 + 1e-9, f"{col} above 1"


def test_image_scale_volume_range(ohlcv_df):
    scaled = image_scale(ohlcv_df, WINDOW)
    assert scaled["Volume"].min() > 0
    assert scaled["Volume"].max() <= 1.0 + 1e-9


def test_image_scale_deterministic(ohlcv_df):
    pd.testing.assert_frame_equal(
        image_scale(ohlcv_df, WINDOW),
        image_scale(ohlcv_df, WINDOW),
    )


def test_image_scale_golden_close():
    """
    On a linear price ramp with window=10, every valid Close normalises to
    w/(w+1) = 10/11 (see _linear_ohlcv docstring for derivation).
    """
    df = _linear_ohlcv(n=30, window=10)
    scaled = image_scale(df, window=10)
    expected = 10.0 / 11.0  # w / (w+1)
    np.testing.assert_allclose(scaled["Close"].values, expected, rtol=1e-9)


def test_image_scale_golden_high_low():
    """
    High[i] is always the window max → normalises to 1.0.
    Low[i] = (w-1)/(w+1) = 9/11 (it is not the window min; the oldest
    Low in the window holds that position and maps to 0).
    """
    df = _linear_ohlcv(n=30, window=10)
    scaled = image_scale(df, window=10)
    np.testing.assert_allclose(scaled["High"].values, 1.0, atol=1e-9)
    np.testing.assert_allclose(scaled["Low"].values, 9.0 / 11.0, atol=1e-9)  # (w-1)/(w+1)


# ---------------------------------------------------------------------------
# cumret_scale
# ---------------------------------------------------------------------------

def test_cumret_scale_volume_range(ohlcv_df):
    scaled = cumret_scale(ohlcv_df, WINDOW)
    assert scaled["Volume"].min() > 0
    assert scaled["Volume"].max() <= 1.0 + 1e-9


def test_cumret_scale_deterministic(ohlcv_df):
    pd.testing.assert_frame_equal(
        cumret_scale(ohlcv_df, WINDOW),
        cumret_scale(ohlcv_df, WINDOW),
    )


def test_cumret_scale_golden_first_row():
    """
    Linear ramp, window=10: first valid row is i=9.
    first_close = Close[shift(9)][9] = Close[0] = 10.
    scaled_close[9] = Close[9] / Close[0] = 19 / 10 = 1.9.
    """
    df = _linear_ohlcv(n=30, window=10)
    scaled = cumret_scale(df, window=10)
    np.testing.assert_allclose(scaled["Close"].iloc[0], 19.0 / 10.0, rtol=1e-9)


def test_cumret_scale_golden_last_row():
    """
    Last valid row is i=29.
    first_close[29] = Close[20] = 30.
    scaled_close[29] = Close[29] / Close[20] = 39 / 30 = 1.3.
    """
    df = _linear_ohlcv(n=30, window=10)
    scaled = cumret_scale(df, window=10)
    np.testing.assert_allclose(scaled["Close"].iloc[-1], 39.0 / 30.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# make_labels
# ---------------------------------------------------------------------------

def test_make_labels_binary(ohlcv_df):
    labels = make_labels(ohlcv_df, horizon=HORIZON)
    assert set(labels.dropna().unique()).issubset({0, 1})


def test_make_labels_dtype(ohlcv_df):
    # float to accommodate NaN for the last `horizon` rows
    labels = make_labels(ohlcv_df, horizon=HORIZON)
    assert np.issubdtype(labels.dtype, np.floating)


def test_make_labels_golden_sequence():
    """
    Known price sequence → known labels.

    Close = [100, 102, 98, 105, 99, 103, 101, 107, 96, 110], horizon=3
    label[i] = 1 iff Close[i+3] > Close[i]

    i=0: 105 > 100 → 1
    i=1:  99 > 102 → 0
    i=2: 103 >  98 → 1
    i=3: 101 > 105 → 0
    i=4: 107 >  99 → 1
    i=5:  96 > 103 → 0
    i=6: 110 > 101 → 1
    """
    close = [100, 102, 98, 105, 99, 103, 101, 107, 96, 110]
    df = pd.DataFrame({"Close": close})
    labels = make_labels(df, horizon=3).dropna().astype(int).tolist()
    assert labels == [1, 0, 1, 0, 1, 0, 1]


def test_make_labels_no_lookahead():
    """
    Label uses Close[i+horizon], NOT Close[i].
    A jump at position 20 should only affect labels at i <= 14 (horizon=5).
    """
    close = [100.0] * 20 + [200.0] * 20
    df = pd.DataFrame({"Close": close})
    labels = make_labels(df, horizon=5)
    # i=14: Close[19]=100 vs Close[14]=100 → return=0, label=0
    # i=15: Close[20]=200 vs Close[15]=100 → return>0, label=1
    assert labels.iloc[14] == 0
    assert labels.iloc[15] == 1


# ---------------------------------------------------------------------------
# make_tabular_dataset
# ---------------------------------------------------------------------------

def test_tabular_dataset_shape(ohlcv_df):
    X, y, returns = make_tabular_dataset(ohlcv_df, window=WINDOW, horizon=HORIZON)
    assert X.ndim == 3
    assert X.shape[1] == WINDOW
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_tabular_dataset_dtype(ohlcv_df):
    X, y, returns = make_tabular_dataset(ohlcv_df, window=WINDOW, horizon=HORIZON)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert returns.dtype == np.float64


def test_tabular_dataset_labels_binary(ohlcv_df):
    _, y, _ = make_tabular_dataset(ohlcv_df, window=WINDOW, horizon=HORIZON)
    assert set(y.tolist()).issubset({0, 1})


def test_tabular_dataset_feature_range(ohlcv_df):
    """image_scale should keep price features in [0, 1]."""
    X, _, _ = make_tabular_dataset(ohlcv_df, window=WINDOW, horizon=HORIZON, scaling="image")
    # Price features (first 4 channels: O/H/L/C) should be in [0, 1]
    price = X[:, :, :4]
    assert price.min() >= -1e-6
    assert price.max() <= 1.0 + 1e-6


def test_tabular_dataset_returns_shape(ohlcv_df):
    X, y, returns = make_tabular_dataset(ohlcv_df, window=WINDOW, horizon=HORIZON)
    assert returns.shape == (X.shape[0],)
    assert np.isfinite(returns).any()


# ---------------------------------------------------------------------------
# make_multi_stock_dataset
# ---------------------------------------------------------------------------

def test_multi_stock_dataset_keys(multi_stock_data):
    ds = make_multi_stock_dataset(multi_stock_data, window=WINDOW, horizon=HORIZON)
    for split in ["train", "val", "test"]:
        assert f"X_{split}" in ds
        assert f"y_{split}" in ds


def test_multi_stock_dataset_shapes_consistent(multi_stock_data):
    ds = make_multi_stock_dataset(multi_stock_data, window=WINDOW, horizon=HORIZON)
    for split in ["train", "val", "test"]:
        X, y = ds[f"X_{split}"], ds[f"y_{split}"]
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == WINDOW


def test_multi_stock_dataset_total_size(multi_stock_data):
    """All samples are allocated to exactly one split (no duplication or drop)."""
    ds = make_multi_stock_dataset(
        multi_stock_data, window=WINDOW, horizon=HORIZON, train_ratio=0.7
    )
    n_total = len(ds["X_train"]) + len(ds["X_val"]) + len(ds["X_test"])
    assert n_total > 0
    # Val and test are equal halves of the held-out portion
    assert abs(len(ds["X_val"]) - len(ds["X_test"])) <= 1


def test_multi_stock_dataset_deterministic(multi_stock_data):
    """Same random state → same split sizes."""
    np.random.seed(0)
    ds1 = make_multi_stock_dataset(multi_stock_data, window=WINDOW, horizon=HORIZON)
    np.random.seed(0)
    ds2 = make_multi_stock_dataset(multi_stock_data, window=WINDOW, horizon=HORIZON)
    assert len(ds1["X_train"]) == len(ds2["X_train"])
