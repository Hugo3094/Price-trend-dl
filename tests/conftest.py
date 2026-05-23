"""
Shared deterministic fixtures.

All fixtures are session-scoped and seed-based — every run on every branch
produces byte-identical DataFrames, so numerical assertions are cross-branch
comparable.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(seed: int, n: int = 120, base_price: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a reproducible random walk."""
    rng = np.random.default_rng(seed)

    log_ret = rng.normal(0.0005, 0.012, n)
    close = base_price * np.cumprod(1.0 + log_ret)

    spread = np.abs(rng.normal(0.0, 0.006, n)) * close
    high = close + spread
    low = np.maximum(close - spread, 1e-4)
    open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)
    volume = rng.integers(1_000_000, 8_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


@pytest.fixture(scope="session")
def ohlcv_df() -> pd.DataFrame:
    """120-row synthetic OHLCV (seed=42). Identical on every branch."""
    return _make_ohlcv(seed=42)


@pytest.fixture(scope="session")
def multi_stock_data() -> dict[str, pd.DataFrame]:
    """Three independent synthetic tickers (seeds 42, 43, 44)."""
    return {
        "AAA": _make_ohlcv(seed=42),
        "BBB": _make_ohlcv(seed=43),
        "CCC": _make_ohlcv(seed=44),
    }


@pytest.fixture(scope="session")
def window_df_20d(ohlcv_df) -> pd.DataFrame:
    """Exactly 20 rows with MA column — ready for generate_ohlc_image(window=20)."""
    df = ohlcv_df.copy()
    df["MA"] = df["Close"].rolling(20).mean()
    return df.dropna().iloc[:20]


@pytest.fixture(scope="session")
def window_df_5d(ohlcv_df) -> pd.DataFrame:
    """Exactly 5 rows with MA column — ready for generate_ohlc_image(window=5)."""
    df = ohlcv_df.copy()
    df["MA"] = df["Close"].rolling(5).mean()
    return df.dropna().iloc[:5]
