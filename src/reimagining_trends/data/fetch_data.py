"""
fetch_data.py
-------------
Downloads and prepares OHLCV data from Yahoo Finance.
Generates return labels (up/down) for binary classification.

Reference: Jiang, Kelly & Xiu (2023) - (Re-)Imag(in)ing Price Trends
"""

import logging
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "GS", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "KO",
    "PEP", "NVDA", "AMD", "INTC", "CSCO",
]

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]

# Chronological split boundaries (aligned with ohlc_chart.py)
DEFAULT_TRAIN_END = "2016-12-31"
DEFAULT_VAL_END   = "2019-12-31"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens yfinance MultiIndex columns to their price-type level.

    Recent yfinance versions return MultiIndex columns even for single-ticker
    downloads, e.g. ('Close', 'AAPL') instead of 'Close'.  This helper
    normalises the DataFrame so all downstream code can use plain string keys.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_ohlcv(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    save_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Downloads OHLCV data for a list of tickers.

    Returns
    -------
    dict  {ticker: DataFrame with columns Open/High/Low/Close/Volume}
    """
    data = {}
    for ticker in tickers:
        logger.info("Downloading %s ...", ticker)
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                logger.warning("%s: EMPTY — skipped", ticker)
                continue
            df = _ensure_flat_columns(df)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            data[ticker] = df
            logger.info("%s: OK (%d days)", ticker, len(df))
        except Exception as e:
            logger.error("%s: ERROR — %s", ticker, e)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ticker, df in data.items():
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        logger.info("Data saved to: %s", save_dir)

    return data


def load_ohlcv(data_dir: str) -> dict[str, pd.DataFrame]:
    """Loads previously downloaded CSV files from a directory."""
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname), index_col=0, parse_dates=True)
            data[ticker] = _ensure_flat_columns(df)
    logger.info("%d tickers loaded from %s", len(data), data_dir)
    return data


def load_parquet(
    path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    permnos: Optional[list] = None,
) -> dict[str, pd.DataFrame]:
    """
    Loads OHLCV data from a WRDS-style long-format Parquet file.

    Expected schema (case-insensitive):
      date, permno, open, high, low, close, volume  (+ optional extra columns)

    ``permno`` (CRSP permanent number) is the primary identifier.  If the file
    has no ``permno`` column the function falls back to ``ticker`` with a warning.

    Parameters
    ----------
    path    : path to the .parquet file
    start   : optional ISO date string — rows before this date are dropped
    end     : optional ISO date string — rows after this date are dropped
    permnos : optional list of PERMNOs to load (None = load all)

    Returns
    -------
    dict  {str(permno): DataFrame with columns Open/High/Low/Close/Volume,
                        DatetimeIndex sorted ascending}
    """
    df = pd.read_parquet(path)

    # If date was saved as the DataFrame index, restore it as a column
    if df.index.name is not None and "date" not in df.columns:
        df = df.reset_index()

    # normalise column names: strip whitespace, lowercase for lookup
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("Parquet file must have a 'date' column.")

    # Determine the security identifier column — prefer permno (CRSP stable ID)
    if "permno" in df.columns:
        id_col = "permno"
    elif "ticker" in df.columns:
        id_col = "ticker"
        logger.warning("No 'permno' column found — falling back to 'ticker' as identifier.")
    else:
        raise ValueError("Parquet file must have a 'permno' or 'ticker' column.")

    # rename OHLCV columns to title-case expected by the rest of the pipeline
    ohlcv_rename = {"open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in ohlcv_rename.items() if k in df.columns})

    df["date"] = pd.to_datetime(df["date"])

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    # Optional PERMNO filter
    if permnos is not None:
        pset = {str(p) for p in permnos}
        df = df[df[id_col].astype(str).isin(pset)]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet file missing columns: {missing}")

    result: dict[str, pd.DataFrame] = {}
    for id_val, grp in df.groupby(id_col):
        ohlcv = (
            grp.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")[required]
            .sort_index()
            .dropna()
        )
        if len(ohlcv) > 0:
            result[str(id_val)] = ohlcv

    logger.info("Loaded %d securities (%s) from parquet: %s", len(result), id_col, path)
    return result


# ---------------------------------------------------------------------------
# Normalisation (image scaling — see paper Section I)
# ---------------------------------------------------------------------------
def image_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalises prices over a rolling window of `window` days.
    The window's max High and min Low are mapped to [0, 1].
    Volume is similarly normalised (max volume in window -> 1).

    This is the exact normalisation used in the paper to make
    all stocks comparable on a common scale.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)

    price_cols = ["Open", "High", "Low", "Close"]
    price_max = df[price_cols].rolling(window).max().max(axis=1)
    price_min = df[price_cols].rolling(window).min().min(axis=1)
    denom = (price_max - price_min).replace(0, np.nan)

    for col in price_cols:
        scaled[col] = df[col].sub(price_min, axis=0).div(denom, axis=0)

    vol_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / vol_max

    return scaled.dropna()


def cumret_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalises prices by the Close at the start of each window.
    Equivalent to a cumulative return — used as a comparison baseline.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)
    first_close = df["Close"].shift(window - 1)
    for col in ["Open", "High", "Low", "Close"]:
        scaled[col] = df[col] / first_close
    vol_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / vol_max
    return scaled.dropna()


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------
def add_moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Adds a `window`-day moving average column (MA)."""
    df = _ensure_flat_columns(df).copy()
    df["MA"] = df["Close"].rolling(window).mean()
    return df.dropna()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def make_labels(
    df: pd.DataFrame,
    horizon: int,
    col: str = "Close",
) -> pd.Series:
    """
    Generates binary labels:
      1    if the `horizon`-day return is positive
      0    otherwise
      NaN  for the last `horizon` rows (future unknown)

    The label at t corresponds to the return between t and t+horizon.
    NaN is preserved so that downstream .dropna() correctly excludes
    the tail rows that have no valid future price.
    """
    df = _ensure_flat_columns(df)
    future_close = df[col].shift(-horizon)
    label = (future_close > df[col]).astype(float)
    label[future_close.isna()] = np.nan
    return label.rename(f"label_{horizon}d")


# ---------------------------------------------------------------------------
# Sliding windows — tabular data (for MLP / LSTM)
# ---------------------------------------------------------------------------
def make_tabular_dataset(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds (X, y, returns) for MLP and LSTM from a single OHLCV series.

    Parameters
    ----------
    df      : OHLCV DataFrame for a single ticker
    window  : input window length (e.g. 5, 20, 60)
    horizon : prediction horizon
    scaling : normalisation method ("image" | "cumret")
    add_ma  : include moving average

    Returns
    -------
    X       : (n_samples, window, n_features)  float32
    y       : (n_samples,)                     int64   binary label
    returns : (n_samples,)                     float64 actual horizon-period return
                                               (Close[t+h] / Close[t] - 1, original prices)
    """
    # preserve original close BEFORE any scaling — used for real-return computation
    original_close = _ensure_flat_columns(df)["Close"].copy()

    if add_ma:
        df = add_moving_average(df, window)

    if scaling == "image":
        df_scaled = image_scale(df, window)
    else:
        df_scaled = cumret_scale(df, window)

    labels = make_labels(df_scaled, horizon)

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma and "MA" in df_scaled.columns:
        feature_cols.append("MA")

    df_scaled = df_scaled.join(labels).dropna()

    # align original close to the rows surviving the scaling / label dropna
    orig_cls = original_close.reindex(df_scaled.index).values

    X_list, y_list, ret_list = [], [], []
    arr = df_scaled[feature_cols].values
    lbl = df_scaled[f"label_{horizon}d"].values
    n   = len(arr)

    for i in range(window, n - horizon):
        X_list.append(arr[i - window:i])
        y_list.append(lbl[i])
        c0, ch = orig_cls[i], orig_cls[i + horizon]
        ret_list.append(ch / c0 - 1 if (c0 > 0 and not np.isnan(c0) and not np.isnan(ch)) else np.nan)

    if not X_list:
        n_feat = len(feature_cols)
        return np.empty((0, window, n_feat)), np.empty(0, dtype=np.int64), np.empty(0)

    return (
        np.array(X_list,  dtype=np.float32),
        np.array(y_list,  dtype=np.int64),
        np.array(ret_list, dtype=np.float64),
    )


def _ratio_split_tabular(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str,
    add_ma: bool,
    train_ratio: float,
    rng: np.random.Generator,
) -> dict:
    """Random ratio-based train/val/test split (fallback for short / synthetic data)."""
    X_all, y_all, ret_all = [], [], []
    for df in data.values():
        X, y, returns = make_tabular_dataset(df, window, horizon, scaling, add_ma)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            ret_all.append(returns)

    if not X_all:
        raise ValueError("No tabular samples generated — check data length, window, and horizon.")

    X       = np.concatenate(X_all,   axis=0)
    y       = np.concatenate(y_all,   axis=0)
    returns = np.concatenate(ret_all, axis=0)

    idx = rng.permutation(len(X))
    X, y, returns = X[idx], y[idx], returns[idx]

    n_train = int(len(X) * train_ratio)
    n_val = (len(X) - n_train) // 2

    return {
        "X_train": X[:n_train],               "y_train": y[:n_train],
        "X_val":   X[n_train:n_train + n_val], "y_val": y[n_train:n_train + n_val],
        "X_test":  X[n_train + n_val:],        "y_test": y[n_train + n_val:],
        "returns_train": returns[:n_train],
        "returns_val":   returns[n_train:n_train + n_val],
        "returns_test":  returns[n_train + n_val:],
        "window": window, "horizon": horizon, "scaling": scaling,
    }


def _chronological_split_tabular(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str,
    add_ma: bool,
    train_end: str,
    val_end: str,
    rng: np.random.Generator,
) -> dict:
    """Date-based train/val/test split — no look-ahead bias."""
    buckets: dict[str, tuple[list, list, list]] = {
        "train": ([], [], []),
        "val":   ([], [], []),
        "test":  ([], [], []),
    }

    for df in data.values():
        splits = {
            "train": df[df.index <= train_end],
            "val":   df[(df.index > train_end) & (df.index <= val_end)],
            "test":  df[df.index > val_end],
        }
        for split_name, df_split in splits.items():
            if len(df_split) < window + horizon:
                continue
            X, y, returns = make_tabular_dataset(df_split, window, horizon, scaling, add_ma)
            if len(X) > 0:
                buckets[split_name][0].append(X)
                buckets[split_name][1].append(y)
                buckets[split_name][2].append(returns)

    def _concat(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_list, y_list, ret_list = buckets[key]
        if not X_list:
            raise ValueError(
                f"Tabular split '{key}' is empty. "
                f"Check train_end='{train_end}', val_end='{val_end}'."
            )
        return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(ret_list)

    X_train, y_train, ret_train = _concat("train")
    # shuffle only train to avoid order bias during SGD
    idx = rng.permutation(len(X_train))
    X_train, y_train, ret_train = X_train[idx], y_train[idx], ret_train[idx]

    X_val,  y_val,  ret_val  = _concat("val")
    X_test, y_test, ret_test = _concat("test")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "returns_train": ret_train,
        "returns_val":   ret_val,
        "returns_test":  ret_test,
        "window": window, "horizon": horizon, "scaling": scaling,
    }


def make_multi_stock_dataset(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
    train_end: Optional[str] = DEFAULT_TRAIN_END,
    val_end: Optional[str] = DEFAULT_VAL_END,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    Build the full tabular dataset (all tickers) with a train/val/test split.

    Split strategy
    --------------
    Chronological (default) — used when the data spans ``train_end`` and
    ``val_end``:
      • train : rows with index ≤ train_end
      • val   : train_end < index ≤ val_end
      • test  : index > val_end

    Ratio fallback — used when the data is too short or synthetic (e.g. tests):
      • train / val / test proportions derived from ``train_ratio``

    Parameters
    ----------
    data        : {ticker: OHLCV DataFrame}
    window      : sequence length in trading days
    horizon     : prediction horizon in trading days
    scaling     : "image" | "cumret"
    add_ma      : include moving-average feature
    train_end   : last date of training set (ISO string)
    val_end     : last date of validation set (ISO string)
    train_ratio : fraction used for training when falling back to ratio split
    seed        : RNG seed for reproducible shuffling
    """
    rng = np.random.default_rng(seed)

    # decide whether chronological split is viable
    use_chrono = False
    if train_end is not None and val_end is not None:
        try:
            has_dt_index = all(isinstance(df.index, pd.DatetimeIndex) for df in data.values())
            if has_dt_index:
                all_dates = pd.concat([df.index.to_series() for df in data.values()])
                use_chrono = (
                    all_dates.max() > pd.Timestamp(train_end)
                    and all_dates.min() <= pd.Timestamp(train_end)
                )
        except Exception:
            pass

    if use_chrono:
        logger.info(
            "Tabular split: chronological (train≤%s | val≤%s | test>%s)",
            train_end, val_end, val_end,
        )
        return _chronological_split_tabular(
            data, window, horizon, scaling, add_ma, train_end, val_end, rng
        )

    logger.info("Tabular split: ratio (train=%.0f%%)", train_ratio * 100)
    return _ratio_split_tabular(data, window, horizon, scaling, add_ma, train_ratio, rng)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== Downloading data ===")
    raw_data = download_ohlcv(
        tickers=DEFAULT_TICKERS,
        start="2000-01-01",
        end="2023-12-31",
        save_dir="data/raw",
    )

    logger.info("=== Building tabular dataset (window=20, horizon=5) ===")
    dataset = make_multi_stock_dataset(
        data=raw_data,
        window=20,
        horizon=5,
        scaling="image",
    )
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        pos = y.mean() * 100
        logger.info("  %s: %s  —  %.1f%% positive", split, X.shape, pos)