"""
data_manager.py
---------------
DataManager — owns the full data pipeline:
  download / load → tabular dataset → image dataset.

All parameters are taken from Config, so nothing is hardcoded.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from reimagining_trends.data.fetch_data import (
    download_ohlcv,
    load_ohlcv,
    load_parquet,
    make_multi_stock_dataset,
)
from reimagining_trends.imaging.ohlc_chart import IMAGE_SPECS, make_image_dataset
from reimagining_trends.utils.cache import (
    file_md5, fingerprints_match, load_json, save_json,
)
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates data download and dataset construction.

    Parameters
    ----------
    config : Config
        Pipeline configuration (tickers, dates, window, horizon, …).

    Attributes
    ----------
    raw_data : dict[str, pd.DataFrame]
        Populated after calling :meth:`download` or :meth:`load`.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.raw_data: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def download(self) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data from the configured source.

        - ``data_source = "yahoo"``   — fetches live from Yahoo Finance
        - ``data_source = "parquet"`` — reads a local WRDS-style Parquet file
        """
        if self.cfg.data_source == "parquet":
            if not self.cfg.parquet_path:
                raise ValueError(
                    "data_source='parquet' requires parquet_path to be set in the config."
                )
            logger.info("Loading parquet: %s", self.cfg.parquet_path)
            self.raw_data = load_parquet(
                self.cfg.parquet_path,
                start=self.cfg.start,
                end=self.cfg.end,
                permnos=self.cfg.permnos,
            )
            if self.cfg.test_mode:
                keys = list(self.raw_data.keys())[:2]
                self.raw_data = {k: self.raw_data[k] for k in keys}
                logger.info("TEST_MODE: limited to %d securities from parquet.", len(self.raw_data))
        else:
            logger.info(
                "Downloading %d tickers from Yahoo Finance (%s → %s)",
                len(self.cfg.tickers), self.cfg.start, self.cfg.end,
            )
            self.raw_data = download_ohlcv(
                tickers=self.cfg.tickers,
                start=self.cfg.start,
                end=self.cfg.end,
                save_dir=self.cfg.data_save_dir,
            )

        logger.info("Data loaded: %d securities.", len(self.raw_data))
        return self.raw_data

    def load(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Load previously downloaded CSVs from *data_dir*."""
        self.raw_data = load_ohlcv(data_dir)
        return self.raw_data

    # ------------------------------------------------------------------
    # Dataset builders
    # ------------------------------------------------------------------

    def build_tabular(
        self,
        raw_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Build the tabular (MLP / LSTM / GRU) dataset.

        Parameters
        ----------
        raw_data : optional override — defaults to ``self.raw_data``

        Returns
        -------
        dict with X_train/val/test, y_train/val/test, window, horizon, scaling
        """
        data = raw_data if raw_data is not None else self.raw_data
        if not data:
            raise RuntimeError("No raw data available. Call download() or load() first.")

        logger.info(
            "Building tabular dataset: window=%d, horizon=%d, scaling=%s",
            self.cfg.window, self.cfg.horizon, self.cfg.scaling,
        )
        ds = make_multi_stock_dataset(
            data=data,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            scaling=self.cfg.scaling,
            add_ma=self.cfg.include_ma,
            train_end=self.cfg.train_end,
            val_end=self.cfg.val_end,
            train_ratio=self.cfg.train_ratio,
            seed=self.cfg.seed,
        )
        self._log_split_sizes("tabular", ds)
        return ds

    def build_image(
        self,
        raw_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Build the OHLC image (CNN) dataset, with fingerprint-based caching.

        On a cache hit the dataset is loaded from a compressed .npz file,
        skipping image generation entirely.  The cache is stored under
        ``{checkpoints_dir}/image_dataset/``.

        Parameters
        ----------
        raw_data : optional override — defaults to ``self.raw_data``

        Returns
        -------
        dict with X_train/val/test (N,H,W,1), y_train/val/test
        """
        data = raw_data if raw_data is not None else self.raw_data
        if not data:
            raise RuntimeError("No raw data available. Call download() or load() first.")

        cache_dir = os.path.join(self.cfg.checkpoints_dir, "image_dataset")
        fp_path   = os.path.join(cache_dir, "fingerprint.json")
        npz_path  = os.path.join(cache_dir, "dataset.npz")

        fp = self._image_fingerprint()
        cached_fp = load_json(fp_path)
        if fingerprints_match(fp, cached_fp) and os.path.exists(npz_path):
            logger.info("Image dataset cache hit — loading from %s", npz_path)
            npz = np.load(npz_path, allow_pickle=False)
            ds: dict = {k: npz[k] for k in npz.files}
            specs = IMAGE_SPECS[self.cfg.window]
            ds["image_shape"] = (specs["height"], specs["width"], 1)
            ds["window"]  = self.cfg.window
            ds["horizon"] = self.cfg.horizon
            self._log_split_sizes("image", ds)
            return ds

        ds = make_image_dataset(
            data=data,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            include_vol=self.cfg.include_vol,
            include_ma=self.cfg.include_ma,
            train_end=self.cfg.train_end,
            val_end=self.cfg.val_end,
            val_ratio=self.cfg.val_ratio,
            seed=self.cfg.seed,
        )

        os.makedirs(cache_dir, exist_ok=True)
        array_keys = [k for k, v in ds.items() if isinstance(v, np.ndarray)]
        np.savez_compressed(npz_path, **{k: ds[k] for k in array_keys})
        save_json(fp, fp_path)
        logger.info("Image dataset cached to %s", npz_path)

        self._log_split_sizes("image", ds)
        return ds

    def _image_fingerprint(self) -> dict:
        """All inputs that can affect the image dataset output."""
        data_hash = None
        if getattr(self.cfg, "parquet_path", None) and os.path.exists(self.cfg.parquet_path):
            data_hash = file_md5(self.cfg.parquet_path)
        return {
            "data_source": self.cfg.data_source,
            "data_hash":   data_hash,
            "start":       self.cfg.start,
            "end":         self.cfg.end,
            "permnos":     sorted(self.cfg.permnos) if self.cfg.permnos else None,
            "window":      self.cfg.window,
            "horizon":     self.cfg.horizon,
            "include_vol": self.cfg.include_vol,
            "include_ma":  self.cfg.include_ma,
            "train_end":   self.cfg.train_end,
            "val_end":     self.cfg.val_end,
            "val_ratio":   self.cfg.val_ratio,
            "seed":        self.cfg.seed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_split_sizes(kind: str, ds: dict) -> None:
        for split in ["train", "val", "test"]:
            X, y = ds[f"X_{split}"], ds[f"y_{split}"]
            logger.info(
                "  %s %s: %s  %.1f%% UP",
                kind, split, X.shape, y.mean() * 100,
            )
