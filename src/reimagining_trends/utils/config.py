"""
config.py
---------
Centralised configuration for the reimagining-price-trends pipeline.

Populated from ``configs/run_pipeline_config.json`` at instantiation.
Every JSON key maps to a Python attribute listed below.

When PIPELINE.TEST_MODE is true the following overrides are applied
automatically so debug runs are fast:
  - tickers    → first 2 tickers only
  - start      → "2021-01-01"
  - epochs     → 2
  - patience   → 1

When PIPELINE.TEST_SMALL_CRSP is true the pipeline switches to the
small WRDS sample (2020-2024) with matching chronological split dates:
  - data_source → "parquet"
  - parquet_path → data/small_wrds_gross_query.parquet
  - start/end   → 2020-01-01 / 2024-12-31
  - train_end   → 2022-06-30
  - val_end     → 2023-06-30
Both flags can be combined (TEST_MODE speed + TEST_SMALL_CRSP data).

Usage
-----
    from reimagining_trends.utils.config import Config
    cfg = Config()                        # loads configs/run_pipeline_config.json
    cfg = Config("path/to/other.json")   # custom path
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Tickers used when TEST_MODE overrides the full list
_TEST_TICKERS = ["AAPL", "MSFT"]
_TEST_START   = "2021-01-01"
_TEST_EPOCHS  = 2
_TEST_PATIENCE = 1

# Small CRSP sample overrides
_SMALL_CRSP_PARQUET   = "data/small_wrds_gross_query.parquet"
_SMALL_CRSP_START     = "2020-01-01"
_SMALL_CRSP_END       = "2024-12-31"
_SMALL_CRSP_TRAIN_END = "2022-06-30"
_SMALL_CRSP_VAL_END   = "2023-06-30"


class Config:
    """
    Pipeline configuration.

    PIPELINE
    --------
    test_mode        : bool   — fast debug run (2 tickers, 2 epochs)
    test_small_crsp  : bool   — switch to small WRDS sample (2020-2024)
    seed             : int    — global random seed
    results_dir      : str    — directory for output figures
    checkpoints_dir  : str    — directory for model checkpoints

    DATA
    ----
    tickers          : list[str]     — ticker symbols to download
    start            : str           — download start date (YYYY-MM-DD)
    end              : str           — download end date (YYYY-MM-DD)
    data_save_dir    : str | None    — if set, cache raw CSVs here

    PREPROCESSING
    -------------
    window           : int    — image / sequence window in trading days (5 | 20 | 60)
    horizon          : int    — prediction horizon in trading days
    scaling          : str    — normalisation method ("image" | "cumret")
    include_vol      : bool   — include volume bars in OHLC images
    include_ma       : bool   — include moving-average overlay
    train_end        : str    — last date of training set for chronological split
    val_end          : str    — last date of validation set for chronological split
    train_ratio      : float  — fallback fraction for training (used when data is too short)
    val_ratio        : float  — fallback fraction for validation

    TRAINING
    --------
    epochs           : int    — maximum training epochs
    batch_size       : int    — mini-batch size
    patience         : int    — early-stopping patience
    lr               : float  — Adam learning rate
    weight_decay     : float  — Adam L2 regularisation
    models_to_train  : list[str]  — which models to run (["MLP","GRU","LSTM","CNN"])

    MODELS
    ------
    mlp_hidden_dims  : list[int]
    mlp_dropout      : float
    lstm_hidden_size : int
    lstm_num_layers  : int
    lstm_dropout     : float
    lstm_bidirectional: bool
    gru_hidden_size  : int
    gru_num_layers   : int
    gru_dropout      : float
    attn_hidden_size : int
    attn_num_layers  : int
    attn_dropout     : float
    cnn_dropout      : float

    EVALUATION
    ----------
    n_deciles        : int    — number of portfolio deciles
    periods_per_year : int    — annualisation factor for Sharpe (52 = weekly)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()

        self._config_path = (
            Path(config_path) if config_path
            else self.ROOT_DIR / "configs" / "run_pipeline_config.json"
        )
        logger.info("Config path: %s", self._config_path)

        # ── PIPELINE defaults ──────────────────────────────────────────────
        self.test_mode: bool = False
        self.test_small_crsp: bool = False
        self.seed: int = 42
        self.results_dir: str = "results"
        self.checkpoints_dir: str = "checkpoints"

        # ── DATA defaults ──────────────────────────────────────────────────
        self.data_source: str = "yahoo"   # "yahoo" | "parquet"
        self.parquet_path: Optional[str] = None
        self.permnos: Optional[List] = None  # if set, restrict parquet load to these PERMNOs
        self.tickers: List[str] = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "TSLA", "JPM", "GS", "BAC", "WMT",
            "XOM", "CVX", "JNJ", "PFE", "KO",
            "PEP", "NVDA", "AMD", "INTC", "CSCO",
        ]
        self.start: str = "2010-01-01"
        self.end: str = "2022-12-31"
        self.data_save_dir: Optional[str] = None

        # ── PREPROCESSING defaults ─────────────────────────────────────────
        self.window: int = 20
        self.horizon: int = 5
        self.scaling: str = "image"
        self.include_vol: bool = True
        self.include_ma: bool = True
        self.train_end: str = "2016-12-31"
        self.val_end: str = "2019-12-31"
        self.train_ratio: float = 0.70   # used only when chronological split falls back
        self.val_ratio: float = 0.15

        # ── TRAINING defaults ──────────────────────────────────────────────
        self.epochs: int = 30
        self.batch_size: int = 64
        self.patience: int = 5
        self.lr: float = 1e-4
        self.weight_decay: float = 1e-5
        self.models_to_train: List[str] = ["MLP", "GRU", "LSTM", "CNN"]

        # ── MODEL architecture defaults ────────────────────────────────────
        self.mlp_hidden_dims: List[int] = [256, 128, 64]
        self.mlp_dropout: float = 0.3
        self.lstm_hidden_size: int = 128
        self.lstm_num_layers: int = 2
        self.lstm_dropout: float = 0.3
        self.lstm_bidirectional: bool = False
        self.gru_hidden_size: int = 128
        self.gru_num_layers: int = 2
        self.gru_dropout: float = 0.3
        self.attn_hidden_size: int = 128
        self.attn_num_layers: int = 2
        self.attn_dropout: float = 0.3
        self.cnn_dropout: float = 0.5

        # ── EVALUATION defaults ────────────────────────────────────────────
        self.n_deciles: int = 10
        self.periods_per_year: int = 52

        # ── BACKTEST defaults ──────────────────────────────────────────────
        self.bt_portfolio_type: str = "both"
        self.bt_weighting: str = "all"
        self.bt_neutrality: list = ["sector", "beta"]
        self.bt_beta_window: int = 252
        self.bt_min_beta_obs: int = 126
        self.bt_cost_bps: float = 10.0
        self.bt_borrow_cost_bps_per_year: float = 100.0
        self.bt_rf_path: Optional[str] = None
        self.bt_benchmarks: List[str] = ["MOM", "STR", "WSTR"]

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Parse the JSON file and overwrite defaults."""
        if not self._config_path.exists():
            logger.warning("Config file not found at %s — using defaults.", self._config_path)
            return

        with open(self._config_path, "r") as f:
            cfg: dict = json.load(f)

        # PIPELINE
        pl = cfg.get("PIPELINE", {})
        if pl.get("TEST_MODE") is not None:
            self.test_mode = bool(pl["TEST_MODE"])
        if pl.get("TEST_SMALL_CRSP") is not None:
            self.test_small_crsp = bool(pl["TEST_SMALL_CRSP"])
        if pl.get("SEED") is not None:
            self.seed = int(pl["SEED"])
        if pl.get("RESULTS_DIR") is not None:
            self.results_dir = pl["RESULTS_DIR"]
        if pl.get("CHECKPOINTS_DIR") is not None:
            self.checkpoints_dir = pl["CHECKPOINTS_DIR"]

        # DATA
        d = cfg.get("DATA", {})
        if d.get("DATA_SOURCE") is not None:
            self.data_source = d["DATA_SOURCE"]
        if "PARQUET_PATH" in d:
            raw = d["PARQUET_PATH"]
            if raw is not None:
                p = Path(raw)
                self.parquet_path = str(p if p.is_absolute() else self.ROOT_DIR / p)
            else:
                self.parquet_path = None
        if "PERMNOS" in d and d["PERMNOS"] is not None:
            self.permnos = list(d["PERMNOS"])
        if d.get("TICKERS"):
            self.tickers = list(d["TICKERS"])
        if d.get("START") is not None:
            self.start = d["START"]
        if d.get("END") is not None:
            self.end = d["END"]
        if "SAVE_DIR" in d:
            self.data_save_dir = d["SAVE_DIR"]

        # PREPROCESSING
        pp = cfg.get("PREPROCESSING", {})
        if pp.get("WINDOW") is not None:
            self.window = int(pp["WINDOW"])
        if pp.get("HORIZON") is not None:
            self.horizon = int(pp["HORIZON"])
        if pp.get("SCALING") is not None:
            self.scaling = pp["SCALING"]
        if pp.get("INCLUDE_VOL") is not None:
            self.include_vol = bool(pp["INCLUDE_VOL"])
        if pp.get("INCLUDE_MA") is not None:
            self.include_ma = bool(pp["INCLUDE_MA"])
        if pp.get("TRAIN_END") is not None:
            self.train_end = pp["TRAIN_END"]
        if pp.get("VAL_END") is not None:
            self.val_end = pp["VAL_END"]
        if pp.get("TRAIN_RATIO") is not None:
            self.train_ratio = float(pp["TRAIN_RATIO"])
        if pp.get("VAL_RATIO") is not None:
            self.val_ratio = float(pp["VAL_RATIO"])

        # TRAINING
        tr = cfg.get("TRAINING", {})
        if tr.get("EPOCHS") is not None:
            self.epochs = int(tr["EPOCHS"])
        if tr.get("BATCH_SIZE") is not None:
            self.batch_size = int(tr["BATCH_SIZE"])
        if tr.get("PATIENCE") is not None:
            self.patience = int(tr["PATIENCE"])
        if tr.get("LR") is not None:
            self.lr = float(tr["LR"])
        if tr.get("WEIGHT_DECAY") is not None:
            self.weight_decay = float(tr["WEIGHT_DECAY"])
        if tr.get("MODELS") is not None:
            self.models_to_train = list(tr["MODELS"])

        # MODELS
        mo = cfg.get("MODELS", {})
        mlp = mo.get("MLP", {})
        if mlp.get("HIDDEN_DIMS") is not None:
            self.mlp_hidden_dims = list(mlp["HIDDEN_DIMS"])
        if mlp.get("DROPOUT") is not None:
            self.mlp_dropout = float(mlp["DROPOUT"])

        lstm = mo.get("LSTM", {})
        if lstm.get("HIDDEN_SIZE") is not None:
            self.lstm_hidden_size = int(lstm["HIDDEN_SIZE"])
        if lstm.get("NUM_LAYERS") is not None:
            self.lstm_num_layers = int(lstm["NUM_LAYERS"])
        if lstm.get("DROPOUT") is not None:
            self.lstm_dropout = float(lstm["DROPOUT"])
        if lstm.get("BIDIRECTIONAL") is not None:
            self.lstm_bidirectional = bool(lstm["BIDIRECTIONAL"])

        gru = mo.get("GRU", {})
        if gru.get("HIDDEN_SIZE") is not None:
            self.gru_hidden_size = int(gru["HIDDEN_SIZE"])
        if gru.get("NUM_LAYERS") is not None:
            self.gru_num_layers = int(gru["NUM_LAYERS"])
        if gru.get("DROPOUT") is not None:
            self.gru_dropout = float(gru["DROPOUT"])

        attn = mo.get("ATTENTION_LSTM", {})
        if attn.get("HIDDEN_SIZE") is not None:
            self.attn_hidden_size = int(attn["HIDDEN_SIZE"])
        if attn.get("NUM_LAYERS") is not None:
            self.attn_num_layers = int(attn["NUM_LAYERS"])
        if attn.get("DROPOUT") is not None:
            self.attn_dropout = float(attn["DROPOUT"])

        cnn = mo.get("CNN", {})
        if cnn.get("DROPOUT") is not None:
            self.cnn_dropout = float(cnn["DROPOUT"])

        # EVALUATION
        ev = cfg.get("EVALUATION", {})
        if ev.get("N_DECILES") is not None:
            self.n_deciles = int(ev["N_DECILES"])
        if ev.get("PERIODS_PER_YEAR") is not None:
            self.periods_per_year = int(ev["PERIODS_PER_YEAR"])

        # BACKTEST
        bt = cfg.get("BACKTEST", {})
        if bt.get("PORTFOLIO_TYPE") is not None:
            self.bt_portfolio_type = bt["PORTFOLIO_TYPE"]
        if bt.get("WEIGHTING") is not None:
            self.bt_weighting = bt["WEIGHTING"]
        if bt.get("NEUTRALITY") is not None:
            self.bt_neutrality = list(bt["NEUTRALITY"])
        if bt.get("BETA_WINDOW") is not None:
            self.bt_beta_window = int(bt["BETA_WINDOW"])
        if bt.get("MIN_BETA_OBS") is not None:
            self.bt_min_beta_obs = int(bt["MIN_BETA_OBS"])
        if bt.get("COST_BPS") is not None:
            self.bt_cost_bps = float(bt["COST_BPS"])
        if bt.get("BORROW_COST_BPS_PER_YEAR") is not None:
            self.bt_borrow_cost_bps_per_year = float(bt["BORROW_COST_BPS_PER_YEAR"])
        if bt.get("RF_PATH") is not None:
            raw = bt["RF_PATH"]
            p = Path(raw)
            self.bt_rf_path = str(p if p.is_absolute() else self.ROOT_DIR / p)
        if bt.get("BENCHMARKS") is not None:
            self.bt_benchmarks = list(bt["BENCHMARKS"])

        # TEST_MODE overrides
        if self.test_mode:
            logger.info("TEST_MODE=True — overriding tickers/epochs/patience for fast run.")
            self.tickers  = _TEST_TICKERS
            self.start    = _TEST_START
            self.epochs   = _TEST_EPOCHS
            self.patience = _TEST_PATIENCE

        # TEST_SMALL_CRSP overrides — applied after TEST_MODE so dates always win
        if self.test_small_crsp:
            logger.info("TEST_SMALL_CRSP=True — switching to small WRDS sample (2020-2024).")
            self.data_source  = "parquet"
            self.parquet_path = str(self.ROOT_DIR / _SMALL_CRSP_PARQUET)
            self.start        = _SMALL_CRSP_START
            self.end          = _SMALL_CRSP_END
            self.train_end    = _SMALL_CRSP_TRAIN_END
            self.val_end      = _SMALL_CRSP_VAL_END

        logger.info(
            "Config loaded: %d tickers | window=%d | horizon=%d | epochs=%d",
            len(self.tickers), self.window, self.horizon, self.epochs,
        )

    def __repr__(self) -> str:
        return (
            f"Config(test={self.test_mode}, tickers={len(self.tickers)}, "
            f"window={self.window}, horizon={self.horizon}, "
            f"epochs={self.epochs}, models={self.models_to_train})"
        )
