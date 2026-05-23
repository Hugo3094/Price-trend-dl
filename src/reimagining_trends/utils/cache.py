"""
cache.py
--------
Fingerprint-based caching utilities.

A fingerprint is a JSON-serializable dict capturing every input that can affect
a computation. Identical fingerprints → safe to reuse cached output.

File hashes are memoized in-process so a large parquet is only read once per run.
All I/O is wrapped in try/except: a corrupted cache falls through to recompute.
"""

import hashlib
import json
import logging
import os
import pickle
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level memoization — avoids re-hashing multi-GB files in the same run
_file_hash_cache: dict[str, str] = {}


# ── Low-level I/O helpers ─────────────────────────────────────────────────────

def file_md5(path: str) -> str:
    """Stream-hash a file in 1-MB chunks (handles multi-GB parquets efficiently)."""
    if path in _file_hash_cache:
        return _file_hash_cache[path]
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    result = h.hexdigest()
    _file_hash_cache[path] = result
    return result


def dict_hash(d: dict) -> str:
    """Stable MD5 of a JSON-serializable dict (keys sorted for determinism)."""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Failed to load fingerprint %s: %s", path, exc)
        return None


def save_json(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_pickle(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.debug("Failed to load pickle %s: %s", path, exc)
        return None


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def fingerprints_match(current: dict, cached: Optional[dict]) -> bool:
    return cached is not None and current == cached


# ── Fingerprint builders ──────────────────────────────────────────────────────

def model_fp(cfg, model_type: str, arch_params: dict) -> dict:
    """
    All inputs that can influence model training outcomes.

    Parameters
    ----------
    cfg         : Config instance
    model_type  : "mlp" | "gru" | "lstm" | "cnn"
    arch_params : model-specific hyperparameters (dropout, hidden_size, …)
    """
    data_hash = None
    if getattr(cfg, "parquet_path", None) and os.path.exists(cfg.parquet_path):
        data_hash = file_md5(cfg.parquet_path)

    return {
        # ── Data ──────────────────────────────────────────────────────────
        "data_source":  cfg.data_source,
        "data_hash":    data_hash,          # parquet content hash (None for Yahoo)
        "start":        cfg.start,
        "end":          cfg.end,
        "permnos":      sorted(cfg.permnos) if cfg.permnos else None,

        # ── Preprocessing ─────────────────────────────────────────────────
        "window":       cfg.window,
        "horizon":      cfg.horizon,
        "scaling":      cfg.scaling,
        "include_vol":  cfg.include_vol,
        "include_ma":   cfg.include_ma,
        "train_end":    cfg.train_end,
        "val_end":      cfg.val_end,
        "train_ratio":  cfg.train_ratio,
        "val_ratio":    cfg.val_ratio,
        "seed":         cfg.seed,

        # ── Architecture ──────────────────────────────────────────────────
        "model_type":   model_type,
        **arch_params,

        # ── Training hyperparameters ───────────────────────────────────────
        "epochs":        cfg.epochs,
        "batch_size":    cfg.batch_size,
        "lr":            cfg.lr,
        "weight_decay":  cfg.weight_decay,
        "patience":      cfg.patience,
    }


def scores_fp(model_fp_dict: dict, checkpoint_path: str) -> dict:
    """
    All inputs that can influence model inference scores.

    Hashes both the training-time fingerprint (what the model was trained on)
    and the checkpoint file itself (the actual weights used at inference time).
    """
    ckpt_hash = file_md5(checkpoint_path) if os.path.exists(checkpoint_path) else None
    return {
        "model_fp_hash": dict_hash(model_fp_dict),
        "checkpoint_hash": ckpt_hash,
    }


def benchmark_fp(signal_name: str, cfg) -> dict:
    """
    All inputs that can influence benchmark signal computation.

    Rolling-window lengths (252/42/21/5) are fixed in the code and identified
    implicitly by signal_name, so they do not need separate fields.
    """
    data_hash = None
    if getattr(cfg, "parquet_path", None) and os.path.exists(cfg.parquet_path):
        data_hash = file_md5(cfg.parquet_path)

    return {
        "signal_name":  signal_name,
        "data_source":  cfg.data_source,
        "data_hash":    data_hash,
        "start":        cfg.start,
        "end":          cfg.end,
        "permnos":      sorted(cfg.permnos) if cfg.permnos else None,
        "val_end":      cfg.val_end,
        "horizon":      cfg.horizon,
    }


def backtest_fp(scores_provider_fp: dict, cfg, weighting: str, port_type: str) -> dict:
    """
    All inputs that can influence a single backtest run (schedule → sim → metrics).

    scores_provider_fp is either a scores_fp (model strategy) or a benchmark_fp
    (benchmark strategy) — both are hashed compactly via dict_hash.
    """
    rf_hash = None
    if getattr(cfg, "bt_rf_path", None) and os.path.exists(cfg.bt_rf_path):
        rf_hash = file_md5(cfg.bt_rf_path)

    # Parquet provides mktcap + sectors used during portfolio construction
    parquet_hash = None
    if getattr(cfg, "parquet_path", None) and os.path.exists(cfg.parquet_path):
        parquet_hash = file_md5(cfg.parquet_path)

    return {
        # What produced the scores
        "scores_hash":                dict_hash(scores_provider_fp),

        # Auxiliary data used during portfolio construction / simulation
        "rf_hash":                    rf_hash,
        "parquet_hash":               parquet_hash,

        # Portfolio construction
        "weighting":                  weighting,
        "port_type":                  port_type,
        "n_deciles":                  cfg.n_deciles,
        "neutrality":                 sorted(cfg.bt_neutrality),
        "beta_window":                cfg.bt_beta_window,
        "min_beta_obs":               cfg.bt_min_beta_obs,

        # Transaction / borrow costs
        "cost_bps":                   cfg.bt_cost_bps,
        "borrow_cost_bps_per_year":   cfg.bt_borrow_cost_bps_per_year,

        # Rebalancing grid
        "horizon":                    cfg.horizon,
        "val_end":                    cfg.val_end,
    }