"""
Training pipeline tests — train.py

Covers:
  - make_dataloader : CNN permutes axes to (N, 1, H, W)
  - Trainer.predict : probabilities in [0,1], rows sum to 1
"""

import numpy as np
import pytest
import torch

from reimagining_trends.models.cnn import build_cnn
from reimagining_trends.models.mlp import build_mlp
from reimagining_trends.training.train import Trainer, make_dataloader

WINDOW = 20
N_FEAT = 6
BATCH = 16


# ---------------------------------------------------------------------------
# make_dataloader
# ---------------------------------------------------------------------------

def test_dataloader_mlp_shape():
    X = np.random.randn(50, WINDOW, N_FEAT).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int64)
    loader = make_dataloader(X, y, batch_size=BATCH, shuffle=False, model_type="mlp")
    xb, yb = next(iter(loader))
    assert xb.shape == (BATCH, WINDOW, N_FEAT)
    assert yb.shape == (BATCH,)


def test_dataloader_cnn_permutes_axes():
    """CNN loader must convert (N, H, W, 1) → (N, 1, H, W)."""
    H, W = 60, 64
    X = np.random.randn(50, H, W, 1).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int64)
    loader = make_dataloader(X, y, batch_size=BATCH, shuffle=False, model_type="cnn")
    xb, _ = next(iter(loader))
    assert xb.shape == (BATCH, 1, H, W), f"Expected (N,1,H,W), got {xb.shape}"


def test_dataloader_mlp_does_not_permute():
    """MLP loader must NOT permute axes."""
    X = np.random.randn(50, WINDOW, N_FEAT).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int64)
    loader = make_dataloader(X, y, batch_size=BATCH, shuffle=False, model_type="mlp")
    xb, _ = next(iter(loader))
    assert xb.shape[1] == WINDOW   # time dimension preserved
    assert xb.shape[2] == N_FEAT   # feature dimension preserved


# ---------------------------------------------------------------------------
# Trainer.predict
# ---------------------------------------------------------------------------

def _quick_trainer(model, model_type: str) -> Trainer:
    return Trainer(model, model_type, save_dir="checkpoints/_test")


def test_predict_probabilities_in_range():
    """All predicted probabilities must lie in [0, 1]."""
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    trainer = _quick_trainer(model, "mlp")
    X = np.random.randn(30, WINDOW, N_FEAT).astype(np.float32)
    probs = trainer.predict(X)
    assert probs.min() >= 0.0 - 1e-6
    assert probs.max() <= 1.0 + 1e-6


def test_predict_rows_sum_to_one():
    """P(down) + P(up) = 1 for every sample."""
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    trainer = _quick_trainer(model, "mlp")
    X = np.random.randn(30, WINDOW, N_FEAT).astype(np.float32)
    probs = trainer.predict(X)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(30), atol=1e-5)


def test_predict_output_shape():
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    trainer = _quick_trainer(model, "mlp")
    X = np.random.randn(30, WINDOW, N_FEAT).astype(np.float32)
    probs = trainer.predict(X)
    assert probs.shape == (30, 2)


def test_predict_cnn_probabilities_in_range():
    H, W = 60, 64
    model = build_cnn(window=WINDOW)
    trainer = _quick_trainer(model, "cnn")
    X = np.random.randn(10, H, W, 1).astype(np.float32)
    probs = trainer.predict(X)
    assert probs.shape == (10, 2)
    assert probs.min() >= 0.0 - 1e-6
    assert probs.max() <= 1.0 + 1e-6
