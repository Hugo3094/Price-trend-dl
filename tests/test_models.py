"""
Model architecture tests — mlp.py / lstm.py / cnn.py

Covers:
  - Output shape (batch, 2) for all architectures
  - AttentionLSTM returns (logits, weights) with correct shapes
  - Attention weights sum to 1 per sample
  - CNN accepts all three window/image-size combinations
  - GradCAM output is in [0, 1] with correct spatial dimensions
"""

import numpy as np
import pytest
import torch

from reimagining_trends.models.cnn import GradCAM, build_cnn
from reimagining_trends.models.lstm import build_attention_lstm, build_gru, build_lstm
from reimagining_trends.models.mlp import build_mlp

BATCH = 8
WINDOW = 20
N_FEAT = 6  # O/H/L/C/V/MA


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_output_shape():
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH, 2)


def test_mlp_accepts_flat_input():
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW * N_FEAT)
    out = model(x)
    assert out.shape == (BATCH, 2)


def test_mlp_output_finite():
    model = build_mlp(window=WINDOW, n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    out = model(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

def test_gru_output_shape():
    model = build_gru(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH, 2)


def test_gru_output_finite():
    model = build_gru(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    assert torch.isfinite(model(x)).all()


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_lstm_output_shape():
    model = build_lstm(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH, 2)


def test_lstm_output_finite():
    model = build_lstm(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    assert torch.isfinite(model(x)).all()


# ---------------------------------------------------------------------------
# AttentionLSTM
# ---------------------------------------------------------------------------

def test_attention_lstm_output_shapes():
    model = build_attention_lstm(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    logits, weights = model(x)
    assert logits.shape == (BATCH, 2)
    assert weights.shape == (BATCH, WINDOW)


def test_attention_lstm_weights_sum_to_one():
    """Softmax over time: each sample's weights must sum to 1."""
    model = build_attention_lstm(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    _, weights = model(x)
    sums = weights.sum(dim=1)
    np.testing.assert_allclose(sums.detach().numpy(), np.ones(BATCH), atol=1e-5)


def test_attention_lstm_weights_non_negative():
    model = build_attention_lstm(n_features=N_FEAT)
    x = torch.randn(BATCH, WINDOW, N_FEAT)
    _, weights = model(x)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# CNN — all three window/image-size combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window,H,W", [
    (5,  15,  32),
    (20, 60,  64),
    (60, 180, 96),
])
def test_cnn_output_shape(window, H, W):
    model = build_cnn(window=window)
    x = torch.randn(BATCH, 1, H, W)
    out = model(x)
    assert out.shape == (BATCH, 2), f"window={window}: {out.shape}"


@pytest.mark.parametrize("window,H,W", [
    (5,  15,  32),
    (20, 60,  64),
    (60, 180, 96),
])
def test_cnn_output_finite(window, H, W):
    model = build_cnn(window=window)
    x = torch.randn(BATCH, 1, H, W)
    assert torch.isfinite(model(x)).all()


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def test_gradcam_output_shape():
    model = build_cnn(window=20)
    gradcam = GradCAM(model)
    x = torch.randn(1, 1, 60, 64)
    cam = gradcam.generate(x)
    assert cam.shape == (60, 64)


def test_gradcam_range():
    """Grad-CAM output normalised to [0, 1]."""
    model = build_cnn(window=20)
    gradcam = GradCAM(model)
    x = torch.randn(1, 1, 60, 64)
    cam = gradcam.generate(x)
    assert float(cam.min()) >= 0.0 - 1e-6
    assert float(cam.max()) <= 1.0 + 1e-6


def test_gradcam_deterministic():
    """Same input → same CAM (model weights fixed, no stochasticity in eval)."""
    torch.manual_seed(0)
    model = build_cnn(window=20)
    model.eval()
    gradcam = GradCAM(model)
    x = torch.randn(1, 1, 60, 64)
    cam1 = gradcam.generate(x.clone())

    gradcam2 = GradCAM(model)
    cam2 = gradcam2.generate(x.clone())
    np.testing.assert_allclose(cam1.detach().numpy(), cam2.detach().numpy(), atol=1e-6)
