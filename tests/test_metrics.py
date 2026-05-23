"""
Metrics tests — evaluation/metrics.py

Covers:
  - compute_ml_metrics  : perfect classifier, known values
  - annualized_sharpe   : formula check, zero-std edge case
  - decile_portfolio_returns : returns 10 deciles, shapes correct
"""

import numpy as np
import pytest

from reimagining_trends.evaluation.metrics import (
    annualized_sharpe,
    compare_models,
    compute_ml_metrics,
    decile_portfolio_returns,
)


# ---------------------------------------------------------------------------
# compute_ml_metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    m = compute_ml_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0
    assert m["brier"] == 0.0


def test_metrics_worst_classifier():
    """Inverse labels → accuracy=0, AUC=0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    m = compute_ml_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 0.0
    assert m["auc"] == 0.0


def test_metrics_golden_values():
    """
    Analytically known case:
      y_true = [1, 0, 1, 0]
      y_pred = [1, 0, 0, 1]   (2 correct, 2 wrong → acc = 0.5)
      y_prob = [0.9, 0.1, 0.4, 0.6]
    AUC: positives are at probs [0.9, 0.4], negatives at [0.1, 0.6].
    Concordant pairs: (0.9,0.1),(0.9,0.6),(0.4,0.1) = 3; discordant: (0.4,0.6) = 1 → AUC=3/4=0.75
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.4, 0.6])
    m = compute_ml_metrics(y_true, y_pred, y_prob)
    np.testing.assert_allclose(m["accuracy"], 0.5, atol=1e-9)
    np.testing.assert_allclose(m["auc"], 0.75, atol=1e-9)


def test_metrics_keys():
    y = np.array([0, 1, 0, 1])
    m = compute_ml_metrics(y, y, y.astype(float))
    assert set(m.keys()) == {"accuracy", "f1", "auc", "brier"}


# ---------------------------------------------------------------------------
# annualized_sharpe
# ---------------------------------------------------------------------------

def test_sharpe_zero_std_returns_zero():
    """Constant returns → std=0 → Sharpe must be 0, not NaN/inf."""
    returns = np.full(52, 0.01)
    result = annualized_sharpe(returns)
    assert result == 0.0
    assert np.isfinite(result)


def test_sharpe_empty_returns_zero():
    """Empty array → Sharpe must be 0, not raise."""
    result = annualized_sharpe(np.array([]))
    assert result == 0.0


def test_sharpe_golden_value():
    """
    52 weekly returns all equal to 0.01 with std=0.02:
    We use seeded random data to get a deterministic Sharpe.
    mean = 0.01, std ≈ 0.02, Sharpe ≈ 0.01/0.02 * sqrt(52) ≈ 3.606
    """
    rng = np.random.default_rng(42)
    returns = rng.normal(0.01, 0.02, 52)
    result = annualized_sharpe(returns, periods_per_year=52)
    expected = (returns.mean() / returns.std()) * np.sqrt(52)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_sharpe_sign():
    """Negative mean returns → negative Sharpe."""
    rng = np.random.default_rng(7)
    returns = rng.normal(-0.02, 0.01, 100)
    assert annualized_sharpe(returns) < 0


# ---------------------------------------------------------------------------
# decile_portfolio_returns
# ---------------------------------------------------------------------------

def test_decile_returns_count():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, n)
    y_prob = rng.uniform(0, 1, n)
    returns = rng.normal(0.001, 0.02, n)
    df = decile_portfolio_returns(y_true, y_prob, returns, n_deciles=10)
    assert len(df) == 10


def test_decile_returns_index():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, n)
    y_prob = rng.uniform(0, 1, n)
    returns = rng.normal(0.001, 0.02, n)
    df = decile_portfolio_returns(y_true, y_prob, returns, n_deciles=10)
    assert list(df.index) == list(range(1, 11))


def test_decile_returns_columns():
    rng = np.random.default_rng(42)
    n = 500
    y_prob = rng.uniform(0, 1, n)
    returns = rng.normal(0.001, 0.02, n)
    df = decile_portfolio_returns(np.zeros(n, dtype=int), y_prob, returns)
    assert {"mean_return", "sharpe", "n"}.issubset(set(df.columns))


def test_decile_sample_counts_sum_to_total():
    rng = np.random.default_rng(42)
    n = 500
    y_prob = rng.uniform(0, 1, n)
    returns = rng.normal(0.001, 0.02, n)
    df = decile_portfolio_returns(np.zeros(n, dtype=int), y_prob, returns)
    assert df["n"].sum() == n


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def test_compare_models_sorted_by_auc():
    results = {
        "A": {"accuracy": 0.5, "f1": 0.5, "auc": 0.6, "brier": 0.25},
        "B": {"accuracy": 0.6, "f1": 0.6, "auc": 0.8, "brier": 0.20},
        "C": {"accuracy": 0.55, "f1": 0.55, "auc": 0.7, "brier": 0.22},
    }
    df = compare_models(results)
    assert list(df.index) == ["B", "C", "A"]
