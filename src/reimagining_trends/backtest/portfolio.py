"""
portfolio.py
------------
Portfolio construction via CVXPY.

Supports:
  - Long-short (LS): cash neutral + optional sector/beta neutral
  - Long-only  (LO): beta = 1 constraint
  - Weighting: equal | proportional | cap_weighted
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    _CVXPY_OK = True
except ImportError:
    _CVXPY_OK = False
    logger.warning("cvxpy not installed — portfolio constraints will be skipped.")


# ---------------------------------------------------------------------------
# Target weight helpers
# ---------------------------------------------------------------------------

def _ls_target(scores, market_caps, long_tickers, short_tickers, weighting):
    n_l, n_s = len(long_tickers), len(short_tickers)
    n = n_l + n_s
    w = np.zeros(n)

    if weighting == "equal":
        w[:n_l] = 1.0 / n_l
        w[n_l:] = -1.0 / n_s

    elif weighting == "proportional":
        ls = np.array([scores[t] for t in long_tickers])
        ss = np.array([1.0 - scores[t] for t in short_tickers])
        w[:n_l] = ls / ls.sum() if ls.sum() > 0 else 1.0 / n_l
        w[n_l:] = -(ss / ss.sum()) if ss.sum() > 0 else -1.0 / n_s

    elif weighting == "cap_weighted":
        lc = np.array([market_caps.get(t, 1.0) for t in long_tickers])
        sc = np.array([market_caps.get(t, 1.0) for t in short_tickers])
        w[:n_l] = lc / lc.sum() if lc.sum() > 0 else 1.0 / n_l
        w[n_l:] = -(sc / sc.sum()) if sc.sum() > 0 else -1.0 / n_s

    return w


def _lo_target(scores, market_caps, long_tickers, weighting):
    n = len(long_tickers)
    if weighting == "equal":
        return np.full(n, 1.0 / n)
    elif weighting == "proportional":
        s = np.array([scores[t] for t in long_tickers])
        return s / s.sum() if s.sum() > 0 else np.full(n, 1.0 / n)
    elif weighting == "cap_weighted":
        c = np.array([market_caps.get(t, 1.0) for t in long_tickers])
        return c / c.sum() if c.sum() > 0 else np.full(n, 1.0 / n)
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def construct_ls_portfolio(
    scores: dict,
    market_caps: dict,
    sectors: dict,
    betas: dict,
    n_decile: int,
    weighting: str,
    neutrality: list,
) -> dict:
    """
    Long-short portfolio — long top decile, short bottom decile.

    Parameters
    ----------
    scores      : {ticker: P(UP)}
    market_caps : {ticker: market_cap at t-1}
    sectors     : {ticker: gsector int}
    betas       : {ticker: beta}  (np.nan if unavailable)
    n_decile    : stocks per leg
    weighting   : "equal" | "proportional" | "cap_weighted"
    neutrality  : subset of ["sector", "beta"]

    Returns
    -------
    {ticker: signed weight}
    """
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    long_t  = ranked[:n_decile]
    short_t = ranked[-n_decile:]
    all_t   = long_t + short_t
    n_l     = len(long_t)
    n       = len(all_t)

    w_target = _ls_target(scores, market_caps, long_t, short_t, weighting)

    if not neutrality or not _CVXPY_OK:
        return dict(zip(all_t, w_target))

    long_mask  = np.zeros(n, dtype=bool); long_mask[:n_l]  = True
    short_mask = np.zeros(n, dtype=bool); short_mask[n_l:] = True
    betas_arr   = np.array([betas.get(t, np.nan) for t in all_t])
    sectors_arr = np.array([sectors.get(t, -1)   for t in all_t])

    w = cp.Variable(n)
    constraints = [
        cp.sum(w[long_mask])  == 1.0,
        cp.sum(w[short_mask]) == -1.0,
        w[long_mask]  >= 0,
        w[short_mask] <= 0,
    ]

    if "sector" in neutrality:
        for s in np.unique(sectors_arr[sectors_arr >= 0]):
            mask = sectors_arr == s
            if mask[long_mask].any() and mask[short_mask].any():
                constraints.append(cp.sum(w[mask]) == 0)

    if "beta" in neutrality:
        valid = ~np.isnan(betas_arr)
        if valid.sum() >= 2:
            constraints.append(w[valid] @ betas_arr[valid] == 0)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w_target)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return dict(zip(all_t, w.value))
    except Exception as exc:
        logger.warning("LS CVXPY failed (%s) — using target weights.", exc)

    return dict(zip(all_t, w_target))


def construct_lo_portfolio(
    scores: dict,
    market_caps: dict,
    betas: dict,
    n_decile: int,
    weighting: str,
    beta_neutral: bool,
) -> dict:
    """
    Long-only portfolio — top decile, optional beta = 1 constraint.

    Returns
    -------
    {ticker: weight}  (positive, sum to 1)
    """
    ranked   = sorted(scores, key=scores.__getitem__, reverse=True)
    long_t   = ranked[:n_decile]
    n        = len(long_t)
    w_target = _lo_target(scores, market_caps, long_t, weighting)

    if not beta_neutral or not _CVXPY_OK:
        return dict(zip(long_t, w_target))

    betas_arr = np.array([betas.get(t, np.nan) for t in long_t])
    valid = ~np.isnan(betas_arr)

    if valid.sum() < 2:
        return dict(zip(long_t, w_target))

    target_beta = float(np.nansum(w_target * betas_arr))
    if abs(target_beta - 1.0) < 0.05:
        return dict(zip(long_t, w_target))

    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        w[valid] @ betas_arr[valid] == 1.0,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w_target)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return dict(zip(long_t, w.value))
    except Exception as exc:
        logger.warning("LO CVXPY failed (%s) — using target weights.", exc)

    return dict(zip(long_t, w_target))
