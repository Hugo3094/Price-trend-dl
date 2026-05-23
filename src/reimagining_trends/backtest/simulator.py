"""
simulator.py
------------
Day-by-day portfolio P&L simulation.

Timeline per period k:
  signal_t   : close of t → model score computed
  entry_t    : close of t+1 → portfolio entered, transaction cost paid
  hold days  : t+2 … t+h → daily returns earned (gross − borrow)
  gap day    : entry_{k+1} = t+h+1 → in cash, earning rf, paying next cost
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_portfolio(
    schedule: list,
    daily_returns: pd.DataFrame,
    rf_series: pd.Series,
    cost_bps: float,
    borrow_rate_daily: float,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    schedule          : list of (entry_date, exit_date, weights_dict)
                        entry_date = signal_t + 1 trading day
                        exit_date  = signal_t + h trading days
                        weights_dict : {ticker: signed weight}
    daily_returns     : DataFrame (date x ticker), r[d] = Close[d]/Close[d-1]-1
    rf_series         : Series, daily risk-free rate indexed by date
    cost_bps          : one-way transaction cost (basis points)
    borrow_rate_daily : daily borrow rate on |short weight|

    Returns
    -------
    DataFrame indexed by date with columns:
        gross_return, net_return, turnover, is_rebalance
        portfolio_value_gross, portfolio_value_net
    """
    # ── Pre-compute drifted weights and build day → action lookup ──────────
    # drifted_weights[k] = weights of period k-1 drifted to its exit date
    # used to compute turnover when entering period k.

    prev_w_drifted: dict = {}   # {ticker: weight} at END of previous period

    # day_map[date] = ("gap", weights, turnover) | ("hold", weights)
    day_map: dict = {}

    for entry_t, exit_t, weights in schedule:
        # Turnover vs drifted weights of previous period
        all_t = set(weights) | set(prev_w_drifted)
        turnover = sum(
            abs(weights.get(t, 0.0) - prev_w_drifted.get(t, 0.0))
            for t in all_t
        )
        day_map[entry_t] = ("gap", weights, turnover)

        # Hold days: entry_t+1 through exit_t
        if exit_t in daily_returns.index and entry_t in daily_returns.index:
            hold_idx = daily_returns.loc[entry_t:exit_t].index[1:]  # skip entry_t
        else:
            hold_idx = pd.DatetimeIndex([])

        for d in hold_idx:
            day_map[d] = ("hold", weights, 0.0)

        # Drift weights to end of hold period for next turnover computation
        if len(hold_idx) > 0:
            hold_rets = daily_returns.loc[hold_idx, list(weights.keys())].fillna(0.0)
            total_r   = (1 + hold_rets).prod() - 1   # cumulative return per ticker
            port_r    = sum(w * total_r.get(t, 0.0) for t, w in weights.items())
            denom     = 1.0 + port_r
            if abs(denom) > 1e-10:
                prev_w_drifted = {
                    t: w * (1.0 + total_r.get(t, 0.0)) / denom
                    for t, w in weights.items()
                }
            else:
                prev_w_drifted = dict(weights)
        else:
            prev_w_drifted = dict(weights)

    # ── Day-by-day simulation ──────────────────────────────────────────────
    all_sim_dates = daily_returns.index.union(pd.DatetimeIndex(list(day_map.keys())))
    all_sim_dates = all_sim_dates.sort_values()

    if len(schedule) == 0:
        return pd.DataFrame()

    start_date = schedule[0][0]   # first entry date
    end_date   = schedule[-1][1]  # last exit date
    sim_dates  = all_sim_dates[(all_sim_dates >= start_date) & (all_sim_dates <= end_date)]

    records = []
    pv_gross = 1.0
    pv_net   = 1.0

    for d in sim_dates:
        if d not in daily_returns.index:
            continue

        action = day_map.get(d)

        if action is None:
            # Between first entry and a hold period — shouldn't normally occur
            rf_raw = rf_series.get(d, 0.0)
            rf = float(rf_raw.iloc[0]) if isinstance(rf_raw, pd.Series) else float(rf_raw)
            records.append(dict(date=d, gross_return=rf, net_return=rf,
                                turnover=0.0, is_rebalance=False))
            pv_gross *= (1.0 + rf)
            pv_net   *= (1.0 + rf)
            continue

        mode, weights, turnover = action

        if mode == "gap":
            # Entry day: pay transaction cost, earn rf on cash
            rf_raw = rf_series.get(d, 0.0)
            rf = float(rf_raw.iloc[0]) if isinstance(rf_raw, pd.Series) else float(rf_raw)
            cost = (cost_bps / 10_000.0) * turnover
            r_net = rf - cost
            records.append(dict(date=d, gross_return=rf, net_return=r_net,
                                turnover=turnover, is_rebalance=True))
            pv_gross *= (1.0 + rf)
            pv_net   *= (1.0 + r_net)

        else:  # "hold"
            r_day = daily_returns.loc[d]
            if isinstance(r_day, pd.DataFrame):
                r_day = r_day.iloc[0]
            r_day = r_day.fillna(0.0)  # NaN return (halt/delisting) → 0 for P&L computation
            r_gross = float(sum(w * float(r_day.get(t, 0.0)) for t, w in weights.items()))
            short_w = sum(abs(w) for w in weights.values() if w < 0)
            r_borrow = borrow_rate_daily * short_w
            r_net    = r_gross - r_borrow
            records.append(dict(date=d, gross_return=r_gross, net_return=r_net,
                                turnover=0.0, is_rebalance=False))
            pv_gross *= (1.0 + r_gross)
            pv_net   *= (1.0 + r_net)

    df = pd.DataFrame(records).set_index("date")
    df["portfolio_value_gross"] = (1.0 + df["gross_return"]).cumprod()
    df["portfolio_value_net"]   = (1.0 + df["net_return"]).cumprod()
    return df