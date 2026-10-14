"""Causal liquidity-cap diagnostics for previously frozen event portfolios.

These are desired weights under a prior-volume budget, not simulated fills.
Two traded notionals (entry and exit) consume the two-minute volume budget.
Only existing selected names are clipped; unused capital remains cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_frozen_portfolio(
    weights: pd.DataFrame, *, capital_usd: float, round_trip_participation: float,
) -> pd.DataFrame:
    if not np.isfinite(capital_usd) or capital_usd <= 0:
        raise ValueError("capital must be finite and positive")
    if not np.isfinite(round_trip_participation) or not 0 < round_trip_participation <= 1:
        raise ValueError("participation must be in (0, 1]")
    required = {"effective_close_date", "ticker", "target_weight",
                "expected_interval_dollar_volume", "liquidity_asof"}
    if not required.issubset(weights):
        raise ValueError(f"missing capacity inputs: {sorted(required - set(weights))}")
    out = weights.copy()
    out["effective_close_date"] = pd.to_datetime(out["effective_close_date"], errors="raise")
    asof = pd.to_datetime(out["liquidity_asof"], errors="raise")
    if out.duplicated(["effective_close_date", "ticker"]).any():
        raise ValueError("duplicate event-security weights")
    if asof.isna().any() or not asof.lt(out["effective_close_date"]).all():
        raise ValueError("liquidity must be known strictly before the event date")
    desired = pd.to_numeric(out["target_weight"], errors="raise")
    liquidity = pd.to_numeric(out["expected_interval_dollar_volume"], errors="raise")
    if not np.isfinite(desired).all() or not np.isfinite(liquidity).all() or (liquidity <= 0).any():
        raise ValueError("finite weights and positive prior liquidity are required")
    out["original_target_weight"] = desired
    out["capacity_weight"] = round_trip_participation * liquidity / (2.0 * capital_usd)
    out["target_weight"] = np.sign(desired) * np.minimum(desired.abs(), out["capacity_weight"])
    for _, group in out.groupby("effective_close_date", sort=True):
        long_idx = group.index[group["target_weight"].gt(0)]
        short_idx = group.index[group["target_weight"].lt(0)]
        long_gross = out.loc[long_idx, "target_weight"].sum()
        short_gross = -out.loc[short_idx, "target_weight"].sum()
        common = min(long_gross, short_gross)
        if common <= 0:
            out.loc[group.index, "target_weight"] = 0.0
        else:
            out.loc[long_idx, "target_weight"] *= common / long_gross
            out.loc[short_idx, "target_weight"] *= common / short_gross
    out["round_trip_participation"] = 2 * capital_usd * out["target_weight"].abs() / liquidity
    out["capital_usd"] = capital_usd
    out["participation_budget"] = round_trip_participation
    out["execution_status"] = "PRIOR_VOLUME_CAP_PROXY_NOT_FILLS"
    return out
