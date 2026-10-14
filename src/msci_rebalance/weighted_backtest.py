"""Portfolio accounting and diagnostics for externally supplied target weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _capped_leg_allocation(
    scores: np.ndarray,
    cap: float | np.ndarray,
    target: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    caps = np.broadcast_to(np.asarray(cap, dtype=float), scores.shape).copy()
    caps = np.where(np.isfinite(caps) & (caps > 0), caps, 0.0)
    allocation = np.zeros(len(scores), dtype=float)
    active = np.flatnonzero((scores > 0) & (caps > 0))
    remaining = float(target)
    while len(active) and remaining > 1e-15:
        proposed = remaining * scores[active] / scores[active].sum()
        room = caps[active] - allocation[active]
        binding = proposed >= room - 1e-15
        if not binding.any():
            allocation[active] += proposed
            break
        bound = active[binding]
        allocation[bound] += room[binding]
        remaining = target - float(allocation.sum())
        active = active[~binding]
    return allocation


def build_capped_leg_weights(
    securities: pd.DataFrame,
    score_column: str | None = None,
    *,
    gross_target: float = 1.0,
    max_name_weight: float = 0.10,
    cap_column: str | None = None,
    require_both_legs: bool = True,
) -> pd.DataFrame:
    """Normalize positive scores within add/delete legs and leave residual cash.

    The output remains dollar neutral whenever both legs exist.  A sparse leg
    can bind the name cap; the other leg is scaled down to the same notional.
    """

    required = {"effective_close_date", "ticker", "action"}
    missing = required.difference(securities.columns)
    if missing:
        raise ValueError(f"security panel lacks fields: {sorted(missing)}")
    if score_column is not None and score_column not in securities:
        raise ValueError(f"score column not found: {score_column}")
    if cap_column is not None and cap_column not in securities:
        raise ValueError(f"cap column not found: {cap_column}")
    if not 0 < max_name_weight <= gross_target / 2:
        raise ValueError("max_name_weight must be positive and no larger than one leg target")
    frame = securities.copy().reset_index(drop=True)
    frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    frame["_score"] = (
        pd.to_numeric(frame[score_column], errors="coerce")
        if score_column is not None
        else 1.0
    )
    frame["effective_cap_weight"] = (
        pd.to_numeric(frame[cap_column], errors="coerce")
        if cap_column is not None
        else float(max_name_weight)
    )
    frame["effective_cap_weight"] = (
        frame["effective_cap_weight"]
        .where(frame["effective_cap_weight"].gt(0), 0.0)
        .clip(upper=float(max_name_weight))
        .fillna(0.0)
    )
    frame["target_weight"] = 0.0
    leg_target = gross_target / 2.0
    for _, positions in frame.groupby("effective_close_date", sort=True).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        group = frame.loc[positions]
        long_positions = group.index[group["action"].eq("add")].to_numpy(int)
        short_positions = group.index[group["action"].eq("delete")].to_numpy(int)
        if require_both_legs and (not len(long_positions) or not len(short_positions)):
            continue
        long_allocation = _capped_leg_allocation(
            frame.loc[long_positions, "_score"].to_numpy(float),
            frame.loc[long_positions, "effective_cap_weight"].to_numpy(float),
            leg_target,
        )
        short_allocation = _capped_leg_allocation(
            frame.loc[short_positions, "_score"].to_numpy(float),
            frame.loc[short_positions, "effective_cap_weight"].to_numpy(float),
            leg_target,
        )
        if len(long_positions) and len(short_positions):
            balanced = min(float(long_allocation.sum()), float(short_allocation.sum()))
            if long_allocation.sum() > 0:
                long_allocation *= balanced / float(long_allocation.sum())
            if short_allocation.sum() > 0:
                short_allocation *= balanced / float(short_allocation.sum())
        frame.loc[long_positions, "target_weight"] = long_allocation
        frame.loc[short_positions, "target_weight"] = -short_allocation
    frame["target_abs_weight"] = frame["target_weight"].abs()
    return frame.drop(columns="_score")


def weighted_date_portfolio(
    weights: pd.DataFrame,
    *,
    underlying_return_column: str = "underlying_return",
    cost_bps: float = 0.0,
    scheduled_dates: pd.Series | list[object] | None = None,
) -> pd.DataFrame:
    """Create one independent P&L observation per implementation date."""

    required = {"effective_close_date", "ticker", "target_weight", underlying_return_column}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"weighted ledger lacks fields: {sorted(missing)}")
    frame = weights.copy()
    frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    frame["_return"] = pd.to_numeric(frame[underlying_return_column], errors="coerce")
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
    missing_selected = frame["target_weight"].abs().gt(0) & ~np.isfinite(frame["_return"])
    if missing_selected.any():
        examples = frame.loc[missing_selected, ["effective_close_date", "ticker"]].head(5)
        raise ValueError(
            "Selected positions have missing realized returns: "
            + ", ".join(
                f"{row.effective_close_date.date()}:{row.ticker}"
                for row in examples.itertuples(index=False)
            )
        )
    frame["_return"] = frame["_return"].fillna(0.0)
    frame["pnl_contribution"] = frame["target_weight"] * frame["_return"]
    records: list[dict[str, object]] = []
    for event_date, group in frame.groupby("effective_close_date", sort=True):
        absolute = group["target_weight"].abs()
        gross_exposure = float(absolute.sum())
        normalized = absolute / gross_exposure if gross_exposure > 0 else absolute
        gross_return = float(group["pnl_contribution"].sum())
        positive_contributions = group["pnl_contribution"].clip(lower=0)
        positive_pnl = positive_contributions.sum()
        absolute_pnl = group["pnl_contribution"].abs().sum()
        top_abs_pnl = float(group["pnl_contribution"].abs().max()) if len(group) else np.nan
        top_positive_pnl = float(positive_contributions.max()) if len(group) else np.nan
        records.append(
            {
                "effective_close_date": event_date,
                "gross_return": gross_return,
                "net_return": gross_return - gross_exposure * float(cost_bps) / 10_000.0,
                "cost_bps": float(cost_bps),
                "gross_exposure": gross_exposure,
                "long_exposure": float(group["target_weight"].clip(lower=0).sum()),
                "short_exposure": float(-group["target_weight"].clip(upper=0).sum()),
                "net_exposure": float(group["target_weight"].sum()),
                "n_names": int(group["target_weight"].ne(0).sum()),
                "max_abs_name_weight": float(absolute.max()) if len(group) else np.nan,
                "weight_hhi": float(np.square(normalized).sum()) if gross_exposure > 0 else np.nan,
                "effective_names": float(1.0 / np.square(normalized).sum())
                if gross_exposure > 0 and np.square(normalized).sum() > 0
                else np.nan,
                "top_name_abs_pnl": top_abs_pnl,
                "top_name_share_of_positive_pnl": top_positive_pnl / float(positive_pnl)
                if positive_pnl > 0
                else np.nan,
                "top_name_share_of_absolute_pnl": top_abs_pnl / float(absolute_pnl)
                if absolute_pnl > 0
                else np.nan,
            }
        )
    output = pd.DataFrame(records)
    if scheduled_dates is not None:
        schedule = pd.DataFrame(
            {
                "effective_close_date": pd.to_datetime(
                    pd.Series(scheduled_dates, dtype="object")
                ).dt.normalize().drop_duplicates().sort_values()
            }
        )
        output = schedule.merge(output, on="effective_close_date", how="left")
        zero_columns = [
            "gross_return",
            "net_return",
            "gross_exposure",
            "long_exposure",
            "short_exposure",
            "net_exposure",
            "n_names",
            "max_abs_name_weight",
            "weight_hhi",
            "effective_names",
            "top_name_abs_pnl",
            "top_name_share_of_positive_pnl",
            "top_name_share_of_absolute_pnl",
        ]
        for column in zero_columns:
            output[column] = output[column].fillna(0.0)
        output["cost_bps"] = output["cost_bps"].fillna(float(cost_bps))
    return output
