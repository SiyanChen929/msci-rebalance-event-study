"""Point-in-time cross-sectional ranking for MSCI rebalance candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_pre_event_features(frame: pd.DataFrame) -> None:
    """Reject feature panels whose declared predictor timestamps reach the event day."""

    required = {
        "effective_close_date",
        "ticker",
        "action",
        "feature_cutoff_date",
        "liquidity_asof",
        "feature_point_in_time_valid",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ranking feature panel lacks fields: {sorted(missing)}")
    if frame.duplicated(["effective_close_date", "ticker"]).any():
        raise ValueError("ranking feature identities are not unique")

    event = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    feature_cutoff = pd.to_datetime(frame["feature_cutoff_date"], errors="coerce")
    liquidity_asof = pd.to_datetime(frame["liquidity_asof"], errors="coerce")
    declared_valid = frame["feature_point_in_time_valid"].fillna(False).astype(bool)
    invalid_asof = declared_valid & (
        feature_cutoff.isna()
        | liquidity_asof.isna()
        | feature_cutoff.ge(event)
        | liquidity_asof.ge(event)
    )
    if invalid_asof.any():
        examples = frame.loc[invalid_asof, ["effective_close_date", "ticker"]].head(5)
        raise ValueError(
            "point-in-time ranking timestamps are invalid: "
            + ", ".join(
                f"{pd.Timestamp(row.effective_close_date).date()}:{row.ticker}"
                for row in examples.itertuples(index=False)
            )
        )


def add_liquidity_volatility_proxy_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicitly labelled proxies when actual tracker flow and market cap are absent.

    ``proxy_pressure`` treats ADV as a synthetic order size. It is not an estimate
    of MSCI passive flow. ``volatility_scaled_pressure_proxy`` applies the common
    square-root-impact shape to that synthetic pressure.
    """

    required = {
        "median_daily_dollar_volume_20d",
        "expected_interval_dollar_volume",
        "daily_log_volatility_20d",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"proxy ranking lacks fields: {sorted(missing)}")
    output = frame.copy()
    adv = pd.to_numeric(output["median_daily_dollar_volume_20d"], errors="coerce")
    interval = pd.to_numeric(
        output["expected_interval_dollar_volume"], errors="coerce"
    )
    volatility = pd.to_numeric(output["daily_log_volatility_20d"], errors="coerce")
    output["proxy_pressure"] = adv / interval
    output["volatility_scaled_pressure_proxy"] = volatility * np.sqrt(
        output["proxy_pressure"]
    )
    valid = (
        adv.gt(0)
        & interval.gt(0)
        & volatility.gt(0)
        & np.isfinite(output["proxy_pressure"])
        & np.isfinite(output["volatility_scaled_pressure_proxy"])
    )
    output.loc[~valid, ["proxy_pressure", "volatility_scaled_pressure_proxy"]] = np.nan
    return output


def add_licensed_flow_scores(
    frame: pd.DataFrame,
    *,
    flow_column: str = "predicted_net_flow_usd",
    market_cap_column: str = "float_adjusted_market_cap_usd",
    interval_column: str = "expected_interval_dollar_volume",
    volatility_column: str = "daily_log_volatility_20d",
) -> pd.DataFrame:
    """Build the requested flow/capital/volatility score from licensed inputs."""

    required = {
        "effective_close_date",
        flow_column,
        market_cap_column,
        interval_column,
        volatility_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"licensed ranking lacks fields: {sorted(missing)}")
    output = frame.copy()
    flow = pd.to_numeric(output[flow_column], errors="coerce")
    market_cap = pd.to_numeric(output[market_cap_column], errors="coerce")
    interval = pd.to_numeric(output[interval_column], errors="coerce")
    volatility = pd.to_numeric(output[volatility_column], errors="coerce")
    output["expected_flow_side"] = np.sign(flow)
    output["flow_to_interval_liquidity"] = flow.abs() / interval
    output["flow_to_float_market_cap"] = flow.abs() / market_cap
    output["flow_impact_score"] = volatility * np.sqrt(
        output["flow_to_interval_liquidity"]
    )

    group = output.groupby(pd.to_datetime(output["effective_close_date"]).dt.normalize())
    output["flow_liquidity_percentile"] = group[
        "flow_to_interval_liquidity"
    ].rank(method="average", pct=True)
    output["flow_market_cap_percentile"] = group["flow_to_float_market_cap"].rank(
        method="average", pct=True
    )
    output["volatility_percentile"] = group[volatility_column].rank(
        method="average", pct=True
    )
    output["licensed_composite_score"] = (
        0.50 * output["flow_liquidity_percentile"]
        + 0.30 * output["flow_market_cap_percentile"]
        + 0.20 * output["volatility_percentile"]
    )
    valid = (
        flow.ne(0)
        & market_cap.gt(0)
        & interval.gt(0)
        & volatility.gt(0)
        & np.isfinite(output["licensed_composite_score"])
    )
    score_columns = [
        "expected_flow_side",
        "flow_to_interval_liquidity",
        "flow_to_float_market_cap",
        "flow_impact_score",
        "flow_liquidity_percentile",
        "flow_market_cap_percentile",
        "volatility_percentile",
        "licensed_composite_score",
    ]
    output.loc[~valid, score_columns] = np.nan
    return output


def select_balanced_top_n(
    frame: pd.DataFrame,
    score_column: str,
    *,
    top_n_total: int = 10,
    minimum_per_side: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select equal counts of additions and deletions with deterministic ties."""

    if top_n_total <= 0 or top_n_total % 2:
        raise ValueError("top_n_total must be a positive even number")
    if minimum_per_side <= 0 or minimum_per_side > top_n_total // 2:
        raise ValueError("minimum_per_side is inconsistent with top_n_total")
    if score_column not in frame:
        raise ValueError(f"score column not found: {score_column}")
    required = {"effective_close_date", "ticker", "action"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selection panel lacks fields: {sorted(missing)}")

    selected: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    target_per_side = top_n_total // 2
    work = frame.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.loc[
        work["action"].isin(["add", "delete"])
        & np.isfinite(work[score_column])
    ].copy()
    for event_date, group in work.groupby("effective_close_date", sort=True):
        side_frames = {
            side: group.loc[group["action"].eq(side)].sort_values(
                [score_column, "ticker"], ascending=[False, True]
            )
            for side in ("add", "delete")
        }
        available = min(len(side_frames["add"]), len(side_frames["delete"]))
        take = min(target_per_side, available)
        eligible = take >= minimum_per_side
        for side, candidates in side_frames.items():
            boundary_tie = False
            if eligible and len(candidates) > take:
                boundary_tie = bool(
                    np.isclose(
                        float(candidates.iloc[take - 1][score_column]),
                        float(candidates.iloc[take][score_column]),
                    )
                )
            diagnostics.append(
                {
                    "effective_close_date": event_date,
                    "score": score_column,
                    "side": side,
                    "available_names": len(candidates),
                    "selected_names": take if eligible else 0,
                    "distinct_scores": int(candidates[score_column].nunique()),
                    "boundary_tie": boundary_tie,
                    "review_eligible": eligible,
                }
            )
            if eligible:
                chosen = candidates.head(take).copy()
                chosen["side_rank"] = np.arange(1, len(chosen) + 1)
                selected.append(chosen)
    output = pd.concat(selected, ignore_index=True) if selected else work.iloc[0:0].copy()
    return output, pd.DataFrame(diagnostics)
