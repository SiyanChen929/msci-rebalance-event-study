"""Point-in-time portfolio weights for the MSCI close-burst strategy.

The module deliberately separates signal strength from executable capacity.
Licensed/proprietary flow forecasts receive first priority.  When they are not
available, the fallback uses the published add/delete direction and liquidity
computed strictly before the implementation date.  Event-day volume is never
read as a proxy input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightingConfig:
    """Frozen sizing controls.

    Weights are fractions of ``capital_usd``.  With both legs present, each leg
    receives half of ``gross_target`` and the smaller capacity-constrained leg
    determines the size of the other, preserving dollar neutrality.
    """

    capital_usd: float = 1_000_000.0
    gross_target: float = 1.0
    max_name_weight: float = 0.10
    max_interval_participation: float = 0.02
    missing_liquidity_max_weight: float = 0.01
    interval_share_of_adv: float = 0.005
    proxy_liquidity_power: float = 0.5
    require_both_legs: bool = True
    require_asof_for_true_inputs: bool = False

    def __post_init__(self) -> None:
        positive = {
            "capital_usd": self.capital_usd,
            "gross_target": self.gross_target,
            "max_name_weight": self.max_name_weight,
            "max_interval_participation": self.max_interval_participation,
            "missing_liquidity_max_weight": self.missing_liquidity_max_weight,
            "interval_share_of_adv": self.interval_share_of_adv,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError("Weighting controls must be positive: " + ", ".join(invalid))
        if self.gross_target > 2.0:
            raise ValueError("gross_target above 2.0 requires an explicitly separate leverage policy")
        if not 0.0 <= self.proxy_liquidity_power <= 1.0:
            raise ValueError("proxy_liquidity_power must be between zero and one")


@dataclass(frozen=True)
class WeightingResult:
    """Per-name target weights plus one aggregate diagnostic row per review."""

    weights: pd.DataFrame
    diagnostics: pd.DataFrame


SIGNAL_CONFIDENCE = {
    "predicted_flow_pressure": 1.00,
    "predicted_flow_pre_event_liquidity": 0.90,
    "predicted_flow_unscaled": 0.75,
    "delta_weight_pressure_proxy": 0.70,
    "delta_weight_unscaled": 0.50,
    "action_liquidity_proxy": 0.35,
    "action_equal_fallback": 0.20,
    "neutral": 0.00,
}


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def estimate_pre_event_liquidity(
    daily: pd.DataFrame,
    effective_close_date: str | pd.Timestamp,
    *,
    lookback_sessions: int = 20,
    interval_share_of_adv: float = 0.005,
) -> dict[str, Any]:
    """Estimate interval capacity from completed sessions before the event.

    The event session and all later rows are removed before the lookback is
    selected.  This makes the result invariant to any event-day volume spike.
    """

    required = {"timestamp", "close", "volume"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError("Daily liquidity input is missing: " + ", ".join(sorted(missing)))
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    if not np.isfinite(interval_share_of_adv) or interval_share_of_adv <= 0:
        raise ValueError("interval_share_of_adv must be positive")

    event_date = pd.Timestamp(effective_close_date).normalize()
    frame = daily.copy()
    frame["_session_date"] = _session_dates(frame)
    frame["_dollar_volume"] = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(
        frame["volume"], errors="coerce"
    )
    history = frame.loc[
        frame["_session_date"].lt(event_date)
        & frame["_dollar_volume"].gt(0)
        & np.isfinite(frame["_dollar_volume"])
    ].sort_values(["_session_date", "timestamp"])
    history = history.drop_duplicates("_session_date", keep="last").tail(lookback_sessions)
    if history.empty:
        return {
            "pre_event_adv20_usd": np.nan,
            "pre_event_interval_dollar_volume": np.nan,
            "liquidity_asof": pd.NaT,
            "liquidity_observations": 0,
        }

    adv = float(history["_dollar_volume"].median())
    return {
        "pre_event_adv20_usd": adv,
        "pre_event_interval_dollar_volume": adv * interval_share_of_adv,
        "liquidity_asof": pd.Timestamp(history["_session_date"].max()),
        "liquidity_observations": int(len(history)),
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _known_by(row: pd.Series, timestamp_column: str, cutoff: pd.Timestamp | None) -> tuple[bool, bool]:
    """Return (usable, verified) for a point-in-time field."""

    if cutoff is None:
        return True, False
    known_at = _timestamp(row.get(timestamp_column))
    if known_at is None:
        return True, False
    return known_at <= cutoff, True


def _pre_event_liquidity(row: pd.Series, config: WeightingConfig) -> tuple[float, str, bool]:
    """Return a safe liquidity forecast, its provenance, and lookahead rejection."""

    event_date = _timestamp(row.get("effective_close_date"))
    liquidity_asof = _timestamp(row.get("liquidity_asof"))
    rejected = bool(
        event_date is not None
        and liquidity_asof is not None
        and liquidity_asof.normalize() >= event_date.normalize()
    )
    if rejected:
        return np.nan, "missing", True
    if _positive(row.get("pre_event_interval_dollar_volume")):
        return float(row["pre_event_interval_dollar_volume"]), "pre_event_interval", False
    if _positive(row.get("pre_event_adv20_usd")):
        return (
            float(row["pre_event_adv20_usd"]) * config.interval_share_of_adv,
            "pre_event_adv20_scaled",
            False,
        )
    if _positive(row.get("pre_event_median_dollar_volume")):
        return (
            float(row["pre_event_median_dollar_volume"]) * config.interval_share_of_adv,
            "pre_event_median_daily_scaled",
            False,
        )
    return np.nan, "missing", False


def _resolve_row(
    row: pd.Series,
    config: WeightingConfig,
    freeze_time: pd.Timestamp | None,
) -> dict[str, Any]:
    row_cutoff = _timestamp(row.get("signal_freeze_ts"))
    if row_cutoff is None:
        row_cutoff = freeze_time
    if config.require_asof_for_true_inputs and row_cutoff is None:
        raise ValueError(
            "require_asof_for_true_inputs=True requires signal_freeze_ts "
            "on every row or an explicit freeze_time"
        )
    signal_usable, signal_verified = _known_by(row, "signal_asof", row_cutoff)
    interval_usable, interval_verified = _known_by(row, "expected_interval_asof", row_cutoff)

    predicted_supplied = _finite(row.get("predicted_net_flow_usd"))
    delta_supplied = _finite(row.get("delta_weight"))
    if config.require_asof_for_true_inputs and row_cutoff is not None:
        if predicted_supplied and not signal_verified:
            signal_usable = False
        if delta_supplied and not signal_verified:
            signal_usable = False
        if _positive(row.get("expected_interval_dollar_volume")) and not interval_verified:
            interval_usable = False

    safe_liquidity, safe_source, liquidity_lookahead_rejected = _pre_event_liquidity(row, config)
    expected_liquidity = (
        float(row["expected_interval_dollar_volume"])
        if interval_usable and _positive(row.get("expected_interval_dollar_volume"))
        else safe_liquidity
    )
    liquidity_source = (
        "expected_interval_dollar_volume"
        if interval_usable and _positive(row.get("expected_interval_dollar_volume"))
        else safe_source
    )

    action = str(row.get("action", "")).strip().lower()
    action_side = 1.0 if action in {"add", "addition"} else -1.0 if action in {"delete", "deletion"} else 0.0
    signal_lookahead_rejected = bool((predicted_supplied or delta_supplied) and not signal_usable)

    if signal_usable and predicted_supplied:
        flow = float(row["predicted_net_flow_usd"])
        side = float(np.sign(flow))
        if side == 0:
            source, strength = "neutral", 0.0
        elif _positive(row.get("expected_interval_dollar_volume")) and interval_usable:
            source, strength = "predicted_flow_pressure", abs(flow) / expected_liquidity
        elif _positive(safe_liquidity):
            source, strength = "predicted_flow_pre_event_liquidity", abs(flow) / safe_liquidity
        else:
            source, strength = "predicted_flow_unscaled", abs(flow)
    elif signal_usable and delta_supplied:
        delta = float(row["delta_weight"])
        side = float(np.sign(delta))
        if side == 0:
            source, strength = "neutral", 0.0
        elif _positive(expected_liquidity):
            source, strength = "delta_weight_pressure_proxy", abs(delta) / expected_liquidity
        else:
            source, strength = "delta_weight_unscaled", abs(delta)
    else:
        side = action_side
        if side == 0:
            source, strength = "neutral", 0.0
        elif _positive(safe_liquidity):
            source = "action_liquidity_proxy"
            strength = safe_liquidity ** (-config.proxy_liquidity_power)
        else:
            source, strength = "action_equal_fallback", 1.0

    if not _positive(strength):
        strength = 0.0
    participation_cap_weight = (
        expected_liquidity * config.max_interval_participation / config.capital_usd
        if _positive(expected_liquidity)
        else np.nan
    )
    effective_cap = min(
        config.max_name_weight,
        participation_cap_weight
        if _positive(participation_cap_weight)
        else config.missing_liquidity_max_weight,
    )
    if side == 0:
        effective_cap = 0.0

    borrow_value = row.get("borrow_available")
    borrow_known = borrow_value is not None and not pd.isna(borrow_value)
    borrow_available = True
    if borrow_known:
        if isinstance(borrow_value, str):
            normalized_borrow = borrow_value.strip().lower()
            if normalized_borrow in {"false", "0", "no", "n", "unavailable"}:
                borrow_available = False
            elif normalized_borrow not in {"true", "1", "yes", "y", "available"}:
                raise ValueError(f"Unrecognized borrow_available value: {borrow_value}")
        else:
            borrow_available = bool(borrow_value)
    borrow_unavailable_rejected = bool(side < 0 and borrow_known and not borrow_available)
    if borrow_unavailable_rejected:
        effective_cap = 0.0

    fallback_reasons: list[str] = []
    if source.startswith("action_"):
        fallback_reasons.append("missing_usable_true_flow")
    if liquidity_source == "missing":
        fallback_reasons.append("missing_pre_event_liquidity")
    if signal_lookahead_rejected:
        fallback_reasons.append("signal_after_freeze_rejected")
    if liquidity_lookahead_rejected:
        fallback_reasons.append("event_day_liquidity_rejected")
    if borrow_unavailable_rejected:
        fallback_reasons.append("borrow_unavailable")

    signal_kind = str(row.get("signal_kind", "unspecified")).strip().lower()

    return {
        "side": side,
        "signal_source": source,
        "signal_confidence": SIGNAL_CONFIDENCE[source],
        "raw_signal_strength": strength,
        "expected_interval_dollar_volume_used": expected_liquidity,
        "liquidity_source": liquidity_source,
        "participation_cap_weight": participation_cap_weight,
        "effective_cap_weight": effective_cap,
        "signal_point_in_time_verified": signal_verified,
        "interval_point_in_time_verified": interval_verified,
        "signal_lookahead_rejected": signal_lookahead_rejected,
        "liquidity_lookahead_rejected": liquidity_lookahead_rejected,
        "flow_direction_conflicts_with_action": bool(side and action_side and side != action_side),
        "input_signal_kind": signal_kind,
        "borrow_known": borrow_known,
        "borrow_unavailable_rejected": borrow_unavailable_rejected,
        "fallback_reason": ";".join(fallback_reasons),
    }


def _allocation_scores(frame: pd.DataFrame) -> np.ndarray:
    """Normalize incomparable fallback sources before combining them."""

    scores = np.zeros(len(frame), dtype=float)
    for source, positions in frame.groupby("signal_source", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        raw = frame.iloc[positions]["raw_signal_strength"].to_numpy(float)
        positive = raw[np.isfinite(raw) & (raw > 0)]
        scale = float(np.median(positive)) if len(positive) else 1.0
        normalized = np.clip(raw / scale, 0.0, 5.0)
        scores[positions] = normalized * float(SIGNAL_CONFIDENCE.get(source, 0.0))
    return scores


def _capped_proportional(scores: np.ndarray, caps: np.ndarray, target: float) -> np.ndarray:
    allocation = np.zeros(len(scores), dtype=float)
    active = np.flatnonzero((scores > 0) & (caps > 0))
    remaining = float(target)
    while len(active) and remaining > 1e-15:
        active_scores = scores[active]
        score_total = float(active_scores.sum())
        if score_total <= 0:
            active_scores = np.ones(len(active), dtype=float)
            score_total = float(len(active))
        proposed = remaining * active_scores / score_total
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


def _identity_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ("effective_close_date", "security_id", "ticker") if column in frame]
    if not any(column in frame for column in ("security_id", "ticker")):
        raise ValueError("Candidates require security_id or ticker")
    return columns


def build_target_weights(
    candidates: pd.DataFrame,
    config: WeightingConfig | None = None,
    *,
    freeze_time: str | pd.Timestamp | None = None,
) -> WeightingResult:
    """Build deterministic, capacity-constrained review-date weights.

    Signal priority is ``predicted_net_flow_usd``, then ``delta_weight``, then
    an add/delete liquidity proxy.  Unsafe event-day columns such as
    ``day_volume`` and ``last10_volume`` are intentionally never referenced.
    """

    config = config or WeightingConfig()
    if candidates.empty:
        return WeightingResult(pd.DataFrame(), pd.DataFrame())
    frame = candidates.copy()
    identity = _identity_columns(frame)
    if frame.duplicated(identity).any():
        raise ValueError("Candidate identity must be unique: " + ", ".join(identity))
    if "effective_close_date" in frame:
        frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    frame = frame.sort_values(identity, kind="mergesort").reset_index(drop=True)
    parsed_freeze = _timestamp(freeze_time)

    resolved = pd.DataFrame(
        [_resolve_row(row, config, parsed_freeze) for _, row in frame.iterrows()]
    )
    frame = pd.concat([frame, resolved], axis=1)
    group_column = "effective_close_date" if "effective_close_date" in frame else None
    groups = frame.groupby(group_column, sort=True, dropna=False) if group_column else [("all", frame)]

    weighted_groups: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    leg_target = config.gross_target / 2.0
    for group_key, group in groups:
        group = group.copy().reset_index(drop=True)
        group["allocation_score"] = _allocation_scores(group)
        caps = group["effective_cap_weight"].to_numpy(float)
        scores = group["allocation_score"].to_numpy(float)
        side = group["side"].to_numpy(float)
        long_mask = side > 0
        short_mask = side < 0
        long_allocation = np.zeros(len(group), dtype=float)
        short_allocation = np.zeros(len(group), dtype=float)
        if long_mask.any():
            long_allocation[long_mask] = _capped_proportional(
                scores[long_mask], caps[long_mask], leg_target
            )
        if short_mask.any():
            short_allocation[short_mask] = _capped_proportional(
                scores[short_mask], caps[short_mask], leg_target
            )

        both_legs = bool(long_mask.any() and short_mask.any())
        if config.require_both_legs and not both_legs:
            long_allocation[:] = 0.0
            short_allocation[:] = 0.0
        elif both_legs:
            balanced_leg = min(float(long_allocation.sum()), float(short_allocation.sum()))
            if long_allocation.sum() > 0:
                long_allocation *= balanced_leg / float(long_allocation.sum())
            if short_allocation.sum() > 0:
                short_allocation *= balanced_leg / float(short_allocation.sum())

        absolute = long_allocation + short_allocation
        signed = long_allocation - short_allocation
        group["target_abs_weight"] = absolute
        group["target_weight"] = signed
        group["target_notional_usd"] = signed * config.capital_usd
        group["capacity_binding"] = (
            group["target_abs_weight"].gt(0)
            & np.isclose(group["target_abs_weight"], group["effective_cap_weight"], atol=1e-12)
        )
        weighted_groups.append(group)

        long_weight = float(long_allocation.sum())
        short_weight = float(short_allocation.sum())
        gross = long_weight + short_weight
        diagnostic_rows.append(
            {
                "effective_close_date": group_key if group_column else pd.NaT,
                "candidate_names": len(group),
                "long_names": int(long_mask.sum()),
                "short_names": int(short_mask.sum()),
                "both_legs_present": both_legs,
                "long_weight": long_weight,
                "short_weight": short_weight,
                "gross_weight": gross,
                "net_weight": long_weight - short_weight,
                "cash_weight": max(0.0, 1.0 - gross),
                "target_gross_weight": config.gross_target,
                "capacity_binding_names": int(group["capacity_binding"].sum()),
                "true_flow_names": int(group["input_signal_kind"].eq("licensed_net_flow").sum()),
                "delta_weight_names": int(group["signal_source"].str.startswith("delta_").sum()),
                "proxy_names": int(
                    (
                        group["input_signal_kind"].str.contains("proxy", na=False)
                        | group["signal_source"].str.startswith("action_")
                    ).sum()
                ),
                "borrow_unavailable_rejected_names": int(
                    group["borrow_unavailable_rejected"].sum()
                ),
                "missing_liquidity_names": int(group["liquidity_source"].eq("missing").sum()),
                "signal_lookahead_rejected_names": int(group["signal_lookahead_rejected"].sum()),
                "liquidity_lookahead_rejected_names": int(
                    group["liquidity_lookahead_rejected"].sum()
                ),
                "balanced_when_two_sided": bool(
                    not both_legs or np.isclose(long_weight, short_weight, atol=1e-12)
                ),
            }
        )

    weights = pd.concat(weighted_groups, ignore_index=True).sort_values(identity, kind="mergesort")
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        "effective_close_date", kind="mergesort", na_position="first"
    )
    return WeightingResult(weights.reset_index(drop=True), diagnostics.reset_index(drop=True))
