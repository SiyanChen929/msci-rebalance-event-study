"""Point-in-time market-state filters for the MSCI 15:49 entry proxy.

The production-oriented feature cutoff is the close of the 15:47 start-stamped
bar, leaving one full minute before the 15:49 entry proxy.  The 15:48 bar is
available only in an explicitly labelled tight-cutoff research sensitivity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STATE_CUTOFF_CLOCK = "15:47"
TIGHT_CUTOFF_CLOCK = "15:48"
MARKET_STRESS_THRESHOLD = 0.0075
SOFT_DOWN_TO_SMALL_HOUR_RETENTION = {
    "down_to_small_hour_soft_075": 0.75,
    "down_to_small_hour_soft_050": 0.50,
    "down_to_small_hour_soft_025": 0.25,
}


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _timestamps(values: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(values)
    if getattr(timestamp.dt, "tz", None) is None:
        timestamp = timestamp.dt.tz_localize("America/New_York")
    else:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp


def _prepare_minute(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"minute data lacks fields: {sorted(missing)}")
    output = frame.copy()
    output["_timestamp"] = _timestamps(output["timestamp"])
    output["_date"] = output["_timestamp"].dt.tz_localize(None).dt.normalize()
    output["_clock"] = output["_timestamp"].dt.strftime("%H:%M")
    output["_minute"] = output["_timestamp"].dt.hour * 60 + output["_timestamp"].dt.minute
    return output.sort_values("_timestamp").drop_duplicates(["_date", "_clock"], keep="last")


def _prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily data lacks fields: {sorted(missing)}")
    output = frame.copy()
    output["_timestamp"] = _timestamps(output["timestamp"])
    output["_date"] = output["_timestamp"].dt.tz_localize(None).dt.normalize()
    return output.sort_values("_timestamp").drop_duplicates("_date", keep="last")


def _clock_minute(clock: str) -> int:
    hour, minute = (int(value) for value in clock.split(":"))
    return hour * 60 + minute


def _intraday_state(
    frame: pd.DataFrame,
    event_date: pd.Timestamp,
    cutoff_clock: str,
) -> dict[str, object] | None:
    prepared = _prepare_minute(frame)
    event_date = pd.Timestamp(event_date).normalize()
    session = prepared.loc[prepared["_date"].eq(event_date)].copy()
    if session.empty:
        return None
    cutoff_minute = _clock_minute(cutoff_clock)
    hour_start_minute = cutoff_minute - 59
    pre = session.loc[
        session["_minute"].between(_clock_minute("09:30"), cutoff_minute)
    ].copy()
    cutoff = pre.loc[pre["_minute"].eq(cutoff_minute)]
    opening = pre.loc[pre["_minute"].eq(_clock_minute("09:30"))]
    hour_open = pre.loc[pre["_minute"].eq(hour_start_minute)]
    if cutoff.empty or opening.empty or hour_open.empty:
        return None
    cutoff_row = cutoff.iloc[-1]
    opening_row = opening.iloc[0]
    hour_row = hour_open.iloc[0]
    event_open = float(opening_row["open"])
    hour_open_price = float(hour_row["open"])
    pre_close = float(cutoff_row["close"])
    if not all(np.isfinite(value) and value > 0 for value in (event_open, hour_open_price, pre_close)):
        return None
    prices = pd.to_numeric(pre["close"], errors="coerce").where(lambda values: values.gt(0))
    log_returns = np.log(prices).diff().dropna()
    dollar_volume = pd.to_numeric(pre["close"], errors="coerce") * pd.to_numeric(
        pre["volume"], errors="coerce"
    )
    hour = pre.loc[pre["_minute"].ge(hour_start_minute)]
    hour_dollar = pd.to_numeric(hour["close"], errors="coerce") * pd.to_numeric(
        hour["volume"], errors="coerce"
    )
    ten = pre.loc[pre["_minute"].ge(cutoff_minute - 9)]
    ten_dollar = pd.to_numeric(ten["close"], errors="coerce") * pd.to_numeric(
        ten["volume"], errors="coerce"
    )
    total_dollar = float(dollar_volume.sum())
    high = float(pd.to_numeric(pre["high"], errors="coerce").max())
    low = float(pd.to_numeric(pre["low"], errors="coerce").min())
    close_location = (pre_close - low) / (high - low) if high > low else np.nan
    history_rows: list[dict[str, float]] = []
    prior = prepared.loc[prepared["_date"].lt(event_date)]
    for _, prior_session in prior.groupby("_date", sort=True):
        prior_pre = prior_session.loc[
            prior_session["_minute"].between(_clock_minute("09:30"), cutoff_minute)
        ]
        if not {
            _clock_minute("09:30"),
            hour_start_minute,
            cutoff_minute,
        }.issubset(set(prior_pre["_minute"])):
            continue
        prior_dollar = pd.to_numeric(
            prior_pre["close"], errors="coerce"
        ) * pd.to_numeric(prior_pre["volume"], errors="coerce")
        prior_hour = prior_pre.loc[prior_pre["_minute"].ge(hour_start_minute)]
        prior_hour_dollar = pd.to_numeric(
            prior_hour["close"], errors="coerce"
        ) * pd.to_numeric(prior_hour["volume"], errors="coerce")
        prior_ten = prior_pre.loc[prior_pre["_minute"].ge(cutoff_minute - 9)]
        prior_ten_dollar = pd.to_numeric(
            prior_ten["close"], errors="coerce"
        ) * pd.to_numeric(prior_ten["volume"], errors="coerce")
        values = (
            float(prior_dollar.sum()),
            float(prior_hour_dollar.sum()),
            float(prior_ten_dollar.sum()),
        )
        if all(np.isfinite(value) and value > 0 for value in values):
            history_rows.append(
                {
                    "day": values[0],
                    "hour": values[1],
                    "ten": values[2],
                }
            )
    history = pd.DataFrame(history_rows).tail(5)
    day_history = float(history["day"].median()) if len(history) else np.nan
    hour_history = float(history["hour"].median()) if len(history) else np.nan
    ten_history = float(history["ten"].median()) if len(history) else np.nan
    event_hour_dollar = float(hour_dollar.sum())
    event_ten_dollar = float(ten_dollar.sum())
    return {
        "feature_asof_timestamp": cutoff_row["_timestamp"],
        "cutoff_clock": cutoff_clock,
        "event_open": event_open,
        "pre_close": pre_close,
        "day_return_pre": pre_close / event_open - 1.0,
        "hour_return_pre": pre_close / hour_open_price - 1.0,
        "intraday_realized_volatility_pre": float(log_returns.std(ddof=1))
        if len(log_returns) >= 2
        else np.nan,
        "pre_day_dollar_volume": total_dollar,
        "last_hour_dollar_volume_share_pre": float(hour_dollar.sum()) / total_dollar
        if total_dollar > 0
        else np.nan,
        "last_10m_dollar_volume_share_pre": float(ten_dollar.sum()) / total_dollar
        if total_dollar > 0
        else np.nan,
        "prior_same_clock_sessions": int(len(history)),
        "same_clock_day_rvol": total_dollar / day_history
        if np.isfinite(day_history) and day_history > 0
        else np.nan,
        "same_clock_hour_rvol": event_hour_dollar / hour_history
        if np.isfinite(hour_history) and hour_history > 0
        else np.nan,
        "same_clock_last_10m_rvol": event_ten_dollar / ten_history
        if np.isfinite(ten_history) and ten_history > 0
        else np.nan,
        "close_location_pre": close_location,
    }


def _daily_state(frame: pd.DataFrame, event_date: pd.Timestamp) -> dict[str, object]:
    prepared = _prepare_daily(frame)
    event_date = pd.Timestamp(event_date).normalize()
    history = prepared.loc[prepared["_date"].lt(event_date)].tail(20).copy()
    closes = pd.to_numeric(history["close"], errors="coerce")
    volumes = pd.to_numeric(history["volume"], errors="coerce")
    valid = closes.gt(0) & volumes.gt(0)
    dollar = (closes * volumes).where(valid)
    week_return = (
        float(closes.iloc[-1] / closes.iloc[-6] - 1.0)
        if len(closes) >= 6 and closes.iloc[-6] > 0 and closes.iloc[-1] > 0
        else np.nan
    )
    return {
        "daily_feature_asof": pd.Timestamp(history["_date"].max()) if len(history) else pd.NaT,
        "daily_sessions": int(len(history)),
        "week_return_pre": week_return,
        "median_daily_dollar_volume_20d_state": float(dollar.median())
        if dollar.notna().any()
        else np.nan,
    }


def extract_security_state(
    event_date: pd.Timestamp,
    ticker: str,
    marketdata_root: str | Path,
    *,
    cutoff_clock: str = STATE_CUTOFF_CLOCK,
) -> dict[str, object]:
    """Extract one security's state without reading any bar after ``cutoff_clock``."""

    event_date = pd.Timestamp(event_date).normalize()
    root = Path(marketdata_root)
    stem = f"{event_date.date()}_{_safe_ticker(str(ticker))}.parquet"
    minute_path = root / "minute" / stem
    daily_path = root / "daily" / stem
    result: dict[str, object] = {
        "minute_state_available": False,
        "daily_state_available": False,
        "cutoff_clock": cutoff_clock,
    }
    if minute_path.exists():
        intraday = _intraday_state(pd.read_parquet(minute_path), event_date, cutoff_clock)
        if intraday is not None:
            result.update(intraday)
            result["minute_state_available"] = True
    if daily_path.exists():
        result.update(_daily_state(pd.read_parquet(daily_path), event_date))
        result["daily_state_available"] = True
    return result


def build_state_feature_panel(
    base_ledger: pd.DataFrame,
    marketdata_root: str | Path,
) -> pd.DataFrame:
    """Attach stock and SPY state features to a frozen position ledger."""

    required = {"effective_close_date", "ticker", "action", "target_weight"}
    missing = required.difference(base_ledger.columns)
    if missing:
        raise ValueError(f"base ledger lacks fields: {sorted(missing)}")
    base = base_ledger.copy()
    base["effective_close_date"] = pd.to_datetime(base["effective_close_date"]).dt.normalize()
    if base.duplicated(["effective_close_date", "ticker"]).any():
        raise ValueError("base state-filter identities are not unique")
    spy_cache: dict[pd.Timestamp, tuple[dict[str, object], dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        if event_date not in spy_cache:
            spy_cache[event_date] = (
                extract_security_state(event_date, "SPY", marketdata_root),
                extract_security_state(
                    event_date,
                    "SPY",
                    marketdata_root,
                    cutoff_clock=TIGHT_CUTOFF_CLOCK,
                ),
            )
        stock = extract_security_state(event_date, str(row.ticker), marketdata_root)
        stock_tight = extract_security_state(
            event_date,
            str(row.ticker),
            marketdata_root,
            cutoff_clock=TIGHT_CUTOFF_CLOCK,
        )
        spy, spy_tight = spy_cache[event_date]
        output = row._asdict()
        for key, value in stock.items():
            output[f"stock_{key}"] = value
        for key, value in spy.items():
            output[f"spy_{key}"] = value
        direction = 1.0 if row.action == "add" else -1.0
        output["direction"] = direction
        for horizon in ("day", "hour", "week"):
            stock_value = stock.get(f"{horizon}_return_pre", np.nan)
            spy_value = spy.get(f"{horizon}_return_pre", np.nan)
            output[f"signed_stock_{horizon}_return_pre"] = (
                direction * float(stock_value) if np.isfinite(stock_value) else np.nan
            )
            output[f"signed_relative_{horizon}_return_pre"] = (
                direction * (float(stock_value) - float(spy_value))
                if np.isfinite(stock_value) and np.isfinite(spy_value)
                else np.nan
            )
        stock_tight_hour = stock_tight.get("hour_return_pre", np.nan)
        spy_tight_hour = spy_tight.get("hour_return_pre", np.nan)
        output["signed_relative_hour_return_pre_tight"] = (
            direction * (float(stock_tight_hour) - float(spy_tight_hour))
            if np.isfinite(stock_tight_hour) and np.isfinite(spy_tight_hour)
            else np.nan
        )
        adv = stock.get("median_daily_dollar_volume_20d_state", np.nan)
        pre_volume = stock.get("pre_day_dollar_volume", np.nan)
        output["pre_day_dollar_volume_to_adv20"] = (
            float(pre_volume) / float(adv)
            if np.isfinite(pre_volume) and np.isfinite(adv) and float(adv) > 0
            else np.nan
        )
        output["state_feature_point_in_time_valid"] = bool(
            stock.get("minute_state_available", False)
            and stock.get("daily_state_available", False)
            and spy.get("minute_state_available", False)
            and spy.get("daily_state_available", False)
        )
        rows.append(output)
    panel = pd.DataFrame(rows)
    percentile_columns = {
        "pre_day_dollar_volume_to_adv20": "volume_to_adv_side_percentile",
        "stock_last_hour_dollar_volume_share_pre": "last_hour_volume_share_side_percentile",
        "stock_intraday_realized_volatility_pre": "intraday_volatility_side_percentile",
        "stock_same_clock_day_rvol": "same_clock_day_rvol_side_percentile",
        "stock_same_clock_hour_rvol": "same_clock_hour_rvol_side_percentile",
    }
    for source, target in percentile_columns.items():
        panel[target] = panel.groupby(["effective_close_date", "action"])[source].rank(
            method="average", pct=True
        )
    return panel


def validate_state_features(panel: pd.DataFrame) -> None:
    """Fail closed when a declared production feature reaches the 15:48 bar."""

    required = {
        "effective_close_date",
        "ticker",
        "stock_feature_asof_timestamp",
        "spy_feature_asof_timestamp",
        "stock_daily_feature_asof",
        "spy_daily_feature_asof",
        "state_feature_point_in_time_valid",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"state feature panel lacks fields: {sorted(missing)}")
    event = pd.to_datetime(panel["effective_close_date"]).dt.normalize()
    cutoff = pd.to_datetime(
        event.dt.strftime("%Y-%m-%d") + " 15:48:00"
    ).dt.tz_localize("America/New_York")
    valid = panel["state_feature_point_in_time_valid"].fillna(False).astype(bool)
    stock_asof = pd.to_datetime(panel["stock_feature_asof_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    spy_asof = pd.to_datetime(panel["spy_feature_asof_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    stock_daily = pd.to_datetime(panel["stock_daily_feature_asof"], errors="coerce")
    spy_daily = pd.to_datetime(panel["spy_daily_feature_asof"], errors="coerce")
    invalid = valid & (
        stock_asof.ge(cutoff)
        | spy_asof.ge(cutoff)
        | stock_daily.ge(event)
        | spy_daily.ge(event)
    )
    if invalid.any():
        sample = panel.loc[invalid, ["effective_close_date", "ticker"]].head(5)
        raise ValueError(
            "state feature timestamp reaches the event cutoff: "
            + ", ".join(
                f"{pd.Timestamp(row.effective_close_date).date()}:{row.ticker}"
                for row in sample.itertuples(index=False)
            )
        )


def rule_mask(
    panel: pd.DataFrame,
    rule: str,
    *,
    market_stress_threshold: float = MARKET_STRESS_THRESHOLD,
) -> pd.Series:
    """Return the pre-registered, fail-closed keep mask for one rule."""

    valid = panel["state_feature_point_in_time_valid"].fillna(False).astype(bool)
    if rule == "baseline_all":
        return pd.Series(True, index=panel.index)
    mappings = {
        "stock_day": "signed_stock_day_return_pre",
        "stock_hour": "signed_stock_hour_return_pre",
        "stock_week": "signed_stock_week_return_pre",
        "relative_day": "signed_relative_day_return_pre",
        "relative_hour": "signed_relative_hour_return_pre",
        "relative_week": "signed_relative_week_return_pre",
    }
    for stem, column in mappings.items():
        if rule == f"{stem}_momentum":
            values = pd.to_numeric(panel[column], errors="coerce")
            return valid & values.notna() & values.gt(0)
        if rule == f"{stem}_reversal":
            values = pd.to_numeric(panel[column], errors="coerce")
            return valid & values.notna() & values.le(0)
    if rule == "volume_above_side_median":
        values = pd.to_numeric(panel["volume_to_adv_side_percentile"], errors="coerce")
        return valid & values.notna() & values.ge(0.5)
    if rule == "last_hour_volume_share_above_side_median":
        values = pd.to_numeric(
            panel["last_hour_volume_share_side_percentile"], errors="coerce"
        )
        return valid & values.notna() & values.ge(0.5)
    if rule == "intraday_volatility_above_side_median":
        values = pd.to_numeric(panel["intraday_volatility_side_percentile"], errors="coerce")
        return valid & values.notna() & values.ge(0.5)
    if rule == "same_clock_day_rvol_above_one":
        history_ok = pd.to_numeric(
            panel["stock_prior_same_clock_sessions"], errors="coerce"
        ).ge(3)
        values = pd.to_numeric(panel["stock_same_clock_day_rvol"], errors="coerce")
        return valid & history_ok & values.notna() & values.gt(1.0)
    if rule == "same_clock_hour_rvol_above_one":
        history_ok = pd.to_numeric(
            panel["stock_prior_same_clock_sessions"], errors="coerce"
        ).ge(3)
        values = pd.to_numeric(panel["stock_same_clock_hour_rvol"], errors="coerce")
        return valid & history_ok & values.notna() & values.gt(1.0)
    if rule == "relative_hour_momentum_same_clock_rvol":
        return rule_mask(panel, "relative_hour_momentum") & rule_mask(
            panel, "same_clock_hour_rvol_above_one"
        )
    if rule == "relative_hour_momentum_high_volume":
        return rule_mask(panel, "relative_hour_momentum") & rule_mask(
            panel, "volume_above_side_median"
        )
    if rule == "relative_day_momentum_high_volume":
        return rule_mask(panel, "relative_day_momentum") & rule_mask(
            panel, "volume_above_side_median"
        )
    down_to_small = panel["action"].eq("delete") & panel["change_type"].eq(
        "down_to_small"
    )
    deletion = panel["action"].eq("delete")
    if rule == "down_to_small_hour_confirmation":
        confirmation = rule_mask(panel, "relative_hour_momentum")
        return valid & (~down_to_small | confirmation)
    if rule == "down_to_small_day_confirmation":
        confirmation = rule_mask(panel, "relative_day_momentum")
        return valid & (~down_to_small | confirmation)
    if rule == "down_to_small_hour_or_day_confirmation":
        confirmation = rule_mask(panel, "relative_hour_momentum") | rule_mask(
            panel, "relative_day_momentum"
        )
        return valid & (~down_to_small | confirmation)
    if rule == "down_to_small_hour_and_day_confirmation":
        confirmation = rule_mask(panel, "relative_hour_momentum") & rule_mask(
            panel, "relative_day_momentum"
        )
        return valid & (~down_to_small | confirmation)
    if rule == "deletion_relative_hour_confirmation":
        confirmation = rule_mask(panel, "relative_hour_momentum")
        return valid & (~deletion | confirmation)
    if rule == "deletion_relative_day_confirmation":
        confirmation = rule_mask(panel, "relative_day_momentum")
        return valid & (~deletion | confirmation)
    spy_day = pd.to_numeric(panel["spy_day_return_pre"], errors="coerce").abs()
    if rule == "calm_market_day":
        return valid & spy_day.notna() & spy_day.le(float(market_stress_threshold))
    if rule == "stressed_market_day":
        return valid & spy_day.notna() & spy_day.gt(float(market_stress_threshold))
    if rule == "relative_hour_momentum_tight":
        values = pd.to_numeric(panel["signed_relative_hour_return_pre_tight"], errors="coerce")
        return valid & values.notna() & values.gt(0)
    raise ValueError(f"unknown state-filter rule: {rule}")


def rule_multiplier(
    panel: pd.DataFrame,
    rule: str,
    *,
    market_stress_threshold: float = MARKET_STRESS_THRESHOLD,
) -> pd.Series:
    """Return a point-in-time position multiplier in the closed interval [0, 1]."""

    if rule not in SOFT_DOWN_TO_SMALL_HOUR_RETENTION:
        return rule_mask(
            panel, rule, market_stress_threshold=market_stress_threshold
        ).astype(float)
    valid = panel["state_feature_point_in_time_valid"].fillna(False).astype(bool)
    down_to_small = panel["action"].eq("delete") & panel["change_type"].eq(
        "down_to_small"
    )
    confirmation = rule_mask(panel, "relative_hour_momentum")
    retention = float(SOFT_DOWN_TO_SMALL_HOUR_RETENTION[rule])
    multiplier = pd.Series(0.0, index=panel.index, dtype=float)
    multiplier.loc[valid & ~down_to_small] = 1.0
    multiplier.loc[valid & down_to_small & confirmation] = 1.0
    multiplier.loc[valid & down_to_small & ~confirmation] = retention
    return multiplier


def apply_filter_and_balance(
    panel: pd.DataFrame,
    rule: str,
    *,
    market_stress_threshold: float = MARKET_STRESS_THRESHOLD,
) -> pd.DataFrame:
    """Apply a gate, retain original caps, and restore dollar neutrality by scaling down."""

    output = panel.copy()
    output["filter_rule"] = rule
    output["filter_multiplier"] = rule_multiplier(
        output, rule, market_stress_threshold=market_stress_threshold
    )
    if output["filter_multiplier"].lt(0).any() or output["filter_multiplier"].gt(1).any():
        raise ValueError("state-filter multiplier must remain between zero and one")
    output["filter_keep"] = output["filter_multiplier"].gt(0)
    base_weight = pd.to_numeric(output["target_weight"], errors="coerce").fillna(0.0)
    output["filtered_target_weight"] = base_weight * output["filter_multiplier"]
    for _, positions in output.groupby("effective_close_date", sort=True).groups.items():
        positions = list(positions)
        weights = output.loc[positions, "filtered_target_weight"]
        long_total = float(weights.clip(lower=0).sum())
        short_total = float(-weights.clip(upper=0).sum())
        balanced = min(long_total, short_total)
        if balanced <= 0:
            output.loc[positions, "filtered_target_weight"] = 0.0
            continue
        long_positions = [position for position in positions if output.at[position, "filtered_target_weight"] > 0]
        short_positions = [position for position in positions if output.at[position, "filtered_target_weight"] < 0]
        output.loc[long_positions, "filtered_target_weight"] *= balanced / long_total
        output.loc[short_positions, "filtered_target_weight"] *= balanced / short_total
    output["filtered_target_abs_weight"] = output["filtered_target_weight"].abs()
    return output
