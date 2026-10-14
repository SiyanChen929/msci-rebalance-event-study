"""Point-in-time portfolio constructions for the MSCI closing-window study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def estimate_pre_event_beta(
    asset_path: str | Path,
    market_path: str | Path,
    cutoff_date: object,
    *,
    lookback_returns: int = 40,
    minimum_observations: int = 20,
    shrink_to_one: float = 0.50,
    beta_floor: float = 0.25,
    beta_cap: float = 2.50,
    fallback_beta: float = 1.0,
) -> dict[str, object]:
    """Estimate a stable market beta using closes available by ``cutoff_date``.

    The raw covariance beta is shrunk toward one and clipped. Missing or short
    histories fail closed to a documented unit-beta fallback.
    """

    if not 0 <= shrink_to_one <= 1:
        raise ValueError("shrink_to_one must be between zero and one")
    if lookback_returns < 2 or minimum_observations < 2:
        raise ValueError("beta windows must contain at least two observations")
    if beta_floor <= 0 or beta_cap < beta_floor or fallback_beta <= 0:
        raise ValueError("beta bounds and fallback must be positive")

    asset_path = Path(asset_path)
    market_path = Path(market_path)
    cutoff = pd.Timestamp(cutoff_date).normalize()
    fallback = {
        "beta_raw": np.nan,
        "beta_used": float(fallback_beta),
        "beta_observations": 0,
        "beta_feature_asof": pd.NaT,
        "beta_source": "unit_beta_fallback",
    }
    if not asset_path.exists() or not market_path.exists():
        return fallback

    asset = pd.read_parquet(asset_path).copy()
    market = pd.read_parquet(market_path).copy()
    required = {"timestamp", "close"}
    if not required.issubset(asset.columns) or not required.issubset(market.columns):
        return fallback
    asset["date"] = _session_dates(asset)
    market["date"] = _session_dates(market)
    asset = (
        asset.loc[asset["date"].le(cutoff), ["date", "close"]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .tail(lookback_returns + 1)
    )
    market = (
        market.loc[market["date"].le(cutoff), ["date", "close"]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .tail(lookback_returns + 1)
    )
    if asset.empty or market.empty:
        return fallback
    asset["asset_return"] = pd.to_numeric(asset["close"], errors="coerce").pct_change()
    market["market_return"] = pd.to_numeric(market["close"], errors="coerce").pct_change()
    aligned = asset[["date", "asset_return"]].merge(
        market[["date", "market_return"]], on="date", how="inner", validate="one_to_one"
    ).dropna()
    observations = int(len(aligned))
    asof = min(asset["date"].max(), market["date"].max())
    if observations < minimum_observations:
        return {
            **fallback,
            "beta_observations": observations,
            "beta_feature_asof": asof,
        }
    variance = float(aligned["market_return"].var(ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        return {
            **fallback,
            "beta_observations": observations,
            "beta_feature_asof": asof,
        }
    raw = float(aligned["asset_return"].cov(aligned["market_return"]) / variance)
    if not np.isfinite(raw):
        return {
            **fallback,
            "beta_observations": observations,
            "beta_feature_asof": asof,
        }
    used = float(np.clip(shrink_to_one * 1.0 + (1.0 - shrink_to_one) * raw, beta_floor, beta_cap))
    return {
        "beta_raw": raw,
        "beta_used": used,
        "beta_observations": observations,
        "beta_feature_asof": asof,
        "beta_source": "pre_event_daily_covariance_shrunk_clipped",
    }


def build_beta_neutral_equal_weights(
    securities: pd.DataFrame,
    *,
    beta_column: str = "beta_used",
    gross_target: float = 1.0,
    max_name_weight: float = 0.10,
) -> pd.DataFrame:
    """Use equal weights within each leg and solve leg notionals for beta neutrality."""

    required = {"effective_close_date", "ticker", "action", beta_column}
    missing = required.difference(securities.columns)
    if missing:
        raise ValueError(f"security panel lacks fields: {sorted(missing)}")
    if gross_target <= 0 or max_name_weight <= 0:
        raise ValueError("gross target and name cap must be positive")
    frame = securities.copy().reset_index(drop=True)
    frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    frame["target_weight"] = 0.0
    beta = pd.to_numeric(frame[beta_column], errors="coerce")
    if (~np.isfinite(beta) | beta.le(0)).any():
        raise ValueError("beta-neutral construction requires finite positive betas")

    for _, index in frame.groupby("effective_close_date", sort=True).groups.items():
        index = list(index)
        group = frame.loc[index]
        longs = group.index[group["action"].eq("add")].to_numpy(int)
        shorts = group.index[group["action"].eq("delete")].to_numpy(int)
        if not len(longs) or not len(shorts):
            continue
        long_beta = float(beta.loc[longs].mean())
        short_beta = float(beta.loc[shorts].mean())
        denominator = long_beta + short_beta
        if denominator <= 0:
            continue
        base_long = gross_target * short_beta / denominator
        base_short = gross_target * long_beta / denominator
        scale = min(
            1.0,
            len(longs) * max_name_weight / base_long,
            len(shorts) * max_name_weight / base_short,
        )
        long_gross = base_long * scale
        short_gross = base_short * scale
        frame.loc[longs, "target_weight"] = long_gross / len(longs)
        frame.loc[shorts, "target_weight"] = -short_gross / len(shorts)

    frame["target_abs_weight"] = frame["target_weight"].abs()
    return frame


def build_unneutralized_signal_weights(
    securities: pd.DataFrame,
    score_column: str,
    *,
    gross_target: float = 1.0,
    max_name_weight: float = 0.10,
    require_both_legs: bool = True,
) -> pd.DataFrame:
    """Allocate across both directions by absolute signal, leaving cap residual in cash.

    Weight clipped by the name cap is deliberately not redistributed; redistribution
    across ten names at a 10% cap would collapse back to equal weights.
    """

    required = {"effective_close_date", "ticker", "action", score_column}
    missing = required.difference(securities.columns)
    if missing:
        raise ValueError(f"security panel lacks fields: {sorted(missing)}")
    if gross_target <= 0 or max_name_weight <= 0:
        raise ValueError("gross target and name cap must be positive")
    frame = securities.copy().reset_index(drop=True)
    frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"]).dt.normalize()
    frame["target_weight"] = 0.0
    score = pd.to_numeric(frame[score_column], errors="coerce")
    score = score.where(np.isfinite(score) & score.gt(0), 0.0)
    for _, index in frame.groupby("effective_close_date", sort=True).groups.items():
        index = list(index)
        group = frame.loc[index]
        if require_both_legs and not {"add", "delete"}.issubset(set(group["action"])):
            continue
        total = float(score.loc[index].sum())
        if total <= 0:
            continue
        absolute = (gross_target * score.loc[index] / total).clip(upper=max_name_weight)
        direction = group["action"].map({"add": 1.0, "delete": -1.0}).fillna(0.0)
        frame.loc[index, "target_weight"] = absolute.to_numpy(float) * direction.to_numpy(float)
    frame["target_abs_weight"] = frame["target_weight"].abs()
    return frame


def split_dollar_neutral_legs(securities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-only and short-only ledgers that exactly decompose base weights."""

    if "target_weight" not in securities:
        raise ValueError("target_weight is required")
    long_only = securities.copy()
    short_only = securities.copy()
    long_only["target_weight"] = pd.to_numeric(
        long_only["target_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    short_only["target_weight"] = pd.to_numeric(
        short_only["target_weight"], errors="coerce"
    ).fillna(0.0).clip(upper=0.0)
    long_only["target_abs_weight"] = long_only["target_weight"].abs()
    short_only["target_abs_weight"] = short_only["target_weight"].abs()
    return long_only, short_only
