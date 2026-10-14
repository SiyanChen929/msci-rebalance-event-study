"""Point-in-time liquidity features for MSCI close-burst weighting.

The public MSCI change lists do not contain pro-forma index weights.  This
module therefore builds *proxy* inputs from market data that was observable
before an implementation session.  Event-day bars are deliberately excluded.
The resulting fields can support research comparisons, but they are not a
substitute for licensed old/new weights, NOS, FIF, and tracked-product AUM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INTERVAL_CLOCKS = ("15:49", "15:50")


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _session_date(values: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(values)
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def _clock(values: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(values)
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp.dt.strftime("%H:%M")


def _dollar_volume(frame: pd.DataFrame) -> pd.Series:
    price = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    return price * volume


def extract_pre_event_features(
    event_date: pd.Timestamp,
    ticker: str,
    marketdata_root: str | Path,
    daily_lookback: int = 20,
    interval_session_lookback: int = 5,
    interval_clocks: tuple[str, ...] = INTERVAL_CLOCKS,
) -> dict[str, object]:
    """Extract features using sessions strictly earlier than ``event_date``."""

    event_date = pd.Timestamp(event_date).normalize()
    root = Path(marketdata_root)
    stem = f"{event_date.date()}_{_safe_ticker(str(ticker))}.parquet"
    daily_path = root / "daily" / stem
    minute_path = root / "minute" / stem
    result: dict[str, object] = {
        "feature_cutoff_date": event_date - pd.Timedelta(days=1),
        "daily_feature_sessions": 0,
        "interval_feature_sessions": 0,
        "daily_liquidity_asof": pd.NaT,
        "interval_liquidity_asof_direct": pd.NaT,
        "median_daily_dollar_volume_20d": np.nan,
        "mean_daily_dollar_volume_20d": np.nan,
        "daily_log_volatility_20d": np.nan,
        "last_pre_event_close": np.nan,
        "expected_interval_dollar_volume_direct": np.nan,
        "expected_interval_share_direct": np.nan,
    }

    if daily_path.exists():
        daily = pd.read_parquet(daily_path).copy()
        daily["_date"] = _session_date(daily["timestamp"])
        daily = daily.loc[daily["_date"].lt(event_date)].sort_values("_date")
        daily = daily.drop_duplicates("_date", keep="last").tail(daily_lookback)
        daily["_dollar_volume"] = _dollar_volume(daily)
        valid_dollar = daily["_dollar_volume"].replace([np.inf, -np.inf], np.nan).dropna()
        closes = pd.to_numeric(daily["close"], errors="coerce")
        log_returns = np.log(closes.where(closes.gt(0))).diff().dropna()
        result.update(
            daily_feature_sessions=int(len(daily)),
            daily_liquidity_asof=pd.Timestamp(daily["_date"].max()) if len(daily) else pd.NaT,
            median_daily_dollar_volume_20d=float(valid_dollar.median())
            if not valid_dollar.empty
            else np.nan,
            mean_daily_dollar_volume_20d=float(valid_dollar.mean())
            if not valid_dollar.empty
            else np.nan,
            daily_log_volatility_20d=float(log_returns.std(ddof=1))
            if len(log_returns) >= 2
            else np.nan,
            last_pre_event_close=float(closes.iloc[-1]) if len(closes) and np.isfinite(closes.iloc[-1]) else np.nan,
        )

    if minute_path.exists():
        minute = pd.read_parquet(minute_path).copy()
        minute["_date"] = _session_date(minute["timestamp"])
        minute["_clock"] = _clock(minute["timestamp"])
        minute = minute.loc[minute["_date"].lt(event_date)].sort_values("timestamp")
        session_rows: list[dict[str, float | pd.Timestamp]] = []
        for session_date, session in minute.groupby("_date", sort=True):
            interval = session.loc[session["_clock"].isin(interval_clocks)]
            clock_counts = interval["_clock"].value_counts()
            if set(clock_counts.index) != set(interval_clocks) or not clock_counts.eq(1).all():
                continue
            interval_dollar = float(_dollar_volume(interval).sum())
            session_dollar = float(_dollar_volume(session).sum())
            if not np.isfinite(interval_dollar) or interval_dollar <= 0:
                continue
            session_rows.append(
                {
                    "session_date": pd.Timestamp(session_date),
                    "interval_dollar": interval_dollar,
                    "interval_share": interval_dollar / session_dollar
                    if np.isfinite(session_dollar) and session_dollar > 0
                    else np.nan,
                }
            )
        history = pd.DataFrame(session_rows).tail(interval_session_lookback)
        if not history.empty:
            shares = pd.to_numeric(history["interval_share"], errors="coerce").dropna()
            result.update(
                interval_feature_sessions=int(len(history)),
                interval_liquidity_asof_direct=pd.Timestamp(history["session_date"].max()),
                expected_interval_dollar_volume_direct=float(history["interval_dollar"].median()),
                expected_interval_share_direct=float(shares.median()) if not shares.empty else np.nan,
            )
    return result


def build_pre_event_feature_panel(
    securities: pd.DataFrame,
    marketdata_root: str | Path,
    minimum_daily_sessions: int = 5,
    default_interval_share: float = 0.005,
) -> pd.DataFrame:
    """Attach causal proxy features and deterministic missing-data fallbacks.

    The interval-share fallback is the same-date cross-sectional median of
    other securities' prior-session histories.  If an entire date lacks those
    histories, the expanding median from earlier implementation dates is used.
    Only the first date can fall through to ``default_interval_share``.
    """

    required = {"effective_close_date", "ticker", "action"}
    missing = required.difference(securities.columns)
    if missing:
        raise ValueError(f"security panel lacks fields: {sorted(missing)}")
    base = securities.copy()
    base["effective_close_date"] = pd.to_datetime(base["effective_close_date"]).dt.normalize()
    feature_rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        feature_rows.append(
            {
                "effective_close_date": row.effective_close_date,
                "ticker": row.ticker,
                **extract_pre_event_features(
                    row.effective_close_date,
                    str(row.ticker),
                    marketdata_root,
                ),
            }
        )
    features = pd.DataFrame(feature_rows)
    panel = base.merge(features, on=["effective_close_date", "ticker"], how="left")

    usable_daily = panel["daily_feature_sessions"].fillna(0).ge(minimum_daily_sessions)
    for column in (
        "median_daily_dollar_volume_20d",
        "mean_daily_dollar_volume_20d",
        "daily_log_volatility_20d",
        "last_pre_event_close",
    ):
        panel.loc[~usable_daily, column] = np.nan

    direct_share = pd.to_numeric(panel["expected_interval_share_direct"], errors="coerce")
    direct_asof = pd.to_datetime(
        panel["interval_liquidity_asof_direct"], errors="coerce"
    )
    same_date_share = panel.assign(_direct_share=direct_share).groupby("effective_close_date")[
        "_direct_share"
    ].transform("median")
    same_date_asof = panel.assign(_direct_asof=direct_asof).groupby("effective_close_date")[
        "_direct_asof"
    ].transform("max")
    date_medians = (
        panel.assign(_direct_share=direct_share, _direct_asof=direct_asof)
        .groupby("effective_close_date", as_index=False)
        .agg(_direct_share=("_direct_share", "median"), _direct_asof=("_direct_asof", "max"))
        .sort_values("effective_close_date")
    )
    date_medians["_prior_date_share"] = (
        date_medians["_direct_share"].expanding().median().shift(1)
    )
    date_medians["_prior_date_asof"] = date_medians["effective_close_date"].shift(1)
    prior_map = date_medians.set_index("effective_close_date")["_prior_date_share"]
    prior_asof_map = date_medians.set_index("effective_close_date")["_prior_date_asof"]
    prior_share = panel["effective_close_date"].map(prior_map)
    prior_asof = pd.to_datetime(
        panel["effective_close_date"].map(prior_asof_map), errors="coerce"
    )
    panel["expected_interval_share"] = (
        direct_share.fillna(same_date_share)
        .fillna(prior_share)
        .fillna(float(default_interval_share))
        .clip(lower=0.0001, upper=0.25)
    )
    panel["interval_liquidity_source"] = np.select(
        [direct_share.notna(), same_date_share.notna(), prior_share.notna()],
        ["same_name_prior_sessions", "same_date_cross_section", "earlier_review_expanding_median"],
        default="fixed_default",
    )
    panel["interval_share_source_asof"] = (
        direct_asof.where(direct_share.notna())
        .fillna(same_date_asof.where(same_date_share.notna()))
        .fillna(prior_asof.where(prior_share.notna()))
        .fillna(pd.to_datetime(panel["feature_cutoff_date"], errors="coerce"))
    )
    median_daily = pd.to_numeric(panel["median_daily_dollar_volume_20d"], errors="coerce")
    direct_interval = pd.to_numeric(
        panel["expected_interval_dollar_volume_direct"], errors="coerce"
    )
    panel["expected_interval_dollar_volume"] = direct_interval.fillna(
        median_daily * panel["expected_interval_share"]
    )
    daily_asof = pd.to_datetime(panel["daily_liquidity_asof"], errors="coerce")
    fallback_asof = pd.concat(
        [daily_asof, panel["interval_share_source_asof"]], axis=1
    ).max(axis=1)
    panel["liquidity_asof"] = direct_asof.where(direct_interval.notna()).fillna(
        fallback_asof
    )
    panel["proxy_predicted_flow_usd_abs"] = median_daily
    panel["proxy_predicted_net_flow_usd"] = (
        np.where(panel["action"].eq("add"), 1.0, -1.0)
        * panel["proxy_predicted_flow_usd_abs"]
    )
    panel["proxy_flow_pressure"] = panel["proxy_predicted_flow_usd_abs"] / panel[
        "expected_interval_dollar_volume"
    ]
    panel["feature_point_in_time_valid"] = (
        panel["feature_cutoff_date"].lt(panel["effective_close_date"])
        & panel["liquidity_asof"].lt(panel["effective_close_date"])
        & median_daily.gt(0)
        & panel["expected_interval_dollar_volume"].gt(0)
        & panel["daily_log_volatility_20d"].gt(0)
    )
    return panel
