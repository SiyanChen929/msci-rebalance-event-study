#!/usr/bin/env python3
"""Validate the motivating SNDK observation against full post-listing history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from msci_rebalance.marketdata import _request_candles, load_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-minute-history", action="store_true")
    args = parser.parse_args()
    token = load_token(args.env_file)
    daily, status, error = _request_candles(
        "SNDK",
        "D",
        pd.Timestamp("2025-02-24"),
        pd.Timestamp("2026-08-31"),
        token,
    )
    if daily.empty:
        raise RuntimeError(f"SNDK daily request failed ({status}): {error}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "sndk_daily_history.csv", index=False)

    minute = pd.read_parquet(args.minute_file)
    minute["clock"] = pd.to_datetime(minute["timestamp"]).dt.strftime("%H:%M")
    entry = float(minute.loc[minute["clock"].eq("15:49"), "open"].iloc[0])
    last_minute_close = float(minute.loc[minute["clock"].eq("15:59"), "close"].iloc[0])
    daily_dates = pd.to_datetime(daily["timestamp"]).dt.tz_localize(None).dt.normalize()
    event_daily = daily.loc[daily_dates.eq(pd.Timestamp("2026-08-31"))].iloc[-1]
    daily_close_proxy = float(event_daily["close"])
    last10 = minute.loc[minute["clock"].between("15:50", "15:59"), "volume"].sum()
    event_volume = float(event_daily["volume"])
    prior = daily.loc[daily_dates.lt(pd.Timestamp("2026-08-31"))].copy()
    prior_max_idx = prior["volume"].idxmax()
    summary = {
        "ticker": "SNDK",
        "event_date": "2026-08-31",
        "daily_sessions_since_listing": len(daily),
        "return_1549_open_to_daily_close_proxy": daily_close_proxy / entry - 1.0,
        "return_1549_open_to_last_minute_diagnostic": last_minute_close / entry - 1.0,
        "entry_1549_open": entry,
        "daily_close_proxy": daily_close_proxy,
        "last_minute_close": last_minute_close,
        "last_minute_vs_daily_close_proxy_bps": (last_minute_close / daily_close_proxy - 1.0) * 10_000.0,
        "last_10_minute_volume": float(last10),
        "day_minute_bar_volume_excluding_uncaptured_auction": float(minute["volume"].sum()),
        "last_10_minute_volume_share": float(last10 / minute["volume"].sum()),
        "last_minute_volume_share": float(
            minute.loc[minute["clock"].eq("15:59"), "volume"].iloc[0] / minute["volume"].sum()
        ),
        "event_daily_volume": event_volume,
        "daily_minus_minute_bar_volume": float(event_volume - minute["volume"].sum()),
        "prior_max_daily_volume": float(prior.loc[prior_max_idx, "volume"]),
        "prior_max_daily_volume_date": str(pd.Timestamp(prior.loc[prior_max_idx, "timestamp"]).date()),
        "event_volume_rank_descending": int(daily["volume"].rank(method="min", ascending=False).iloc[-1]),
        "event_volume_vs_prior_max": float(event_volume / prior.loc[prior_max_idx, "volume"]),
    }
    if args.full_minute_history:
        minute_chunks = []
        for start, end in (("2025-02-24", "2026-02-23"), ("2026-02-24", "2026-08-28")):
            chunk, chunk_status, chunk_error = _request_candles(
                "SNDK", "1", pd.Timestamp(start), pd.Timestamp(end), token
            )
            if chunk.empty:
                raise RuntimeError(f"SNDK minute request failed ({chunk_status}): {chunk_error}")
            minute_chunks.append(chunk)
        minute_history = pd.concat([*minute_chunks, minute.drop(columns=["clock"])], ignore_index=True)
        minute_history["session"] = pd.to_datetime(minute_history["timestamp"]).dt.date
        minute_history["clock"] = pd.to_datetime(minute_history["timestamp"]).dt.strftime("%H:%M")
        by_session = minute_history.groupby("session").agg(
            full_day_volume=("volume", "sum"),
            last_minute_volume=("volume", lambda values: float(values.iloc[-1])),
        )
        last10 = (
            minute_history.loc[minute_history["clock"].between("15:50", "15:59")]
            .groupby("session")["volume"]
            .sum()
            .rename("last10_volume")
        )
        by_session = by_session.join(last10)
        event_key = pd.Timestamp("2026-08-31").date()
        event_session = by_session.loc[event_key]
        prior_sessions = by_session.drop(index=event_key)
        for label in ("last_minute_volume", "last10_volume"):
            summary[f"event_{label}_rank_descending"] = int(
                by_session[label].rank(method="min", ascending=False).loc[event_key]
            )
            summary[f"event_{label}_vs_prior_max"] = float(
                event_session[label] / prior_sessions[label].max()
            )
        by_session.reset_index().to_csv(args.output_dir / "sndk_intraday_volume_history.csv", index=False)
    (args.output_dir / "sndk_case_study.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
