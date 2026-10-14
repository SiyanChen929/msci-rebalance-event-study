#!/usr/bin/env python3
"""Combine audited manual, Yahoo, and SEC symbol mappings conservatively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--yahoo", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--event-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    yahoo = pd.read_csv(args.yahoo)
    merged = events.merge(yahoo, on="security_name", how="left")
    override_payload = json.loads(args.overrides.read_text())
    overrides = {str(k).upper().strip(): str(v).upper() for k, v in override_payload.items()}
    event_overrides = (
        {str(k).upper().strip(): str(v).upper() for k, v in json.loads(args.event_overrides.read_text()).items()}
        if args.event_overrides and args.event_overrides.exists()
        else {}
    )

    final_ticker: list[str] = []
    method: list[str] = []
    for row in merged.itertuples(index=False):
        key = str(row.security_name).upper().strip()
        event_key = f"{str(row.effective_close_date)[:10]}|{key}"
        yahoo_ticker = "" if pd.isna(row.yahoo_ticker) else str(row.yahoo_ticker).upper()
        sec_ticker = "" if pd.isna(row.ticker) else str(row.ticker).upper()
        yahoo_score = 0.0 if pd.isna(row.yahoo_match_score) else float(row.yahoo_match_score)
        sec_score = 0.0 if pd.isna(row.ticker_match_score) else float(row.ticker_match_score)
        if event_key in event_overrides:
            final_ticker.append(event_overrides[event_key])
            method.append("event_date_override")
        elif key in overrides:
            final_ticker.append(overrides[key])
            method.append("audited_override")
        elif yahoo_ticker and yahoo_score >= 0.80:
            final_ticker.append(yahoo_ticker)
            method.append("yahoo_strong")
        elif sec_ticker and sec_score >= 0.97:
            final_ticker.append(sec_ticker)
            method.append("sec_exact")
        else:
            final_ticker.append("")
            method.append("unresolved")

    merged = merged.rename(columns={"ticker": "sec_ticker"})
    merged["ticker"] = final_ticker
    merged["ticker_resolution"] = method
    merged["ticker_resolved"] = merged["ticker"].ne("")
    merged["mapping_conflict"] = (
        merged["sec_ticker"].fillna("").ne(merged["yahoo_ticker"].fillna(""))
        & merged["sec_ticker"].fillna("").ne("")
        & merged["yahoo_ticker"].fillna("").ne("")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        f"events={len(merged)} resolved={merged['ticker_resolved'].sum()} "
        f"unique_names={merged['security_name'].nunique()} "
        f"unresolved_names={merged.loc[~merged.ticker_resolved, 'security_name'].nunique()}"
    )
    if (~merged["ticker_resolved"]).any():
        print(merged.loc[~merged.ticker_resolved, ["security_name"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
