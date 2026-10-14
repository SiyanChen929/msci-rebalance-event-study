#!/usr/bin/env python3
"""Build the auditable MSCI USA Standard constituent-change universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from msci_rebalance.events import (
    build_standard_event_frame,
    load_sec_ticker_map,
    parse_us_review_pdf,
    resolve_tickers,
)


BASE = "https://www.msci.com/eqb/gimi"


def source_url(path: Path) -> str:
    directory = "smallcap" if "_SC" in path.name else "stdindex"
    return f"{BASE}/{directory}/{path.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--sec-tickers", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calendar-output", type=Path)
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("MSCI_*_?CPublicList.pdf")) + sorted(
        args.pdf_dir.glob("MSCI_*_STPublicList.pdf")
    )
    # De-duplicate because the first glob can also match ST on some shells.
    pdfs = sorted(set(pdfs))
    reviews = [parse_us_review_pdf(path, source_url(path)) for path in pdfs]
    events = build_standard_event_frame(reviews)
    overrides = json.loads(args.overrides.read_text()) if args.overrides and args.overrides.exists() else {}
    resolved = resolve_tickers(events, load_sec_ticker_map(args.sec_tickers), overrides=overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(args.output, index=False)
    calendar_path = args.calendar_output or args.output.with_name("review_calendar.csv")
    calendar = pd.DataFrame(
        [
            {
                "announcement_date": review.announcement_date,
                "effective_close_date": review.effective_close_date,
                "standard_additions": len(review.additions),
                "standard_deletions": len(review.deletions),
                "source_url": review.source_url,
            }
            for review in reviews
            if review.segment == "standard"
        ]
    ).sort_values("effective_close_date")
    calendar.to_csv(calendar_path, index=False)

    print(f"reviews={len(calendar)} events={len(resolved)} resolved={resolved['ticker'].ne('').sum()}")
    unresolved = resolved.loc[resolved["ticker"].eq(""), ["effective_close_date", "action", "security_name"]]
    if not unresolved.empty:
        print("UNRESOLVED")
        print(unresolved.to_string(index=False))


if __name__ == "__main__":
    main()
