#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from msci_rebalance.marketdata import download_event_windows, load_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-date")
    parser.add_argument(
        "--only-ticker",
        action="append",
        dest="only_tickers",
        help="Restrict downloads to one ticker; repeat for multiple tickers.",
    )
    args = parser.parse_args()
    events = pd.read_csv(args.events)
    audit = download_event_windows(
        events,
        args.output_dir,
        token=load_token(args.env_file),
        workers=args.workers,
        overwrite=args.overwrite,
        only_date=args.only_date,
        only_tickers=set(args.only_tickers or []),
    )
    print(
        f"jobs={len(audit)} minute_ok={(audit.minute_rows.fillna(0) > 0).sum()} "
        f"daily_ok={(audit.daily_rows.fillna(0) > 0).sum()}"
    )


if __name__ == "__main__":
    main()
