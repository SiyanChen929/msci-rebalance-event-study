#!/usr/bin/env python3
"""Supplemental sensitivity using independently sourced missing daily closes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from msci_rebalance.backtest import aggregate_date_portfolio, performance_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--secondary-closes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trades = pd.read_csv(args.ledger)
    trades["effective_close_date"] = pd.to_datetime(trades["effective_close_date"])
    secondary = pd.read_csv(args.secondary_closes)
    secondary["effective_close_date"] = pd.to_datetime(secondary["effective_close_date"])
    filled = trades.merge(
        secondary[["effective_close_date", "ticker", "daily_close_proxy_secondary"]],
        on=["effective_close_date", "ticker"],
        how="left",
    )
    use_secondary = filled["flow_return_gross"].isna() & filled["daily_close_proxy_secondary"].notna()
    direction = filled["action"].map({"add": 1.0, "delete": -1.0})
    filled.loc[use_secondary, "flow_return_gross"] = direction.loc[use_secondary] * (
        filled.loc[use_secondary, "daily_close_proxy_secondary"]
        / filled.loc[use_secondary, "entry_1549"]
        - 1.0
    )
    portfolio = aggregate_date_portfolio(filled, "flow_return_gross", cost_bps=10.0)
    metrics = pd.DataFrame(
        [
            {
                "sensitivity": "eight_secondary_daily_closes_filled",
                "filled_security_events": int(use_secondary.sum()),
                **performance_metrics(portfolio),
            }
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(args.output_dir / "missing_close_sensitivity_portfolio.csv", index=False)
    metrics.to_csv(args.output_dir / "missing_close_sensitivity_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
