#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from msci_rebalance.backtest import (
    PLACEBO_OFFSETS,
    add_equity_columns,
    aggregate_date_portfolio,
    extract_trade_ledger,
    performance_metrics,
)


STRATEGIES = {
    "first_minute_1549_open_to_1550_close_proxy": ("first_minute_alpha_return_gross", 10.0),
    "flow_1549_to_daily_close_proxy": ("flow_return_gross", 10.0),
    "flow_1549_to_last_minute_diagnostic": ("last_minute_flow_return_gross", 10.0),
    "auction_reversal_to_next_open": ("reversal_return_gross", 20.0),
    "announcement_to_implementation": ("announcement_drift_return_gross", 20.0),
}
SCHEMA_VERSION = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(root.glob("*/*.parquet"))
    for path in files:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return len(files), digest.hexdigest()


def provenance(events_path: Path, marketdata_dir: Path) -> dict[str, object]:
    file_count, cache_hash = cache_fingerprint(marketdata_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "events_sha256": sha256_file(events_path),
        "backtest_code_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src" / "msci_rebalance" / "backtest.py"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "marketdata_cache_file_count": file_count,
        "marketdata_cache_fingerprint": cache_hash,
    }


def period_label(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2023-05-01"):
        return "pre_comprehensive_regime"
    return "quarterly_comprehensive_regime"


def build_coverage(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(events["effective_close_date"].drop_duplicates())
    grid = pd.MultiIndex.from_product(
        [dates, ["add", "delete"]],
        names=["effective_close_date", "action"],
    ).to_frame(index=False)
    expected = (
        events.groupby(["effective_close_date", "action"])
        .size()
        .rename("expected_security_events")
        .reset_index()
    )
    extracted = (
        trades.groupby(["effective_close_date", "action"])
        .agg(
            minute_entry_usable=("entry_1549", "count"),
            daily_close_proxy_usable=("flow_return_gross", "count"),
        )
        .reset_index()
    )
    coverage = grid.merge(expected, how="left").merge(extracted, how="left")
    count_columns = ["expected_security_events", "minute_entry_usable", "daily_close_proxy_usable"]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    denominator = coverage["expected_security_events"].replace(0, np.nan)
    coverage["minute_entry_coverage"] = coverage["minute_entry_usable"] / denominator
    coverage["daily_close_proxy_coverage"] = coverage["daily_close_proxy_usable"] / denominator
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-ledger", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    events["effective_close_date"] = pd.to_datetime(events["effective_close_date"])
    ledger_path = args.output_dir / "trade_ledger.csv"
    failure_path = args.output_dir / "extraction_failures.csv"
    manifest_path = args.output_dir / "run_manifest.json"
    current_provenance = provenance(args.events, args.marketdata_dir)
    if args.reuse_ledger and ledger_path.exists():
        if not manifest_path.exists():
            raise RuntimeError("Cannot reuse ledger without run_manifest.json")
        previous = json.loads(manifest_path.read_text())
        mismatches = [
            key
            for key, value in current_provenance.items()
            if previous.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "Refusing stale ledger reuse; provenance changed: " + ", ".join(mismatches)
            )
        trades = pd.read_csv(ledger_path)
        failures = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
    else:
        trades, failures = extract_trade_ledger(events, args.marketdata_dir)
    trades["effective_close_date"] = pd.to_datetime(trades["effective_close_date"])
    trades["regime"] = trades["effective_close_date"].map(period_label)
    trades.to_csv(ledger_path, index=False)
    failures.to_csv(failure_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                **current_provenance,
                "generated_utc": datetime.now(UTC).isoformat(),
                "events_file": str(args.events),
                "marketdata_dir": str(args.marketdata_dir),
            },
            indent=2,
        )
    )

    portfolio_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for strategy, (column, base_cost) in STRATEGIES.items():
        portfolio = aggregate_date_portfolio(trades, column, cost_bps=base_cost)
        portfolio.insert(0, "strategy", strategy)
        portfolio_frames.append(add_equity_columns(portfolio))
        for sample, mask in {
            "full": np.ones(len(portfolio), dtype=bool),
            "early_sample_through_2023": portfolio.effective_close_date.lt("2024-01-01"),
            "retrospective_2024_2026": portfolio.effective_close_date.ge("2024-01-01"),
            "full_nos_fif_qcir_from_may_2023": portfolio.effective_close_date.ge("2023-05-01"),
            "excluding_aug_2026_discovery": portfolio.effective_close_date.ne("2026-08-31"),
            "two_sided_dates_only": portfolio["n_adds"].gt(0) & portfolio["n_deletes"].gt(0),
        }.items():
            metrics = performance_metrics(portfolio.loc[mask])
            metrics.update(strategy=strategy, sample=sample, assumed_cost_bps=base_cost)
            metrics_rows.append(metrics)
        for cost in (0.0, 5.0, 10.0, 20.0, 40.0):
            scenario = aggregate_date_portfolio(trades, column, cost_bps=cost)
            metrics = performance_metrics(scenario)
            sensitivity_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost,
                    **metrics,
                }
            )

    # Matched-session negative controls for the exact 15:49 strategy.
    placebo_rows: list[dict[str, object]] = []
    placebo_portfolios: list[pd.DataFrame] = []
    for offset in PLACEBO_OFFSETS:
        column = f"placebo_{offset:+d}_gross"
        portfolio = aggregate_date_portfolio(trades, column, cost_bps=0.0)
        portfolio = portfolio[["effective_close_date", "gross_return"]].rename(
            columns={"gross_return": f"placebo_{offset:+d}"}
        )
        placebo_portfolios.append(portfolio)
        metrics = performance_metrics(
            portfolio.rename(columns={f"placebo_{offset:+d}": "gross_return"}), "gross_return"
        )
        placebo_rows.append({"session_offset": offset, **metrics})

    placebo_columns = [f"placebo_{offset:+d}_gross" for offset in PLACEBO_OFFSETS]
    paired_security = trades.dropna(subset=["flow_return_gross", *placebo_columns]).copy()
    paired_security["matched_placebo_mean"] = paired_security[placebo_columns].mean(axis=1)
    paired_security["paired_abnormal_return_gross"] = (
        paired_security["flow_return_gross"] - paired_security["matched_placebo_mean"]
    )
    paired = aggregate_date_portfolio(
        paired_security,
        "paired_abnormal_return_gross",
        cost_bps=0.0,
    )
    paired.to_csv(args.output_dir / "paired_placebo_abnormal_returns.csv", index=False)
    pd.DataFrame([performance_metrics(paired, "gross_return")]).to_csv(
        args.output_dir / "paired_placebo_metrics.csv", index=False
    )
    paired_coverage = (
        trades.groupby("effective_close_date")
        .agg(event_names=("flow_return_gross", "count"))
        .join(paired_security.groupby("effective_close_date").size().rename("complete_paired_names"))
        .fillna(0)
        .reset_index()
    )
    paired_coverage["complete_pair_coverage"] = (
        paired_coverage["complete_paired_names"] / paired_coverage["event_names"].replace(0, np.nan)
    )
    paired_coverage.to_csv(args.output_dir / "paired_placebo_coverage.csv", index=False)

    subgroup_rows: list[dict[str, object]] = []
    subgroups = {
        "all": np.ones(len(trades), dtype=bool),
        "size_migrations": trades["change_type"].isin(["up_from_small", "down_to_small"]),
        "non_migration_entries_exits": trades["change_type"].isin(["new_to_standard", "exit_standard"]),
        "additions_only": trades["action"].eq("add"),
        "deletions_only": trades["action"].eq("delete"),
    }
    for subgroup, mask in subgroups.items():
        for sample, date_mask in {
            "full": np.ones(int(mask.sum()), dtype=bool),
            "retrospective_2024_2026": trades.loc[mask, "effective_close_date"].ge("2024-01-01").to_numpy(),
            "full_nos_fif_qcir_from_may_2023": trades.loc[mask, "effective_close_date"].ge("2023-05-01").to_numpy(),
        }.items():
            subset = trades.loc[mask].iloc[np.flatnonzero(date_mask)]
            portfolio = aggregate_date_portfolio(
                subset,
                "flow_return_gross",
                cost_bps=10.0,
                single_leg_weight=1.0,
            )
            subgroup_rows.append(
                {"subgroup": subgroup, "sample": sample, "security_events": len(subset), **performance_metrics(portfolio)}
            )

    yearly_rows: list[dict[str, object]] = []
    for strategy, (column, base_cost) in STRATEGIES.items():
        portfolio = aggregate_date_portfolio(trades, column, cost_bps=base_cost)
        for year, group in portfolio.groupby(portfolio["effective_close_date"].dt.year):
            returns = group["net_return"]
            yearly_rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "event_dates": len(group),
                    "total_return": float((1.0 + returns).prod() - 1.0),
                    "mean_return_per_event": float(returns.mean()),
                    "median_return_per_event": float(returns.median()),
                    "hit_rate": float(returns.gt(0).mean()),
                    "worst_event": float(returns.min()),
                    "best_event": float(returns.max()),
                }
            )

    # Anchored, grouped chronological folds.  No parameter is fitted: the
    # strategy and costs are fixed before each fold and every constituent from
    # a review date remains in the same test block.
    primary_column, primary_cost = STRATEGIES["flow_1549_to_daily_close_proxy"]
    primary = aggregate_date_portfolio(trades, primary_column, cost_bps=primary_cost)
    walk_forward_rows: list[dict[str, object]] = []
    walk_forward_tests: list[pd.DataFrame] = []
    for test_year in range(2023, 2027):
        train = primary.loc[primary["effective_close_date"].dt.year.lt(test_year)]
        test = primary.loc[primary["effective_close_date"].dt.year.eq(test_year)]
        if train.empty or test.empty:
            continue
        walk_forward_tests.append(test.assign(fold_test_year=test_year))
        walk_forward_rows.append(
            {
                "fold_test_year": test_year,
                "train_start": train["effective_close_date"].min(),
                "train_end": train["effective_close_date"].max(),
                "train_event_dates": len(train),
                "test_start": test["effective_close_date"].min(),
                "test_end": test["effective_close_date"].max(),
                **{f"test_{key}": value for key, value in performance_metrics(test).items()},
            }
        )
    walk_forward_portfolio = pd.concat(walk_forward_tests, ignore_index=True)
    walk_forward_portfolio.to_csv(args.output_dir / "walk_forward_portfolio.csv", index=False)
    pd.DataFrame(walk_forward_rows).to_csv(args.output_dir / "walk_forward_metrics.csv", index=False)

    coverage = build_coverage(events, trades)
    coverage.to_csv(args.output_dir / "data_quality_coverage.csv", index=False)
    trades.loc[trades["price_scale_reconciled"].fillna(False)].to_csv(
        args.output_dir / "price_scale_reconciliations.csv",
        index=False,
    )

    pd.concat(portfolio_frames, ignore_index=True).to_csv(args.output_dir / "portfolio_equity_curves.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "performance_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    pd.DataFrame(placebo_rows).to_csv(args.output_dir / "placebo_metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(args.output_dir / "yearly_stability.csv", index=False)

    print(
        f"events={len(events)} trades={len(trades)} failures={len(failures)} "
        f"dates={trades['effective_close_date'].nunique()}"
    )
    headline = pd.DataFrame(metrics_rows)
    print(
        headline.loc[headline["sample"].eq("full"), [
            "strategy", "event_dates", "total_return", "cagr", "max_drawdown",
            "event_sharpe_sqrt4", "mean_return_per_event", "hit_rate",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
