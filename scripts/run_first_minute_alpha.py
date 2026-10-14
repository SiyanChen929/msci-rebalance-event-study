#!/usr/bin/env python3
"""Run the frozen bar-proxy validation for the discovered 15:50 burst."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from msci_rebalance.backtest import add_equity_columns, aggregate_date_portfolio, performance_metrics
from msci_rebalance.first_minute import (
    CONTROL_OFFSETS,
    build_event_and_control_ledgers,
    build_time_window_portfolios,
    matched_control_analysis,
    summarize_time_windows,
)


STRATEGY = "first_minute_1549_open_to_1550_close_proxy_v1"
COST_GRID_BPS = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--event-time-security", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--reuse-event-control-ledgers", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(args.ledger)
    trades["effective_close_date"] = pd.to_datetime(trades["effective_close_date"])
    trades["announcement_date"] = pd.to_datetime(trades["announcement_date"])
    security_path = args.output_dir / "first_minute_alpha_security_ledger.csv"
    controls_path = args.output_dir / "first_minute_alpha_control_ledger.csv"
    if args.reuse_event_control_ledgers and security_path.exists() and controls_path.exists():
        security = pd.read_csv(
            security_path,
            parse_dates=["effective_close_date", "announcement_date", "session_date"],
        )
        controls = pd.read_csv(
            controls_path,
            parse_dates=[
                "effective_close_date",
                "announcement_date",
                "control_session_date",
                "session_date",
            ],
        )
    else:
        security, controls = build_event_and_control_ledgers(trades, args.marketdata_dir)
        security.to_csv(security_path, index=False)
        controls.to_csv(controls_path, index=False)

    cost_rows: list[dict[str, object]] = []
    portfolio_frames: list[pd.DataFrame] = []
    for cost in COST_GRID_BPS:
        portfolio = aggregate_date_portfolio(security, "gross_return", cost_bps=cost)
        portfolio.insert(0, "strategy", STRATEGY)
        portfolio_frames.append(add_equity_columns(portfolio))
        cost_rows.append({"strategy": STRATEGY, "cost_bps": cost, **performance_metrics(portfolio)})
    portfolios = pd.concat(portfolio_frames, ignore_index=True)
    portfolios.to_csv(args.output_dir / "first_minute_alpha_portfolio.csv", index=False)
    cost_metrics = pd.DataFrame(cost_rows)
    cost_metrics.to_csv(args.output_dir / "first_minute_alpha_cost_sensitivity.csv", index=False)

    regime_masks = {
        "full": np.ones(len(security), dtype=bool),
        "2020_through_2022": security["effective_close_date"].lt("2023-01-01"),
        "2023_through_2026": security["effective_close_date"].ge("2023-01-01"),
        "pre_full_qcir_before_may_2023": security["effective_close_date"].lt("2023-05-01"),
        "full_qcir_from_may_2023": security["effective_close_date"].ge("2023-05-01"),
        "retrospective_2024_2026": security["effective_close_date"].ge("2024-01-01"),
        "excluding_aug_2026_discovery": security["effective_close_date"].ne("2026-08-31"),
    }
    regime_rows: list[dict[str, object]] = []
    for sample, mask in regime_masks.items():
        subset = security.loc[mask]
        for cost in (0.0, 10.0):
            portfolio = aggregate_date_portfolio(subset, "gross_return", cost_bps=cost)
            regime_rows.append(
                {"sample": sample, "cost_bps": cost, "security_events": len(subset), **performance_metrics(portfolio)}
            )
    pd.DataFrame(regime_rows).to_csv(
        args.output_dir / "first_minute_alpha_regime_metrics.csv", index=False
    )

    subgroup_rows: list[dict[str, object]] = []
    subgroup_masks = {
        "all": np.ones(len(security), dtype=bool),
        "additions_only": security["action"].eq("add"),
        "deletions_only": security["action"].eq("delete"),
        "size_migrations": security["change_type"].isin(["up_from_small", "down_to_small"]),
        "new_entries_and_full_exits": security["change_type"].isin(["new_to_standard", "exit_standard"]),
    }
    for subgroup, mask in subgroup_masks.items():
        subset = security.loc[mask]
        for cost in (0.0, 10.0):
            portfolio = aggregate_date_portfolio(
                subset,
                "gross_return",
                cost_bps=cost,
                single_leg_weight=1.0
                if subgroup in {"additions_only", "deletions_only"}
                else 0.5,
            )
            subgroup_rows.append(
                {
                    "subgroup": subgroup,
                    "cost_bps": cost,
                    "security_events": len(subset),
                    **performance_metrics(
                        portfolio, "gross_return" if cost == 0.0 else "net_return"
                    ),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(
        args.output_dir / "first_minute_alpha_subgroup_metrics.csv", index=False
    )

    paired, control_portfolios, control_metrics = matched_control_analysis(
        security, controls, CONTROL_OFFSETS
    )
    paired.to_csv(args.output_dir / "first_minute_alpha_paired_controls.csv", index=False)
    control_portfolios.to_csv(
        args.output_dir / "first_minute_alpha_control_portfolios.csv", index=False
    )
    control_metrics.to_csv(
        args.output_dir / "first_minute_alpha_control_metrics.csv", index=False
    )

    event_time = pd.read_csv(args.event_time_security)
    time_security, time_portfolios = build_time_window_portfolios(event_time)
    time_metrics, time_summary = summarize_time_windows(
        time_portfolios, samples=args.randomization_samples
    )
    time_security.to_csv(
        args.output_dir / "first_minute_alpha_time_window_security.csv", index=False
    )
    time_portfolios.to_csv(
        args.output_dir / "first_minute_alpha_time_window_portfolios.csv", index=False
    )
    time_metrics.to_csv(
        args.output_dir / "first_minute_alpha_time_window_metrics.csv", index=False
    )
    time_summary.to_csv(
        args.output_dir / "first_minute_alpha_selection_adjusted_tests.csv", index=False
    )

    walk_rows: list[dict[str, object]] = []
    for year in range(2023, 2027):
        train = security.loc[security["effective_close_date"].dt.year.lt(year)]
        test = security.loc[security["effective_close_date"].dt.year.eq(year)]
        if train.empty or test.empty:
            continue
        test_portfolio = aggregate_date_portfolio(test, "gross_return", cost_bps=10.0)
        walk_rows.append(
            {
                "test_year": year,
                "train_review_dates": train["effective_close_date"].nunique(),
                "test_security_events": len(test),
                "retrospective_discovery_warning": True,
                **{f"test_{key}": value for key, value in performance_metrics(test_portfolio).items()},
            }
        )
    pd.DataFrame(walk_rows).to_csv(
        args.output_dir / "first_minute_alpha_chronological_folds.csv", index=False
    )

    system_spec = {
        "strategy_version": STRATEGY,
        "status": "post_hoc_research_candidate_not_deployable",
        "bar_semantics": {
            "provider": "MarketData minute OHLCV",
            "timestamps": "bar start",
            "entry_proxy": "open of 15:49:00-15:49:59 bar",
            "exit_proxy": "close of 15:50:00-15:50:59 bar",
            "warning": "bar open/close are trades, not simultaneous executable NBBO fills",
        },
        "signal": {
            "baseline": "+1 for published Standard addition; -1 for deletion",
            "feature_cutoff": "15:48:59.999 America/New_York",
            "portfolio": "50% additions / 50% deletions; absent leg remains cash",
            "noii_use": "none in v1; any post-15:50 NOII response is a separate strategy",
        },
        "production_replay_contract": {
            "signal_freeze": "15:48:30.000",
            "entry_send": "15:49:00.000",
            "exit_send": "15:50:59.000",
            "hard_flat": "15:50:59.500",
            "entry_fill": "first fresh executable ask for buys / bid for shorts after send plus latency",
            "exit_fill": "first fresh executable bid for sells / ask for covers after send plus latency",
            "latency_stress_ms": [10, 100, 500],
            "mandatory_inputs": [
                "licensed point-in-time MSCI old/new weights, NOS and FIF",
                "consolidated NBBO, trades, sizes and conditions",
                "NYSE auction imbalances and Nasdaq NOII with event timestamps",
                "borrow locates and fees",
                "venue fees, rejects, partial fills and halts",
            ],
        },
        "validation": {
            "independent_unit": "implementation review date",
            "selection_adjustment": "date-level max-t sign randomization over all inspected clock windows",
            "controls": "four prior same-name same-clock sessions",
            "true_holdout": "first MSCI review after 2026-08-31",
        },
        "options_extension": {
            "status": "not_backtested",
            "reason": "no historical intraday OPRA executable bid/ask chain in current cache",
            "required_fill_rule": "buy call and put at ask; liquidate both at bid before hard-flat deadline",
        },
    }
    (args.output_dir / "first_minute_alpha_system_spec.json").write_text(
        json.dumps(system_spec, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "strategy": STRATEGY,
        "ledger_sha256": sha256_file(args.ledger),
        "event_time_security_sha256": sha256_file(args.event_time_security),
        "module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src" / "msci_rebalance" / "first_minute.py"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "randomization_samples": args.randomization_samples,
        "discovery_status": "post_hoc",
    }
    (args.output_dir / "first_minute_alpha_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    full_10 = cost_metrics.loc[cost_metrics["cost_bps"].eq(10.0)].iloc[0]
    print(
        f"security_events={len(security)} dates={security['effective_close_date'].nunique()} "
        f"net_10bp={full_10['total_return']:.6f} mean_10bp={full_10['mean_return_per_event']:.6f}"
    )


if __name__ == "__main__":
    main()
