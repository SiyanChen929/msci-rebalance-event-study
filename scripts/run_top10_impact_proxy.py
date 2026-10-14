#!/usr/bin/env python3
"""Evaluate a capped, balanced top-10 proxy for the MSCI 15:50 burst."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from msci_rebalance.backtest import (
    active_trade_metrics,
    add_equity_columns,
    moving_block_bootstrap_mean,
    performance_metrics,
)
from msci_rebalance.first_minute import block_max_t_sign_flip_pvalue
from msci_rebalance.ranking import (
    add_liquidity_volatility_proxy_scores,
    select_balanced_top_n,
    validate_pre_event_features,
)
from msci_rebalance.weighted_backtest import (
    build_capped_leg_weights,
    weighted_date_portfolio,
)


COSTS = (0.0, 10.0)
TOP_N_GRID = (6, 10, 20)
SCORES = ("proxy_pressure", "volatility_scaled_pressure_proxy")
CONTROL = "same_gate_full_universe_10pct_cap"
PRIMARY = "volatility_scaled_pressure_proxy_top10_10pct_cap"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strategy_name(score: str, top_n: int) -> str:
    return f"{score}_top{top_n}_10pct_cap"


def _eligible_dates(
    frame: pd.DataFrame,
    minimum_per_side: int,
) -> set[pd.Timestamp]:
    counts = frame.groupby(["effective_close_date", "action"]).size().unstack(fill_value=0)
    for side in ("add", "delete"):
        if side not in counts:
            counts[side] = 0
    return {
        pd.Timestamp(date).normalize()
        for date, row in counts.iterrows()
        if min(int(row["add"]), int(row["delete"])) >= minimum_per_side
    }


def _weights(
    scored: pd.DataFrame,
    score: str,
    top_n: int,
    max_name_weight: float,
    minimum_per_side: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, diagnostics = select_balanced_top_n(
        scored,
        score,
        top_n_total=top_n,
        minimum_per_side=minimum_per_side,
    )
    weights = build_capped_leg_weights(
        selected,
        None,
        gross_target=1.0,
        max_name_weight=max_name_weight,
        require_both_legs=True,
    )
    strategy = _strategy_name(score, top_n)
    weights.insert(0, "strategy", strategy)
    diagnostics.insert(0, "strategy", strategy)
    return weights, diagnostics


def _control_weights(
    scored: pd.DataFrame,
    max_name_weight: float,
    minimum_per_side: int,
) -> pd.DataFrame:
    dates = _eligible_dates(scored, minimum_per_side)
    eligible = scored.loc[
        pd.to_datetime(scored["effective_close_date"]).dt.normalize().isin(dates)
    ].copy()
    weights = build_capped_leg_weights(
        eligible,
        None,
        gross_target=1.0,
        max_name_weight=max_name_weight,
        require_both_legs=True,
    )
    weights.insert(0, "strategy", CONTROL)
    return weights


def _evaluate(
    ledgers: list[pd.DataFrame],
    scheduled_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_rows: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for ledger in ledgers:
        strategy = str(ledger["strategy"].iloc[0])
        for cost in COSTS:
            portfolio = weighted_date_portfolio(
                ledger,
                cost_bps=cost,
                scheduled_dates=scheduled_dates,
            )
            portfolio.insert(0, "strategy", strategy)
            portfolio = add_equity_columns(portfolio)
            portfolio_rows.append(portfolio)
            summary = performance_metrics(portfolio, "net_return")
            active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
            metrics.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost,
                    **summary,
                    **active_trade_metrics(portfolio, "net_return"),
                    "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
                    "active_review_dates": len(active),
                    "mean_return_per_active_review": float(active["net_return"].mean())
                    if len(active)
                    else np.nan,
                    "active_hit_rate": float(active["net_return"].gt(0).mean())
                    if len(active)
                    else np.nan,
                    "mean_names": float(portfolio["n_names"].mean()),
                    "mean_max_abs_name_weight": float(
                        portfolio["max_abs_name_weight"].mean()
                    ),
                    "max_abs_name_weight": float(portfolio["max_abs_name_weight"].max()),
                    "mean_effective_names": float(portfolio["effective_names"].mean()),
                }
            )
    return pd.concat(portfolio_rows, ignore_index=True), pd.DataFrame(metrics)


def _paired_tests(portfolios: pd.DataFrame) -> pd.DataFrame:
    gross = portfolios.loc[portfolios["cost_bps"].eq(0.0)].pivot(
        index="effective_close_date", columns="strategy", values="gross_return"
    )
    differences: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for strategy in sorted(set(gross.columns).difference({CONTROL})):
        pair = gross[[strategy, CONTROL]].dropna()
        difference = pair[strategy] - pair[CONTROL]
        active = pair.loc[
            portfolios.loc[
                portfolios["strategy"].eq(strategy)
                & portfolios["cost_bps"].eq(0.0),
                ["effective_close_date", "gross_exposure"],
            ]
            .set_index("effective_close_date")["gross_exposure"]
            .reindex(pair.index)
            .gt(0)
        ]
        active_difference = active[strategy] - active[CONTROL]
        low, high = moving_block_bootstrap_mean(difference.to_numpy(float), block_size=4)
        test = stats.ttest_1samp(difference, 0.0, alternative="greater")
        rows.append(
            {
                "strategy": strategy,
                "comparator": CONTROL,
                "review_dates": len(difference),
                "mean_difference": float(difference.mean()),
                "median_difference": float(difference.median()),
                "positive_difference_rate": float(difference.gt(0).mean()),
                "active_review_dates": len(active_difference),
                "mean_difference_per_active_review": float(active_difference.mean()),
                "active_positive_difference_rate": float(
                    active_difference.gt(0).mean()
                ),
                "one_sided_t_pvalue": float(test.pvalue),
                "block4_mean_ci_low": low,
                "block4_mean_ci_high": high,
            }
        )
        differences.append(
            pd.DataFrame(
                {
                    "effective_close_date": difference.index,
                    "window": strategy,
                    "gross_return": difference.to_numpy(float),
                }
            )
        )
    family = pd.concat(differences, ignore_index=True)
    output = pd.DataFrame(rows)
    for strategy in output["strategy"]:
        pvalue, dates, candidates = block_max_t_sign_flip_pvalue(
            family,
            strategy,
            block_size=4,
        )
        mask = output["strategy"].eq(strategy)
        output.loc[mask, "block4_max_t_familywise_pvalue"] = pvalue
        output.loc[mask, "family_complete_dates"] = dates
        output.loc[mask, "family_candidates"] = candidates
    return output


def _regimes(portfolios: pd.DataFrame) -> pd.DataFrame:
    gross = portfolios.loc[portfolios["cost_bps"].isin(COSTS)].copy()
    gross["effective_close_date"] = pd.to_datetime(gross["effective_close_date"])
    regimes = {
        "2020_through_2022": lambda date: date.lt("2023-01-01"),
        "2023_through_2026": lambda date: date.ge("2023-01-01"),
        "pre_may_2023": lambda date: date.lt("2023-05-01"),
        "may_2023_plus": lambda date: date.ge("2023-05-01"),
        "excluding_aug_2026": lambda date: ~date.eq("2026-08-31"),
    }
    rows: list[dict[str, object]] = []
    for (strategy, cost), group in gross.groupby(["strategy", "cost_bps"], sort=True):
        for label, rule in regimes.items():
            sample = group.loc[rule(group["effective_close_date"])].copy()
            summary = performance_metrics(sample, "net_return")
            active = sample.loc[sample["gross_exposure"].gt(0)]
            rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost,
                    "sample": label,
                    **summary,
                    **active_trade_metrics(sample, "net_return"),
                    "mean_gross_exposure": float(sample["gross_exposure"].mean()),
                    "active_review_dates": len(active),
                    "mean_return_per_active_review": float(active["net_return"].mean())
                    if len(active)
                    else np.nan,
                    "active_hit_rate": float(active["net_return"].gt(0).mean())
                    if len(active)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _sndk_sensitivity(ledgers: list[pd.DataFrame], scheduled_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ledger in ledgers:
        name = ledger.loc[
            ledger["ticker"].eq("SNDK")
            & pd.to_datetime(ledger["effective_close_date"]).dt.normalize().eq("2026-08-31")
        ]
        contribution = (
            float(name.iloc[0]["target_weight"] * name.iloc[0]["underlying_return"])
            if len(name)
            else 0.0
        )
        total = float((ledger["target_weight"] * ledger["underlying_return"]).sum())
        rows.append(
            {
                "strategy": str(ledger["strategy"].iloc[0]),
                "sndk_selected": bool(len(name)),
                "sndk_weight": float(name.iloc[0]["target_weight"]) if len(name) else 0.0,
                "sndk_pnl_contribution": contribution,
                "full_mean_gross_return": total / scheduled_count,
                "full_mean_without_sndk_cash": (total - contribution) / scheduled_count,
            }
        )
    return pd.DataFrame(rows)


def _chart(portfolios: pd.DataFrame, output: Path) -> None:
    selected = portfolios.loc[
        portfolios["cost_bps"].eq(10.0)
        & portfolios["strategy"].isin(
            [
                CONTROL,
                "proxy_pressure_top10_10pct_cap",
                PRIMARY,
            ]
        )
    ].copy()
    labels = {
        CONTROL: "All eligible names · 10% cap",
        "proxy_pressure_top10_10pct_cap": "Top 5+5 · pressure proxy",
        PRIMARY: "Top 5+5 · volatility-scaled proxy",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for strategy, group in selected.groupby("strategy"):
        ax.plot(
            pd.to_datetime(group["effective_close_date"]),
            group["equity"],
            linewidth=2.1,
            label=labels[strategy],
        )
    ax.axhline(1.0, color="#657486", linewidth=0.8)
    ax.set_title("Balanced top-10 proxy · cash-funded equity after 10 bp")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-panel",
        type=Path,
        default=Path("outputs/baseline/flow_weighting_feature_panel.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--max-name-weight", type=float, default=0.10)
    parser.add_argument("--minimum-per-side", type=int, default=3)
    args = parser.parse_args()
    if args.max_name_weight > 0.10:
        raise ValueError("top-10 test hard-caps names at 10% or less")

    features = pd.read_csv(args.feature_panel, parse_dates=["effective_close_date"])
    validate_pre_event_features(features)
    scheduled_dates = sorted(
        pd.to_datetime(features["effective_close_date"]).dt.normalize().unique()
    )
    valid = features["feature_point_in_time_valid"].fillna(False).astype(bool)
    scored = add_liquidity_volatility_proxy_scores(features.loc[valid].copy())
    scored = scored.loc[
        np.isfinite(scored["proxy_pressure"])
        & np.isfinite(scored["volatility_scaled_pressure_proxy"])
    ].copy()

    control = _control_weights(scored, args.max_name_weight, args.minimum_per_side)
    ledgers = [control]
    diagnostics: list[pd.DataFrame] = []
    for score in SCORES:
        for top_n in TOP_N_GRID:
            weights, detail = _weights(
                scored,
                score,
                top_n,
                args.max_name_weight,
                args.minimum_per_side,
            )
            ledgers.append(weights)
            diagnostics.append(detail)

    portfolios, metrics = _evaluate(ledgers, scheduled_dates)
    paired = _paired_tests(portfolios)
    regimes = _regimes(portfolios)
    sndk = _sndk_sensitivity(ledgers, len(scheduled_dates))
    weights = pd.concat(ledgers, ignore_index=True)
    selection = pd.concat(diagnostics, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir / "top10_impact_security_weights.csv", index=False)
    selection.to_csv(args.output_dir / "top10_impact_selection_diagnostics.csv", index=False)
    portfolios.to_csv(args.output_dir / "top10_impact_date_portfolios.csv", index=False)
    metrics.to_csv(args.output_dir / "top10_impact_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "top10_impact_paired_tests.csv", index=False)
    regimes.to_csv(args.output_dir / "top10_impact_regime_metrics.csv", index=False)
    sndk.to_csv(args.output_dir / "top10_impact_sndk_sensitivity.csv", index=False)
    _chart(portfolios, chart_dir / "top10_impact_equity.png")

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "feature_panel": str(args.feature_panel),
        "feature_panel_sha256": _sha256(args.feature_panel),
        "status": "proxy_research_only",
        "exact_requested_test": "not_run_missing_actual_net_tracker_flow_and_historical_float_market_cap",
        "primary_proxy": PRIMARY,
        "proxy_definition": "daily_volatility_20d * sqrt(ADV20 / expected_pre_event_15:49-15:50_dollar_volume)",
        "warning": "ADV is a synthetic order-size proxy, not MSCI passive capital",
        "selection": "top five additions and top five deletions; equal weights; 10% name cap",
        "minimum_per_side": args.minimum_per_side,
        "sparse_review_policy": "skip when either side has fewer than minimum_per_side; otherwise scale both legs and leave cash",
        "candidate_family": [
            _strategy_name(score, top_n) for score in SCORES for top_n in TOP_N_GRID
        ],
        "costs_bps": list(COSTS),
        "information_policy": "predictors must be dated before the implementation session; event returns and volumes are outcomes only",
        "required_exact_inputs": [
            "licensed point-in-time net tracker flow by security",
            "historical float-adjusted market cap with as-of timestamp",
            "product AUM and replication fractions",
            "pre/post index weights netted across overlapping benchmarks",
        ],
    }
    (args.output_dir / "top10_impact_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


if __name__ == "__main__":
    main()
