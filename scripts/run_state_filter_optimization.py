#!/usr/bin/env python3
"""Bounded retrospective optimization of the pre-15:49 MSCI state filter."""

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
from msci_rebalance.state_filters import apply_filter_and_balance, validate_state_features
from msci_rebalance.state_optimization import (
    expanding_walk_forward,
    select_rule,
    training_scores,
)
from msci_rebalance.weighted_backtest import weighted_date_portfolio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portfolio(
    filtered: pd.DataFrame,
    scheduled_dates: list[pd.Timestamp],
    cost_bps: float,
) -> pd.DataFrame:
    working = filtered.copy()
    working["target_weight"] = working["filtered_target_weight"]
    return add_equity_columns(
        weighted_date_portfolio(
            working,
            cost_bps=float(cost_bps),
            scheduled_dates=scheduled_dates,
        )
    )


def _portfolio_metric(
    rule: str,
    cost_bps: float,
    portfolio: pd.DataFrame,
    *,
    sample: str = "all",
) -> dict[str, object]:
    active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
    return {
        "filter_rule": rule,
        "sample": sample,
        "cost_bps": float(cost_bps),
        **performance_metrics(portfolio, "net_return"),
        **active_trade_metrics(portfolio, "net_return"),
        "arithmetic_gross_return_bp": float(portfolio["gross_return"].sum() * 10_000.0),
        "arithmetic_net_return_bp": float(portfolio["net_return"].sum() * 10_000.0),
        "active_review_dates": int(len(active)),
        "cash_review_dates": int(len(portfolio) - len(active)),
        "active_review_hit_rate": float(active["net_return"].gt(0).mean())
        if len(active)
        else np.nan,
        "scheduled_review_hit_rate": float(portfolio["net_return"].gt(0).mean()),
        "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
        "mean_active_gross_exposure": float(active["gross_exposure"].mean())
        if len(active)
        else np.nan,
        "mean_names_per_scheduled_review": float(portfolio["n_names"].mean()),
    }


def _paired_tests(
    portfolios: pd.DataFrame,
    candidate_rules: list[str],
    primary_cost_bps: float,
) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost_bps)]
    wide = sample.pivot(
        index="effective_close_date", columns="filter_rule", values="net_return"
    ).sort_index()
    rows: list[dict[str, object]] = []
    family: list[pd.DataFrame] = []
    for rule in candidate_rules:
        if rule == "baseline_all":
            continue
        difference = (wide[rule] - wide["baseline_all"]).dropna()
        low, high = moving_block_bootstrap_mean(
            difference.to_numpy(float), block_size=4, seed=20260901
        )
        test = stats.ttest_1samp(difference, 0.0, alternative="greater")
        rows.append(
            {
                "filter_rule": rule,
                "cost_bps": float(primary_cost_bps),
                "review_dates": int(len(difference)),
                "mean_paired_increment_bp": float(difference.mean() * 10_000.0),
                "median_paired_increment_bp": float(difference.median() * 10_000.0),
                "positive_increment_rate": float(difference.gt(0).mean()),
                "one_sided_t_pvalue": float(test.pvalue),
                "block4_increment_ci_low_bp": float(low * 10_000.0),
                "block4_increment_ci_high_bp": float(high * 10_000.0),
            }
        )
        family.append(
            pd.DataFrame(
                {
                    "effective_close_date": difference.index,
                    "window": rule,
                    "gross_return": difference.to_numpy(float),
                }
            )
        )
    family_frame = pd.concat(family, ignore_index=True)
    for row in rows:
        pvalue, dates, candidates = block_max_t_sign_flip_pvalue(
            family_frame,
            str(row["filter_rule"]),
            block_size=4,
        )
        row["block4_max_t_familywise_pvalue"] = pvalue
        row["family_complete_dates"] = dates
        row["family_candidates"] = candidates
    return pd.DataFrame(rows)


def _regime_metrics(
    portfolios: pd.DataFrame,
    rules: list[str],
    primary_cost_bps: float,
) -> pd.DataFrame:
    samples = {
        "2020-2022": lambda date: date.lt("2023-01-01"),
        "2023-plus": lambda date: date.ge("2023-01-01"),
        "excluding-2026": lambda date: date.lt("2026-01-01"),
    }
    rows: list[dict[str, object]] = []
    for rule in rules:
        portfolio = portfolios.loc[
            portfolios["filter_rule"].eq(rule)
            & portfolios["cost_bps"].eq(primary_cost_bps)
        ].copy()
        date = pd.to_datetime(portfolio["effective_close_date"])
        for label, condition in samples.items():
            rows.append(
                _portfolio_metric(
                    rule,
                    primary_cost_bps,
                    portfolio.loc[condition(date)].copy(),
                    sample=label,
                )
            )
    return pd.DataFrame(rows)


def _walk_metrics(
    walk_portfolios: pd.DataFrame,
    fixed_portfolios: pd.DataFrame,
    primary_cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_dates = sorted(pd.to_datetime(walk_portfolios["effective_close_date"]).dt.normalize().unique())
    baseline = fixed_portfolios.loc[
        fixed_portfolios["filter_rule"].eq("baseline_all")
        & fixed_portfolios["cost_bps"].eq(primary_cost_bps)
        & pd.to_datetime(fixed_portfolios["effective_close_date"])
        .dt.normalize()
        .isin(test_dates)
    ].copy()
    rows.append(
        _portfolio_metric(
            "matched_baseline",
            primary_cost_bps,
            baseline,
            sample="pseudo_oos_after_first_8_reviews",
        )
    )
    for policy, portfolio in walk_portfolios.groupby("selection_policy", sort=True):
        ordered = add_equity_columns(portfolio.sort_values("effective_close_date").copy())
        rows.append(
            _portfolio_metric(
                str(policy),
                primary_cost_bps,
                ordered,
                sample="pseudo_oos_after_first_8_reviews",
            )
        )
    result = pd.DataFrame(rows)
    baseline_row = result.loc[result["filter_rule"].eq("matched_baseline")].iloc[0]
    challenger = ~result["filter_rule"].eq("matched_baseline")
    result.loc[challenger, "net_return_retention_vs_matched_baseline"] = (
        result.loc[challenger, "arithmetic_net_return_bp"]
        / float(baseline_row["arithmetic_net_return_bp"])
    )
    result.loc[challenger, "active_hit_rate_improvement"] = (
        result.loc[challenger, "active_review_hit_rate"]
        - float(baseline_row["active_review_hit_rate"])
    )
    return result


def _walk_paired_tests(
    walk_portfolios: pd.DataFrame,
    fixed_portfolios: pd.DataFrame,
    primary_cost_bps: float,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(walk_portfolios["effective_close_date"]).dt.normalize().unique())
    baseline = fixed_portfolios.loc[
        fixed_portfolios["filter_rule"].eq("baseline_all")
        & fixed_portfolios["cost_bps"].eq(primary_cost_bps)
    ].copy()
    baseline["effective_close_date"] = pd.to_datetime(
        baseline["effective_close_date"]
    ).dt.normalize()
    baseline = baseline.set_index("effective_close_date").reindex(dates)["net_return"]
    rows: list[dict[str, object]] = []
    family: list[pd.DataFrame] = []
    for policy, group in walk_portfolios.groupby("selection_policy", sort=True):
        group = group.copy()
        group["effective_close_date"] = pd.to_datetime(
            group["effective_close_date"]
        ).dt.normalize()
        realized = group.set_index("effective_close_date").reindex(dates)["net_return"]
        difference = realized - baseline
        low, high = moving_block_bootstrap_mean(
            difference.to_numpy(float), block_size=4, seed=20260901
        )
        test = stats.ttest_1samp(difference, 0.0, alternative="greater")
        rows.append(
            {
                "selection_policy": policy,
                "cost_bps": primary_cost_bps,
                "review_dates": len(difference),
                "mean_paired_increment_bp": float(difference.mean() * 10_000.0),
                "block4_increment_ci_low_bp": float(low * 10_000.0),
                "block4_increment_ci_high_bp": float(high * 10_000.0),
                "one_sided_t_pvalue": float(test.pvalue),
            }
        )
        family.append(
            pd.DataFrame(
                {
                    "effective_close_date": difference.index,
                    "window": str(policy),
                    "gross_return": difference.to_numpy(float),
                }
            )
        )
    family_frame = pd.concat(family, ignore_index=True)
    for row in rows:
        pvalue, complete_dates, candidates = block_max_t_sign_flip_pvalue(
            family_frame, str(row["selection_policy"]), block_size=4
        )
        row["block4_max_t_familywise_pvalue"] = pvalue
        row["family_complete_dates"] = complete_dates
        row["family_candidates"] = candidates
    return pd.DataFrame(rows)


def _mechanism_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    working = panel.loc[
        panel["state_feature_point_in_time_valid"].fillna(False).astype(bool)
        & panel["action"].eq("delete")
        & panel["change_type"].eq("down_to_small")
    ].copy()
    working["hour_confirmation"] = pd.to_numeric(
        working["signed_relative_hour_return_pre"], errors="coerce"
    ).gt(0)
    working["day_confirmation"] = pd.to_numeric(
        working["signed_relative_day_return_pre"], errors="coerce"
    ).gt(0)
    working["confirmation_pattern"] = np.select(
        [
            working["hour_confirmation"] & working["day_confirmation"],
            working["hour_confirmation"] & ~working["day_confirmation"],
            ~working["hour_confirmation"] & working["day_confirmation"],
        ],
        ["hour_and_day", "hour_only", "day_only"],
        default="neither",
    )
    working["regime"] = np.where(
        pd.to_datetime(working["effective_close_date"]).lt("2023-01-01"),
        "2020-2022",
        "2023-plus",
    )
    rows: list[dict[str, object]] = []
    for regime, regime_frame in [("all", working), *working.groupby("regime")]:
        for pattern, group in regime_frame.groupby("confirmation_pattern"):
            date_mean = group.groupby("effective_close_date")[
                "directional_security_return"
            ].mean()
            rows.append(
                {
                    "regime": regime,
                    "confirmation_pattern": pattern,
                    "review_dates": int(len(date_mean)),
                    "security_rows": int(len(group)),
                    "date_balanced_mean_signed_return_bp": float(
                        date_mean.mean() * 10_000.0
                    ),
                    "date_hit_rate": float(date_mean.gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_out(
    portfolios: pd.DataFrame,
    optimization_rules: list[str],
    intervention_rank: dict[str, int],
    walk_spec: dict[str, object],
    primary_cost_bps: float,
) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost_bps)].copy()
    dates = sorted(pd.to_datetime(sample["effective_close_date"]).dt.normalize().unique())
    rows: list[dict[str, object]] = []
    for excluded in dates:
        train_dates = [date for date in dates if date != excluded]
        scores = training_scores(
            sample,
            train_dates,
            optimization_rules,
            intervention_rank,
            minimum_active_reviews=int(walk_spec["minimum_active_training_reviews"]),
            minimum_return_retention=float(
                walk_spec["minimum_net_return_retention_vs_baseline"]
            ),
            minimum_hit_improvement=float(
                walk_spec["minimum_active_hit_rate_improvement"]
            ),
            minimum_exposure_ratio=float(
                walk_spec["minimum_gross_exposure_ratio_to_baseline"]
            ),
        )
        rows.append(
            {
                "excluded_review": pd.Timestamp(excluded),
                "constrained_selected_rule": select_rule(
                    scores, "constrained_minimum_intervention"
                ),
                "maximum_mean_selected_rule": select_rule(
                    scores, "maximum_training_mean"
                ),
            }
        )
    return pd.DataFrame(rows)


def _chart(
    portfolios: pd.DataFrame,
    metrics: pd.DataFrame,
    walk_portfolios: pd.DataFrame,
    output: Path,
    primary_cost_bps: float,
) -> None:
    soft_rules = [
        "baseline_all",
        "down_to_small_hour_soft_075",
        "down_to_small_hour_soft_050",
        "down_to_small_hour_soft_025",
        "down_to_small_hour_confirmation",
    ]
    labels = {
        "baseline_all": "Baseline",
        "down_to_small_hour_soft_075": "Retain 75%",
        "down_to_small_hour_soft_050": "Retain 50%",
        "down_to_small_hour_soft_025": "Retain 25%",
        "down_to_small_hour_confirmation": "Hard gate",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.5))
    selected = portfolios.loc[
        portfolios["cost_bps"].eq(primary_cost_bps)
        & portfolios["filter_rule"].isin(soft_rules)
    ]
    for rule, group in selected.groupby("filter_rule"):
        axes[0, 0].plot(
            pd.to_datetime(group["effective_close_date"]),
            group["equity"],
            label=labels[rule],
            linewidth=1.8,
        )
    axes[0, 0].axhline(1.0, color="#77828d", linewidth=0.8)
    axes[0, 0].set_title(f"Fixed soft-gate equity at {primary_cost_bps:.0f} bp")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    q_map = {
        "baseline_all": 1.0,
        "down_to_small_hour_soft_075": 0.75,
        "down_to_small_hour_soft_050": 0.50,
        "down_to_small_hour_soft_025": 0.25,
        "down_to_small_hour_confirmation": 0.0,
    }
    surface = metrics.loc[
        metrics["cost_bps"].eq(primary_cost_bps)
        & metrics["filter_rule"].isin(soft_rules)
    ].copy()
    surface["unconfirmed_retention"] = surface["filter_rule"].map(q_map)
    surface = surface.sort_values("unconfirmed_retention")
    axes[0, 1].plot(
        surface["unconfirmed_retention"],
        surface["arithmetic_net_return_bp"],
        marker="o",
        color="#2f7d57",
        label="Net bp",
    )
    twin = axes[0, 1].twinx()
    twin.plot(
        surface["unconfirmed_retention"],
        surface["active_review_hit_rate"] * 100.0,
        marker="s",
        color="#527aa3",
        label="Active hit %",
    )
    axes[0, 1].set_title("Soft-gate profit / reliability frontier")
    axes[0, 1].set_xlabel("Weight retained for unconfirmed down-to-small shorts")
    axes[0, 1].set_ylabel("Arithmetic net bp")
    twin.set_ylabel("Active-date hit rate (%)")
    axes[0, 1].grid(alpha=0.25)

    cost_table = metrics.loc[metrics["filter_rule"].isin(soft_rules)]
    for rule, group in cost_table.groupby("filter_rule"):
        axes[1, 0].plot(
            group["cost_bps"],
            group["arithmetic_net_return_bp"],
            marker="o",
            linewidth=1.6,
            label=labels[rule],
        )
    axes[1, 0].axhline(0.0, color="#77828d", linewidth=0.8)
    axes[1, 0].set_title("Cost stress")
    axes[1, 0].set_xlabel("All-in round-trip haircut (bp)")
    axes[1, 0].set_ylabel("Arithmetic net bp")
    axes[1, 0].grid(alpha=0.25)

    for policy, group in walk_portfolios.groupby("selection_policy"):
        ordered = add_equity_columns(group.sort_values("effective_close_date").copy())
        axes[1, 1].plot(
            pd.to_datetime(ordered["effective_close_date"]),
            ordered["equity"],
            linewidth=1.8,
            label=str(policy).replace("_", " "),
        )
    walk_dates = pd.to_datetime(walk_portfolios["effective_close_date"]).dt.normalize().unique()
    baseline_walk = portfolios.loc[
        portfolios["filter_rule"].eq("baseline_all")
        & portfolios["cost_bps"].eq(primary_cost_bps)
        & pd.to_datetime(portfolios["effective_close_date"]).dt.normalize().isin(walk_dates)
    ].sort_values("effective_close_date")
    baseline_walk = baseline_walk.copy()
    baseline_walk["equity"] = (1.0 + baseline_walk["net_return"]).cumprod()
    axes[1, 1].plot(
        pd.to_datetime(baseline_walk["effective_close_date"]),
        baseline_walk["equity"],
        color="#67727d",
        linestyle="--",
        linewidth=1.5,
        label="matched baseline",
    )
    axes[1, 1].axhline(1.0, color="#77828d", linewidth=0.8)
    axes[1, 1].set_title("Retrospective expanding pseudo-OOS")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/state_filter_optimization_spec.json"),
    )
    parser.add_argument(
        "--feature-panel",
        type=Path,
        default=Path("outputs/state_filter_extension/state_filter_feature_panel.csv"),
    )
    parser.add_argument(
        "--date-ledger",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_date_trade_log.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/state_filter_optimization")
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    panel = pd.read_csv(args.feature_panel, parse_dates=["effective_close_date"])
    validate_state_features(panel)
    date_ledger = pd.read_csv(args.date_ledger, parse_dates=["effective_close_date"])
    scheduled_dates = sorted(
        pd.to_datetime(date_ledger["effective_close_date"]).dt.normalize().unique()
    )
    candidate_rules = list(spec["candidate_rules"])
    optimization_rules = list(spec["optimization_rules"])
    intervention_rank = {
        str(key): int(value) for key, value in spec["rule_intervention_rank"].items()
    }
    primary_cost_bps = float(spec["primary_cost_bps"])

    filtered_ledgers: dict[str, pd.DataFrame] = {}
    portfolio_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for rule in candidate_rules:
        filtered = apply_filter_and_balance(panel, rule)
        filtered_ledgers[rule] = filtered
        for cost in spec["costs_bps"]:
            portfolio = _portfolio(filtered, scheduled_dates, float(cost))
            portfolio.insert(0, "filter_rule", rule)
            portfolio_rows.append(portfolio)
            metric_rows.append(_portfolio_metric(rule, float(cost), portfolio))
    portfolios = pd.concat(portfolio_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    paired = _paired_tests(portfolios, candidate_rules, primary_cost_bps)
    regimes = _regime_metrics(portfolios, candidate_rules, primary_cost_bps)

    walk_spec = spec["walk_forward"]
    primary_sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost_bps)].copy()
    walk_selection: list[pd.DataFrame] = []
    walk_scores: list[pd.DataFrame] = []
    walk_portfolios: list[pd.DataFrame] = []
    for policy in ("constrained_minimum_intervention", "maximum_training_mean"):
        selection, scores, realized = expanding_walk_forward(
            primary_sample,
            optimization_rules,
            intervention_rank,
            minimum_training_reviews=int(walk_spec["minimum_training_reviews"]),
            minimum_active_reviews=int(walk_spec["minimum_active_training_reviews"]),
            minimum_return_retention=float(
                walk_spec["minimum_net_return_retention_vs_baseline"]
            ),
            minimum_hit_improvement=float(
                walk_spec["minimum_active_hit_rate_improvement"]
            ),
            minimum_exposure_ratio=float(
                walk_spec["minimum_gross_exposure_ratio_to_baseline"]
            ),
            policy=policy,
        )
        walk_selection.append(selection)
        walk_scores.append(scores)
        walk_portfolios.append(realized)
    selections = pd.concat(walk_selection, ignore_index=True)
    training_score_panel = pd.concat(walk_scores, ignore_index=True)
    walk_portfolio = pd.concat(walk_portfolios, ignore_index=True)
    walk_metrics = _walk_metrics(walk_portfolio, portfolios, primary_cost_bps)
    walk_paired = _walk_paired_tests(
        walk_portfolio, portfolios, primary_cost_bps
    )
    leave_one_out = _leave_one_out(
        portfolios,
        optimization_rules,
        intervention_rank,
        walk_spec,
        primary_cost_bps,
    )

    full_scores = training_scores(
        primary_sample,
        scheduled_dates,
        optimization_rules,
        intervention_rank,
        minimum_active_reviews=int(walk_spec["minimum_active_training_reviews"]),
        minimum_return_retention=float(
            walk_spec["minimum_net_return_retention_vs_baseline"]
        ),
        minimum_hit_improvement=float(
            walk_spec["minimum_active_hit_rate_improvement"]
        ),
        minimum_exposure_ratio=float(
            walk_spec["minimum_gross_exposure_ratio_to_baseline"]
        ),
    )
    final_selection = select_rule(full_scores, "constrained_minimum_intervention")
    full_scores["selected_for_future_shadow"] = full_scores["filter_rule"].eq(
        final_selection
    )
    mechanism_buckets = _mechanism_buckets(panel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    portfolios.to_csv(args.output_dir / "state_optimization_date_portfolios.csv", index=False)
    metrics.to_csv(args.output_dir / "state_optimization_rule_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "state_optimization_paired_tests.csv", index=False)
    regimes.to_csv(args.output_dir / "state_optimization_regime_metrics.csv", index=False)
    selections.to_csv(
        args.output_dir / "state_optimization_walk_forward_selection.csv", index=False
    )
    training_score_panel.to_csv(
        args.output_dir / "state_optimization_walk_forward_training_scores.csv",
        index=False,
    )
    walk_portfolio.to_csv(
        args.output_dir / "state_optimization_walk_forward_portfolio.csv", index=False
    )
    walk_metrics.to_csv(
        args.output_dir / "state_optimization_walk_forward_metrics.csv", index=False
    )
    walk_paired.to_csv(
        args.output_dir / "state_optimization_walk_forward_paired_tests.csv",
        index=False,
    )
    leave_one_out.to_csv(
        args.output_dir / "state_optimization_leave_one_review_out.csv", index=False
    )
    full_scores.to_csv(
        args.output_dir / "state_optimization_final_training_scores.csv", index=False
    )
    mechanism_buckets.to_csv(
        args.output_dir / "state_optimization_mechanism_buckets.csv", index=False
    )
    filtered_ledgers[final_selection].to_csv(
        args.output_dir / "state_optimization_selected_trade_log.csv", index=False
    )
    _chart(
        portfolios,
        metrics,
        walk_portfolio,
        chart_dir / "state_filter_optimization.png",
        primary_cost_bps,
    )

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "research_status": spec["research_status"],
        "spec": str(args.spec),
        "spec_sha256": _sha256(args.spec),
        "feature_panel": str(args.feature_panel),
        "feature_panel_sha256": _sha256(args.feature_panel),
        "date_ledger": str(args.date_ledger),
        "date_ledger_sha256": _sha256(args.date_ledger),
        "candidate_rules": candidate_rules,
        "optimization_rules": optimization_rules,
        "attempted_rule_count": len(candidate_rules),
        "scheduled_review_dates": len(scheduled_dates),
        "information_cutoff": spec["information_cutoff"],
        "primary_cost_bps": primary_cost_bps,
        "final_full_sample_constrained_selection": final_selection,
        "future_holdout": spec["future_holdout"],
        "warnings": [
            "all 2020-2026 results are retrospective because the mechanism family was designed after the sample existed",
            "the expanding test is pseudo-OOS and cannot replace the untouched next eight reviews",
            "minute-bar endpoints are research fill proxies without NBBO, impact, borrow, queue, or partial-fill replay",
            "licensed MSCI delta weights and tracker AUM remain unavailable",
        ],
    }
    (args.output_dir / "state_optimization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"optimization_rules={len(candidate_rules)} reviews={len(scheduled_dates)} "
        f"walk_reviews={len(walk_portfolio) // 2} selected={final_selection}"
    )


if __name__ == "__main__":
    main()
