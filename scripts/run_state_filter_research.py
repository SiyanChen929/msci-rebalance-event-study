#!/usr/bin/env python3
"""Evaluate pre-15:49 stock/market-state gates on the frozen Top-10 ledger."""

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
from msci_rebalance.state_filters import (
    apply_filter_and_balance,
    build_state_feature_panel,
    validate_state_features,
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
    portfolio = weighted_date_portfolio(
        working,
        cost_bps=cost_bps,
        scheduled_dates=scheduled_dates,
    )
    return add_equity_columns(portfolio)


def _metric_row(
    rule: str,
    cost_bps: float,
    portfolio: pd.DataFrame,
    filtered: pd.DataFrame,
    *,
    sample: str = "all",
) -> dict[str, object]:
    summary = performance_metrics(portfolio, "net_return")
    active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
    positions = filtered.loc[filtered["filtered_target_abs_weight"].gt(0)].copy()
    position_net = (
        np.sign(positions["filtered_target_weight"])
        * pd.to_numeric(positions["underlying_return"], errors="coerce")
        - float(cost_bps) / 10_000.0
    )
    return {
        "filter_rule": rule,
        "sample": sample,
        "cost_bps": float(cost_bps),
        **summary,
        **active_trade_metrics(portfolio, "net_return"),
        "arithmetic_gross_return_bp": float(portfolio["gross_return"].sum() * 10_000.0),
        "arithmetic_net_return_bp": float(portfolio["net_return"].sum() * 10_000.0),
        "active_review_dates": int(len(active)),
        "cash_review_dates": int(len(portfolio) - len(active)),
        "active_review_hit_rate": float(active["net_return"].gt(0).mean())
        if len(active)
        else np.nan,
        "scheduled_review_hit_rate": float(portfolio["net_return"].gt(0).mean()),
        "mean_net_return_per_active_review": float(active["net_return"].mean())
        if len(active)
        else np.nan,
        "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
        "mean_gross_exposure_active": float(active["gross_exposure"].mean())
        if len(active)
        else np.nan,
        "selected_position_rows": int(len(positions)),
        "position_win_rate": float(position_net.gt(0).mean()) if len(position_net) else np.nan,
        "mean_names_per_scheduled_review": float(portfolio["n_names"].mean()),
        "mean_names_per_active_review": float(active["n_names"].mean()) if len(active) else np.nan,
    }


def _paired_rule_tests(
    portfolios: pd.DataFrame,
    candidate_rules: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost in sorted(portfolios["cost_bps"].unique()):
        sample = portfolios.loc[portfolios["cost_bps"].eq(cost)]
        wide = sample.pivot(
            index="effective_close_date", columns="filter_rule", values="net_return"
        ).sort_index()
        differences: list[pd.DataFrame] = []
        for rule in candidate_rules:
            if rule == "baseline_all":
                continue
            pair = wide[[rule, "baseline_all"]].dropna()
            difference = pair[rule] - pair["baseline_all"]
            low, high = moving_block_bootstrap_mean(
                difference.to_numpy(float), block_size=4, seed=20260901
            )
            test = stats.ttest_1samp(difference, 0.0, alternative="greater")
            rows.append(
                {
                    "filter_rule": rule,
                    "cost_bps": float(cost),
                    "review_dates": int(len(difference)),
                    "mean_paired_increment_bp": float(difference.mean() * 10_000.0),
                    "median_paired_increment_bp": float(difference.median() * 10_000.0),
                    "positive_increment_rate": float(difference.gt(0).mean()),
                    "one_sided_t_pvalue": float(test.pvalue),
                    "block4_increment_ci_low_bp": float(low * 10_000.0),
                    "block4_increment_ci_high_bp": float(high * 10_000.0),
                }
            )
            differences.append(
                pd.DataFrame(
                    {
                        "effective_close_date": difference.index,
                        "window": rule,
                        "gross_return": difference.to_numpy(float),
                    }
                )
            )
        family = pd.concat(differences, ignore_index=True)
        family_rows = pd.DataFrame(rows)
        for rule in candidate_rules:
            if rule == "baseline_all":
                continue
            pvalue, dates, candidates = block_max_t_sign_flip_pvalue(
                family,
                rule,
                block_size=4,
            )
            mask = (
                family_rows["filter_rule"].eq(rule)
                & family_rows["cost_bps"].eq(float(cost))
            )
            for row_index in family_rows.index[mask]:
                rows[row_index]["block4_max_t_familywise_pvalue"] = pvalue
                rows[row_index]["family_complete_dates"] = dates
                rows[row_index]["family_candidates"] = candidates
    return pd.DataFrame(rows)


def _regime_metrics(
    portfolios: pd.DataFrame,
    filtered_ledgers: dict[str, pd.DataFrame],
    rules: list[str],
) -> pd.DataFrame:
    samples = {
        "2020-2022": lambda date: date.lt("2023-01-01"),
        "2023_plus": lambda date: date.ge("2023-01-01"),
        "retrospective_train_through_2023": lambda date: date.le("2023-11-30"),
        "retrospective_test_2024_plus": lambda date: date.ge("2024-02-29"),
        "excluding_2026": lambda date: date.lt("2026-01-01"),
    }
    rows: list[dict[str, object]] = []
    for rule in rules:
        for cost in (0.0, 10.0):
            portfolio = portfolios.loc[
                portfolios["filter_rule"].eq(rule) & portfolios["cost_bps"].eq(cost)
            ].copy()
            filtered = filtered_ledgers[rule]
            for label, condition in samples.items():
                include = condition(pd.to_datetime(portfolio["effective_close_date"]))
                subset = portfolio.loc[include].copy()
                dates = set(pd.to_datetime(subset["effective_close_date"]).dt.normalize())
                ledger_subset = filtered.loc[
                    pd.to_datetime(filtered["effective_close_date"]).dt.normalize().isin(dates)
                ]
                rows.append(
                    _metric_row(rule, cost, subset, ledger_subset, sample=label)
                )
    return pd.DataFrame(rows)


def _walk_forward(
    portfolios: pd.DataFrame,
    candidate_rules: list[str],
    *,
    minimum_training_reviews: int,
    minimum_active_training_reviews: int,
    minimum_exposure_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = portfolios.loc[portfolios["cost_bps"].eq(10.0)].copy()
    dates = sorted(pd.to_datetime(sample["effective_close_date"]).dt.normalize().unique())
    rule_frames = {
        rule: sample.loc[sample["filter_rule"].eq(rule)].set_index("effective_close_date")
        for rule in candidate_rules
    }
    selections: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    for index in range(minimum_training_reviews, len(dates)):
        test_date = pd.Timestamp(dates[index]).normalize()
        train_dates = [pd.Timestamp(value).normalize() for value in dates[:index]]
        baseline_train = rule_frames["baseline_all"].reindex(train_dates)
        baseline_exposure = float(baseline_train["gross_exposure"].mean())
        candidate_scores: list[dict[str, object]] = []
        for rule in candidate_rules:
            train = rule_frames[rule].reindex(train_dates)
            active = train.loc[train["gross_exposure"].gt(0)]
            exposure_ratio = (
                float(train["gross_exposure"].mean()) / baseline_exposure
                if baseline_exposure > 0
                else 0.0
            )
            eligible = bool(
                len(active) >= minimum_active_training_reviews
                and exposure_ratio >= minimum_exposure_ratio
            )
            candidate = {
                "test_date": test_date,
                "train_start": min(train_dates),
                "train_end": max(train_dates),
                "training_reviews": len(train_dates),
                "filter_rule": rule,
                "train_mean_net_bp_per_scheduled": float(train["net_return"].mean() * 10_000.0),
                "train_active_reviews": int(len(active)),
                "train_active_hit_rate": float(active["net_return"].gt(0).mean())
                if len(active)
                else np.nan,
                "train_exposure_ratio_to_baseline": exposure_ratio,
                "eligible": eligible,
            }
            candidate_scores.append(candidate)
            scores.append(candidate)
        eligible = [row for row in candidate_scores if row["eligible"]]
        eligible.sort(
            key=lambda row: (
                -row["train_mean_net_bp_per_scheduled"],
                -row["train_active_hit_rate"] if np.isfinite(row["train_active_hit_rate"]) else np.inf,
                row["filter_rule"],
            )
        )
        chosen = eligible[0] if eligible else next(
            row for row in candidate_scores if row["filter_rule"] == "baseline_all"
        )
        selections.append({**chosen, "selection_asof": max(train_dates)})
        realized = rule_frames[chosen["filter_rule"]].loc[test_date].to_dict()
        test_rows.append(
            {
                "effective_close_date": test_date,
                "selected_filter_rule": chosen["filter_rule"],
                **realized,
            }
        )
    portfolio = pd.DataFrame(test_rows).sort_values("effective_close_date")
    portfolio = add_equity_columns(portfolio)
    return pd.DataFrame(selections), pd.DataFrame(scores), portfolio


def _factor_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    factors = [
        "signed_stock_day_return_pre",
        "signed_stock_hour_return_pre",
        "signed_stock_week_return_pre",
        "signed_relative_day_return_pre",
        "signed_relative_hour_return_pre",
        "signed_relative_week_return_pre",
        "pre_day_dollar_volume_to_adv20",
        "stock_last_hour_dollar_volume_share_pre",
        "stock_intraday_realized_volatility_pre",
        "stock_same_clock_day_rvol",
        "stock_same_clock_hour_rvol",
    ]
    rows: list[dict[str, object]] = []
    for factor in factors:
        working = panel.dropna(subset=[factor, "directional_security_return"]).copy()
        working["factor_side_percentile"] = working.groupby(
            ["effective_close_date", "action"]
        )[factor].rank(method="average", pct=True)
        working["bucket"] = np.where(
            working["factor_side_percentile"].ge(0.5), "high_half", "low_half"
        )
        date_means = (
            working.groupby(["bucket", "effective_close_date"], as_index=False)
            .agg(
                signed_return=("directional_security_return", "mean"),
                names=("ticker", "size"),
            )
        )
        for bucket, group in date_means.groupby("bucket"):
            rows.append(
                {
                    "factor": factor,
                    "bucket": bucket,
                    "review_dates": int(len(group)),
                    "security_rows": int(group["names"].sum()),
                    "date_balanced_mean_signed_return_bp": float(
                        group["signed_return"].mean() * 10_000.0
                    ),
                    "date_hit_rate": float(group["signed_return"].gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _blend_frontier(
    baseline: pd.DataFrame,
    filtered: pd.DataFrame,
    scheduled_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Trace the in-sample risk/return frontier between baseline and one gate."""

    rows: list[dict[str, object]] = []
    baseline_portfolio = _portfolio(baseline, scheduled_dates, 10.0)
    baseline_net = float(baseline_portfolio["net_return"].sum())
    baseline_active = baseline_portfolio.loc[baseline_portfolio["gross_exposure"].gt(0)]
    baseline_hit = float(baseline_active["net_return"].gt(0).mean())
    for gate_weight in np.linspace(0.0, 1.0, 11):
        blended = baseline.copy()
        blended["filtered_target_weight"] = (
            (1.0 - gate_weight)
            * pd.to_numeric(baseline["filtered_target_weight"], errors="coerce")
            + gate_weight
            * pd.to_numeric(filtered["filtered_target_weight"], errors="coerce")
        )
        blended["filtered_target_abs_weight"] = blended["filtered_target_weight"].abs()
        portfolio = _portfolio(blended, scheduled_dates, 10.0)
        metric = _metric_row(
            f"baseline_gate_blend_{gate_weight:.1f}",
            10.0,
            portfolio,
            blended,
        )
        retention = (
            float(portfolio["net_return"].sum()) / baseline_net if baseline_net > 0 else np.nan
        )
        hit_improvement = float(metric["active_review_hit_rate"] - baseline_hit)
        rows.append(
            {
                "gate_weight": float(gate_weight),
                **metric,
                "net_return_retention_vs_baseline": retention,
                "active_hit_rate_improvement": hit_improvement,
                "meets_80pct_return_and_5pp_hit_constraint": bool(
                    retention >= 0.80 and hit_improvement >= 0.05
                ),
                "warning": "same-sample blend frontier; threshold selection is post hoc",
            }
        )
    return pd.DataFrame(rows)


def _chart(
    portfolios: pd.DataFrame,
    walk_forward: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
    primary_rule: str,
) -> None:
    available = set(metrics["filter_rule"])
    preferred = [
        "baseline_all",
        primary_rule,
        "relative_hour_momentum_high_volume",
        "relative_hour_momentum_same_clock_rvol",
        "same_clock_hour_rvol_above_one",
    ]
    rules = []
    for rule in preferred:
        if rule in available and rule not in rules:
            rules.append(rule)
        if len(rules) == 3:
            break
    labels = {rule: rule.replace("_", " ") for rule in rules}
    labels["baseline_all"] = "Baseline Top-10"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    selected = portfolios.loc[
        portfolios["cost_bps"].eq(10.0) & portfolios["filter_rule"].isin(rules)
    ]
    for rule, group in selected.groupby("filter_rule"):
        axes[0].plot(
            pd.to_datetime(group["effective_close_date"]),
            group["equity"],
            linewidth=2.0,
            label=labels[rule],
        )
    if len(walk_forward):
        axes[0].plot(
            pd.to_datetime(walk_forward["effective_close_date"]),
            walk_forward["equity"],
            linestyle="--",
            linewidth=1.8,
            label="Adaptive walk-forward",
        )
    axes[0].axhline(1.0, color="#617083", linewidth=0.8)
    axes[0].set_title("Cash-funded equity at 10 bp")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    table = metrics.loc[
        metrics["cost_bps"].eq(10.0) & metrics["filter_rule"].isin(rules)
    ].set_index("filter_rule").reindex(rules)
    x = np.arange(len(rules))
    axes[1].bar(
        x - 0.18,
        table["mean_return_per_event"] * 10_000.0,
        width=0.36,
        label="Mean bp / scheduled review",
        color="#2f7d57",
    )
    axes[1].bar(
        x + 0.18,
        table["active_review_hit_rate"] * 100.0,
        width=0.36,
        label="Active hit rate (%)",
        color="#527aa3",
    )
    axes[1].set_xticks(x, [labels[rule] for rule in rules], rotation=12, ha="right")
    axes[1].set_title("Return, hit rate and coverage trade-off")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ledger",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_trade_log.csv"),
    )
    parser.add_argument(
        "--date-ledger",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_date_trade_log.csv"),
    )
    parser.add_argument(
        "--marketdata-dir", type=Path, default=Path("data/marketdata")
    )
    parser.add_argument(
        "--reuse-feature-panel",
        type=Path,
        default=None,
        help="Reuse an already generated state feature panel after validating its timestamps.",
    )
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/state_filter_spec.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    candidate_rules = list(spec["candidate_rules"])
    base = pd.read_csv(args.base_ledger, parse_dates=["effective_close_date"])
    date_ledger = pd.read_csv(args.date_ledger, parse_dates=["effective_close_date"])
    scheduled_dates = sorted(
        pd.to_datetime(date_ledger["effective_close_date"]).dt.normalize().unique()
    )
    feature_panel = (
        pd.read_csv(args.reuse_feature_panel, parse_dates=["effective_close_date"])
        if args.reuse_feature_panel is not None
        else build_state_feature_panel(base, args.marketdata_dir)
    )
    validate_state_features(feature_panel)
    valid_count = int(feature_panel["state_feature_point_in_time_valid"].sum())

    filtered_ledgers: dict[str, pd.DataFrame] = {}
    portfolio_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for rule in candidate_rules:
        filtered = apply_filter_and_balance(
            feature_panel,
            rule,
            market_stress_threshold=float(spec["market_day_absolute_return_threshold"]),
        )
        filtered_ledgers[rule] = filtered
        for cost in spec["costs_bps"]:
            portfolio = _portfolio(filtered, scheduled_dates, float(cost))
            portfolio.insert(0, "filter_rule", rule)
            portfolio_rows.append(portfolio)
            metric_rows.append(_metric_row(rule, float(cost), portfolio, filtered))
    portfolios = pd.concat(portfolio_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    paired = _paired_rule_tests(portfolios, candidate_rules)
    primary_rules = ["baseline_all", str(spec["primary_filter"])]
    regimes = _regime_metrics(portfolios, filtered_ledgers, primary_rules)

    walk_spec = spec["walk_forward"]
    selection, walk_scores, walk_portfolio = _walk_forward(
        portfolios,
        candidate_rules,
        minimum_training_reviews=int(walk_spec["minimum_training_reviews"]),
        minimum_active_training_reviews=int(walk_spec["minimum_active_training_reviews"]),
        minimum_exposure_ratio=float(walk_spec["minimum_gross_exposure_ratio_to_baseline"]),
    )
    walk_dates = set(pd.to_datetime(walk_portfolio["effective_close_date"]).dt.normalize())
    matched_baseline = portfolios.loc[
        portfolios["filter_rule"].eq("baseline_all")
        & portfolios["cost_bps"].eq(10.0)
        & pd.to_datetime(portfolios["effective_close_date"]).dt.normalize().isin(walk_dates)
    ].copy()
    walk_metric_rows: list[dict[str, object]] = []
    for label, portfolio in (
        ("adaptive_walk_forward", walk_portfolio),
        ("matched_baseline", matched_baseline),
    ):
        active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
        walk_metric_rows.append(
            {
                "filter_rule": label,
                "sample": "pseudo_oos_after_first_8_reviews",
                "cost_bps": 10.0,
                **performance_metrics(portfolio, "net_return"),
                **active_trade_metrics(portfolio, "net_return"),
                "arithmetic_net_return_bp": float(portfolio["net_return"].sum() * 10_000.0),
                "active_review_dates": int(len(active)),
                "active_review_hit_rate": float(active["net_return"].gt(0).mean())
                if len(active)
                else np.nan,
                "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
            }
        )
    walk_metrics = pd.DataFrame(walk_metric_rows)
    baseline_walk = walk_metrics.loc[walk_metrics["filter_rule"].eq("matched_baseline")].iloc[0]
    adaptive_walk = walk_metrics["filter_rule"].eq("adaptive_walk_forward")
    walk_metrics.loc[adaptive_walk, "net_return_retention_vs_matched_baseline"] = (
        walk_metrics.loc[adaptive_walk, "arithmetic_net_return_bp"]
        / float(baseline_walk["arithmetic_net_return_bp"])
    )
    walk_metrics.loc[adaptive_walk, "active_hit_rate_improvement"] = (
        walk_metrics.loc[adaptive_walk, "active_review_hit_rate"]
        - float(baseline_walk["active_review_hit_rate"])
    )

    cutoff_rows: list[dict[str, object]] = []
    cutoff_ledgers = [
        ("safe_cutoff_1547", filtered_ledgers[str(spec["primary_filter"])])
    ]
    if str(spec["primary_filter"]) == "relative_hour_momentum":
        cutoff_ledgers.append(
            (
                "tight_cutoff_1548_research_only",
                apply_filter_and_balance(feature_panel, "relative_hour_momentum_tight"),
            )
        )
    for label, filtered in cutoff_ledgers:
        for cost in spec["costs_bps"]:
            portfolio = _portfolio(filtered, scheduled_dates, float(cost))
            cutoff_rows.append(_metric_row(label, float(cost), portfolio, filtered))
    cutoff_sensitivity = pd.DataFrame(cutoff_rows)
    buckets = _factor_buckets(feature_panel)
    blend_frontier = _blend_frontier(
        filtered_ledgers["baseline_all"],
        filtered_ledgers[str(spec["primary_filter"])],
        scheduled_dates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    feature_panel.to_csv(args.output_dir / "state_filter_feature_panel.csv", index=False)
    portfolios.to_csv(args.output_dir / "state_filter_date_portfolios.csv", index=False)
    metrics.to_csv(args.output_dir / "state_filter_rule_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "state_filter_paired_tests.csv", index=False)
    regimes.to_csv(args.output_dir / "state_filter_regime_metrics.csv", index=False)
    filtered_ledgers[str(spec["primary_filter"])].to_csv(
        args.output_dir / "state_filter_primary_trade_log.csv", index=False
    )
    selection.to_csv(args.output_dir / "state_filter_walk_forward_selection.csv", index=False)
    walk_scores.to_csv(args.output_dir / "state_filter_walk_forward_training_scores.csv", index=False)
    walk_portfolio.to_csv(args.output_dir / "state_filter_walk_forward_portfolio.csv", index=False)
    walk_metrics.to_csv(args.output_dir / "state_filter_walk_forward_metrics.csv", index=False)
    cutoff_sensitivity.to_csv(args.output_dir / "state_filter_cutoff_sensitivity.csv", index=False)
    buckets.to_csv(args.output_dir / "state_filter_factor_buckets.csv", index=False)
    blend_frontier.to_csv(args.output_dir / "state_filter_blend_frontier.csv", index=False)
    _chart(
        portfolios,
        walk_portfolio,
        metrics,
        chart_dir / "state_filter_comparison.png",
        str(spec["primary_filter"]),
    )

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "retrospective_research_proxy_not_executable",
        "spec": str(args.spec),
        "spec_sha256": _sha256(args.spec),
        "base_ledger": str(args.base_ledger),
        "base_ledger_sha256": _sha256(args.base_ledger),
        "base_security_rows": len(base),
        "reused_feature_panel": str(args.reuse_feature_panel)
        if args.reuse_feature_panel is not None
        else None,
        "scheduled_review_dates": len(scheduled_dates),
        "complete_state_feature_rows": valid_count,
        "incomplete_state_feature_rows": int(len(feature_panel) - valid_count),
        "candidate_rules": candidate_rules,
        "primary_filter": spec["primary_filter"],
        "information_cutoff": spec["information_cutoff"],
        "execution_buffer": spec["execution_buffer"],
        "tight_cutoff_status": "research_sensitivity_only",
        "inference_unit": "MSCI implementation close",
        "warnings": [
            "candidate family was proposed after the historical sample existed",
            "rules requiring state features fail closed when the 15:47 or daily history is incomplete; the baseline remains unchanged",
            "minute-bar entry and exit endpoints are not broker fills",
            "SPY adjustment is unhedged raw relative return, not a fitted beta residual",
            "volume-to-ADV is not same-clock RVOL because the cache does not contain 20 complete matching event-clock histories for every name",
            "real flow pressure still requires licensed delta weights and tracker AUM",
            "the next eight reviews must remain untouched for confirmation",
        ],
    }
    (args.output_dir / "state_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"state_filter_rows={len(feature_panel)} reviews={len(scheduled_dates)} "
        f"rules={len(candidate_rules)} valid={valid_count}"
    )


if __name__ == "__main__":
    main()
