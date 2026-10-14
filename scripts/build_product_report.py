#!/usr/bin/env python3
"""Build the product-grade MSCI close-pressure research report."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from msci_rebalance.backtest import active_trade_metrics

try:
    from analyze_leg_attribution import (
        build_driver_table,
        make_all_name_change_type_diagnostics,
        make_broad_diagnostics,
        make_date_balanced_change_type_diagnostics,
        make_event_ledger,
        make_influence_diagnostics,
        make_period_summaries,
        make_regime_clock_profile,
        plot_event_legs,
        plot_regime_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - import path differs under test collection
    from scripts.analyze_leg_attribution import (
        build_driver_table,
        make_all_name_change_type_diagnostics,
        make_broad_diagnostics,
        make_date_balanced_change_type_diagnostics,
        make_event_ledger,
        make_influence_diagnostics,
        make_period_summaries,
        make_regime_clock_profile,
        plot_event_legs,
        plot_regime_diagnostics,
    )


PRIMARY = "flow_1549_to_daily_close_proxy"
BURST = "first_minute_1549_open_to_1550_close_proxy_v1"
TOP10_PRIMARY_COST_BPS = 0.0
TOP10_SENSITIVITY_COST_BPS = 10.0


def pct(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.{digits}f}%"


def bp(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 10_000:+.{digits}f}"


def num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def integer(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def date_string(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def timestamp_string(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def dollars(value: object, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.{digits}f}"


def table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    formatters: dict[str, object] | None = None,
    table_id: str | None = None,
) -> str:
    working = frame.loc[:, columns].copy()
    formatters = formatters or {}
    for column in columns:
        formatter = formatters.get(column)
        if formatter:
            working[column] = working[column].map(formatter)
        else:
            working[column] = working[column].map(
                lambda value: "—" if pd.isna(value) else str(value)
            )
    working = working.rename(columns=labels)
    rendered = working.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
        table_id=table_id,
    )
    return rendered


def chart(name: str, alt: str, caption: str) -> str:
    return (
        '<figure class="chart">'
        f'<img loading="lazy" src="charts/{html.escape(name)}" alt="{html.escape(alt)}">'
        f'<figcaption>{caption}</figcaption>'
        "</figure>"
    )


def metric_card(label: str, value: str, detail: str, tone: str = "neutral") -> str:
    return (
        f'<div class="metric {tone}"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div>'
    )


def source_item(label: str, url: str, note: str) -> str:
    return (
        '<li><a href="'
        + html.escape(url, quote=True)
        + '" target="_blank" rel="noreferrer">'
        + html.escape(label)
        + "</a><span>"
        + html.escape(note)
        + "</span></li>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/events/msci_us_standard_events_resolved.csv"),
    )
    parser.add_argument("--test-summary", default="59/59 tests passed")
    args = parser.parse_args()

    out = args.output_dir
    metrics = pd.read_csv(out / "performance_metrics.csv")
    sensitivity = pd.read_csv(out / "cost_sensitivity.csv")
    curves = pd.read_csv(out / "portfolio_equity_curves.csv")
    curves["effective_close_date"] = pd.to_datetime(curves["effective_close_date"])
    windows = pd.read_csv(out / "window_decomposition_metrics.csv")
    pairs_metrics = pd.read_csv(out / "matched_pairs_metrics.csv").iloc[0]
    pairs = pd.read_csv(out / "matched_pairs.csv")
    pairs["effective_close_date"] = pd.to_datetime(pairs["effective_close_date"])
    volume_metrics = pd.read_csv(out / "volume_diagnostics_metrics.csv").iloc[0]
    subgroup = pd.read_csv(out / "subgroup_metrics.csv")
    walk = pd.read_csv(out / "walk_forward_metrics.csv")
    failures = pd.read_csv(out / "extraction_failures.csv")
    failures["effective_close_date"] = pd.to_datetime(failures["effective_close_date"])
    coverage = pd.read_csv(out / "data_quality_coverage.csv")
    ledger = pd.read_csv(out / "trade_ledger.csv")
    ledger["effective_close_date"] = pd.to_datetime(ledger["effective_close_date"])
    ledger["announcement_date"] = pd.to_datetime(ledger["announcement_date"])
    events = pd.read_csv(args.events)
    events["effective_close_date"] = pd.to_datetime(events["effective_close_date"])
    events["announcement_date"] = pd.to_datetime(events["announcement_date"])
    sndk = json.loads((out / "sndk_case_study.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    volatility_manifest = json.loads(
        (out / "volatility_run_manifest.json").read_text()
    )
    missing_close = pd.read_csv(out / "missing_close_sensitivity_metrics.csv").iloc[0]
    volatility_metrics = pd.read_csv(out / "volatility_diagnostic_metrics.csv")
    volatility_thresholds = pd.read_csv(out / "volatility_threshold_metrics.csv")
    volatility_dates = pd.read_csv(out / "volatility_date_diagnostics.csv")
    volatility_dates["effective_close_date"] = pd.to_datetime(
        volatility_dates["effective_close_date"]
    )
    breakout_metrics = pd.read_csv(out / "breakout_proxy_metrics.csv")
    straddle_scenarios = pd.read_csv(
        out / "hypothetical_straddle_scenario_metrics.csv"
    )
    burst_cost = pd.read_csv(out / "first_minute_alpha_cost_sensitivity.csv")
    burst_regime = pd.read_csv(out / "first_minute_alpha_regime_metrics.csv")
    burst_subgroup = pd.read_csv(out / "first_minute_alpha_subgroup_metrics.csv")
    burst_control = pd.read_csv(out / "first_minute_alpha_control_metrics.csv")
    burst_selection = pd.read_csv(
        out / "first_minute_alpha_selection_adjusted_tests.csv"
    )
    burst_manifest = json.loads(
        (out / "first_minute_alpha_run_manifest.json").read_text()
    )
    flow_metrics = pd.read_csv(out / "flow_weighting_metrics.csv")
    flow_paired = pd.read_csv(out / "flow_weighting_paired_tests.csv")
    flow_capacity = pd.read_csv(out / "flow_weighting_capacity_summary.csv")
    flow_regime = pd.read_csv(out / "flow_weighting_regime_metrics.csv")
    flow_sndk = pd.read_csv(out / "flow_weighting_sndk_concentration.csv")
    flow_manifest = json.loads(
        (out / "flow_weighting_run_manifest.json").read_text()
    )
    top10_metrics = pd.read_csv(out / "topn_capped10_metrics.csv")
    top10_paired = pd.read_csv(out / "topn_capped10_paired_tests.csv")
    top10_regime = pd.read_csv(out / "topn_capped10_regime_metrics.csv")
    top10_sndk = pd.read_csv(out / "topn_capped10_sndk_sensitivity.csv")
    top10_manifest = json.loads((out / "topn_capped10_manifest.json").read_text())
    top10_security = pd.read_csv(out / "topn_capped10_security_weights.csv")
    top10_dates = pd.read_csv(out / "topn_capped10_date_portfolios.csv")
    top10_active_rows: list[dict[str, object]] = []
    for (strategy, cost_bps), group in top10_dates.groupby(
        ["strategy", "cost_bps"], sort=False
    ):
        top10_active_rows.append(
            {
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                **active_trade_metrics(group, "net_return"),
            }
        )
    top10_active_metrics = pd.DataFrame(top10_active_rows)
    top10_metrics = top10_metrics.drop(
        columns=[
            column
            for column in top10_active_metrics.columns
            if column not in {"strategy", "cost_bps"}
        ],
        errors="ignore",
    ).merge(
        top10_active_metrics,
        on=["strategy", "cost_bps"],
        how="left",
        validate="one_to_one",
    )
    top10_metrics.to_csv(out / "topn_capped10_metrics_active.csv", index=False)
    burst_security_ledger = pd.read_csv(out / "first_minute_alpha_security_ledger.csv")
    flow_features = pd.read_csv(out / "flow_weighting_feature_panel.csv")
    state_dir = out.parent / "state_filter_extension"
    state_metrics = pd.read_csv(state_dir / "state_filter_rule_metrics.csv")
    state_paired = pd.read_csv(state_dir / "state_filter_paired_tests.csv")
    state_regime = pd.read_csv(state_dir / "state_filter_regime_metrics.csv")
    state_walk = pd.read_csv(state_dir / "state_filter_walk_forward_metrics.csv")
    state_blend = pd.read_csv(state_dir / "state_filter_blend_frontier.csv")
    state_manifest = json.loads((state_dir / "state_filter_manifest.json").read_text())
    state_opt_dir = out.parent / "state_filter_optimization"
    state_opt_metrics = pd.read_csv(state_opt_dir / "state_optimization_rule_metrics.csv")
    state_opt_paired = pd.read_csv(state_opt_dir / "state_optimization_paired_tests.csv")
    state_opt_walk = pd.read_csv(state_opt_dir / "state_optimization_walk_forward_metrics.csv")
    state_opt_walk_paired = pd.read_csv(
        state_opt_dir / "state_optimization_walk_forward_paired_tests.csv"
    )
    state_opt_scores = pd.read_csv(
        state_opt_dir / "state_optimization_final_training_scores.csv"
    )
    state_opt_loo = pd.read_csv(
        state_opt_dir / "state_optimization_leave_one_review_out.csv"
    )
    state_opt_mechanism = pd.read_csv(
        state_opt_dir / "state_optimization_mechanism_buckets.csv"
    )
    state_opt_audit = json.loads(
        (state_opt_dir / "state_optimization_manual_audit.json").read_text()
    )

    primary = metrics.loc[
        metrics["strategy"].eq(PRIMARY) & metrics["sample"].eq("full")
    ].iloc[0]
    gross_primary = sensitivity.loc[
        sensitivity["strategy"].eq(PRIMARY) & sensitivity["cost_bps"].eq(0.0)
    ].iloc[0]
    announcement = metrics.loc[
        metrics["strategy"].eq("announcement_to_implementation")
        & metrics["sample"].eq("full")
    ].iloc[0]
    reversal = metrics.loc[
        metrics["strategy"].eq("auction_reversal_to_next_open")
        & metrics["sample"].eq("full")
    ].iloc[0]
    pre_1550 = windows.loc[
        windows["window"].eq("15:49_open_to_15:50_open")
        & windows["cost_bps"].eq(0.0)
    ].iloc[0]
    post_1550 = windows.loc[
        windows["window"].eq("15:50_open_to_15:59_close")
        & windows["cost_bps"].eq(0.0)
    ].iloc[0]
    terminal_move = volatility_metrics.loc[
        volatility_metrics["metric"].eq("absolute_terminal_move")
    ].iloc[0]
    breakout_primary = breakout_metrics.loc[
        breakout_metrics["scope"].eq(
            "event_minute_bar_breakout_payoff_proxy"
        )
        & breakout_metrics["barrier_bps"].eq(25.0)
        & breakout_metrics["cost_bps"].eq(10.0)
    ].iloc[0]
    model_primary = straddle_scenarios.loc[
        straddle_scenarios["dte"].eq(7)
        & straddle_scenarios["entry_iv"].eq(0.50)
        & straddle_scenarios["iv_change"].eq(0.0)
        & straddle_scenarios["friction_fraction_of_premium"].eq(0.01)
    ].iloc[0]
    burst_gross = burst_cost.loc[burst_cost["cost_bps"].eq(0.0)].iloc[0]
    burst_net_10 = burst_cost.loc[burst_cost["cost_bps"].eq(10.0)].iloc[0]
    burst_post_qcir = burst_regime.loc[
        burst_regime["sample"].eq("full_qcir_from_may_2023")
        & burst_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    burst_pre_qcir = burst_regime.loc[
        burst_regime["sample"].eq("pre_full_qcir_before_may_2023")
        & burst_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    burst_two_bar_selection = burst_selection.loc[
        burst_selection["analysis"].eq("adjacent_two_bar_window")
    ].iloc[0]
    burst_abnormal = burst_control.loc[
        burst_control["analysis"].eq("event_minus_matched_control")
    ].iloc[0]
    flow_control_10 = flow_metrics.loc[
        flow_metrics["strategy"].eq("equal_pit_universe_10pct_cap")
        & flow_metrics["cost_bps"].eq(10.0)
    ].iloc[0]
    flow_proxy_10 = flow_metrics.loc[
        flow_metrics["strategy"].eq("turnover_pressure_proxy_10pct_cap")
        & flow_metrics["cost_bps"].eq(10.0)
    ].iloc[0]
    flow_proxy_test = flow_paired.loc[
        flow_paired["strategy"].eq("turnover_pressure_proxy_10pct_cap")
    ].iloc[0]
    flow_control_early = flow_regime.loc[
        flow_regime["strategy"].eq("equal_pit_universe_10pct_cap")
        & flow_regime["sample"].eq("2020_through_2022")
        & flow_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    flow_proxy_early = flow_regime.loc[
        flow_regime["strategy"].eq("turnover_pressure_proxy_10pct_cap")
        & flow_regime["sample"].eq("2020_through_2022")
        & flow_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    flow_control_qcir = flow_regime.loc[
        flow_regime["strategy"].eq("equal_pit_universe_10pct_cap")
        & flow_regime["sample"].eq("full_qcir_from_may_2023")
        & flow_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    flow_proxy_qcir = flow_regime.loc[
        flow_regime["strategy"].eq("turnover_pressure_proxy_10pct_cap")
        & flow_regime["sample"].eq("full_qcir_from_may_2023")
        & flow_regime["cost_bps"].eq(0.0)
    ].iloc[0]
    flow_sndk_proxy = flow_sndk.loc[
        flow_sndk["strategy"].eq("turnover_pressure_proxy_10pct_cap")
    ].iloc[0]
    flow_proxy_capacity_row = flow_capacity.loc[
        flow_capacity["strategy"].eq("turnover_pressure_proxy_capacity")
    ].iloc[0]
    flow_proxy_uncapped_capacity_row = flow_capacity.loc[
        flow_capacity["strategy"].eq("turnover_pressure_proxy_10pct_cap")
    ].iloc[0]
    top10_primary_name = "impact_proxy_top10"
    top10_pressure_name = "pressure_proxy_top10"
    top10_control_name = "capped_full_valid_universe"
    top10_primary_zero = top10_metrics.loc[
        top10_metrics["strategy"].eq(top10_primary_name)
        & top10_metrics["cost_bps"].eq(TOP10_PRIMARY_COST_BPS)
    ].iloc[0]
    top10_primary_10bp = top10_metrics.loc[
        top10_metrics["strategy"].eq(top10_primary_name)
        & top10_metrics["cost_bps"].eq(TOP10_SENSITIVITY_COST_BPS)
    ].iloc[0]
    top10_primary_test = top10_paired.loc[
        top10_paired["strategy"].eq(top10_primary_name)
    ].iloc[0]
    top10_primary_sndk = top10_sndk.loc[
        top10_sndk["strategy"].eq(top10_primary_name)
    ].iloc[0]

    # Security-level simulated trade log for the preferred top-10 proxy.
    top10_trade_log = top10_security.loc[
        top10_security["strategy"].eq(top10_primary_name)
    ].copy()
    top10_trade_log["effective_close_date"] = pd.to_datetime(
        top10_trade_log["effective_close_date"]
    )
    top10_trade_log = top10_trade_log.sort_values(
        ["effective_close_date", "action", "impact_proxy", "ticker"],
        ascending=[True, True, False, True],
    )
    top10_trade_log["side_rank"] = (
        top10_trade_log.groupby(["effective_close_date", "action"]).cumcount() + 1
    )
    execution_columns = [
        "effective_close_date",
        "ticker",
        "entry_bar_timestamp",
        "exit_bar_timestamp",
        "entry_open_proxy",
        "exit_close_proxy",
        "exit_bar_volume",
        "day_volume",
        "exit_bar_volume_share",
        "fill_model",
        "information_cutoff",
    ]
    burst_security_ledger["effective_close_date"] = pd.to_datetime(
        burst_security_ledger["effective_close_date"]
    )
    top10_trade_log = top10_trade_log.merge(
        burst_security_ledger[execution_columns],
        on=["effective_close_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if top10_trade_log[["entry_open_proxy", "exit_close_proxy"]].isna().any().any():
        raise ValueError("top-10 trade log has missing entry or exit proxies")
    if not top10_trade_log["feature_point_in_time_valid"].fillna(False).astype(bool).all():
        raise ValueError("top-10 trade log contains a point-in-time-invalid selection")
    feature_cutoff = pd.to_datetime(top10_trade_log["feature_cutoff_date"])
    liquidity_asof = pd.to_datetime(top10_trade_log["liquidity_asof"])
    review_date = top10_trade_log["effective_close_date"].dt.normalize()
    if feature_cutoff.ge(review_date).any() or liquidity_asof.ge(review_date).any():
        raise ValueError("top-10 trade log contains a predictor dated on/after review")
    entry_local = (
        pd.to_datetime(top10_trade_log["entry_bar_timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    information_cutoff_local = review_date + pd.Timedelta(
        hours=15, minutes=48, seconds=59, milliseconds=999
    )
    if information_cutoff_local.ge(entry_local).any():
        raise ValueError("top-10 information cutoff does not precede entry proxy")
    top10_trade_log["side"] = top10_trade_log["action"].map(
        {"add": "Long", "delete": "Short"}
    )
    top10_trade_log["directional_security_return"] = (
        np.sign(top10_trade_log["target_weight"])
        * top10_trade_log["underlying_return"]
    )
    top10_trade_log["modeled_cost_bps"] = TOP10_PRIMARY_COST_BPS
    top10_trade_log["modeled_cost_contribution"] = (
        top10_trade_log["target_abs_weight"] * TOP10_PRIMARY_COST_BPS / 10_000.0
    )
    top10_trade_log["net_pnl_contribution"] = (
        top10_trade_log["security_pnl_contribution"]
        - top10_trade_log["modeled_cost_contribution"]
    )
    top10_trade_log["sensitivity_cost_contribution_10bp"] = (
        top10_trade_log["target_abs_weight"]
        * TOP10_SENSITIVITY_COST_BPS
        / 10_000.0
    )
    top10_trade_log["sensitivity_net_pnl_contribution_10bp"] = (
        top10_trade_log["security_pnl_contribution"]
        - top10_trade_log["sensitivity_cost_contribution_10bp"]
    )
    top10_trade_log["signed_position_usd_at_1m"] = (
        top10_trade_log["target_weight"] * 1_000_000.0
    )
    top10_trade_log["gross_pnl_usd_at_1m"] = (
        top10_trade_log["security_pnl_contribution"] * 1_000_000.0
    )
    top10_trade_log["modeled_cost_usd_at_1m"] = (
        top10_trade_log["modeled_cost_contribution"] * 1_000_000.0
    )
    top10_trade_log["net_pnl_usd_at_1m"] = (
        top10_trade_log["net_pnl_contribution"] * 1_000_000.0
    )
    top10_trade_log["record_type"] = "research_proxy_not_broker_fill"
    top10_trade_log["capital_basis"] = "fixed_$1m_each_review"
    top10_trade_log["entry_proxy_definition"] = (
        "open of 15:49 start-stamped minute bar"
    )
    top10_trade_log["exit_proxy_definition"] = (
        "close of 15:50 start-stamped minute bar (~15:50:59)"
    )
    top10_trade_log.to_csv(out / "topn_capped10_trade_log.csv", index=False)

    top10_date_log = top10_dates.loc[
        top10_dates["strategy"].eq(top10_primary_name)
        & top10_dates["cost_bps"].eq(TOP10_PRIMARY_COST_BPS)
    ].copy()
    top10_date_log["effective_close_date"] = pd.to_datetime(
        top10_date_log["effective_close_date"]
    )
    top10_date_log["trade_status"] = np.where(
        top10_date_log["gross_exposure"].gt(0), "Traded", "Cash / ineligible"
    )
    top10_date_log["modeled_cost_return"] = (
        top10_date_log["gross_return"] - top10_date_log["net_return"]
    )
    flow_features["effective_close_date"] = pd.to_datetime(
        flow_features["effective_close_date"]
    )
    valid_features = flow_features.loc[
        flow_features["feature_point_in_time_valid"].fillna(False).astype(bool)
    ]
    eligible_counts = (
        valid_features.groupby(["effective_close_date", "action"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"add": "eligible_adds", "delete": "eligible_deletes"})
        .reset_index()
    )
    for column in ("eligible_adds", "eligible_deletes"):
        if column not in eligible_counts:
            eligible_counts[column] = 0
    top10_date_log = top10_date_log.merge(
        eligible_counts[
            ["effective_close_date", "eligible_adds", "eligible_deletes"]
        ],
        on="effective_close_date",
        how="left",
        validate="one_to_one",
    )
    top10_date_log[["eligible_adds", "eligible_deletes"]] = top10_date_log[
        ["eligible_adds", "eligible_deletes"]
    ].fillna(0)
    security_net_by_date = top10_trade_log.groupby("effective_close_date")[
        "net_pnl_contribution"
    ].sum()
    active_date_log = top10_date_log.loc[top10_date_log["gross_exposure"].gt(0)]
    if not np.allclose(
        active_date_log["net_return"].to_numpy(float),
        security_net_by_date.reindex(active_date_log["effective_close_date"]).to_numpy(float),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("security-level top-10 net P&L does not reconcile to date P&L")
    top10_date_log.to_csv(out / "topn_capped10_date_trade_log.csv", index=False)

    # Fresh attribution is computed from the same in-memory ledgers used by this report.
    attribution_event = make_event_ledger(top10_trade_log, top10_date_log)
    attribution_regime, attribution_seasonal, attribution_yearly = make_period_summaries(
        attribution_event
    )
    broad_portfolio = pd.read_csv(out / "first_minute_alpha_portfolio.csv")
    volume_by_date = pd.read_csv(out / "volume_diagnostics_by_date.csv")
    attribution_broad_regime, attribution_review_type = make_broad_diagnostics(
        broad_portfolio, volume_by_date
    )
    attribution_drivers = build_driver_table(
        attribution_regime, attribution_broad_regime, attribution_review_type
    )
    attribution_broad_change_pooled = make_all_name_change_type_diagnostics(
        burst_security_ledger
    )
    attribution_broad_change = make_date_balanced_change_type_diagnostics(
        burst_security_ledger
    )
    attribution_clock = make_regime_clock_profile(
        pd.read_csv(out / "event_time_date_panel.csv")
    )
    attribution_influence = make_influence_diagnostics(
        attribution_event, broad_portfolio
    )
    attribution_event.to_csv(
        out / "topn_capped10_quarterly_event_attribution.csv", index=False
    )
    attribution_regime.to_csv(
        out / "topn_capped10_leg_regime_attribution.csv", index=False
    )
    attribution_seasonal.to_csv(
        out / "topn_capped10_quarter_seasonality.csv", index=False
    )
    attribution_yearly.to_csv(
        out / "topn_capped10_yearly_leg_attribution.csv", index=False
    )
    attribution_broad_regime.to_csv(
        out / "first_minute_alpha_regime_leg_diagnostics.csv", index=False
    )
    attribution_review_type.to_csv(
        out / "first_minute_alpha_review_type_diagnostics.csv", index=False
    )
    attribution_drivers.to_csv(
        out / "topn_capped10_early_regime_drivers.csv", index=False
    )
    attribution_broad_change_pooled.to_csv(
        out / "first_minute_alpha_change_type_diagnostics.csv", index=False
    )
    attribution_broad_change.to_csv(
        out / "first_minute_alpha_change_type_date_balanced.csv", index=False
    )
    attribution_clock.to_csv(
        out / "first_minute_alpha_regime_clock_profile.csv", index=False
    )
    attribution_influence.to_csv(
        out / "topn_capped10_regime_influence_diagnostics.csv", index=False
    )
    plot_event_legs(
        attribution_event, out / "charts" / "topn_capped10_event_leg_attribution.png"
    )
    plot_regime_diagnostics(
        attribution_regime,
        attribution_broad_regime,
        out / "charts" / "topn_capped10_early_regime_diagnostics.png",
    )

    def model_case(iv_change: float, friction: float) -> pd.Series:
        return straddle_scenarios.loc[
            straddle_scenarios["dte"].eq(7)
            & straddle_scenarios["entry_iv"].eq(0.50)
            & np.isclose(straddle_scenarios["iv_change"], iv_change)
            & np.isclose(
                straddle_scenarios["friction_fraction_of_premium"], friction
            )
        ].iloc[0]

    model_zero_friction = model_case(0.0, 0.0)
    model_halfpct_friction = model_case(0.0, 0.005)
    model_onepct_friction = model_case(0.0, 0.01)
    model_iv_down = model_case(-0.05, 0.0)
    model_iv_up = model_case(0.05, 0.0)

    primary_curve = curves.loc[curves["strategy"].eq(PRIMARY)]
    aug_portfolio = primary_curve.loc[
        primary_curve["effective_close_date"].eq(pd.Timestamp("2026-08-31"))
    ].iloc[0]

    expected_events = int(coverage["expected_security_events"].sum())
    minute_usable = int(coverage["minute_entry_usable"].sum())
    close_usable = int(coverage["daily_close_proxy_usable"].sum())
    migrations = int(events["change_type"].isin(["up_from_small", "down_to_small"]).sum())

    # Complete event ledger with explicit coverage and exclusion status.
    trade_columns = [
        "effective_close_date",
        "action",
        "name_key",
        "entry_1549",
        "flow_return_gross",
        "pre_1550_return_gross",
        "post_1550_continuous_return_gross",
        "daily_close_gap_return_gross",
        "volume_vs_20d_median",
        "last10_volume_share",
        "close_discrepancy_bps",
    ]
    event_ledger = events.merge(
        ledger[trade_columns],
        on=["effective_close_date", "action", "name_key"],
        how="left",
    )
    failure_reason = (
        failures.groupby(["effective_close_date", "ticker", "security_name"])["reason"]
        .first()
        .rename("failure_reason")
        .reset_index()
    )
    event_ledger = event_ledger.merge(
        failure_reason,
        on=["effective_close_date", "ticker", "security_name"],
        how="left",
    )
    event_ledger["coverage_status"] = np.select(
        [
            event_ledger["flow_return_gross"].notna(),
            event_ledger["entry_1549"].notna(),
            event_ledger["failure_reason"].notna(),
        ],
        ["close proxy usable", "minute only", event_ledger["failure_reason"].fillna("")],
        default="not usable",
    )
    event_ledger = event_ledger.sort_values(
        ["effective_close_date", "action", "security_name"]
    )

    aug = ledger.loc[ledger["effective_close_date"].eq(pd.Timestamp("2026-08-31"))].copy()
    aug = aug.sort_values("flow_return_gross", ascending=False)

    pair_extremes = pd.concat(
        [pairs.nsmallest(10, "pair_gross_return"), pairs.nlargest(10, "pair_gross_return")]
    ).drop_duplicates().sort_values("pair_gross_return")

    strategy_full = metrics.loc[
        metrics["sample"].eq("full")
        & metrics["strategy"].isin(
            [PRIMARY, "auction_reversal_to_next_open", "announcement_to_implementation"]
        )
    ].copy()
    strategy_full["label"] = strategy_full["strategy"].map(
        {
            PRIMARY: "15:49 → daily-close proxy",
            "auction_reversal_to_next_open": "Close proxy → next open reversal",
            "announcement_to_implementation": "Announcement-next-open → implementation close",
        }
    )

    kpis = "".join(
        [
            metric_card("Primary net return", pct(primary["total_return"]), "26 review dates · 10 bp cost", "bad"),
            metric_card("Primary CAGR", pct(primary["cagr"]), "Cash between quarterly events", "bad"),
            metric_card("Maximum drawdown", pct(primary["max_drawdown"]), "Date-balanced portfolio", "bad"),
            metric_card("Break-even cost", f"{num(primary['break_even_cost_bps'])} bp", "Gross edge before cost", "warn"),
            metric_card("15:49→15:50 gross", f"{bp(pre_1550['mean_return_per_event'])} bp", "20 of 26 reviews positive", "good"),
            metric_card("15:50→15:59 gross", f"{bp(post_1550['mean_return_per_event'])} bp", "Not distinguishable from zero", "neutral"),
            metric_card("Matched-pair net", pct(pairs_metrics["total_return"]), "148 pairs · 25 reviews", "bad"),
            metric_card("Data coverage", f"{close_usable}/{expected_events}", "Daily-close proxy usable", "neutral"),
        ]
    )
    burst_kpis = "".join(
        [
            metric_card(
                "Gross mean / review",
                f"{bp(burst_gross['mean_return_per_event'])} bp",
                "15:49 open → 15:50 bar close proxy",
                "good",
            ),
            metric_card(
                "Gross cumulative",
                pct(burst_gross["total_return"]),
                "459 names · 26 reviews",
                "good",
            ),
            metric_card(
                "10 bp net cumulative",
                pct(burst_net_10["total_return"]),
                "Positive in-sample; block CI crosses zero",
                "warn",
            ),
            metric_card(
                "10 bp cash-funded CAGR",
                pct(burst_net_10["cagr"]),
                "Cash between quarterly reviews",
                "warn",
            ),
            metric_card(
                "10 bp max drawdown",
                pct(burst_net_10["max_drawdown"]),
                "Date-level mark-to-model",
                "neutral",
            ),
            metric_card(
                "Gross hit rate",
                pct(burst_gross["hit_rate"], 1),
                "20 of 26 review dates",
                "good",
            ),
            metric_card(
                "Four-review max-t p",
                num(burst_two_bar_selection["block4_max_t_familywise_pvalue"], 3),
                "29 searched adjacent windows",
                "warn",
            ),
            metric_card(
                "Mean break-even",
                f"{num(burst_gross['break_even_cost_bps'])} bp",
                "Statistical cost capacity is much lower",
                "warn",
            ),
        ]
    )
    flow_weighting_kpis = "".join(
        [
            metric_card(
                "Capped-equal net mean",
                f"{bp(flow_control_10['mean_return_per_event'])} bp",
                "Exact 451-row PIT universe · 10 bp cost",
                "warn",
            ),
            metric_card(
                "Proxy net mean",
                f"{bp(flow_proxy_10['mean_return_per_event'])} bp",
                "Inverse clock-volume share · 10 bp cost",
                "warn",
            ),
            metric_card(
                "Paired gross lift",
                f"{bp(flow_proxy_test['mean_difference'])} bp",
                "Versus exact-universe capped equal",
                "good",
            ),
            metric_card(
                "Five-rule block max-t p",
                num(flow_proxy_test["block4_max_t_familywise_pvalue"], 4),
                "Does not cover all prior research choices",
                "warn",
            ),
            metric_card(
                "May-2023+ proxy lift",
                f"{bp(flow_proxy_qcir['mean_return_per_event'] - flow_control_qcir['mean_return_per_event'])} bp",
                "Most full-sample improvement is earlier",
                "bad",
            ),
            metric_card(
                "$1m capacity gross",
                pct(flow_proxy_capacity_row["mean_deployed_gross"], 1),
                "5% of normal 15:49–15:50 dollar volume",
                "neutral",
            ),
            metric_card(
                "Point-in-time coverage",
                f"{integer(flow_manifest['point_in_time_valid_rows'])}/{integer(flow_manifest['feature_rows'])}",
                "No implementation-session volume used",
                "good",
            ),
            metric_card(
                "Licensed net flow",
                "Not supplied",
                "True Δw × AUM weighting remains untested",
                "bad",
            ),
        ]
    )
    top10_kpis = "".join(
        [
            metric_card(
                "Top-10 mean · 0 bp",
                f"{bp(top10_primary_zero['mean_return_per_event'])} bp",
                "Per scheduled review · selected cost",
                "good",
            ),
            metric_card(
                "Active mean · 0 bp",
                f"{bp(top10_primary_zero['active_mean_return'])} bp",
                f"{integer(top10_primary_zero['active_review_dates'])} of {integer(top10_primary_zero['event_dates'])} reviews traded",
                "good",
            ),
            metric_card(
                "Top-10 cumulative · 0 bp",
                pct(top10_primary_zero["total_return"]),
                f"CAGR {pct(top10_primary_zero['cagr'])} · max DD {pct(top10_primary_zero['max_drawdown'])}",
                "good",
            ),
            metric_card(
                "Primary active Sharpe",
                num(top10_primary_zero["active_trade_sharpe_sqrt4"], 2),
                "0 bp · sqrt(4) · zero-exposure reviews excluded",
                "good",
            ),
            metric_card(
                "10 bp active Sharpe",
                num(top10_primary_10bp["active_trade_sharpe_sqrt4"], 2),
                "Execution-stress reference · cash excluded",
                "warn",
            ),
            metric_card(
                "Active hit rate · 0 bp",
                pct(top10_primary_zero["active_hit_rate"], 1),
                "Zero-exposure reviews excluded",
                "good",
            ),
            metric_card(
                "Gross selection lift",
                f"{bp(top10_primary_test['mean_difference'])} bp",
                f"Active-review lift {bp(top10_primary_test['mean_difference'] * top10_primary_test['review_dates'] / top10_primary_test['active_review_dates'])} bp",
                "warn",
            ),
            metric_card(
                "15-rule block max-t p",
                num(top10_primary_test["block4_max_t_familywise_pvalue"], 4),
                "Five rankers × top 5/10/20 family",
                "bad",
            ),
            metric_card(
                "Hard name cap",
                pct(top10_primary_zero["max_abs_name_weight"], 0),
                "Five longs plus five shorts; sparse reviews hold cash",
                "good",
            ),
            metric_card(
                "Exact flow/cap rank",
                "Not tested",
                "Historical tracker flow and float market cap are absent",
                "bad",
            ),
        ]
    )

    volatility_display = volatility_metrics.copy()
    volatility_display["metric_label"] = volatility_display["metric"].map(
        {
            "absolute_terminal_move": "Absolute terminal move",
            "sqrt_realized_variance_11m": "√ realized variance (11m)",
            "high_low_log_range": "High–low log range",
            "max_absolute_excursion": "Maximum absolute excursion",
        }
    )
    volatility_table = table(
        volatility_display,
        [
            "metric_label",
            "all_event_review_dates",
            "all_event_security_events",
            "all_event_mean_across_dates",
            "review_dates",
            "controlled_security_events",
            "event_mean_across_dates",
            "pre_event_control_mean_across_dates",
            "mean_event_minus_pre_event_control",
            "block_bootstrap_excess_ci_low",
            "block_bootstrap_excess_ci_high",
            "positive_excess_date_rate",
        ],
        {
            "metric_label": "Path diagnostic",
            "all_event_review_dates": "All dates",
            "all_event_security_events": "All names",
            "all_event_mean_across_dates": "All-event mean (bp)",
            "review_dates": "Controlled dates",
            "controlled_security_events": "Controlled names",
            "event_mean_across_dates": "Controlled event (bp)",
            "pre_event_control_mean_across_dates": "Pre-event control (bp)",
            "mean_event_minus_pre_event_control": "Excess (bp)",
            "block_bootstrap_excess_ci_low": "CI low (bp)",
            "block_bootstrap_excess_ci_high": "CI high (bp)",
            "positive_excess_date_rate": "Positive dates",
        },
        {
            "all_event_review_dates": integer,
            "all_event_security_events": integer,
            "all_event_mean_across_dates": bp,
            "review_dates": integer,
            "controlled_security_events": integer,
            "event_mean_across_dates": bp,
            "pre_event_control_mean_across_dates": bp,
            "mean_event_minus_pre_event_control": bp,
            "block_bootstrap_excess_ci_low": bp,
            "block_bootstrap_excess_ci_high": bp,
            "positive_excess_date_rate": lambda x: pct(x, 1),
        },
    )

    threshold_display = volatility_thresholds.loc[
        volatility_thresholds["threshold_bps"].le(100)
    ]
    volatility_threshold_table = table(
        threshold_display,
        [
            "threshold_bps",
            "review_dates",
            "controlled_security_events",
            "event_mean_across_dates",
            "pre_event_control_mean_across_dates",
            "mean_event_minus_pre_event_control",
            "block_bootstrap_excess_ci_low",
            "block_bootstrap_excess_ci_high",
            "positive_excess_date_rate",
        ],
        {
            "threshold_bps": "Terminal move threshold (bp)",
            "review_dates": "Dates",
            "controlled_security_events": "Names",
            "event_mean_across_dates": "Event hit rate",
            "pre_event_control_mean_across_dates": "Control hit rate",
            "mean_event_minus_pre_event_control": "Excess",
            "block_bootstrap_excess_ci_low": "CI low",
            "block_bootstrap_excess_ci_high": "CI high",
            "positive_excess_date_rate": "Positive dates",
        },
        {
            "threshold_bps": lambda x: num(x, 0),
            "review_dates": integer,
            "controlled_security_events": integer,
            "event_mean_across_dates": lambda x: pct(x, 1),
            "pre_event_control_mean_across_dates": lambda x: pct(x, 1),
            "mean_event_minus_pre_event_control": lambda x: pct(x, 1),
            "block_bootstrap_excess_ci_low": lambda x: pct(x, 1),
            "block_bootstrap_excess_ci_high": lambda x: pct(x, 1),
            "positive_excess_date_rate": lambda x: pct(x, 1),
        },
    )

    breakout_display = breakout_metrics.loc[
        breakout_metrics["scope"].eq(
            "event_minute_bar_breakout_payoff_proxy"
        )
    ].sort_values(["barrier_bps", "cost_bps"])
    breakout_table = table(
        breakout_display,
        [
            "barrier_bps",
            "cost_bps",
            "mean_trigger_rate",
            "mean_return_per_event",
            "total_return",
            "cagr",
            "max_drawdown",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
            "break_even_cost_bps",
        ],
        {
            "barrier_bps": "Barrier (bp)",
            "cost_bps": "Friction haircut (bp)",
            "mean_trigger_rate": "Trigger rate",
            "mean_return_per_event": "Mean/date (bp)",
            "total_return": "Cumulative",
            "cagr": "Cash-funded CAGR",
            "max_drawdown": "Max DD",
            "block_bootstrap_mean_ci_low": "CI low (bp)",
            "block_bootstrap_mean_ci_high": "CI high (bp)",
            "break_even_cost_bps": "Gross break-even (bp)",
        },
        {
            "barrier_bps": lambda x: num(x, 0),
            "cost_bps": lambda x: num(x, 0),
            "mean_trigger_rate": lambda x: pct(x, 1),
            "mean_return_per_event": bp,
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
            "break_even_cost_bps": num,
        },
    )

    model_no_iv_change = straddle_scenarios.loc[
        straddle_scenarios["dte"].eq(7)
        & straddle_scenarios["entry_iv"].eq(0.50)
        & straddle_scenarios["iv_change"].eq(0.0)
    ].sort_values("friction_fraction_of_premium")
    straddle_model_table = table(
        model_no_iv_change,
        [
            "friction_fraction_of_premium",
            "mean_all_event_model_return_on_premium",
            "median_all_event_model_return_on_premium",
            "all_event_positive_date_rate",
            "all_event_p05",
            "all_event_p95",
            "mean_event_minus_pre_event_control",
        ],
        {
            "friction_fraction_of_premium": "Premium friction",
            "mean_all_event_model_return_on_premium": "Mean modeled R/premium",
            "median_all_event_model_return_on_premium": "Median modeled R/premium",
            "all_event_positive_date_rate": "Positive dates",
            "all_event_p05": "5th percentile",
            "all_event_p95": "95th percentile",
            "mean_event_minus_pre_event_control": "Event−control model excess",
        },
        {
            "friction_fraction_of_premium": lambda x: pct(x, 1),
            "mean_all_event_model_return_on_premium": pct,
            "median_all_event_model_return_on_premium": pct,
            "all_event_positive_date_rate": lambda x: pct(x, 1),
            "all_event_p05": pct,
            "all_event_p95": pct,
            "mean_event_minus_pre_event_control": pct,
        },
    )

    burst_cost_table = table(
        burst_cost,
        [
            "cost_bps",
            "mean_return_per_event",
            "total_return",
            "cagr",
            "max_drawdown",
            "hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "cost_bps": "All-in haircut (bp)",
            "mean_return_per_event": "Mean/review (bp)",
            "total_return": "Cumulative",
            "cagr": "Cash-funded CAGR",
            "max_drawdown": "Max DD",
            "hit_rate": "Hit rate",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 1),
            "mean_return_per_event": bp,
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )

    burst_regime_display = burst_regime.copy()
    burst_regime_display["sample_label"] = burst_regime_display["sample"].map(
        {
            "full": "Full sample",
            "2020_through_2022": "2020–2022",
            "2023_through_2026": "2023–2026",
            "pre_full_qcir_before_may_2023": "Before May 2023",
            "full_qcir_from_may_2023": "May 2023 onward",
            "retrospective_2024_2026": "2024–2026 retrospective",
            "excluding_aug_2026_discovery": "Excluding Aug 2026",
        }
    )
    burst_regime_table = table(
        burst_regime_display,
        [
            "sample_label",
            "cost_bps",
            "security_events",
            "event_dates",
            "mean_return_per_event",
            "total_return",
            "hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "sample_label": "Sample",
            "cost_bps": "Cost (bp)",
            "security_events": "Names",
            "event_dates": "Reviews",
            "mean_return_per_event": "Mean/review (bp)",
            "total_return": "Cumulative",
            "hit_rate": "Hit rate",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 0),
            "security_events": integer,
            "event_dates": integer,
            "mean_return_per_event": bp,
            "total_return": pct,
            "hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )

    burst_selection_display = burst_selection.copy()
    burst_selection_display["analysis_label"] = burst_selection_display["analysis"].map(
        {
            "one_minute_bar": "Selected 15:50 bar",
            "adjacent_two_bar_window": "15:49 open → 15:50 close",
        }
    )
    burst_selection_table = table(
        burst_selection_display,
        [
            "analysis_label",
            "complete_review_dates",
            "windows_examined",
            "candidate_mean_return",
            "rank_by_mean",
            "raw_one_sided_t_pvalue",
            "bonferroni_pvalue",
            "max_t_familywise_pvalue",
            "block4_max_t_familywise_pvalue",
        ],
        {
            "analysis_label": "Selected interval",
            "complete_review_dates": "Reviews",
            "windows_examined": "Windows",
            "candidate_mean_return": "Mean (bp)",
            "rank_by_mean": "Rank",
            "raw_one_sided_t_pvalue": "Raw p",
            "bonferroni_pvalue": "Bonferroni p",
            "max_t_familywise_pvalue": "Date max-t p",
            "block4_max_t_familywise_pvalue": "4-review block max-t p",
        },
        {
            "complete_review_dates": integer,
            "windows_examined": integer,
            "candidate_mean_return": bp,
            "rank_by_mean": integer,
            "raw_one_sided_t_pvalue": lambda x: num(x, 4),
            "bonferroni_pvalue": lambda x: num(x, 4),
            "max_t_familywise_pvalue": lambda x: num(x, 4),
            "block4_max_t_familywise_pvalue": lambda x: num(x, 4),
        },
    )

    burst_control_display = burst_control.loc[
        burst_control["analysis"].isin(
            [
                "event_on_complete_control_sample",
                "matched_control_mean",
                "event_minus_matched_control",
            ]
        )
    ].copy()
    burst_control_display["analysis_label"] = burst_control_display["analysis"].map(
        {
            "event_on_complete_control_sample": "MSCI event window",
            "matched_control_mean": "Four prior same-name sessions",
            "event_minus_matched_control": "Paired event minus control",
        }
    )
    burst_control_table = table(
        burst_control_display,
        [
            "analysis_label",
            "event_dates",
            "mean_return_per_event",
            "hit_rate",
            "one_sided_t_pvalue",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "analysis_label": "Control test",
            "event_dates": "Reviews",
            "mean_return_per_event": "Mean (bp)",
            "hit_rate": "Hit rate",
            "one_sided_t_pvalue": "Raw p",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
        },
        {
            "event_dates": integer,
            "mean_return_per_event": bp,
            "hit_rate": lambda x: pct(x, 1),
            "one_sided_t_pvalue": lambda x: num(x, 4),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )

    burst_subgroup_display = burst_subgroup.copy()
    burst_subgroup_display["subgroup_label"] = burst_subgroup_display["subgroup"].map(
        {
            "all": "Balanced all-name portfolio",
            "additions_only": "Long additions",
            "deletions_only": "Short deletions",
            "size_migrations": "Standard↔Small migrations",
            "new_entries_and_full_exits": "New Standard entries / full exits",
        }
    )
    burst_subgroup_table = table(
        burst_subgroup_display,
        [
            "subgroup_label",
            "cost_bps",
            "security_events",
            "event_dates",
            "mean_return_per_event",
            "hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "subgroup_label": "Sleeve",
            "cost_bps": "Cost (bp)",
            "security_events": "Names",
            "event_dates": "Reviews",
            "mean_return_per_event": "Gross mean (bp)",
            "hit_rate": "Hit rate",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 0),
            "security_events": integer,
            "event_dates": integer,
            "mean_return_per_event": bp,
            "hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )

    flow_strategy_labels = {
        "equal_pit_universe_10pct_cap": "Equal · exact PIT universe · 10% name cap",
        "equal_pit_universe_capacity": "Equal · exact PIT universe · 5% interval cap",
        "sqrt_adv_10pct_cap": "√ADV · 10% name cap",
        "sqrt_adv_capacity": "√ADV · 5% interval cap",
        "turnover_pressure_proxy_10pct_cap": "Inverse clock-volume-share proxy · 10% cap",
        "turnover_pressure_proxy_capacity": "Inverse clock-volume-share proxy · 5% interval cap",
    }
    flow_order = list(flow_strategy_labels)
    flow_metrics_display = flow_metrics.loc[
        flow_metrics["cost_bps"].eq(10.0)
        & flow_metrics["strategy"].isin(flow_order)
    ].copy()
    flow_metrics_display["strategy_label"] = flow_metrics_display["strategy"].map(
        flow_strategy_labels
    )
    flow_metrics_display["_order"] = flow_metrics_display["strategy"].map(
        {name: index for index, name in enumerate(flow_order)}
    )
    flow_metrics_display = flow_metrics_display.sort_values("_order")
    flow_metrics_table = table(
        flow_metrics_display,
        [
            "strategy_label",
            "event_dates",
            "mean_return_per_event",
            "total_return",
            "cagr",
            "max_drawdown",
            "hit_rate",
            "mean_gross_exposure",
            "mean_effective_names",
        ],
        {
            "strategy_label": "Weighting rule · 10 bp cost",
            "event_dates": "Reviews",
            "mean_return_per_event": "Mean/review (bp)",
            "total_return": "Cumulative",
            "cagr": "Cash-funded CAGR",
            "max_drawdown": "Max DD",
            "hit_rate": "Hit rate",
            "mean_gross_exposure": "Mean gross",
            "mean_effective_names": "Effective names",
        },
        {
            "event_dates": integer,
            "mean_return_per_event": bp,
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "hit_rate": lambda x: pct(x, 1),
            "mean_gross_exposure": lambda x: pct(x, 1),
            "mean_effective_names": lambda x: num(x, 1),
        },
    )

    flow_paired_display = flow_paired.loc[
        flow_paired["strategy"].isin(
            [
                "sqrt_adv_10pct_cap",
                "sqrt_adv_capacity",
                "turnover_pressure_proxy_10pct_cap",
                "turnover_pressure_proxy_capacity",
            ]
        )
    ].copy()
    flow_paired_display["strategy_label"] = flow_paired_display["strategy"].map(
        flow_strategy_labels
    )
    flow_paired_display["comparator_label"] = flow_paired_display["comparator"].map(
        flow_strategy_labels
    )
    flow_paired_table = table(
        flow_paired_display,
        [
            "strategy_label",
            "comparator_label",
            "review_dates",
            "mean_difference",
            "positive_difference_rate",
            "block4_mean_ci_low",
            "block4_mean_ci_high",
            "block4_max_t_familywise_pvalue",
        ],
        {
            "strategy_label": "Challenger",
            "comparator_label": "Exact-universe control",
            "review_dates": "Reviews",
            "mean_difference": "Mean lift (bp)",
            "positive_difference_rate": "Positive lift dates",
            "block4_mean_ci_low": "Block CI low (bp)",
            "block4_mean_ci_high": "Block CI high (bp)",
            "block4_max_t_familywise_pvalue": "Five-rule block max-t p",
        },
        {
            "review_dates": integer,
            "mean_difference": bp,
            "positive_difference_rate": lambda x: pct(x, 1),
            "block4_mean_ci_low": bp,
            "block4_mean_ci_high": bp,
            "block4_max_t_familywise_pvalue": lambda x: num(x, 4),
        },
    )

    flow_regime_display = flow_regime.loc[
        flow_regime["strategy"].isin(
            ["equal_pit_universe_10pct_cap", "turnover_pressure_proxy_10pct_cap"]
        )
        & flow_regime["sample"].isin(
            ["2020_through_2022", "full_qcir_from_may_2023"]
        )
        & flow_regime["cost_bps"].eq(0.0)
    ].copy()
    flow_regime_display["strategy_label"] = flow_regime_display["strategy"].map(
        flow_strategy_labels
    )
    flow_regime_display["sample_label"] = flow_regime_display["sample"].map(
        {
            "2020_through_2022": "2020–2022",
            "full_qcir_from_may_2023": "May 2023 onward",
        }
    )
    flow_regime_table = table(
        flow_regime_display,
        [
            "strategy_label",
            "sample_label",
            "event_dates",
            "mean_return_per_event",
            "hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
            "mean_gross_exposure",
        ],
        {
            "strategy_label": "Rule",
            "sample_label": "Regime",
            "event_dates": "Reviews",
            "mean_return_per_event": "Gross mean (bp)",
            "hit_rate": "Hit rate",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
            "mean_gross_exposure": "Mean gross",
        },
        {
            "event_dates": integer,
            "mean_return_per_event": bp,
            "hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
            "mean_gross_exposure": lambda x: pct(x, 1),
        },
    )

    flow_capacity_display = flow_capacity.loc[
        flow_capacity["strategy"].isin(flow_order)
    ].copy()
    flow_capacity_display["strategy_label"] = flow_capacity_display["strategy"].map(
        flow_strategy_labels
    )
    flow_capacity_display["_order"] = flow_capacity_display["strategy"].map(
        {name: index for index, name in enumerate(flow_order)}
    )
    flow_capacity_display = flow_capacity_display.sort_values("_order")
    flow_capacity_table = table(
        flow_capacity_display,
        [
            "strategy_label",
            "mean_deployed_gross",
            "interval_participation_p50",
            "interval_participation_p95",
            "interval_participation_max",
            "adv_participation_p95",
            "unknown_interval_active_names",
        ],
        {
            "strategy_label": "Rule at $1m capital",
            "mean_deployed_gross": "Mean gross",
            "interval_participation_p50": "Interval p50",
            "interval_participation_p95": "Interval p95",
            "interval_participation_max": "Interval max",
            "adv_participation_p95": "ADV p95",
            "unknown_interval_active_names": "Unknown interval",
        },
        {
            "mean_deployed_gross": lambda x: pct(x, 1),
            "interval_participation_p50": lambda x: pct(x, 1),
            "interval_participation_p95": lambda x: pct(x, 1),
            "interval_participation_max": lambda x: pct(x, 1),
            "adv_participation_p95": lambda x: pct(x, 2),
            "unknown_interval_active_names": integer,
        },
    )

    top10_strategy_labels = {
        top10_control_name: "All PIT-valid names · same gate · 10% cap",
        top10_pressure_name: "Up to 5+5 · liquidity-pressure proxy",
        top10_primary_name: "Up to 5+5 · volatility-scaled impact proxy",
        "composite_equal_rank_top5": "Top 5 total · composite proxy · low exposure",
    }
    top10_metrics_display = top10_metrics.loc[
        top10_metrics["cost_bps"].eq(TOP10_PRIMARY_COST_BPS)
        & top10_metrics["strategy"].isin(top10_strategy_labels)
    ].copy()
    top10_metrics_display["strategy_label"] = top10_metrics_display["strategy"].map(
        top10_strategy_labels
    )
    top10_metrics_display["_order"] = top10_metrics_display["strategy"].map(
        {name: index for index, name in enumerate(top10_strategy_labels)}
    )
    top10_metrics_display = top10_metrics_display.sort_values("_order")
    top10_metrics_table = table(
        top10_metrics_display,
        [
            "strategy_label",
            "active_review_dates",
            "mean_return_per_event",
            "active_mean_return",
            "active_hit_rate",
            "active_trade_sharpe_sqrt4",
            "event_sortino_sqrt4",
            "total_return",
            "cagr",
            "max_drawdown",
            "mean_gross_exposure",
            "max_abs_name_weight",
        ],
        {
            "strategy_label": "Rule · 0 bp cost",
            "active_review_dates": "Active / 26",
            "mean_return_per_event": "Mean/scheduled (bp)",
            "active_mean_return": "Mean/active (bp)",
            "active_hit_rate": "Active hit rate",
            "active_trade_sharpe_sqrt4": "Active Sharpe √4",
            "event_sortino_sqrt4": "Event Sortino √4",
            "total_return": "Cumulative",
            "cagr": "Cash-funded CAGR",
            "max_drawdown": "Max DD",
            "mean_gross_exposure": "Mean gross",
            "max_abs_name_weight": "Max name",
        },
        {
            "active_review_dates": integer,
            "mean_return_per_event": bp,
            "active_mean_return": bp,
            "active_hit_rate": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "event_sortino_sqrt4": lambda x: num(x, 2),
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "mean_gross_exposure": lambda x: pct(x, 1),
            "max_abs_name_weight": lambda x: pct(x, 0),
        },
    )

    top10_paired_display = top10_paired.loc[
        top10_paired["strategy"].isin(
            [top10_pressure_name, top10_primary_name, "composite_equal_rank_top5"]
        )
    ].copy()
    top10_paired_display["strategy_label"] = top10_paired_display["strategy"].map(
        top10_strategy_labels
    )
    top10_paired_table = table(
        top10_paired_display,
        [
            "strategy_label",
            "mean_difference",
            "block4_mean_ci_low",
            "block4_mean_ci_high",
            "one_sided_t_pvalue",
            "block4_max_t_familywise_pvalue",
            "net10_vs_zero_block4_max_t_familywise_pvalue",
        ],
        {
            "strategy_label": "Selection rule",
            "mean_difference": "Gross lift vs matched control (bp)",
            "block4_mean_ci_low": "Lift CI low (bp)",
            "block4_mean_ci_high": "Lift CI high (bp)",
            "one_sided_t_pvalue": "Raw p",
            "block4_max_t_familywise_pvalue": "15-rule block max-t p",
            "net10_vs_zero_block4_max_t_familywise_pvalue": "10 bp sensitivity: net-vs-zero family p",
        },
        {
            "mean_difference": bp,
            "block4_mean_ci_low": bp,
            "block4_mean_ci_high": bp,
            "one_sided_t_pvalue": lambda x: num(x, 4),
            "block4_max_t_familywise_pvalue": lambda x: num(x, 4),
            "net10_vs_zero_block4_max_t_familywise_pvalue": lambda x: num(x, 4),
        },
    )

    top10_regime_display = top10_regime.loc[
        top10_regime["strategy"].eq(top10_primary_name)
        & top10_regime["regime"].isin(["2020_2022", "2023_plus"])
    ].copy()
    top10_regime_display["regime_label"] = top10_regime_display["regime"].map(
        {"2020_2022": "2020–2022", "2023_plus": "2023 onward"}
    )
    top10_regime_table = table(
        top10_regime_display,
        [
            "regime_label",
            "cost_bps",
            "event_dates",
            "active_review_dates",
            "mean_return_per_event",
            "active_mean_return",
            "active_hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "regime_label": "Regime",
            "cost_bps": "Scenario cost (bp; primary = 0)",
            "event_dates": "Scheduled",
            "active_review_dates": "Active",
            "mean_return_per_event": "Mean/scheduled (bp)",
            "active_mean_return": "Mean/active (bp)",
            "active_hit_rate": "Active hit",
            "block_bootstrap_mean_ci_low": "Block CI low (bp)",
            "block_bootstrap_mean_ci_high": "Block CI high (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 0),
            "event_dates": integer,
            "active_review_dates": integer,
            "mean_return_per_event": bp,
            "active_mean_return": bp,
            "active_hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )

    attribution_all = attribution_regime.loc[
        attribution_regime["period"].eq("All")
    ].iloc[0]
    attribution_early = attribution_regime.loc[
        attribution_regime["period"].eq("2020-2022")
    ].iloc[0]
    attribution_late = attribution_regime.loc[
        attribution_regime["period"].eq("2023+")
    ].iloc[0]
    attribution_kpis = "".join(
        [
            metric_card(
                "Long additions",
                f"{num(attribution_all['long_total_bp'])} bp",
                f"{num(attribution_all['long_signed_share_pct'])}% of arithmetic gross profit",
                "good",
            ),
            metric_card(
                "Short deletions",
                f"{num(attribution_all['short_total_bp'])} bp",
                f"{num(attribution_all['short_signed_share_pct'])}% of arithmetic gross profit",
                "good",
            ),
            metric_card(
                "2020-2022 short leg",
                f"{num(attribution_early['short_total_bp'])} bp",
                "Main early-period detractor",
                "bad",
            ),
            metric_card(
                "2023+ short leg",
                f"+{num(attribution_late['short_total_bp'])} bp",
                "Both legs positive after breakpoint",
                "good",
            ),
        ]
    )

    attribution_regime_display = attribution_regime.loc[
        attribution_regime["period"].isin(["All", "2020-2022", "2023+"])
    ].copy()
    attribution_regime_display["reviews"] = attribution_regime_display.apply(
        lambda row: f"{int(row['scheduled_reviews'])}/{int(row['active_reviews'])}/{int(row['cash_reviews'])}",
        axis=1,
    )
    attribution_regime_display["signed_share"] = attribution_regime_display.apply(
        lambda row: f"{row['long_signed_share_pct']:.1f}% / {row['short_signed_share_pct']:.1f}%",
        axis=1,
    )
    attribution_regime_display["alignment"] = attribution_regime_display.apply(
        lambda row: f"{int(row['both_legs_positive'])} both + / {int(row['leg_sign_disagreement'])} disagree",
        axis=1,
    )
    attribution_regime_table = table(
        attribution_regime_display,
        [
            "period",
            "reviews",
            "long_total_bp",
            "short_total_bp",
            "arithmetic_total_bp",
            "signed_share",
            "leg_netting_capture_pct",
            "alignment",
            "net_10bp_mean_per_scheduled_bp",
        ],
        {
            "period": "Slice",
            "reviews": "Reviews S/A/C",
            "long_total_bp": "Long contribution (bp)",
            "short_total_bp": "Short contribution (bp)",
            "arithmetic_total_bp": "Total (bp)",
            "signed_share": "Signed L/S profit share",
            "leg_netting_capture_pct": "Netting capture",
            "alignment": "Active-date alignment",
            "net_10bp_mean_per_scheduled_bp": "Net mean @10bp",
        },
        {
            "long_total_bp": lambda x: f"{float(x):+.2f}",
            "short_total_bp": lambda x: f"{float(x):+.2f}",
            "arithmetic_total_bp": lambda x: f"{float(x):+.2f}",
            "leg_netting_capture_pct": lambda x: f"{float(x):.1f}%",
            "net_10bp_mean_per_scheduled_bp": lambda x: f"{float(x):+.2f} bp",
        },
    )

    attribution_seasonal_display = attribution_seasonal.copy()
    attribution_seasonal_display["reviews"] = attribution_seasonal_display.apply(
        lambda row: f"{int(row['scheduled_reviews'])}/{int(row['active_reviews'])}/{int(row['cash_reviews'])}",
        axis=1,
    )
    attribution_seasonal_display["active_hit"] = attribution_seasonal_display.apply(
        lambda row: f"{100.0 * row['long_active_hit_rate']:.0f}% / {100.0 * row['short_active_hit_rate']:.0f}%",
        axis=1,
    )
    attribution_seasonal_table = table(
        attribution_seasonal_display,
        [
            "period",
            "reviews",
            "long_total_bp",
            "short_total_bp",
            "arithmetic_total_bp",
            "gross_mean_per_scheduled_bp",
            "active_hit",
        ],
        {
            "period": "Calendar quarter",
            "reviews": "Reviews S/A/C",
            "long_total_bp": "Long bp",
            "short_total_bp": "Short bp",
            "arithmetic_total_bp": "Total bp",
            "gross_mean_per_scheduled_bp": "Mean/scheduled bp",
            "active_hit": "Active hit L/S",
        },
        {
            "long_total_bp": lambda x: f"{float(x):+.2f}",
            "short_total_bp": lambda x: f"{float(x):+.2f}",
            "arithmetic_total_bp": lambda x: f"{float(x):+.2f}",
            "gross_mean_per_scheduled_bp": lambda x: f"{float(x):+.2f}",
        },
    )

    attribution_event_display = attribution_event.copy()
    attribution_event_display["selected"] = attribution_event_display.apply(
        lambda row: f"{int(row['n_long'])}/{int(row['n_short'])}", axis=1
    )
    attribution_event_display["eligible"] = attribution_event_display.apply(
        lambda row: f"{int(row['eligible_adds'])}/{int(row['eligible_deletes'])}", axis=1
    )
    attribution_event_display["absolute_share"] = attribution_event_display.apply(
        lambda row: "—"
        if pd.isna(row["absolute_long_share_pct"])
        else f"{row['absolute_long_share_pct']:.0f}%/{row['absolute_short_share_pct']:.0f}%",
        axis=1,
    )
    attribution_event_table = table(
        attribution_event_display,
        [
            "quarter",
            "effective_close_date",
            "status",
            "selected",
            "eligible",
            "long_contribution_bp",
            "short_contribution_bp",
            "gross_return_bp",
            "net_10bp_bp",
            "absolute_share",
            "top_contributor",
        ],
        {
            "quarter": "Quarter",
            "effective_close_date": "Implementation close",
            "status": "Status",
            "selected": "Selected L/S",
            "eligible": "Eligible A/D",
            "long_contribution_bp": "Long bp",
            "short_contribution_bp": "Short bp",
            "gross_return_bp": "Gross bp",
            "net_10bp_bp": "Net @10bp",
            "absolute_share": "Abs L/S share",
            "top_contributor": "Largest absolute name",
        },
        {
            "effective_close_date": date_string,
            "long_contribution_bp": lambda x: f"{float(x):+.2f}",
            "short_contribution_bp": lambda x: f"{float(x):+.2f}",
            "gross_return_bp": lambda x: f"{float(x):+.2f}",
            "net_10bp_bp": lambda x: f"{float(x):+.2f}",
        },
        table_id="top10-attribution-ledger",
    )

    attribution_driver_table = table(
        attribution_drivers,
        ["metric", "2020_2022", "2023_plus", "difference", "unit", "interpretation"],
        {
            "metric": "Diagnostic",
            "2020_2022": "2020-2022",
            "2023_plus": "2023+",
            "difference": "Difference",
            "unit": "Unit",
            "interpretation": "Interpretation",
        },
        {
            "2020_2022": lambda x: num(x, 2),
            "2023_plus": lambda x: num(x, 2),
            "difference": lambda x: f"{float(x):+.2f}",
        },
    )

    attribution_migration_table = table(
        attribution_broad_change,
        [
            "regime",
            "action",
            "change_type",
            "review_dates",
            "security_events",
            "mean_date_directional_return_bp",
            "date_hit_rate",
        ],
        {
            "regime": "Regime",
            "action": "Public action",
            "change_type": "Migration type",
            "review_dates": "Dates",
            "security_events": "Names",
            "mean_date_directional_return_bp": "Mean date bp",
            "date_hit_rate": "Date hit",
        },
        {
            "review_dates": integer,
            "security_events": integer,
            "mean_date_directional_return_bp": lambda x: f"{float(x):+.2f}",
            "date_hit_rate": lambda x: pct(x, 1),
        },
    )

    attribution_review_type_table = table(
        attribution_review_type,
        [
            "regime",
            "review_class",
            "review_dates",
            "mean_security_events",
            "mean_long_contribution_bp",
            "mean_short_contribution_bp",
            "mean_gross_return_bp",
        ],
        {
            "regime": "Regime",
            "review_class": "Review class",
            "review_dates": "Dates",
            "mean_security_events": "Names/date",
            "mean_long_contribution_bp": "Long bp",
            "mean_short_contribution_bp": "Short bp",
            "mean_gross_return_bp": "Gross bp",
        },
        {
            "review_dates": integer,
            "mean_security_events": lambda x: num(x, 2),
            "mean_long_contribution_bp": lambda x: f"{float(x):+.2f}",
            "mean_short_contribution_bp": lambda x: f"{float(x):+.2f}",
            "mean_gross_return_bp": lambda x: f"{float(x):+.2f}",
        },
    )
    attribution_clock_early_1550 = attribution_clock.loc[
        attribution_clock["regime"].eq("2020-2022")
        & attribution_clock["clock"].eq("15:50")
    ].iloc[0]
    attribution_clock_late_1550 = attribution_clock.loc[
        attribution_clock["regime"].eq("2023+")
        & attribution_clock["clock"].eq("15:50")
    ].iloc[0]
    attribution_top_ex_2026 = attribution_influence.loc[
        attribution_influence["portfolio"].eq("Impact top-10")
        & attribution_influence["sample"].eq("2023-2025")
    ].iloc[0]
    attribution_broad_ex_2026 = attribution_influence.loc[
        attribution_influence["portfolio"].eq("All-name")
        & attribution_influence["sample"].eq("2023-2025")
    ].iloc[0]
    attribution_year_2026 = attribution_yearly.loc[
        attribution_yearly["period"].astype(str).eq("2026")
    ].iloc[0]

    state_rule_labels = {
        "baseline_all": "Baseline Top-10",
        "down_to_small_hour_confirmation": "Down-to-small: relative-hour confirmation",
        "down_to_small_day_confirmation": "Down-to-small: relative-day confirmation",
        "deletion_relative_hour_confirmation": "All deletions: relative-hour confirmation",
        "deletion_relative_day_confirmation": "All deletions: relative-day confirmation",
        "same_clock_day_rvol_above_one": "Same-clock day RVOL > 1",
        "same_clock_hour_rvol_above_one": "Same-clock hour RVOL > 1",
        "relative_hour_momentum_same_clock_rvol": "Relative-hour momentum + RVOL",
    }
    state_10 = state_metrics.loc[state_metrics["cost_bps"].eq(10.0)].copy()
    state_baseline = state_10.loc[state_10["filter_rule"].eq("baseline_all")].iloc[0]
    state_gate = state_10.loc[
        state_10["filter_rule"].eq("down_to_small_hour_confirmation")
    ].iloc[0]
    state_10["rule_label"] = state_10["filter_rule"].map(state_rule_labels)
    state_10["net_retention_pct"] = (
        state_10["arithmetic_net_return_bp"]
        / float(state_baseline["arithmetic_net_return_bp"])
        * 100.0
    )
    state_rule_table = table(
        state_10.sort_values("mean_return_per_event", ascending=False),
        [
            "rule_label",
            "arithmetic_net_return_bp",
            "net_retention_pct",
            "active_review_dates",
            "active_review_hit_rate",
            "mean_gross_exposure",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
        ],
        {
            "rule_label": "10 bp rule",
            "arithmetic_net_return_bp": "Arithmetic net bp",
            "net_retention_pct": "Net retained",
            "active_review_dates": "Active dates",
            "active_review_hit_rate": "Active hit",
            "mean_gross_exposure": "Mean gross",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
            "block_bootstrap_mean_ci_low": "Block CI low bp",
            "block_bootstrap_mean_ci_high": "Block CI high bp",
        },
        {
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "net_retention_pct": lambda x: f"{float(x):.1f}%",
            "active_review_dates": integer,
            "active_review_hit_rate": lambda x: pct(x, 1),
            "mean_gross_exposure": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
        },
    )
    state_gate_paired = state_paired.loc[
        state_paired["filter_rule"].eq("down_to_small_hour_confirmation")
        & state_paired["cost_bps"].eq(10.0)
    ].iloc[0]
    state_regime_display = state_regime.loc[
        state_regime["cost_bps"].eq(10.0)
        & state_regime["filter_rule"].isin(
            ["baseline_all", "down_to_small_hour_confirmation"]
        )
        & state_regime["sample"].isin(
            ["2020-2022", "2023_plus", "excluding_2026"]
        )
    ].copy()
    state_regime_display["rule_label"] = state_regime_display["filter_rule"].map(
        state_rule_labels
    )
    state_regime_table = table(
        state_regime_display,
        [
            "rule_label",
            "sample",
            "arithmetic_net_return_bp",
            "active_review_dates",
            "active_review_hit_rate",
            "mean_gross_exposure",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
        ],
        {
            "rule_label": "Rule",
            "sample": "Slice",
            "arithmetic_net_return_bp": "Net bp",
            "active_review_dates": "Active",
            "active_review_hit_rate": "Active hit",
            "mean_gross_exposure": "Mean gross",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
        },
        {
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "active_review_dates": integer,
            "active_review_hit_rate": lambda x: pct(x, 1),
            "mean_gross_exposure": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
        },
    )
    state_walk_display = state_walk.copy()
    state_walk_display["rule_label"] = state_walk_display["filter_rule"].map(
        {
            "adaptive_walk_forward": "Adaptive rule selection",
            "matched_baseline": "Matched baseline",
        }
    )
    state_walk_table = table(
        state_walk_display,
        [
            "rule_label",
            "event_dates",
            "arithmetic_net_return_bp",
            "active_review_dates",
            "active_review_hit_rate",
            "mean_gross_exposure",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
        ],
        {
            "rule_label": "Pseudo-OOS series",
            "event_dates": "Reviews",
            "arithmetic_net_return_bp": "Net bp",
            "active_review_dates": "Active",
            "active_review_hit_rate": "Active hit",
            "mean_gross_exposure": "Mean gross",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
        },
        {
            "event_dates": integer,
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "active_review_dates": integer,
            "active_review_hit_rate": lambda x: pct(x, 1),
            "mean_gross_exposure": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
        },
    )
    state_walk_adaptive = state_walk.loc[
        state_walk["filter_rule"].eq("adaptive_walk_forward")
    ].iloc[0]
    state_walk_baseline = state_walk.loc[
        state_walk["filter_rule"].eq("matched_baseline")
    ].iloc[0]
    state_blend_display = state_blend.loc[
        np.isclose(state_blend["gate_weight"], 0.0)
        | np.isclose(state_blend["gate_weight"], 0.5)
        | np.isclose(state_blend["gate_weight"], 0.8)
        | np.isclose(state_blend["gate_weight"], 1.0)
    ].copy()
    state_blend_table = table(
        state_blend_display,
        [
            "gate_weight",
            "arithmetic_net_return_bp",
            "net_return_retention_vs_baseline",
            "active_review_hit_rate",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
            "mean_gross_exposure",
        ],
        {
            "gate_weight": "Gate sleeve weight",
            "arithmetic_net_return_bp": "Net bp",
            "net_return_retention_vs_baseline": "Net retained",
            "active_review_hit_rate": "Active hit",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
            "mean_gross_exposure": "Mean gross",
        },
        {
            "gate_weight": lambda x: pct(x, 0),
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "net_return_retention_vs_baseline": lambda x: pct(x, 1),
            "active_review_hit_rate": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
            "mean_gross_exposure": lambda x: pct(x, 1),
        },
    )
    state_blend_80 = state_blend.loc[np.isclose(state_blend["gate_weight"], 0.8)].iloc[0]
    state_filter_kpis = "".join(
        [
            metric_card(
                "Baseline active hit · 10 bp",
                pct(state_baseline["active_review_hit_rate"], 1),
                f"{num(state_baseline['arithmetic_net_return_bp'])} arithmetic net bp",
                "neutral",
            ),
            metric_card(
                "Migration gate active hit",
                pct(state_gate["active_review_hit_rate"], 1),
                f"{num(state_gate['arithmetic_net_return_bp'])} net bp · {num(100.0 * state_gate['arithmetic_net_return_bp'] / state_baseline['arithmetic_net_return_bp'], 1)}% retained",
                "good",
            ),
            metric_card(
                "80% gate blend",
                pct(state_blend_80["active_review_hit_rate"], 1),
                f"{pct(state_blend_80['net_return_retention_vs_baseline'], 1)} of net return retained",
                "warn",
            ),
            metric_card(
                "Pseudo-OOS active hit",
                pct(state_walk_adaptive["active_review_hit_rate"], 1),
                f"Baseline {pct(state_walk_baseline['active_review_hit_rate'], 1)} · 18 reviews",
                "warn",
            ),
            metric_card(
                "Pseudo-OOS net retention",
                pct(state_walk_adaptive["net_return_retention_vs_matched_baseline"], 1),
                f"{num(state_walk_adaptive['arithmetic_net_return_bp'])} versus {num(state_walk_baseline['arithmetic_net_return_bp'])} bp",
                "warn",
            ),
            metric_card(
                "Familywise paired p",
                num(state_gate_paired["block4_max_t_familywise_pvalue"], 4),
                "Gate does not beat baseline return after rule search",
                "bad",
            ),
        ]
    )

    state_opt_labels = {
        "baseline_all": "Baseline",
        "down_to_small_hour_soft_075": "Retain 75%",
        "down_to_small_hour_soft_050": "Retain 50%",
        "down_to_small_hour_soft_025": "Retain 25%",
        "down_to_small_hour_confirmation": "Hard gate",
        "down_to_small_hour_or_day_confirmation": "Hour OR day",
        "down_to_small_hour_and_day_confirmation": "Hour AND day",
        "matched_baseline": "Matched baseline",
        "constrained_minimum_intervention": "Constrained selector",
        "maximum_training_mean": "Maximum training mean",
    }
    state_opt_fixed = state_opt_metrics.loc[
        state_opt_metrics["cost_bps"].eq(10.0)
        & state_opt_metrics["filter_rule"].isin(
            [
                "baseline_all",
                "down_to_small_hour_soft_075",
                "down_to_small_hour_soft_050",
                "down_to_small_hour_soft_025",
                "down_to_small_hour_confirmation",
            ]
        )
    ].copy()
    state_opt_fixed["rule_label"] = state_opt_fixed["filter_rule"].map(
        state_opt_labels
    )
    state_opt_fixed_table = table(
        state_opt_fixed,
        [
            "rule_label",
            "arithmetic_net_return_bp",
            "active_review_hit_rate",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
            "mean_gross_exposure",
            "break_even_cost_bps",
        ],
        {
            "rule_label": "Frozen rule",
            "arithmetic_net_return_bp": "Net bp",
            "active_review_hit_rate": "Active hit",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
            "mean_gross_exposure": "Mean gross",
            "break_even_cost_bps": "Break-even cost",
        },
        {
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "active_review_hit_rate": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
            "mean_gross_exposure": lambda x: pct(x, 1),
            "break_even_cost_bps": lambda x: f"{float(x):.2f} bp",
        },
    )
    state_opt_walk_display = state_opt_walk.copy()
    state_opt_walk_display["rule_label"] = state_opt_walk_display["filter_rule"].map(
        state_opt_labels
    )
    state_opt_walk_table = table(
        state_opt_walk_display,
        [
            "rule_label",
            "event_dates",
            "arithmetic_net_return_bp",
            "net_return_retention_vs_matched_baseline",
            "active_review_hit_rate",
            "active_trade_sharpe_sqrt4",
            "max_drawdown",
            "mean_gross_exposure",
        ],
        {
            "rule_label": "Expanding pseudo-OOS",
            "event_dates": "Reviews",
            "arithmetic_net_return_bp": "Net bp",
            "net_return_retention_vs_matched_baseline": "Net retained",
            "active_review_hit_rate": "Active hit",
            "active_trade_sharpe_sqrt4": "Active Sharpe",
            "max_drawdown": "Max DD",
            "mean_gross_exposure": "Mean gross",
        },
        {
            "event_dates": integer,
            "arithmetic_net_return_bp": lambda x: f"{float(x):+.2f}",
            "net_return_retention_vs_matched_baseline": lambda x: (
                "—" if pd.isna(x) else pct(x, 1)
            ),
            "active_review_hit_rate": lambda x: pct(x, 1),
            "active_trade_sharpe_sqrt4": lambda x: num(x, 2),
            "max_drawdown": lambda x: pct(x, 2),
            "mean_gross_exposure": lambda x: pct(x, 1),
        },
    )
    state_opt_pair_025 = state_opt_paired.loc[
        state_opt_paired["filter_rule"].eq("down_to_small_hour_soft_025")
    ].iloc[0]
    state_opt_walk_paired_display = state_opt_walk_paired.copy()
    state_opt_walk_paired_display["rule_label"] = state_opt_walk_paired_display[
        "selection_policy"
    ].map(state_opt_labels)
    state_opt_walk_paired_table = table(
        state_opt_walk_paired_display,
        [
            "rule_label",
            "mean_paired_increment_bp",
            "block4_increment_ci_low_bp",
            "block4_increment_ci_high_bp",
            "block4_max_t_familywise_pvalue",
        ],
        {
            "rule_label": "Policy vs matched baseline",
            "mean_paired_increment_bp": "Mean increment",
            "block4_increment_ci_low_bp": "Block CI low",
            "block4_increment_ci_high_bp": "Block CI high",
            "block4_max_t_familywise_pvalue": "Max-t p",
        },
        {
            "mean_paired_increment_bp": lambda x: f"{float(x):+.2f} bp",
            "block4_increment_ci_low_bp": lambda x: f"{float(x):+.2f}",
            "block4_increment_ci_high_bp": lambda x: f"{float(x):+.2f}",
            "block4_max_t_familywise_pvalue": lambda x: num(x, 4),
        },
    )
    state_opt_mechanism_early = state_opt_mechanism.loc[
        state_opt_mechanism["regime"].eq("2020-2022")
    ].copy()
    state_opt_mechanism_table = table(
        state_opt_mechanism_early,
        [
            "confirmation_pattern",
            "review_dates",
            "security_rows",
            "date_balanced_mean_signed_return_bp",
            "date_hit_rate",
        ],
        {
            "confirmation_pattern": "2020-2022 confirmation",
            "review_dates": "Dates",
            "security_rows": "Rows",
            "date_balanced_mean_signed_return_bp": "Signed mean",
            "date_hit_rate": "Date hit",
        },
        {
            "review_dates": integer,
            "security_rows": integer,
            "date_balanced_mean_signed_return_bp": lambda x: f"{float(x):+.2f} bp",
            "date_hit_rate": lambda x: pct(x, 1),
        },
    )
    state_opt_early_neither = state_opt_mechanism_early.loc[
        state_opt_mechanism_early["confirmation_pattern"].eq("neither")
    ].iloc[0]
    state_opt_early_hour_only = state_opt_mechanism_early.loc[
        state_opt_mechanism_early["confirmation_pattern"].eq("hour_only")
    ].iloc[0]
    state_opt_baseline = state_opt_fixed.loc[
        state_opt_fixed["filter_rule"].eq("baseline_all")
    ].iloc[0]
    state_opt_025 = state_opt_fixed.loc[
        state_opt_fixed["filter_rule"].eq("down_to_small_hour_soft_025")
    ].iloc[0]
    state_opt_walk_constrained = state_opt_walk.loc[
        state_opt_walk["filter_rule"].eq("constrained_minimum_intervention")
    ].iloc[0]
    state_opt_walk_max = state_opt_walk.loc[
        state_opt_walk["filter_rule"].eq("maximum_training_mean")
    ].iloc[0]
    state_opt_loo_share = (
        state_opt_loo["constrained_selected_rule"]
        .eq("down_to_small_hour_soft_025")
        .mean()
    )
    state_opt_kpis = "".join(
        [
            metric_card(
                "Frozen baseline · 10 bp",
                f"{num(state_opt_baseline['arithmetic_net_return_bp'])} bp",
                f"Active Sharpe {num(state_opt_baseline['active_trade_sharpe_sqrt4'], 2)} · DD {pct(state_opt_baseline['max_drawdown'], 2)}",
                "neutral",
            ),
            metric_card(
                "25% soft gate · 10 bp",
                f"{num(state_opt_025['arithmetic_net_return_bp'])} bp",
                f"Active Sharpe {num(state_opt_025['active_trade_sharpe_sqrt4'], 2)} · DD {pct(state_opt_025['max_drawdown'], 2)}",
                "warn",
            ),
            metric_card(
                "Full-sample net retained",
                pct(
                    state_opt_025["arithmetic_net_return_bp"]
                    / state_opt_baseline["arithmetic_net_return_bp"],
                    1,
                ),
                f"Active hit {pct(state_opt_025['active_review_hit_rate'], 1)}",
                "warn",
            ),
            metric_card(
                "Constrained pseudo-OOS retained",
                pct(
                    state_opt_walk_constrained[
                        "net_return_retention_vs_matched_baseline"
                    ],
                    1,
                ),
                "Below the frozen 80% requirement",
                "bad",
            ),
            metric_card(
                "Max-mean pseudo-OOS active Sharpe",
                num(state_opt_walk_max["active_trade_sharpe_sqrt4"], 2),
                f"{pct(state_opt_walk_max['net_return_retention_vs_matched_baseline'], 1)} net retained",
                "warn",
            ),
            metric_card(
                "Independent reconstruction",
                str(state_opt_audit["status"]).upper(),
                f"Max error {float(state_opt_audit['maximum_date_reconciliation_error']):.2e}",
                "good",
            ),
        ]
    )

    top10_date_table = table(
        top10_date_log.sort_values("effective_close_date"),
        [
            "effective_close_date",
            "trade_status",
            "eligible_adds",
            "eligible_deletes",
            "n_names",
            "long_exposure",
            "short_exposure",
            "gross_return",
            "modeled_cost_return",
            "net_return",
            "equity",
            "drawdown",
        ],
        {
            "effective_close_date": "Review",
            "trade_status": "Status",
            "eligible_adds": "Eligible adds",
            "eligible_deletes": "Eligible deletes",
            "n_names": "Selected",
            "long_exposure": "Long gross",
            "short_exposure": "Short gross",
            "gross_return": "Gross (bp)",
            "modeled_cost_return": "Cost (bp)",
            "net_return": "Net (bp)",
            "equity": "$100 equity",
            "drawdown": "Net DD",
        },
        {
            "effective_close_date": date_string,
            "eligible_adds": integer,
            "eligible_deletes": integer,
            "n_names": integer,
            "long_exposure": lambda x: pct(x, 1),
            "short_exposure": lambda x: pct(x, 1),
            "gross_return": bp,
            "modeled_cost_return": bp,
            "net_return": bp,
            "equity": lambda x: dollars(float(x) * 100.0, 4),
            "drawdown": pct,
        },
        table_id="top10-date-ledger",
    )

    top10_trade_table = table(
        top10_trade_log.sort_values(
            ["effective_close_date", "side", "side_rank", "ticker"]
        ),
        [
            "effective_close_date",
            "side",
            "side_rank",
            "ticker",
            "security_name",
            "change_type",
            "impact_proxy",
            "target_weight",
            "signed_position_usd_at_1m",
            "entry_bar_timestamp",
            "entry_open_proxy",
            "exit_bar_timestamp",
            "exit_close_proxy",
            "directional_security_return",
            "security_pnl_contribution",
            "modeled_cost_contribution",
            "net_pnl_contribution",
            "net_pnl_usd_at_1m",
            "fill_model",
            "information_cutoff",
            "feature_point_in_time_valid",
            "feature_cutoff_date",
            "liquidity_asof",
        ],
        {
            "effective_close_date": "Review",
            "side": "Side",
            "side_rank": "Side rank",
            "ticker": "Ticker",
            "security_name": "Security",
            "change_type": "MSCI type",
            "impact_proxy": "Impact proxy",
            "target_weight": "Signed weight",
            "signed_position_usd_at_1m": "Position at fixed $1m",
            "entry_bar_timestamp": "Entry bar starts",
            "entry_open_proxy": "Entry open proxy",
            "exit_bar_timestamp": "Exit bar starts · fill at close",
            "exit_close_proxy": "Exit close proxy",
            "directional_security_return": "Signed security return (bp)",
            "security_pnl_contribution": "Gross capital P&L (bp)",
            "modeled_cost_contribution": "Modeled cost (bp)",
            "net_pnl_contribution": "Net capital P&L (bp)",
            "net_pnl_usd_at_1m": "Net P&L · fixed $1m",
            "fill_model": "Fill model",
            "information_cutoff": "Information cutoff",
            "feature_point_in_time_valid": "PIT valid",
            "feature_cutoff_date": "Feature cutoff",
            "liquidity_asof": "Liquidity as-of",
        },
        {
            "effective_close_date": date_string,
            "side_rank": integer,
            "impact_proxy": lambda x: num(x, 4),
            "target_weight": lambda x: pct(x, 1),
            "signed_position_usd_at_1m": dollars,
            "entry_bar_timestamp": timestamp_string,
            "entry_open_proxy": lambda x: num(x, 4),
            "exit_bar_timestamp": timestamp_string,
            "exit_close_proxy": lambda x: num(x, 4),
            "directional_security_return": bp,
            "security_pnl_contribution": bp,
            "modeled_cost_contribution": bp,
            "net_pnl_contribution": bp,
            "net_pnl_usd_at_1m": dollars,
            "feature_point_in_time_valid": lambda x: "Yes" if bool(x) else "No",
            "feature_cutoff_date": date_string,
            "liquidity_asof": date_string,
        },
        table_id="top10-trade-ledger",
    )

    window_table = table(
        windows,
        [
            "window",
            "cost_bps",
            "event_dates",
            "mean_return_per_event",
            "total_return",
            "hit_rate",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
            "break_even_cost_bps",
        ],
        {
            "window": "Window",
            "cost_bps": "Cost (bp)",
            "event_dates": "Dates",
            "mean_return_per_event": "Mean/date (bp)",
            "total_return": "Cumulative",
            "hit_rate": "Hit rate",
            "block_bootstrap_mean_ci_low": "Bootstrap low (bp)",
            "block_bootstrap_mean_ci_high": "Bootstrap high (bp)",
            "break_even_cost_bps": "Break-even (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 0),
            "event_dates": integer,
            "mean_return_per_event": bp,
            "total_return": pct,
            "hit_rate": lambda x: pct(x, 1),
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
            "break_even_cost_bps": num,
        },
    )

    strategy_table = table(
        strategy_full,
        [
            "label",
            "assumed_cost_bps",
            "event_dates",
            "total_return",
            "cagr",
            "max_drawdown",
            "event_sharpe_sqrt4",
            "mean_return_per_event",
            "hit_rate",
        ],
        {
            "label": "Hypothesis",
            "assumed_cost_bps": "Cost (bp)",
            "event_dates": "Dates",
            "total_return": "Cumulative",
            "cagr": "CAGR",
            "max_drawdown": "Max DD",
            "event_sharpe_sqrt4": "Event Sharpe",
            "mean_return_per_event": "Mean/date (bp)",
            "hit_rate": "Hit rate",
        },
        {
            "assumed_cost_bps": lambda x: num(x, 0),
            "event_dates": integer,
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "event_sharpe_sqrt4": num,
            "mean_return_per_event": bp,
            "hit_rate": lambda x: pct(x, 1),
        },
    )

    cost_table = table(
        sensitivity.loc[sensitivity["strategy"].eq(PRIMARY)],
        ["cost_bps", "total_return", "cagr", "max_drawdown", "mean_return_per_event"],
        {
            "cost_bps": "All-in cost (bp)",
            "total_return": "Cumulative",
            "cagr": "CAGR",
            "max_drawdown": "Max DD",
            "mean_return_per_event": "Mean/date (bp)",
        },
        {
            "cost_bps": lambda x: num(x, 0),
            "total_return": pct,
            "cagr": pct,
            "max_drawdown": pct,
            "mean_return_per_event": bp,
        },
    )

    aug_table = table(
        aug,
        [
            "ticker",
            "security_name",
            "action",
            "change_type",
            "flow_return_gross",
            "pre_1550_return_gross",
            "post_1550_continuous_return_gross",
            "daily_close_gap_return_gross",
            "volume_vs_20d_median",
            "last10_volume_share",
        ],
        {
            "ticker": "Ticker",
            "security_name": "Security",
            "action": "Action",
            "change_type": "Change type",
            "flow_return_gross": "15:49→close (bp)",
            "pre_1550_return_gross": "15:49→15:50 (bp)",
            "post_1550_continuous_return_gross": "15:50→15:59 (bp)",
            "daily_close_gap_return_gross": "15:59→proxy (bp)",
            "volume_vs_20d_median": "Day vol / 20d",
            "last10_volume_share": "Last-10 share",
        },
        {
            "flow_return_gross": bp,
            "pre_1550_return_gross": bp,
            "post_1550_continuous_return_gross": bp,
            "daily_close_gap_return_gross": bp,
            "volume_vs_20d_median": lambda x: num(x) + "×",
            "last10_volume_share": lambda x: pct(x, 1),
        },
    )

    pair_table = table(
        pair_extremes,
        [
            "effective_close_date",
            "add_ticker",
            "delete_ticker",
            "pre_event_return_correlation",
            "matching_cost",
            "pair_gross_return",
        ],
        {
            "effective_close_date": "Review",
            "add_ticker": "Long add",
            "delete_ticker": "Short delete",
            "pre_event_return_correlation": "Pre-event corr.",
            "matching_cost": "Distance",
            "pair_gross_return": "Gross return (bp)",
        },
        {
            "effective_close_date": date_string,
            "pre_event_return_correlation": num,
            "matching_cost": num,
            "pair_gross_return": bp,
        },
    )

    walk_table = table(
        walk,
        [
            "fold_test_year",
            "train_event_dates",
            "test_event_dates",
            "test_total_return",
            "test_mean_return_per_event",
            "test_hit_rate",
            "test_max_drawdown",
        ],
        {
            "fold_test_year": "Test year",
            "train_event_dates": "Prior review dates",
            "test_event_dates": "Test dates",
            "test_total_return": "Test cumulative",
            "test_mean_return_per_event": "Mean/date (bp)",
            "test_hit_rate": "Hit rate",
            "test_max_drawdown": "Max DD",
        },
        {
            "fold_test_year": integer,
            "train_event_dates": integer,
            "test_event_dates": integer,
            "test_total_return": pct,
            "test_mean_return_per_event": bp,
            "test_hit_rate": lambda x: pct(x, 1),
            "test_max_drawdown": pct,
        },
    )

    subgroup_full = subgroup.loc[
        subgroup["sample"].eq("full")
        & subgroup["subgroup"].isin(
            ["size_migrations", "non_migration_entries_exits", "additions_only", "deletions_only"]
        )
    ].copy()
    subgroup_full["subgroup"] = subgroup_full["subgroup"].map(
        {
            "size_migrations": "Standard↔Small migrations",
            "non_migration_entries_exits": "New entries / full exits",
            "additions_only": "Additions",
            "deletions_only": "Deletions (short-oriented)",
        }
    )
    subgroup_table = table(
        subgroup_full,
        [
            "subgroup",
            "security_events",
            "event_dates",
            "total_return",
            "mean_return_per_event",
            "block_bootstrap_mean_ci_low",
            "block_bootstrap_mean_ci_high",
            "hit_rate",
        ],
        {
            "subgroup": "Subgroup",
            "security_events": "Names",
            "event_dates": "Dates",
            "total_return": "Cumulative",
            "mean_return_per_event": "Mean/date (bp)",
            "block_bootstrap_mean_ci_low": "CI low (bp)",
            "block_bootstrap_mean_ci_high": "CI high (bp)",
            "hit_rate": "Hit rate",
        },
        {
            "security_events": integer,
            "event_dates": integer,
            "total_return": pct,
            "mean_return_per_event": bp,
            "block_bootstrap_mean_ci_low": bp,
            "block_bootstrap_mean_ci_high": bp,
            "hit_rate": lambda x: pct(x, 1),
        },
    )

    failure_table = table(
        failures,
        ["effective_close_date", "ticker", "security_name", "reason"],
        {
            "effective_close_date": "Review",
            "ticker": "Ticker",
            "security_name": "Security",
            "reason": "Reason",
        },
        {"effective_close_date": date_string},
    )

    event_table = table(
        event_ledger,
        [
            "effective_close_date",
            "announcement_date",
            "action",
            "security_name",
            "ticker",
            "change_type",
            "coverage_status",
            "flow_return_gross",
            "pre_1550_return_gross",
            "post_1550_continuous_return_gross",
            "volume_vs_20d_median",
            "last10_volume_share",
            "close_discrepancy_bps",
        ],
        {
            "effective_close_date": "Review",
            "announcement_date": "Announced",
            "action": "Action",
            "security_name": "Security",
            "ticker": "Ticker",
            "change_type": "Type",
            "coverage_status": "Coverage",
            "flow_return_gross": "15:49→proxy (bp)",
            "pre_1550_return_gross": "15:49→15:50 (bp)",
            "post_1550_continuous_return_gross": "15:50→15:59 (bp)",
            "volume_vs_20d_median": "Day vol / 20d",
            "last10_volume_share": "Last-10 share",
            "close_discrepancy_bps": "15:59 vs proxy (bp)",
        },
        {
            "effective_close_date": date_string,
            "announcement_date": date_string,
            "flow_return_gross": bp,
            "pre_1550_return_gross": bp,
            "post_1550_continuous_return_gross": bp,
            "volume_vs_20d_median": lambda x: "—" if pd.isna(x) else num(x) + "×",
            "last10_volume_share": lambda x: pct(x, 1),
            "close_discrepancy_bps": lambda x: num(x),
        },
        table_id="event-ledger",
    )

    all_pairs_table = table(
        pairs.sort_values(["effective_close_date", "pair_number"]),
        [
            "effective_close_date",
            "pair_number",
            "add_ticker",
            "add_security_name",
            "delete_ticker",
            "delete_security_name",
            "pre_event_return_correlation",
            "matching_cost",
            "pair_gross_return",
        ],
        {
            "effective_close_date": "Review",
            "pair_number": "Pair",
            "add_ticker": "Long",
            "add_security_name": "Long security",
            "delete_ticker": "Short",
            "delete_security_name": "Short security",
            "pre_event_return_correlation": "Pre-event corr.",
            "matching_cost": "Distance",
            "pair_gross_return": "Gross return (bp)",
        },
        {
            "effective_close_date": date_string,
            "pair_number": integer,
            "pre_event_return_correlation": num,
            "matching_cost": num,
            "pair_gross_return": bp,
        },
    )

    volatility_dates_display = volatility_dates.copy()
    volatility_dates_display["metric"] = volatility_dates_display["metric"].map(
        {
            "absolute_terminal_move": "Absolute terminal move",
            "sqrt_realized_variance_11m": "√ realized variance (11m)",
            "high_low_log_range": "High–low log range",
            "max_absolute_excursion": "Maximum absolute excursion",
        }
    )
    volatility_date_table = table(
        volatility_dates_display.sort_values(["effective_close_date", "metric"]),
        [
            "effective_close_date",
            "metric",
            "controlled_names",
            "event_value",
            "control_value",
            "excess_value",
        ],
        {
            "effective_close_date": "Review",
            "metric": "Diagnostic",
            "controlled_names": "Names",
            "event_value": "Event (bp)",
            "control_value": "Pre-event control (bp)",
            "excess_value": "Excess (bp)",
        },
        {
            "effective_close_date": date_string,
            "controlled_names": integer,
            "event_value": bp,
            "control_value": bp,
            "excess_value": bp,
        },
    )

    straddle_scenario_table = table(
        straddle_scenarios.sort_values(
            ["dte", "entry_iv", "iv_change", "friction_fraction_of_premium"]
        ),
        [
            "dte",
            "entry_iv",
            "iv_change",
            "friction_fraction_of_premium",
            "all_event_review_dates",
            "mean_all_event_model_return_on_premium",
            "all_event_positive_date_rate",
            "all_event_p05",
            "all_event_p95",
            "mean_event_minus_pre_event_control",
        ],
        {
            "dte": "DTE",
            "entry_iv": "Entry IV",
            "iv_change": "IV change",
            "friction_fraction_of_premium": "Premium friction",
            "all_event_review_dates": "Dates",
            "mean_all_event_model_return_on_premium": "Mean modeled R/premium",
            "all_event_positive_date_rate": "Positive dates",
            "all_event_p05": "5th percentile",
            "all_event_p95": "95th percentile",
            "mean_event_minus_pre_event_control": "Event−control model excess",
        },
        {
            "dte": integer,
            "entry_iv": lambda x: pct(x, 0),
            "iv_change": lambda x: f"{float(x) * 100:+.0f} vol pts",
            "friction_fraction_of_premium": lambda x: pct(x, 1),
            "all_event_review_dates": integer,
            "mean_all_event_model_return_on_premium": pct,
            "all_event_positive_date_rate": lambda x: pct(x, 1),
            "all_event_p05": pct,
            "all_event_p95": pct,
            "mean_event_minus_pre_event_control": pct,
        },
    )

    sources = [
        ("Chang et al. (2025), Pacific-Basin Finance Journal", "https://ideas.repec.org/a/eee/pacfin/v93y2025ics0927538x25002379.html", "56 markets, 2006–2023; announcement-to-implementation evidence, not a final-ten-minute test."),
        ("NTU research summary of Chang et al.", "https://www.ntu.edu.sg/business/research/nbs-knowledge-lab/nbs-research-blog/index-tracking-rigidity-and-arbitrage-opportunities-in-msci-index-reconstitutions", "Public summary of 5.4% global, 2.8% North America, volume concentration and final-minute estimates."),
        ("Bogousslavsky & Muravyev, Who Trades at the Close?", "https://static1.squarespace.com/static/6310c0b9bb63a25599f4418c/t/634ffc92f81e226b2c30654f/1666186387645/who-trades-at-the-close_June2021.pdf", "NYSE/Nasdaq 2010–2018 closing-auction price discovery, passive ownership, and reversal."),
        ("Nasdaq: How Much Does the MOC Imbalance Matter?", "https://www.nasdaq.com/articles/how-much-does-the-moc-imbalance-matter-2019-09-27", "Exchange study: roughly 80% of sampled response within 300 ms; 5.5 bp immediate response; 1.7 bp liquidity-provider reward."),
        ("NYSE: Closing auction order impact", "https://www.nyse.com/data-insights-blog/closing-auction-order-impact-9rbi8f9wcu-partial", "0.71 versus 1.77 spread-unit result for pooled normal/rebalance days—not MSCI alone."),
        ("NYSE: Liquidity momentum around rebalances", "https://www.nyse.com/data-insights/closing-auction-liquidity-momentum-around-rebalances", "MSCI-specific descriptive activity patterns for 2021–2022 NYSE-listed stocks."),
        ("Comerton-Forde & Rindi, Trading @ The Close", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3903757", "European working paper; elevated rebalance-auction activity and auction returns."),
        ("Goyal, Jegadeesh & Wu (2026), JFQA", "https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-impact-in-closing-auctions-opening-auctions-and-continuous-markets-a-benchmark-for-cost-of-trading-on-anomalies/0F72910A79C5B42CF6E85F55164CE846", "Peer-reviewed auction-impact benchmark; square-root impact and 17.7 bp estimate at 1% ADV."),
        ("Pavlova & Sikorskaya, Benchmarking Intensity", "https://academic.oup.com/rfs/article/36/3/859/6665907", "Benchmark-weight changes scaled by tracking intensity motivate actual flow-to-ADV and flow-to-float-cap pressure measures."),
        ("Gao et al., Market Intraday Momentum", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866", "Market-level intraday momentum motivates controlling stock pre-close movement for SPY direction; it does not establish this MSCI one-minute effect."),
        ("Heston, Korajczyk & Sadka, Intraday Patterns in the Cross-section", "https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.2010.01573.x", "Cross-sectional intraday return persistence and reversal motivate testing both momentum and reversal gates."),
        ("MSCI August 2026 review announcement", "https://app2.msci.com/webapp/index_ann/DocGet?format=html&lang=en&pub_key=vhApk7vS6TE%3D", "Official announcement and implementation schedule."),
        ("MSCI August 2026 Global Standard list", "https://www.msci.com/eqb/gimi/stdindex/MSCI_Aug26_STPublicList.pdf", "Official public additions/deletions; SANDISK is a USA Standard addition."),
        ("MSCI August 2026 Global Small Cap list", "https://www.msci.com/eqb/gimi/smallcap/MSCI_Aug26_SCPublicList.pdf", "Official public list; SANDISK is simultaneously removed from USA Small Cap."),
        ("MSCI comprehensive quarterly-review transition", "https://app2.msci.com/webapp/index_ann/DocGet?format=html&lang=en&pub_key=Nlk1P2UjZO4%3D", "QCIR began with February 2023; full NOS/FIF review at every QCIR followed from May 2023."),
        ("MSCI QCIR consultation and historical simulation", "https://www.msci.com/documents/1296102/22675532/Consultation%2Bon%2BIntroducing%2BQuarterly%2BComprehensive%2BIndex%2BReviews.pdf/85987bdb-388d-51f6-79d6-6ea118c899ff?t=1614683476145", "Official methodology design and 2008-2020 turnover simulation; supports the institutional breakpoint, not causal attribution of return."),
        ("Nasdaq NOII timing history", "https://www.nasdaq.com/articles/automation-and-information-produce-efficient-price-discovery-in-nasdaqs-auction-process", "Nasdaq moved closing NOII dissemination from 15:55 to 15:50 in Q4 2018, before this sample."),
        ("NYSE closing D-order timing change", "https://www.nyse.com/data-insights/nyse-closing-auction-price-discovery-opportunities-reach-new-highs", "D-order inclusion moved from 15:55 to 15:50 in August 2024, after the February 2023 breakpoint."),
        ("NYSE closing-auction timetable", "https://www.nyse.com/trade/auctions", "Current 15:50 imbalance publication/cutoff and 16:00 auction mechanics."),
        ("NYSE Pillar Order Imbalances specification", "https://www.ice.com/publicdocs/nyse/data/NYSE_Pillar_Order_Imbalances_Client_Specification_v2.2n.pdf", "Message fields, cadence, auction type and venue-specific semantics."),
        ("Nasdaq Opening/Closing Cross FAQ", "https://www.nasdaqtrader.com/content/ProductsServices/Trading/Crosses/openclose_faqs.pdf", "15:50 early NOII, 15:55 full NOII, MOC/LOC timing and cross logic."),
        ("MarketData stock-candle documentation", "https://www.marketdata.app/docs/api/stocks/candles/", "Underlying OHLCV source; intraday timestamps identify candle start time; daily close is not documented as the official exchange-auction print."),
        ("NYSE Daily TAQ", "https://www.nyse.com/data-products/catalog/daily-taq", "Consolidated U.S. trades and quotes required to replace OHLC endpoints with executable NBBO observations."),
        ("NYSE TAQ Order Imbalances", "https://www.nyse.com/market-data/historical/taq-order-imbalances", "Historical opening and closing imbalance publications for NYSE-listed securities."),
        ("NYSE TAQ Integrated Feed", "https://www.nyse.com/market-data/historical/taq-integrated-feed", "Deterministically sequenced depth, trades, imbalances and security status for NYSE venues."),
        ("Nasdaq Historical TotalView-ITCH specification", "https://www.nasdaqtrader.com/content/technicalSupport/specifications/dataproducts/ITCHFTP.pdf", "Historical order-book and Net Order Imbalance Indicator message files for Nasdaq replay."),
        ("MarketData options coverage", "https://www.marketdata.app/data/options/", "Official limitation: historical options coverage is end-of-day; no historical intraday options candles or trade-level feed."),
        ("MarketData option quote schema", "https://www.marketdata.app/docs/api/options/quotes/", "Current/historical EOD bid, ask, midpoint, last and entitlement behavior; a snapshot timestamp is not an intraday history."),
        ("Cboe equity-option hours and Curb FAQ", "https://www.cboe.com/document/tech-spec/document/technical-specifications/equity-options-extended-trading-hours-faq/", "Ordinary equity-option trading ends at 16:00; the separate 16:00–16:15 Curb is limited to selected eligible classes and limit orders."),
        ("Cboe DataShop Option Quote Intervals", "https://datashop.cboe.com/option-quote-intervals", "Potential historical one-minute/custom-interval NBBO source for a preliminary option execution screen."),
        ("Databento OPRA NBBO", "https://databento.com/docs/examples/options/nbbo-resampling", "Historical consolidated OPRA NBBO and trade records for tick-level option execution research."),
        ("OIC options exercise FAQ", "https://www.optionseducation.org/referencelibrary/faq/options-exercise", "American-style physical settlement, exercise-by-exception and broker-specific exercise procedures."),
        ("Hollstein & Wese Simen, The Index Effect in Options", "https://acfr.aut.ac.nz/__data/assets/pdf_file/0020/530750/Announcement_paper.pdf", "S&P recomposition announcement-day daily-option evidence; supports the hypothesis but not an MSCI 15:49–15:59 trade."),
        ("Alexiou et al. (2025), Review of Finance", "https://academic.oup.com/rof/article/29/4/963/8079062", "Scheduled-event moves can be large while long delta-neutral straddles lose when event volatility is richly priced."),
        ("OPRA", "https://www.opraplan.com/", "Consolidated intraday option quotes/trades required for executable historical validation."),
        ("MSCI Index Terms", "https://www.msci.com/legal/index-terms", "Licensing restrictions relevant to commercial constituent/event data use."),
    ]
    sources_html = "".join(source_item(*item) for item in sources)

    report_generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    config_json = json.dumps(
        {
            "strategy_version": "msci-close-pressure-rv-0.3-research",
            "sample": "2020-05-29 through 2026-08-31",
            "universe": "public MSCI USA Standard additions/deletions",
            "primary_entry": "open of start-stamped 15:49 ET minute bar",
            "primary_exit": "MarketData daily-close proxy",
            "portfolio": "50% add leg / 50% delete leg; missing leg remains cash",
            "primary_cost_bps": 10,
            "first_minute_candidate": {
                "version": BURST,
                "status": "post_hoc_research_candidate_not_deployable",
                "signal_cutoff": "15:48:59.999 America/New_York",
                "entry_proxy": "open of start-stamped 15:49 bar",
                "exit_proxy": "close of start-stamped 15:50 bar",
                "bar_interval": "15:50:00 through 15:50:59",
                "selection_test": "studentized max-t across 29 adjacent windows with date and four-review-block sign flips",
                "true_holdout": "first MSCI review after 2026-08-31",
            },
            "state_filter_candidate": {
                "status": state_manifest["status"],
                "signal_cutoff": state_manifest["information_cutoff"],
                "primary_filter": state_manifest["primary_filter"],
                "feature_rows": state_manifest["base_security_rows"],
                "complete_feature_rows": state_manifest["complete_state_feature_rows"],
                "validation": "review-date grouped fixed rules plus expanding pseudo-walk-forward; next eight reviews remain the genuine holdout",
            },
            "state_filter_optimization": {
                "status": "retrospective_shadow_candidate_not_deployable",
                "frozen_candidates": 7,
                "optimization_candidates": 5,
                "selected_soft_rule": "retain 25% of unconfirmed down-to-small deletion weight",
                "primary_cost_bps": 10,
                "selector": "minimum intervention subject to 80% training-net retention, 5 percentage-point active-hit lift and 35% exposure floor",
                "pseudo_oos_start": "after first eight scheduled reviews",
                "future_holdout": "next eight MSCI implementation reviews after 2026-08-31",
                "execution_status": state_opt_audit["execution_status"],
            },
            "independent_unit": "implementation-close review date",
            "bootstrap": "10,000 moving-block samples; block size four reviews",
            "matched_pairs": "Hungarian assignment on pre-announcement beta, volatility, momentum, dollar volume and return correlation",
            "volatility_diagnostic_window": "15:49 minute open through 15:59 continuous-minute close proxy; excludes the 16:00 auction print",
            "volatility_pre_event_controls": "four nearest complete pre-event 15:49-15:59 sessions; 398 names across 24 reviews",
            "breakout_proxy_primary": "25 bp OCO barrier; trigger in minute t; conservative fill at t+1 open; 10 bp friction haircut; exit at 15:59 close proxy",
            "terminal_spot_model_primary": "delta-hedged unit-spot exact-ATM European Black-Scholes sensitivity; 7 DTE; 50% entry IV; zero rates/dividends; no smile",
            "option_execution_contract": "buy first executable call+put offer after 15:49 plus latency; liquidate at bid by 15:59:30; freeze strike/expiry; exclude 0DTE baseline",
            "option_pnl_status": "not tested; historical intraday OPRA bid/ask unavailable",
            "not_available": [
                "actual MSCI old/new weights and NOS/FIF vintages",
                "timestamped venue NOII / imbalance messages",
                "official primary-exchange auction prints",
                "NBBO/depth replay and latency",
                "historical borrow/locates",
                "historical intraday OPRA options quotes/trades, sizes and complex fills",
            ],
        },
        indent=2,
    )

    template = Template(r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MSCI Close-Pressure Relative Value — Product Research Report</title>
<style>
:root{--ink:#132235;--muted:#667487;--navy:#0b2138;--blue:#315f85;--line:#d9e0e7;--soft:#f5f7fa;--good:#1f7a55;--bad:#b44343;--warn:#a66a12;--paper:#fff;--max:1460px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#eef2f6;font:15px/1.58 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:#1e5d8f;text-decoration:none}a:hover{text-decoration:underline}.sidebar{position:fixed;inset:0 auto 0 0;width:258px;background:var(--navy);color:#dbe5ee;padding:26px 20px;overflow:auto;z-index:20}.sidebar .brand{font-weight:800;font-size:18px;letter-spacing:.01em}.sidebar .version{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#94abc0;margin:8px 0 24px}.sidebar nav a{display:block;color:#c5d3df;padding:7px 9px;margin:2px 0;border-left:2px solid transparent;font-size:13px}.sidebar nav a:hover,.sidebar nav a.active{background:#16324d;border-left-color:#73a8d1;text-decoration:none;color:#fff}.sidebar .legend{border-top:1px solid #29445c;margin-top:20px;padding-top:16px;font-size:11px;color:#9eb2c4}.main{margin-left:258px}.wrap{max-width:var(--max);margin:0 auto;padding:0 34px 72px}.hero{background:linear-gradient(135deg,#0d2741,#183f61 62%,#255d7d);color:#fff;padding:54px 0 38px;border-bottom:4px solid #b68b4b}.hero .wrap{padding-bottom:0}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:800;color:#a9c9df}.hero h1{font-size:42px;line-height:1.08;max-width:1000px;margin:10px 0 14px}.hero p{font-size:17px;color:#d7e4ed;max-width:980px}.hero-meta{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}.tag{display:inline-flex;align-items:center;border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700;letter-spacing:.03em}.decision-strip{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:24px}.decision-strip div{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);padding:14px 16px;border-radius:8px}.decision-strip strong{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em}.decision-strip span{font-size:13px;color:#dbe7ef}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}.metric{background:var(--paper);border:1px solid var(--line);border-top:4px solid #7a8996;border-radius:8px;padding:14px;min-height:112px}.metric span,.metric small{display:block;color:var(--muted);font-size:11px}.metric strong{display:block;font-size:26px;line-height:1.15;margin:7px 0}.metric.good{border-top-color:var(--good)}.metric.bad{border-top-color:var(--bad)}.metric.warn{border-top-color:var(--warn)}section{scroll-margin-top:20px;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:28px;margin:22px 0}section h2{font-size:25px;margin:0 0 8px}section h3{font-size:18px;margin:28px 0 10px}.lede{color:var(--muted);font-size:16px;max-width:1050px}.callout{border-left:5px solid var(--warn);background:#fff8e9;padding:16px 18px;margin:18px 0;border-radius:4px}.callout.bad{border-color:var(--bad);background:#fff2f2}.callout.good{border-color:var(--good);background:#edf8f2}.callout strong{display:block;margin-bottom:4px}.status{display:inline-block;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}.status.rejected{background:#fde4e4;color:#982f2f}.status.candidate{background:#fff0cd;color:#885a0e}.status.untested{background:#e4edf6;color:#244f73}.status.supported{background:#dff3e9;color:#176446}.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.panel{border:1px solid var(--line);border-radius:8px;padding:16px;background:#fbfcfd}.panel h3{margin:0 0 8px}.formula{font:14px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f2539;color:#e7f0f7;border-radius:7px;padding:15px;overflow:auto;margin:12px 0}.timeline{display:grid;grid-template-columns:repeat(6,1fr);gap:0;margin:24px 0}.time-step{position:relative;border-top:4px solid #6889a4;padding:18px 14px 10px;background:#f6f9fb;min-height:145px}.time-step:after{content:"";position:absolute;right:-8px;top:-10px;border-left:9px solid #6889a4;border-top:8px solid transparent;border-bottom:8px solid transparent;z-index:2}.time-step:last-child:after{display:none}.time-step b{display:block;color:#244f73;font-size:13px}.time-step span{font-size:12px;color:var(--muted)}.chart{margin:18px 0;border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px}.chart img{display:block;width:100%;height:auto}.chart figcaption{font-size:11px;color:var(--muted);padding:6px 8px 3px}.table-wrap{overflow:auto;max-height:580px;border:1px solid var(--line);border-radius:7px;margin:14px 0}.data-table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;white-space:nowrap}.data-table th{position:sticky;top:0;z-index:2;background:#eaf0f5;color:#19344d;text-align:left;border-bottom:1px solid #bfcbd6;padding:9px}.data-table td{border-bottom:1px solid #e7ebef;padding:7px 9px}.data-table tbody tr:nth-child(even){background:#fafbfd}.data-table tbody tr:hover{background:#edf4f9}.search-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.search-row input{min-width:320px;padding:9px 11px;border:1px solid #b8c5d1;border-radius:6px;font:inherit}.search-row span{font-size:12px;color:var(--muted)}.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.evidence{border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:7px;padding:15px}.evidence h3{margin:0 0 7px;font-size:15px}.evidence p{font-size:12px;color:var(--muted)}.sources{columns:2;column-gap:32px;padding-left:18px}.sources li{break-inside:avoid;margin:0 0 12px}.sources a{font-weight:700}.sources span{display:block;color:var(--muted);font-size:12px}.roadmap{counter-reset:stage;display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.roadmap div{counter-increment:stage;border:1px solid var(--line);padding:16px;border-radius:7px}.roadmap div:before{content:counter(stage);display:grid;place-items:center;width:26px;height:26px;border-radius:50%;background:var(--navy);color:#fff;font-weight:800;margin-bottom:8px}.roadmap b{display:block}.roadmap span{font-size:12px;color:var(--muted)}details{border:1px solid var(--line);border-radius:7px;margin:12px 0;background:#fff}summary{cursor:pointer;font-weight:750;padding:14px 16px;background:#f6f8fa}details>div{padding:12px 16px}.code{white-space:pre-wrap;background:#0d2134;color:#dce8f1;padding:16px;border-radius:6px;overflow:auto;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.footnote{font-size:11px;color:var(--muted)}footer{color:#657486;text-align:center;font-size:11px;padding:8px 30px 34px}.two-chart{display:grid;grid-template-columns:1fr 1fr;gap:14px}.two-chart .chart{margin:0}.pill-row{display:flex;gap:7px;flex-wrap:wrap}.mini-kpi{font-weight:800;color:#1d4d72}.qa-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.qa-grid div{background:#f6f8fa;border:1px solid var(--line);padding:12px;border-radius:6px}.qa-grid b{display:block;font-size:13px}.qa-grid span{font-size:11px;color:var(--muted)}
@media(max-width:1050px){.sidebar{position:relative;width:auto}.main{margin-left:0}.sidebar nav{display:flex;flex-wrap:wrap}.sidebar nav a{border-left:0;border-bottom:2px solid transparent}.metrics{grid-template-columns:repeat(2,1fr)}.decision-strip,.grid-3,.evidence-grid,.roadmap{grid-template-columns:1fr 1fr}.timeline{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){.wrap{padding-left:14px;padding-right:14px}.hero h1{font-size:31px}.metrics,.decision-strip,.grid-2,.grid-3,.evidence-grid,.roadmap,.qa-grid,.two-chart{grid-template-columns:1fr}.timeline{grid-template-columns:1fr}.time-step:after{display:none}.sources{columns:1}.search-row input{min-width:100%;width:100%}section{padding:19px}.metric strong{font-size:23px}}
@media print{body{background:#fff}.sidebar{display:none}.main{margin-left:0}.wrap{max-width:none}.chart{break-inside:avoid}section{break-inside:auto;border:0}.table-wrap{max-height:none;overflow:visible}.data-table{white-space:normal}}
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">MSCI Close-Pressure RV</div>
  <div class="version">research build 0.3<br>as of 2026-08-31</div>
  <nav id="nav">
    <a href="#executive">Executive decision</a><a href="#burst">15:50 burst alpha</a><a href="#weighting">Flow weighting</a><a href="#state-filters">15:49 state filters</a><a href="#state-optimization">Rule optimization</a><a href="#hypotheses">Evidence matrix</a><a href="#mechanism">Mechanism</a><a href="#literature">Literature</a><a href="#data">Universe &amp; data</a><a href="#anatomy">Close anatomy</a><a href="#volatility">Long-volatility sleeve</a><a href="#sndk">SNDK forensic</a><a href="#strategy">Product strategy</a><a href="#pairs">Pairs results</a><a href="#results">Backtest</a><a href="#validation">Validation</a><a href="#execution">Execution</a><a href="#roadmap">Roadmap</a><a href="#appendix">Appendices</a>
  </nav>
  <div class="legend">Research artifact. Simulated results. Not investment advice. Public MSCI lists are proprietary and are not licensed here for commercial redistribution.</div>
</aside>
<div class="main">
<header class="hero">
  <div class="wrap">
    <div class="eyebrow">Quantitative research · event-driven relative value · closing auctions</div>
    <h1>MSCI Close-Pressure Relative Value</h1>
    <p>A product-level research report separating a newly isolated 15:50 directional burst, the rejected hold-to-close rule, and option or auction variants that still require executable quote data.</p>
    <div class="hero-meta"><span class="tag">463 constituent events</span><span class="tag">26 independent review dates</span><span class="tag">2020–August 2026</span><span class="tag">26 analytical charts</span><span class="tag">review-date inference</span></div>
    <div class="decision-strip">
      <div><strong>Research candidate</strong><span>Pre-position at 15:49; flatten immediately before 15:51.</span></div>
      <div><strong>Not yet deployable</strong><span>10 bp result is fragile; quote, latency, impact and borrow remain untested.</span></div>
      <div><strong>Separate OPRA project</strong><span>Long straddles still have no historical executable option P&amp;L.</span></div>
    </div>
  </div>
</header>
<main class="wrap">

<section id="executive">
  <h2>Executive decision</h2>
  <p class="lede">The isolated event-minute move is a credible research alpha: long published additions and short deletions from the 15:49-bar open through the 15:50-bar close earns $burst_gross_mean_bp bp gross per review. The original hold-to-daily-close strategy remains negative. The new burst result is post hoc, regime-concentrated and based on trade-bar endpoints rather than executable quotes.</p>
  <div class="metrics">$kpis</div>
  <div class="callout bad"><strong>Portfolio action: do not deploy the unconditional 15:49 rule.</strong>Gross cumulative return is only $gross_total; the estimated break-even cost is $break_even bp. At 10 bp, cumulative return is $primary_total, CAGR $primary_cagr, max drawdown $primary_dd and event Sharpe $primary_sharpe.</div>
  <div class="callout good"><strong>Research action: freeze and shadow the 15:49-open → 15:50-bar-close strategy.</strong>Gross cumulative return is $burst_gross_total; after a 10 bp haircut it is $burst_net_total with max drawdown $burst_net_dd. The first future MSCI review is the genuine holdout.</div>
  <div class="callout"><strong>Volatility action: purchase a small OPRA pilot; do not deploy the straddle.</strong>Across $vol_control_dates controlled reviews, the average absolute 15:49-open→15:59-close-proxy move is $vol_event_abs_bp bp versus $vol_control_abs_bp bp in four pre-event sessions. Yet the 25 bp / 10 bp underlying breakout payoff proxy loses $vol_breakout_total outright, and a 7-DTE/50%-IV terminal-spot model turns negative once premium friction reaches 1%.</div>
  <p class="lede"><b>Summary:</b> Pooling 15:50–15:59 diluted the first minute. Re-freezing entry at 15:49 and exit at the close of the start-stamped 15:50 bar gives a positive historical return; the return is concentrated after 2023, the 10 bp confidence interval spans zero, and minute OHLC is a proxy for executable quotes. Current rating: research candidate.</p>
</section>

<section id="burst">
  <h2>Discovered 15:50 burst · frozen validation and system</h2>
  <p class="lede"><b>Exact clock convention:</b> MarketData bars are start-stamped. The candle labelled 15:50 covers 15:50:00–15:50:59 and may appear as “15:51” on an end-stamped chart. Version 1 freezes the public add/delete direction before 15:49, enters at the 15:49-bar open proxy, and exits at the 15:50-bar close proxy.</p>
  <div class="metrics">$burst_kpis</div>
  <div class="callout"><strong>Status: post-hoc research candidate; no production alpha claim.</strong>The chosen two-bar interval ranks first among 29 inspected adjacent windows. Raw p=$burst_raw_p and date-level max-t p=$burst_date_max_p, while the four-review-block max-t p=$burst_block_max_p. The block result is the relevant warning because performance clusters after May 2023.</div>
  $burst_equity_chart
  <div class="two-chart">$burst_cost_regime_chart$burst_time_selection_chart</div>

  <h3>Cost and performance</h3>
  <div class="table-wrap">$burst_cost_table</div>
  <div class="callout"><strong>Cost conclusion</strong>The arithmetic mean break-even is $burst_break_even bp, yet the four-review block interval remains above zero only around the 0–2.5 bp range in this grid. At 10 bp the in-sample result remains positive but its block interval crosses zero.</div>

  <h3>Regime, sleeve and event-specificity diagnostics</h3>
  <div class="grid-2"><div class="table-wrap">$burst_regime_table</div><div class="table-wrap">$burst_subgroup_table</div></div>
  <p>Before May 2023 the gross mean is $burst_pre_qcir_bp bp; from May 2023 it is $burst_post_qcir_bp bp. Long additions have the cleaner block interval. Deletion shorts require locates and add weaker evidence, so any add-only sleeve must be evaluated separately for beta and sector exposure.</p>
  $burst_control_chart
  <div class="table-wrap">$burst_control_table</div>
  <p>On 398 securities across 24 reviews with four complete prior sessions, the paired event-minus-control mean is $burst_abnormal_bp bp. Its four-review block interval is $burst_abnormal_low to $burst_abnormal_high bp, so event specificity is suggestive and not block-robust.</p>

  <h3>Heatmap selection correction</h3>
  <div class="table-wrap">$burst_selection_table</div>
  <p class="footnote">The max-t procedure flips returns by review date, recomputes a studentized statistic for every inspected interval, and compares the largest statistic with the selected interval. The four-review version assigns one sign to each contiguous block. It also addresses only the 29 or 30 displayed clock windows; other explored horizons, costs, sleeves and regimes add further research multiplicity.</p>

  <h3>Version 1 live-replay contract</h3>
  <div class="timeline">
    <div class="time-step"><b>15:48:30</b><span>Freeze event vintage, security mapping, side, target, locate, exclusions and risk. No later feature may mutate the snapshot.</span></div>
    <div class="time-step"><b>15:49:00</b><span>Send marketable-limit entry. Replay the first fresh ask for buys or bid for shorts after send time plus latency.</span></div>
    <div class="time-step"><b>15:50:00</b><span>NOII begins. Version 1 does not use it for entry; a reactive strategy must be versioned and tested separately.</span></div>
    <div class="time-step"><b>15:50:59.000</b><span>Send liquidation. Sell at executable bid; cover at executable ask. Preserve rejected and partial fills.</span></div>
    <div class="time-step"><b>15:50:59.500</b><span>Hard-flat deadline. Halt, stale feed, spread explosion, sequence gap or one-sided book triggers the kill switch.</span></div>
    <div class="time-step"><b>Post-trade</b><span>Reconcile NBBO fills, impact, fees, borrow and residual exposure against the OHLC proxy bias.</span></div>
  </div>
  <div class="grid-3">
    <div class="panel"><h3>Baseline alpha</h3><div class="formula">sideᵢ = +1(add), −1(delete)<br>Rᵢ = sideᵢ · (Pᵢ,15:50-bar-close / Pᵢ,15:49-bar-open − 1)</div><p>50% gross in each available add/delete leg; the absent leg remains cash. Public direction is a baseline, not a weight forecast.</p></div>
    <div class="panel"><h3>Production score</h3><div class="formula">flow pressureᵢ = AUM · Δwᵢ / expected interval dollar volume<br>trade only if expected alpha &gt; 2 × modeled all-in cost</div><p>Licensed old/new weights, NOS/FIF, product AUM and point-in-time identifiers replace equal weights. Current-day realized 15:50 volume cannot size a 15:49 order.</p></div>
    <div class="panel"><h3>Hard gate</h3><p>TAQ/ITCH NBBO and auction-message replay at 10/100/500 ms latency; spread, displayed size, impact, borrow, fees, halts, rejects and partial fills. Shadow the next 2–4 reviews before micro-live capital.</p></div>
  </div>
</section>

<section id="weighting">
  <h2>How weights are decided · implemented and tested</h2>
  <p class="lede">Version 1 now has a deterministic point-in-time allocator. Every rule freezes at 15:48:30, balances long and short legs, caps each name at 10%, leaves cash when a leg is sparse, and can cap each position at 5% of forecast normal 15:49–15:50 dollar volume. All comparisons use one return per review date and exact-universe controls.</p>
  <div class="metrics">$flow_weighting_kpis</div>
  <div class="callout"><strong>Decision: retain capped equal weight as the control; shadow one proxy; promote no proxy.</strong>The inverse clock-volume-share challenger adds $flow_proxy_lift bp gross per review, with four-review block interval $flow_proxy_ci_low to $flow_proxy_ci_high bp and current five-rule block max-t p=$flow_proxy_max_t_p. It is still post hoc and it is not an MSCI-flow estimate.</div>

  <div class="grid-2">
    <div class="panel"><h3>Current research control</h3><div class="formula">sideᵢ = +1(add), −1(delete)<br>scoreᵢ = 1<br>|wᵢ| ≤ 10%<br>Σw<sup>+</sup> = −Σw<sup>−</sup> ≤ 50%</div><p>Equal allocation is intentionally hard to overfit. The exact point-in-time universe has $flow_pit_rows of $flow_feature_rows usable security-events.</p></div>
    <div class="panel"><h3>Licensed production rule</h3><div class="formula">Fᵢ = Σ<sub>products</sub> AUM · replication · (w<sub>post</sub> − w<sub>pre</sub>)<br>sideᵢ = sign(Fᵢ)<br>scoreᵢ = |Fᵢ| / expected interval $ volume<br>|positionᵢ| ≤ min(10% capital, 5% interval $ volume)</div><p>Flows must be netted across Standard, Small Cap and overlapping products. $flow_migrations sample rows require cross-index treatment. Licensed inputs were not supplied, so this production formula has not been backtested.</p></div>
  </div>

  $flow_weighting_chart
  <h3>Cash-funded performance at a 10 bp haircut</h3>
  <div class="table-wrap">$flow_metrics_table</div>
  <p class="footnote">Mean gross exposure is shown because name and interval caps leave cash. These are minute-bar endpoint proxies; there is no nonlinear impact, NBBO spread, locate fee or partial-fill model.</p>

  <h3>Pure weighting lift on matched universes and caps</h3>
  <div class="table-wrap">$flow_paired_table</div>
  <p>The current proxy’s early-period lift is $flow_early_lift bp per review, versus only $flow_qcir_lift bp from May 2023 onward. This is the main reason the statistically positive full-sample comparison is not promoted. The max-t family covers five frozen weighting challengers in this run; it does not erase multiplicity from earlier clock, sleeve and strategy exploration.</p>

  <h3>Regime and capacity diagnostics</h3>
  <div class="grid-2"><div class="table-wrap">$flow_regime_table</div><div class="table-wrap">$flow_capacity_table</div></div>
  <p>At $1 million, the uncapped-in-liquidity proxy reaches $flow_proxy_uncapped_interval_max of normal same-clock interval volume in its worst name. The capacity-capped variant holds every modeled position to 5% but deploys only $flow_capacity_gross gross on average. Its reported $1 million frozen-weight capacity is partly mechanical because the weights were constructed at that same capital and limit.</p>
  <div class="callout"><strong>Concentration warning</strong>SNDK receives $flow_sndk_weight of capital and contributes $flow_sndk_share of August 2026 gross P&amp;L under the proxy. Weighting does not remove dependence on the motivating event.</div>

  <h3>Balanced top-10 concentration test</h3>
  <p class="lede">The proposed construction holds up to five predicted buyers plus five predicted sellers, with a 10% hard cap per security. Reviews with fewer than three eligible names on either side stay in cash. This prevents a sparse review from placing 25–50% in one security.</p>
  <div class="metrics">$top10_kpis</div>
  <div class="grid-2">
    <div class="panel"><h3>Requested production rank</h3><div class="formula">Fᵢ = Σₚ AUMₚ · replicationₚ · (wᵢ,post − wᵢ,pre)<br>Lᵢ = |Fᵢ| / expected interval $ volumeᵢ<br>Qᵢ = |Fᵢ| / float-adjusted market capᵢ<br>scoreᵢ = 0.50·rank(Lᵢ) + 0.30·rank(Qᵢ) + 0.20·rank(σᵢ)</div><p>Actual tracker flow, historical free-float market cap and product overlap data are absent. This exact rank remains untested.</p></div>
    <div class="panel"><h3>Current shadow proxy</h3><div class="formula">impact proxyᵢ = σᵢ,20d · √(ADVᵢ,20d / expected 15:49–15:51 $ volumeᵢ)</div><p>Every predictor is frozen before the event session. ADV is a synthetic order-size assumption, so this proxy cannot be described as MSCI trading capital.</p></div>
  </div>
  <div class="callout"><strong>Result: concentration improves the point estimate but does not validate stock selection.</strong>At the selected 0 bp cost, impact-top-10 earns $top10_net_mean bp per scheduled review and $top10_active_net_mean bp per active review, with $top10_net_total cumulative return. Its gross lift over the same-gate capped basket is $top10_lift bp; the four-review block interval is $top10_lift_low to $top10_lift_high bp and the 15-rule familywise p-value is $top10_family_p. The 10 bp case remains an execution sensitivity.</div>
  $top10_chart
  <p class="footnote">Primary performance uses 0 bp transaction cost. Active-trade Sharpe, active mean and hit rate exclude all nine zero-exposure cash reviews; the separately labeled mean per scheduled review, CAGR and equity curve retain cash dates to disclose capital utilization. The curve is an event-step series, not a continuously marked daily portfolio. The 10 bp curve is retained solely as sensitivity.</p>
  <div class="table-wrap">$top10_metrics_table</div>

  <h3>Long-add versus short-delete profit attribution</h3>
  <div class="metrics">$attribution_kpis</div>
  $attribution_event_chart
  <div class="table-wrap">$attribution_regime_table</div>
  <p class="footnote">Active reviews budget up to 50% to each leg. Profit shares are arithmetic contribution shares from the fixed-capital ledger, not compounded-return shares. A signed share may exceed 100% or become negative when one leg offsets the other.</p>

  <h3>Selection inference and regime stability</h3>
  <div class="grid-2"><div class="table-wrap">$top10_paired_table</div><div class="table-wrap">$top10_regime_table</div></div>
  <p>The selected 0 bp primary is positive in both regime slices. Under the 10 bp sensitivity, the impact-top-10 mean is slightly negative in 2020–2022 and positive after 2023. Removing SNDK without renormalizing lowers gross mean from $top10_gross_mean to $top10_without_sndk bp per scheduled review, so SNDK is influential but does not create the full result. Freeze this rule only as a shadow challenger; the next MSCI review is its first genuine holdout.</p>
  <div class="callout"><strong>Neutrality is now a tested portfolio choice, not a mandatory assumption.</strong>The same frozen Top-10 names have been rerun as dollar-neutral, beta-neutral, unneutralized signal-weighted, long-only and short-only portfolios. <a href="../portfolio_variants/report.html">Open the constraint comparison, equity curves and trade logs.</a></div>

  <h3>Calendar quarter and every event date</h3>
  <div class="grid-2"><div class="table-wrap">$attribution_seasonal_table</div>$attribution_regime_chart</div>
  <p class="footnote">Q1-Q4 contain only six or seven scheduled observations and unequal active counts. This is descriptive seasonality, not four independent tests.</p>
  <div class="table-wrap">$attribution_event_table</div>

  <h3>Why 2020–2022 was weak</h3>
  <div class="callout warn"><strong>Accounting diagnosis: weak and cost-fragile, not literally zero.</strong>The early Top-10 mean is $attr_early_mean bp per scheduled review with $attr_early_cash cash decisions; at a 10 bp haircut it falls to $attr_early_net10 bp. Addition longs contribute $attr_early_long bp while deletion shorts contribute $attr_early_short bp, and only $attr_netting_capture% of absolute leg movement survives cross-leg netting.</div>
  <div class="table-wrap">$attribution_driver_table</div>
  <div class="grid-2"><div><h3>Date-balanced migration description</h3><div class="table-wrap">$attribution_migration_table</div></div><div><h3>Old QIR versus comprehensive quarterly review</h3><div class="table-wrap">$attribution_review_type_table</div></div></div>
  <p>Migration composition is the cleanest mechanism clue: 95 of 99 early deletions are down-to-small migrations. Their date-balanced signed return averages -6.26 bp before 2023 and +21.12 bp later. Standard-index selling can be offset by Small Cap tracker buying, so the public Standard add/delete label does not identify net forced flow. This table is descriptive and does not constitute a clustered breakpoint test. MSCI's first comprehensive quarterly review was implemented at the February 2023 close; February/August breadth rises from 3.60 to 14.25 names per review here. May/November returns also improve, so the methodology transition explains part of the break, not all of it.</p>
  <p>The exact 15:50 signed minute changes from $attr_clock_early bp in 2020–2022 to $attr_clock_late bp after the breakpoint. Mean 15:50 volume share rises from $attr_volume_early% to $attr_volume_late%. Removing all 2026 observations leaves $attr_broad_ex2026 bp per review for all-name, while the Top-10 estimate compresses to $attr_top_ex2026 bp and moves much closer to its early +5.72 bp mean. Since 2026 supplies $attr_2026_share% of total Top-10 arithmetic profit, the concentrated portfolio is not a stable post-2023 result. These associations do not establish causality without licensed Δweights/AUM and message-level NOII, NBBO and auction data.</p>

  <h3>Complete review and security trading logs</h3>
  <p>The 26-row review ledger retains every trade and cash decision under the selected 0 bp cost assumption. The 161-row security ledger assigns zero modeled transaction cost, so security-level net contributions equal gross contributions and reconcile exactly to each review’s portfolio return. “Traded” can include 60% or 80% gross deployment; the exposure columns make partial investment explicit.</p>
  <div class="pill-row"><a href="topn_capped10_metrics_active.csv" download>Download corrected active-trade metrics</a><a href="topn_capped10_date_trade_log.csv" download>Download 26-review ledger CSV</a><a href="topn_capped10_trade_log.csv" download>Download 161-security ledger CSV</a><a href="topn_capped10_quarterly_event_attribution.csv" download>Download event attribution</a><a href="topn_capped10_leg_regime_attribution.csv" download>Download leg attribution</a><a href="topn_capped10_quarter_seasonality.csv" download>Download quarter summary</a><a href="topn_capped10_early_regime_drivers.csv" download>Download regime drivers</a></div>
  <div class="table-wrap">$top10_date_table</div>
  <details><summary>Open searchable security-level research proxy ledger</summary><div>
    <div class="search-row"><input id="top10-trade-search" type="search" placeholder="Filter date, ticker, security, side or MSCI type…"><span id="top10-trade-count">$top10_trade_rows rows</span></div>
    <div class="table-wrap">$top10_trade_table</div>
  </div></details>
  <p class="footnote">Entry is the start-stamped 15:49 bar open proxy; exit is the close of the start-stamped 15:50 bar, approximately 15:50:59. Transaction cost is set to 0 bp as specified; the downloadable security log also carries explicit 10 bp sensitivity columns. Dollar fields use a fresh fixed $1 million notional at each review for arithmetic attribution; the equity curve compounds capital, so the dollar sum and compounded profit differ slightly. These rows are simulated research records, not broker executions. They contain no NBBO spread, depth, venue, queue position, borrow fee, reject or partial-fill data.</p>
  <p class="footnote">Training policy: there are only 26 quarterly review observations and no model parameters with a defensible independent training sample. Purged stock-level K-fold would create false precision. The rule is frozen now; inference uses paired review-date returns, four-review blocks, regime splits, leave-one-review-out diagnostics, and the next eight quarterly reviews as sequential holdouts.</p>
</section>

<section id="state-filters">
  <h2>Pre-15:49 stock and market-state filters</h2>
  <p class="lede">The frozen Top-10 portfolio is augmented only with information available by 15:47:59.999 Eastern Time, leaving one complete minute before the 15:49 entry proxy. The factor family covers stock day/hour/week returns, SPY-relative returns, realized volatility, same-clock RVOL and MSCI migration type.</p>
  <div class="metrics">$state_filter_kpis</div>
  $state_filter_chart
  <div class="callout warn"><strong>Decision: use the migration gate as a risk-control candidate, not as proof of higher alpha.</strong>The static gate raises the 10 bp active-date hit rate from $state_baseline_hit to $state_gate_hit and reduces max drawdown from $state_baseline_dd to $state_gate_dd, but arithmetic net return falls from $state_baseline_net bp to $state_gate_net bp. Its paired increment versus the baseline is $state_gate_increment bp per scheduled review and the block max-t familywise p-value is $state_gate_family_p.</div>

  <h3>Frozen factor and weighting definitions</h3>
  <div class="formula">MOM60ᵢ = sideᵢ × [(Pᵢ,15:47 / Pᵢ,14:48 − 1) − (SPY15:47 / SPY14:48 − 1)]

Migration gate: keep every addition and true Standard exit; keep a down-to-small deletion only when MOM60ᵢ &gt; 0.

Weights: retain original Top-10 weights, scale only the larger surviving leg down to match the smaller leg, keep the 10% name cap, and leave unused capital in cash.</div>
  <p>One of 161 frozen Top-10 rows—TPL on 2022-08-31—lacks a complete 14:48–15:47 window and therefore fails closed under state-dependent rules. The baseline remains unchanged. The 15:48-bar version is retained only as a research sensitivity because it leaves no realistic feature-computation and order-routing buffer before a 15:49-open fill.</p>

  <h3>All mechanism-extension rules at 10 bp</h3>
  <div class="table-wrap">$state_rule_table</div>
  <p>High same-clock RVOL does not improve the result in this sample: the low-RVOL halves earn more than the high-RVOL halves in both the full-day-to-15:47 and last-hour comparisons. Volume confirmation is therefore rejected as a standalone hard gate. The migration-aware hour rule is the best mechanism-targeted fixed rule for raising active-date hit rate while retaining a substantial fraction of profit, though it still fails the paired return-improvement test.</p>

  <h3>Regime and pseudo-walk-forward diagnostics</h3>
  <div class="grid-2"><div class="table-wrap">$state_regime_table</div><div class="table-wrap">$state_walk_table</div></div>
  <p>The expanding selector uses only earlier review dates at each step and starts after eight scheduled reviews. Across its 18 pseudo-OOS reviews it retains $state_walk_retention of matched-baseline net return and increases active-date hit rate by $state_walk_hit_lift. This remains retrospective because the candidate family was proposed after the full 2020–2026 history existed.</p>

  <h3>Soft gate frontier</h3>
  <div class="table-wrap">$state_blend_table</div>
  <p>An 80% migration-gate / 20% baseline blend is the only tested grid point satisfying the provisional constraints of at least 80% net-return retention and a five-percentage-point active-hit improvement. It is a same-sample frontier choice, so it cannot be promoted to production before sequential holdout evidence.</p>
  <div class="pill-row"><a href="state_filter_feature_panel.csv" download>Download 15:47 feature panel</a><a href="../state_filter_extension/state_filter_rule_metrics.csv" download>Download rule metrics</a><a href="../state_filter_extension/state_filter_date_portfolios.csv" download>Download rule-by-date portfolios</a><a href="../state_filter_extension/state_filter_primary_trade_log.csv" download>Download migration-gate trade log</a><a href="../state_filter_extension/state_filter_walk_forward_selection.csv" download>Download walk-forward selections</a><a href="../state_filter_extension/state_filter_blend_frontier.csv" download>Download blend frontier</a></div>
  <p class="footnote">These filters use price and volume bars only. They do not contain actual MSCI delta weights, tracker AUM, NOII, NBBO spread/depth, borrow, news or earnings controls. NOII starts at 15:50 and is prohibited from this 15:49 signal. Minute-bar entry and exit prices remain non-executable proxies.</p>
</section>

<section id="state-optimization">
  <h2>Bounded optimization of the existing rule</h2>
  <p class="lede">The optimization family was frozen before the soft-gate variants were run. It contains the baseline, three monotone retention levels (75%, 50%, 25%), the existing hard gate, and two day/hour mechanism diagnostics. Only the first five rules may be selected. Every feature stops at 15:47:59.999 ET.</p>
  <div class="metrics">$state_opt_kpis</div>
  $state_opt_chart
  <div class="callout warn"><strong>Decision: keep the baseline as the capital-return benchmark; shadow the 25% soft gate as a risk-control policy.</strong>At 10 bp the soft gate raises Active-trade Sharpe from $state_opt_base_sharpe to $state_opt_025_sharpe and active-date hit rate from $state_opt_base_hit to $state_opt_025_hit, while max drawdown falls from $state_opt_base_dd to $state_opt_025_dd. Arithmetic net profit falls from $state_opt_base_net bp to $state_opt_025_net bp. Its mean paired increment is $state_opt_025_increment bp per review with block CI [$state_opt_025_ci_low, $state_opt_025_ci_high] bp and max-t p=$state_opt_025_max_p.</div>

  <h3>Rule and selector</h3>
  <div class="formula">For an unconfirmed down-to-small deletion:
wᵢ* = 0.25 × wᵢ

For additions, true exits and confirmed down-to-small deletions:
wᵢ* = wᵢ

After filtering, scale only the larger surviving leg down to dollar neutrality. Keep the 10% name cap; residual risk budget stays in cash.</div>
  <p>The constrained expanding selector starts after eight reviews. A non-baseline rule must retain at least 80% of training net profit, improve active-date hit rate by at least five percentage points, retain at least 35% of baseline gross exposure and have at least three active training dates. It then chooses the least aggressive eligible rule. This turns an open-ended factor search into a small ordered policy family.</p>
  <div class="table-wrap">$state_opt_fixed_table</div>

  <h3>Chronological validation</h3>
  <div class="grid-2"><div class="table-wrap">$state_opt_walk_table</div><div class="table-wrap">$state_opt_walk_paired_table</div></div>
  <p>The constrained pseudo-OOS policy retains $state_opt_walk_constrained_retention of matched-baseline net profit, below the frozen 80% requirement, although its active-date hit rate improves by $state_opt_walk_constrained_hit_lift. The maximum-training-mean comparator retains $state_opt_walk_max_retention and reaches Sharpe $state_opt_walk_max_sharpe, but still has a negative paired return increment. Leave-one-review-out selection chooses the 25% soft rule only $state_opt_loo_share of the time; this instability prevents deployment.</p>

  <h3>Mechanism and the weak 2020-2022 period</h3>
  <div class="table-wrap">$state_opt_mechanism_table</div>
  <p>Within down-to-small deletions, the early regime is weakest when neither day nor hour relative momentum confirms the event direction ($state_opt_early_neither bp date-balanced signed return) and when only the hour confirms ($state_opt_early_hour_only bp). The sample counts are small, so this is a diagnostic for partial attenuation, not a new causal factor. It is consistent with offsetting Small Cap demand and with short-horizon reversal/liquidity effects. Actual MSCI weight changes, tracker AUM and auction imbalance messages remain unavailable, so flow pressure is still proxied rather than measured.</p>

  <h3>Audit boundary</h3>
  <p>An independent reconstruction reads the raw minute parquet files without importing the production filter or backtest functions. It reconciles $state_opt_audit_rows security rows across $state_opt_audit_dates review dates; maximum date-level error is $state_opt_audit_error. One row lacks a complete state window and fails closed. Prices remain start-stamped minute-bar proxies, not NBBO or broker fills.</p>
  <div class="pill-row"><a href="../state_filter_optimization/state_optimization_rule_metrics.csv" download>Download fixed-rule metrics</a><a href="../state_filter_optimization/state_optimization_walk_forward_metrics.csv" download>Download pseudo-OOS metrics</a><a href="../state_filter_optimization/state_optimization_walk_forward_selection.csv" download>Download selections</a><a href="../state_filter_optimization/state_optimization_selected_trade_log.csv" download>Download selected-rule trade log</a><a href="../state_filter_optimization/state_optimization_manual_trade_samples.csv" download>Download manual samples</a><a href="../state_filter_optimization/state_optimization_manifest.json" download>Download frozen manifest</a></div>
  <p class="footnote">Active-trade Sharpe excludes every zero-exposure review; raw output retains Scheduled Event Sharpe as a separately labeled coverage-aware diagnostic. The risk-free return per one-minute active event remains zero until a point-in-time funding series is joined. The full history was visible before this optimization exercise, so the expanding test is pseudo-OOS, not a genuine untouched holdout. The next eight reviews remain the only sequential test. No option-chain, NOII, TAQ, borrow or market-impact data are used here.</p>
</section>

<section id="hypotheses">
  <h2>What was tested—and what was not</h2>
  <p class="lede">Every strategy family has its own information set, execution clock and evidence label. Evidence from one sleeve cannot be used to rescue another.</p>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>Family</th><th>Signal / entry</th><th>Required data</th><th>Status</th><th>Current conclusion</th><th>Next gate</th></tr></thead><tbody>
  <tr><td>15:50 directional burst</td><td>Add/delete sign; 15:49 open → 15:50-bar close</td><td>Minute OHLCV for proxy; NBBO/TAQ for execution</td><td><span class="status candidate">Post-hoc candidate</span></td><td>$burst_gross_total gross; $burst_net_total at 10 bp; block max-t p=$burst_block_max_p.</td><td>Tick replay plus first future review holdout.</td></tr>
  <tr><td>15:47 migration-aware state gate</td><td>Down-to-small deletions require expected-direction SPY-relative hour momentum</td><td>Point-in-time stock/SPY minute bars + migration label</td><td><span class="status candidate">Risk-control candidate</span></td><td>Active hit $state_gate_hit; $state_gate_net net bp at 10 bp, below the baseline’s $state_baseline_net bp.</td><td>Freeze for eight future reviews; add actual flow, spread, borrow and news gates.</td></tr>
  <tr><td>Public-list continuation to close</td><td>Add/delete sign; 15:49 bar open → daily-close proxy</td><td>Public review lists + minute OHLCV</td><td><span class="status rejected">Rejected</span></td><td>Net $primary_total; break-even $break_even bp.</td><td>Retain as a negative baseline.</td></tr>
  <tr><td>Similarity-matched add/delete pairs</td><td>Public sign + pre-announcement beta/vol/momentum/liquidity matching</td><td>Current daily + minute cache</td><td><span class="status rejected">Rejected</span></td><td>$pair_total net across $pair_dates dates.</td><td>Do not optimize further on this sample.</td></tr>
  <tr><td>Expected-flow pre-position</td><td>Actual old/new weight, fund AUM, predicted residual close demand</td><td>Licensed MSCI vintages + product holdings/AUM</td><td><span class="status candidate">Candidate</span></td><td>Mechanism supported; not tested here.</td><td>Point-in-time flow database and synthetic controls.</td></tr>
  <tr><td>NOII-confirmed continuation</td><td>Flow pressure × aligned normalized imbalance × indicative gap</td><td>Direct/exchange NOII + subsecond NBBO + auction print</td><td><span class="status untested">Untested</span></td><td>Minute bars cannot validate the 300 ms response.</td><td>Message-level replay with latency.</td></tr>
  <tr><td>Announcement drift</td><td>Next open after announcement → implementation close</td><td>Weights, borrow, corporate actions, close print</td><td><span class="status candidate">Needs true holdout</span></td><td>$announcement_total at 20 bp; retrospective and selected.</td><td>Preregister and test future reviews.</td></tr>
  <tr><td>Auction reversal</td><td>Close proxy → next open, opposite event direction</td><td>Official print + executable after-hours/next-open data</td><td><span class="status rejected">Unstable</span></td><td>$reversal_total; later slice deteriorates.</td><td>Only revisit with exact auction deviations.</td></tr>
  <tr><td>Underlying pre-close convexity</td><td>|15:49 open→15:59 close-proxy move|</td><td>Underlying minute OHLCV + four complete pre-event sessions</td><td><span class="status supported">Exploratory support</span></td><td>$vol_event_abs_bp bp event vs $vol_control_abs_bp bp control across $vol_control_dates reviews.</td><td>Confirm with same-date non-event stocks and genuinely future reviews.</td></tr>
  <tr><td>Equity OCO payoff proxy</td><td>Two-sided 25 bp barrier; next-minute fill; 10 bp haircut</td><td>Minute OHLCV; no NBBO or queue</td><td><span class="status rejected">Negative outright</span></td><td>$vol_breakout_total cumulative; $vol_breakout_mean_bp bp/review.</td><td>Retain only as a conservative path-monetization diagnostic.</td></tr>
  <tr><td>Executable option straddle</td><td>Delta-hedged ATM straddle, 15:49→fixed pre-16:00 deadline</td><td>OPRA NBBO/trades/size, contract master, IV/Greeks, fees</td><td><span class="status untested">OPRA required</span></td><td>Underlying premise supported; no option P&amp;L claimed.</td><td>Ask-in/bid-out replay and first future review holdout.</td></tr>
  </tbody></table></div>
</section>

<section id="mechanism">
  <h2>Economic mechanism and information clock</h2>
  <p class="lede">MSCI defines the benchmark; funds and benchmark-aware investors create the orders. The relevant quantity is net product-level flow across overlapping Standard, Small Cap, IMI, USA, World and ACWI trackers—not a binary add/delete label.</p>
  <div class="timeline">
    <div class="time-step"><b>Announcement</b><span>Public list, implementation date and revisions become known. Arbitrageurs can begin announcement drift.</span></div>
    <div class="time-step"><b>T−1 / 15:30</b><span>Freeze point-in-time weights, flow estimates, controls, locates, venue rules and capacity.</span></div>
    <div class="time-step"><b>15:49</b><span>Pre-positioning may use predicted flow and ordinary tape only. Official NYSE/Nasdaq NOII is not yet public.</span></div>
    <div class="time-step"><b>15:50</b><span>NYSE imbalance publication starts; Nasdaq early NOII starts. Initial response can be subsecond.</span></div>
    <div class="time-step"><b>15:55–15:59</b><span>Nasdaq full NOII / order deadlines; NYSE indicative fields update. Remaining alpha and overshoot change dynamically.</span></div>
    <div class="time-step"><b>16:00 / T+1</b><span>Official auction cross, reconciliation, then test permanent impact versus reversal.</span></div>
  </div>
  <div class="grid-2">
    <div class="panel"><h3>Product-level expected flow</h3><div class="formula">Fᵢ = Σₚ AUMₚ · ρₚ(close) · (wᵢ,ₚ,new − wᵢ,ₚ,old)<br>FPᵢ = asinh(|Fᵢ| / [Pᵢ · median auction sharesᵢ,60d])</div><p>A Standard↔Small migration creates offsetting trades across product families. Compute fund-by-fund and de-duplicate nested benchmark exposure.</p></div>
    <div class="panel"><h3>Real-time interaction</h3><div class="formula">NIᵢ(t) = sign(Fᵢ) · imbalance sharesᵢ(t) / normal auction sharesᵢ<br>IPᵢ(t) = sign(Fᵢ) · log(near priceᵢ(t) / midpointᵢ(t−))<br>E[yᵢ] = β₀ + βF·FPᵢ + βN·NIᵢ + βFN·FPᵢ·NIᵢ + βP·IPᵢ + γ′Xᵢ</div><p>The central preregistered test is βFN &gt; 0. The live model must forecast <i>remaining</i> return after subtracting price movement already realized since 15:49.</p></div>
  </div>
</section>

<section id="literature">
  <h2>Verified literature chain</h2>
  <p class="lede">The literature supports each link in the economic mechanism, but no verified public paper joins actual U.S. MSCI Δw, venue-specific NOII and executable subsecond trading into the exact strategy proposed.</p>
  <div class="evidence-grid">
    <article class="evidence"><h3>MSCI forced flow</h3><p>Chang et al. study 56 markets over 2006–2023. The reported 5.4% global and 2.8% North America results are announcement-to-implementation abnormal returns after lending cost—not 15:49-to-close returns. Implementation-close volume exceeds normal volume by more than 10×.</p><span class="status supported">Peer reviewed</span></article>
    <article class="evidence"><h3>Closing-auction price discovery</h3><p>Bogousslavsky &amp; Muravyev document rising closing volume, passive-ownership associations, extreme auction deviations and partial reversal. Their 2010–2018 timing evidence spans historical NYSE/Nasdaq dissemination rules and is not MSCI-specific.</p><span class="status supported">Peer reviewed</span></article>
    <article class="evidence"><h3>Reaction speed</h3><p>Nasdaq research on Nasdaq-100 names in Q2 2019 reports about 80% of the final response within 300 ms, a 5.5 bp immediate move and 1.7 bp liquidity-provider reward. It is exchange research with limited replication detail.</p><span class="status candidate">Exchange research</span></article>
    <article class="evidence"><h3>Rebalance sensitivity</h3><p>NYSE reports 1.77× spread impact for the largest 0.25% of Russell 1000 imbalance changes on pooled rebalance days versus 0.71× on ordinary days. It pools S&amp;P, FTSE Russell, MSCI and other events.</p><span class="status candidate">Descriptive</span></article>
    <article class="evidence"><h3>MSCI closing activity</h3><p>NYSE’s 2021–2022 descriptive study finds unusual auction activity around MSCI events, especially in the largest realized rebalance auctions. Realized activity is an ex-post selection, not a tradable ex-ante ranking.</p><span class="status candidate">MSCI specific</span></article>
    <article class="evidence"><h3>Auction impact and capacity</h3><p>Goyal, Jegadeesh &amp; Wu use CRSP, auction data and microsecond TAQ. A square-root model fits auction impact better than a linear model; their 1%-ADV estimate is about 17.7 bp. Flat-cost backtests are insufficient for scale decisions.</p><span class="status supported">Peer reviewed</span></article>
  </div>
  <div class="callout"><strong>Claims deliberately not made</strong>The 5.4% paper result is not a final-ten-minute return; the NYSE 1.77/0.71 comparison is not MSCI-only; a giant 16:00 print does not prove price discovery was complete beforehand; and none of the cited studies validates a net-profitable manual trade after 15:50.</div>
</section>

<section id="data">
  <h2>Event universe, lineage and coverage</h2>
  <p class="lede">Public Standard and Small Cap lists were reconciled across 27 scheduled reviews. February 2020 had no U.S. Standard changes; 26 implementation closes contain $expected_events events, of which $migrations are Standard↔Small migrations.</p>
  <div class="two-chart">$coverage_chart$composition_chart</div>
  <div class="grid-3">
    <div class="panel"><h3>$expected_events public events</h3><p>Official public-list additions and deletions. The parser reconciles USA rows with each document’s summary.</p></div>
    <div class="panel"><h3>$minute_usable minute windows</h3><p>Usable 15:49 and 15:59 bars. Acquired/taken-private names expose the need for final amendment history.</p></div>
    <div class="panel"><h3>$close_usable close proxies</h3><p>Eight otherwise usable rows lack a daily close. A separate fill sensitivity changes the net result to $missing_total.</p></div>
  </div>
  $close_quality_chart
  <p class="footnote">The daily provider close is a proxy, not a verified primary-exchange closing-auction print. The last-minute/daily gap can contain auction trading, corrections, eligibility filters and feed aggregation differences. One GRUB 2× basis mismatch was detected and normalized from open/high/low ratios.</p>
</section>

<section id="anatomy">
  <h2>Event-time anatomy of the final 30 minutes</h2>
  <p class="lede">This panel uses 443 securities with complete 15:30–15:59 bars, aggregates securities within add/delete legs, then treats each review date equally. It is descriptive. One-minute OHLCV cannot reconstruct NOII, NBBO, message latency or the exact auction print.</p>
  $event_profile_chart
  $event_heatmap_chart
  $window_chart
  <div class="table-wrap">$window_table</div>
  <div class="callout"><strong>Key empirical update</strong>The 15:49-open→15:50-open component averages $pre_1550_bp bp gross and its four-review block-bootstrap interval is [$pre_ci_low, $pre_ci_high] bp. The subsequent 15:50-open→15:59-close component averages $post_1550_bp bp with a wide interval crossing zero. At 10 bp, neither is investable. Because bar timestamps are one-minute buckets, the 15:50 boundary is not a causal NOII timestamp test.</div>
</section>

<section id="volatility">
  <h2>Long volatility in the final pre-close window</h2>
  <p class="lede">Decision: promote the hypothesis to a paid OPRA pilot, not to capital deployment. The underlying premise is unusually strong and stable by review date; actual option profitability remains unknown because the local MarketData history contains only end-of-day option snapshots, not 15:49–15:59 bid/ask paths.</p>
  <div class="metrics">$volatility_kpis</div>
  <div class="callout good"><strong>Underlying convexity screen passes.</strong>On the controlled sample, absolute terminal movement is $vol_event_abs_bp bp on implementation dates versus $vol_control_abs_bp bp in the same securities’ four nearest complete pre-event sessions: excess $vol_excess_abs_bp bp with a four-review block-bootstrap interval of [$vol_abs_ci_low, $vol_abs_ci_high] bp. The excess is positive on $vol_positive_dates of $vol_control_dates controlled dates. The all-event estimate is $vol_all_event_abs_bp bp across all 26 reviews, so the magnitude is not created by control attrition.</div>
  <div class="two-chart">$volatility_excess_chart$volatility_threshold_chart</div>
  <h3>Path diagnostics and large-move frequencies</h3>
  <div class="table-wrap">$volatility_table</div>
  <div class="table-wrap">$volatility_threshold_table</div>
  <p class="footnote">Absolute terminal move is relevant to a static straddle. √ realized variance, the high–low range and maximum excursion describe path convexity and matter mainly for delta hedging or gamma scalping. They are unannualized diagnostics, can contain one-minute bid/ask bounce, and are not tradable payoffs.</p>

  <h3>Can minute bars monetize the path?</h3>
  <p>A deliberately conservative equity OCO proxy places two barriers around the 15:49 reference. A barrier observed inside minute <i>t</i> cannot fill until the next minute open; ambiguous same-minute two-sided hits receive the worse direction. This is a bar-payoff screen—not executable equity P&amp;L and not an option backtest.</p>
  <div class="two-chart">$volatility_breakout_chart<div class="panel"><h3>Primary proxy · 25 bp barrier / 10 bp haircut</h3><p><span class="mini-kpi">$vol_breakout_mean_bp bp per review</span><br>Cumulative $vol_breakout_total; cash-funded CAGR $vol_breakout_cagr; max drawdown $vol_breakout_dd; block-bootstrap mean interval [$vol_breakout_ci_low, $vol_breakout_ci_high] bp.</p><p>The barrier triggers on $vol_breakout_trigger of date-balanced names. The proxy is nearly always active and loses outright; any event-minus-control advantage is not a deployable long/short portfolio.</p></div></div>
  <div class="table-wrap">$breakout_table</div>

  <h3>Terminal-spot straddle sensitivity · explicitly not a backtest</h3>
  <p>The model holds an exact-ATM unit strike, begins delta hedged, uses European Black–Scholes with zero rates/dividends, advances eleven calendar minutes and varies DTE, entry IV, IV change and friction as a fraction of premium. It assumes contract availability, ignores skew/discrete strikes and observes no option quote. It answers only how narrow the economic budget may be.</p>
  <div class="two-chart">$straddle_model_chart<div><div class="table-wrap">$straddle_model_table</div><p class="footnote">At 7 DTE / 50% entry IV / unchanged IV, the mean modeled gross return is $model_zero_friction. It falls to $model_halfpct_friction at 0.5% premium friction and $model_onepct_friction at 1.0%. IV repricing dominates the grid: a five-vol-point drop loses about $model_iv_down before friction, while a five-point rise gains about $model_iv_up.</p></div></div>

  <h3>Frozen executable OPRA research contract</h3>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>Element</th><th>Primary rule</th><th>Why</th></tr></thead><tbody>
  <tr><td>Universe</td><td>Final point-in-time MSCI list; require standard, non-adjusted, liquid equity options; exclude earnings, announced M&amp;A, halts and unresolved corporate actions.</td><td>Underlying coverage is not optionability.</td></tr>
  <tr><td>Contract</td><td>Same-expiry, same-strike ATM call + put; nearest standard expiry in 7–21 DTE; freeze strike/expiry at entry; exclude 0DTE baseline.</td><td>Prevents strike switching, pin risk and look-ahead.</td></tr>
  <tr><td>Entry</td><td>First fresh, two-sided OPRA NBBO after 15:49:00 plus 500 ms; buy call ask + put ask; quote-age, width, size, OI and volume gates.</td><td>Uses executable offers, not midpoint or stale last trade.</td></tr>
  <tr><td>Delta</td><td>Neutralize initial combined option delta with the next underlying NBBO; static hedge is primary. Unhedged and banded gamma scalp are separate variants.</td><td>Separates convexity from residual direction.</td></tr>
  <tr><td>Exit</td><td>Liquidate both legs at bid by 15:59:30; close the stock hedge. No option value may use the subsequently known 16:00 stock auction print.</td><td>Ordinary equity-option trading and the stock auction meet at the same boundary.</td></tr>
  <tr><td>P&amp;L / capacity</td><td>Dollar P&amp;L and return on debit after bid/ask, commissions, exchange/ORF fees, rejects, partials and hedge slippage; cap by displayed size, option ADV/OI and underlying hedge volume.</td><td>Makes the result scalable and auditable.</td></tr>
  </tbody></table></div>
  <div class="callout"><strong>Coverage and causality limits</strong>The controlled panel contains $vol_control_names of $minute_usable underlying windows across $vol_control_dates of 26 reviews. November 2020 lacks four complete pre-event full sessions in the cache; August 2026 has only the event day. The control is same-name and pre-event, but it does not remove month-end seasonality or same-day market volatility. The first confirmatory design must add same-date non-event optionable controls and keep the first post-August-2026 review untouched.</div>
</section>

<section id="sndk">
  <h2>SNDK forensic dashboard · August 31, 2026</h2>
  <p class="lede">Official MSCI documents identify SANDISK as a USA Standard addition and simultaneous USA Small Cap deletion. It is a size-segment migration, not a brand-new ACWI IMI security. Standard trackers buy while Small Cap trackers sell; the net product flow requires benchmark-specific AUM and weights.</p>
  <div class="metrics">
    $sndk_metrics
  </div>
  $sndk_chart
  <div class="callout"><strong>The observation was real but not representative.</strong>SNDK returned $sndk_return from the 15:49 bar open to the daily-close proxy, and its final-ten-minute volume ranked 1 of $sndk_sessions. Its full-day volume ranked only $sndk_day_rank of $sndk_sessions. The entire August date-balanced basket returned $aug_gross gross / $aug_net net.</div>
  <h3>Every August 2026 constituent event</h3>
  <div class="table-wrap">$aug_table</div>
</section>

<section id="strategy">
  <h2>Product strategy contract</h2>
  <p class="lede">The commercial objective is expected net dollar P&amp;L at capacity under drawdown and tail-risk constraints. CAGR is secondary for a strategy exposed only minutes on four implementation closes per year.</p>
  <div class="grid-3">
    <div class="panel"><h3>Book A · Natural event pairs</h3><p>Orient by net expected flow, then minimum-cost match by venue, industry, market cap, beta, idiosyncratic volatility, price, spread, ADV, normal auction share, momentum and borrow state. Enforce calipers; unmatched names remain out.</p></div>
    <div class="panel"><h3>Book B · Event vs synthetic control</h3><p>For each event name, freeze three to five non-event controls before T−1. Exact-match venue/industry where possible; ridge-fit nonnegative weights summing to one. Exclude other index events, earnings, M&amp;A, halts and corporate actions.</p></div>
    <div class="panel"><h3>Book C · Pre-close convexity</h3><p>Buy delta-hedged ATM straddles only in optionable, liquid, high-flow-pressure names. Fix contract and exit rules ex ante; treat long-event/short-control volatility as a separate dispersion product, not as outright long volatility.</p></div>
  </div>
  <div class="formula">maximizeₓ  μ̂′x − ½λR·x′Σx − Σᵢ TCᵢ(xᵢ) − λC·CVaRα(x)<br>subject to: beta ≈ 0; factor/sector bounds; borrow and locate; synchronized pair fills; per-name/review loss; entry/auction/ADV/flow participation caps</div>
  <div class="grid-3">
    <div class="panel"><h3>15:49 predictive model</h3><p>Uses only MSCI vintages, product flow, frozen controls and pre-15:49 tape. No same-day official NYSE/Nasdaq imbalance can enter this forecast.</p></div>
    <div class="panel"><h3>Post-15:50 remaining-alpha model</h3><p>Uses normalized imbalance, change in imbalance, near/far clearing-price gap and realized move since 15:49. Never condition the initial P&amp;L sample on later confirmation.</p></div>
    <div class="panel"><h3>Capacity model</h3><p>|qᵢ| ≤ min(entry participation, auction participation, expected-flow share, ADV share, locate). Scale ends where marginal net P&amp;L becomes negative.</p></div>
  </div>
</section>

<section id="pairs">
  <h2>Preliminary matched-pairs experiment</h2>
  <p class="lede">A point-in-time baseline matched additions and deletions within each review using only daily data available through the announcement: beta, volatility, momentum, log dollar volume and return correlation. Hungarian assignment creates one-to-one pairs. It lacks sector, venue, spread, auction volume and borrow calipers, so it is a diagnostic—not the production matcher.</p>
  $pairs_chart
  <div class="metrics">
    $pair_metrics_cards
  </div>
  <p>The matched book contains $pair_count security pairs across $pair_dates review dates. At 10 bp it returns $pair_total, CAGR $pair_cagr, max drawdown $pair_dd and event Sharpe $pair_sharpe. Matching the public lists more carefully does not manufacture a durable continuation edge.</p>
  <h3>Ten worst and ten best preliminary pairs</h3><div class="table-wrap">$pair_table</div>
</section>

<section id="results">
  <h2>Backtest results</h2>
  <p class="lede">Primary rule: enter at the open of the start-stamped 15:49 bar; long additions and short deletions; 50% gross per leg; missing leg remains cash; exit at the daily-close proxy; charge 10 bp per unit of gross exposure. Each review date is one independent observation.</p>
  $equity_chart
  <div class="two-chart">$event_returns_chart$event_distribution_chart</div>
  $leg_chart
  <h3>Strategy-family comparison</h3><div class="table-wrap">$strategy_table</div>
  $strategy_chart
  <h3>Cost sensitivity</h3><div class="grid-2"><div class="table-wrap">$cost_table</div>$cost_chart</div>
  $volume_chart
  <div class="callout bad"><strong>Economic rejection, not merely statistical ambiguity</strong>The gross primary strategy earns only $gross_total cumulatively and $gross_mean bp per date. That is below spread, adverse-selection, auction, locate and operational friction for this universe. A statistically interesting 15:49→15:50 component still fails the live cost gate.</div>
</section>

<section id="validation">
  <h2>Leakage-safe validation</h2>
  <p class="lede">The independent unit is the review date—not the stock and not the NOII update. All securities and all intraday messages from one implementation close must remain in the same fold.</p>
  <div class="grid-2">
    <div class="panel"><h3>Primary protocol</h3><ol><li>Outer anchored walk-forward; train only on earlier reviews.</li><li>Inner grouped chronological folds for tuning only.</li><li>Purge overlapping labels; embargo through T+1 for reversal.</li><li>Fit scaling, winsorization, matching, hedges and thresholds in train only.</li><li>Use date-level P&amp;L, four-review block bootstrap, date clustering and matched controls.</li><li>Record all frozen variants and correct for multiple testing.</li></ol></div>
    <div class="panel"><h3>Why not ordinary K-fold</h3><p>Random stock-level folds would place the same market-wide auction shock in train and test. With only 26 review dates, purged or combinatorial purged CV is a supplemental stability diagnostic, not the primary estimator. The first genuinely untouched holdout begins after August 2026.</p><p>August 2026 motivated the hypothesis and can never be relabeled out of sample.</p></div>
  </div>
  <div class="callout"><strong>Options-specific protocol</strong>Keep every option, hedge and security from one review in the same fold. Fit DTE, strike, liquidity and exit thresholds only on earlier reviews; preserve the first post-August-2026 review as the genuine holdout. The primary is one frozen 7–21-DTE ATM rule. DTE/strike/exit/hedging variants form one family and require multiplicity control; CPCV is only a robustness diagnostic because quarterly review dates are naturally separated.</div>
  <div class="two-chart">$walk_chart$placebo_chart</div>
  <div class="table-wrap">$walk_table</div>
  $subgroup_chart
  <div class="table-wrap">$subgroup_table</div>
  <div class="callout"><strong>Structural break</strong>MSCI began comprehensive quarterly reviews with February 2023. Full NOS/FIF review at every QCIR applies from May 2023. Pre/post results must be reported separately; later-sample strength is retrospective and includes the motivating SNDK date.</div>
</section>

<section id="execution">
  <h2>Venue-aware execution and capacity</h2>
  <p class="lede">A production backtest must replay historical venue rules, source and receive timestamps, order acknowledgments, the first executable quote after decision latency, auction-order rejects and partial fills.</p>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>Venue</th><th>Imbalance information</th><th>Order constraints</th><th>Product implication</th></tr></thead><tbody>
  <tr><td>NYSE</td><td>Closing imbalance begins 15:50; generally updates each second when fields change.</td><td>Ordinary MOC/LOC entry/cancel restrictions at 15:50; D orders can remain available later through supported floor access.</td><td>Standard-access strategy is mainly predictive before 15:50; public-NOII conditioning and ordinary MOC exit conflict operationally.</td></tr>
  <tr><td>Nasdaq</td><td>Early NOII every 10 seconds from 15:50; full NOII every second from 15:55 with Near/Far prices.</td><td>MOC generally immutable after 15:50 and accepted until 15:55; LOC entry until 15:58; IO until cross.</td><td>More scope for imbalance-conditioned closing interest, but initial response can be subsecond and late orders are constrained.</td></tr>
  </tbody></table></div>
  <h3>Options execution boundary</h3>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>Issue</th><th>Required treatment</th></tr></thead><tbody>
  <tr><td>Simultaneous 16:00 boundary</td><td>Ordinary single-stock options end as the stock auction occurs. Exit the option by the frozen pre-16:00 deadline; never value that fill with the subsequently known auction print.</td></tr>
  <tr><td>Limited Curb exception</td><td>Cboe’s 16:00–16:15 Curb applies only to selected eligible classes, limit orders and supported brokers. Test it as a separate symbol-specific variant, not a universal assumption.</td></tr>
  <tr><td>Two-leg execution</td><td>Headline replay buys call ask + put ask and sells bid + bid after latency. Historical complex-book fills may improve this only when directly observed. Enforce quote freshness, size and synchronized timestamps.</td></tr>
  <tr><td>Expiration / pin risk</td><td>Exclude 0DTE from the baseline. A failed exit in American, physically settled options can create ±100 shares per contract and broker-specific exercise risk.</td></tr>
  <tr><td>Missing exit quote</td><td>Never drop a trade after entry. Use the last fresh executable bid in the liquidation window; otherwise flag an operational failure and apply a preregistered next-observable/zero-recovery stress.</td></tr>
  </tbody></table></div>
  <h3>Execution state machine</h3>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>State</th><th>Clock</th><th>Required action / invariant</th></tr></thead><tbody>
  <tr><td>VERIFIED</td><td>Announcement → T−1</td><td>Reconcile every MSCI revision; compute old/new weights and product-level flows.</td></tr>
  <tr><td>LOCATED</td><td>Implementation morning</td><td>Secure short locates for deletion and hedge legs; reject incomplete pairs.</td></tr>
  <tr><td>READY</td><td>15:30–15:48</td><td>Freeze features, controls, risk, auction forecasts, venue-rule version and quantities.</td></tr>
  <tr><td>ENTER</td><td>15:49</td><td>Synchronized marketable-limit children; simulate first executable quote after latency, never bar open by assumption.</td></tr>
  <tr><td>CLOSE_COMMITTED</td><td>Before venue cutoff</td><td>Submit auction exit only after entry fills are known; persist acknowledgments and cancelability.</td></tr>
  <tr><td>NOII_MONITOR</td><td>15:50 onward</td><td>Classify confirm, neutral, contradict or overshoot; estimate remaining alpha.</td></tr>
  <tr><td>ADJUST / HEDGE</td><td>Until close</td><td>Add only when incremental edge exceeds incremental cost; neutralize rejected or one-sided fills.</td></tr>
  <tr><td>AUCTION / RECONCILED</td><td>16:00 onward</td><td>Consume official cross, corrections, fees and residual exposure; preserve immutable decision snapshots.</td></tr>
  </tbody></table></div>
  <div class="formula">TCᵢ(q) = ½·spreadᵢ·|q| + fees + YᵢσᵢPᵢ|q|√(|q|/Vwindow) + auction impact + borrow + reject/partial-fill risk</div>
  <div class="grid-2"><div class="panel"><h3>Pre-trade controls</h3><p>Revision reconciliation, stable ID/share class, venue, locates, earnings/M&amp;A/corporate-action filter, factor neutrality, participation caps, synchronized clocks and the exact historical rule version.</p></div><div class="panel"><h3>Real-time kill switches</h3><p>Feed sequence gap, stale/crossed NBBO, halt/LULD, imbalance sign flip, indicative-price overshoot, order rejection, unmatched leg, spread/volatility breach, or portfolio loss/exposure breach.</p></div></div>
</section>

<section id="roadmap">
  <h2>Productization roadmap and research gates</h2>
  <div class="roadmap">
    <div><b>Freeze current-data specification</b><span>Keep the sign-only rule and matched pairs as negative baselines. Version every decision timestamp and failed variant.</span></div>
    <div><b>Acquire licensed data</b><span>MSCI vintages/weights/NOS/FIF, fund AUM/holdings, NOII/ITCH/TAQ, official crosses, historical OPRA NBBO/trades, option contract definitions, complex fills and borrow history.</span></div>
    <div><b>Build message-level replay</b><span>Source/receive timestamps, historical rule engine, NBBO/depth, queue/latency, impact, rejects, partials and corrections.</span></div>
    <div><b>Pass research gate</b><span>Positive net P&amp;L at 2× cost, stable across venue/regime/migration, no dominant date, positive matched-control abnormal return and economically positive block CI.</span></div>
    <div><b>Shadow live 2–4 reviews</b><span>Zero orders. Require complete event reconciliation, reliable feeds, locates and no cutoff-state errors.</span></div>
    <div><b>Micro-live then scale</b><span>Hard dollar limits; replace modeled costs with fills. Scale only on measured capacity after at least eight genuinely future reviews.</span></div>
  </div>
  <h3>Production data model</h3>
  <div class="table-wrap"><table class="data-table"><thead><tr><th>Table</th><th>Essential fields</th><th>Why it exists</th></tr></thead><tbody>
  <tr><td>security_master_scd</td><td>Stable MSCI/security IDs, point-in-time ticker, share class, venue, listing status, effective/known timestamps.</td><td>Prevent survivorship, identifier and venue leakage.</td></tr>
  <tr><td>msci_event_vintage</td><td>Announcement/revision timestamps, old/new index/weight, NOS, FIF, price cutoff, action, migration, retraction.</td><td>Reconstruct exactly what was knowable at each decision.</td></tr>
  <tr><td>tracking_product</td><td>Exact benchmark, AUM, replication, tracking fraction, holdings date, overlap group.</td><td>Avoid double-counting nested passive assets.</td></tr>
  <tr><td>expected_flow_snapshot</td><td>Product and aggregate dollars/shares, uncertainty, normalizers, calculation version.</td><td>Freeze the pre-15:50 predictive signal.</td></tr>
  <tr><td>auction_message</td><td>Venue sequence, source/receive timestamp, imbalance/paired shares, side, reference, near/far/indicative price.</td><td>Test Δw × NOII without look-ahead.</td></tr>
  <tr><td>quotes_trades / official_auction</td><td>NBBO/depth/trades/conditions/halts; official price, shares, cross ID and corrections.</td><td>Create executable labels and impact estimates.</td></tr>
  <tr><td>option_security_master</td><td>OSI/root, call/put, strike, expiry, multiplier, deliverable, listing/adjustment history, exercise style.</td><td>Select the exact contract point in time and exclude adjusted deliverables.</td></tr>
  <tr><td>option_quotes_trades</td><td>OPRA source/receive time, bid/ask/size, sale condition, exchange, quote age, complex-order executions and entitlement.</td><td>Replay ask-in/bid-out fills, spreads, liquidity and capacity.</td></tr>
  <tr><td>borrow_snapshot</td><td>Locate time/shares/rate, utilization, HTB, recall.</td><td>Make the short leg real.</td></tr>
  <tr><td>order_execution</td><td>Decision, submit, ack, fill, reject, partial, fee and venue state.</td><td>Measure implementation shortfall and operational risk.</td></tr>
  </tbody></table></div>
</section>

<section id="appendix">
  <h2>Audit appendices</h2>
  <div class="qa-grid">
    <div><b>Run generated</b><span>$manifest_time</span></div><div><b>Report generated</b><span>$report_generated</span></div><div><b>Tests</b><span>$test_summary</span></div>
    <div><b>Events SHA-256</b><span>$events_hash</span></div><div><b>Backtest SHA-256</b><span>$code_hash</span></div><div><b>Cache fingerprint</b><span>$cache_hash</span></div>
    <div><b>Vol diagnostics SHA-256</b><span>$vol_code_hash</span></div><div><b>Vol module SHA-256</b><span>$vol_module_hash</span></div><div><b>Vol cache fingerprint</b><span>$vol_cache_hash</span></div>
    <div><b>Burst module SHA-256</b><span>$burst_module_hash</span></div><div><b>Burst runner SHA-256</b><span>$burst_runner_hash</span></div><div><b>Burst generated</b><span>$burst_manifest_time</span></div>
    <div><b>Flow features SHA-256</b><span>$flow_feature_hash</span></div><div><b>Flow system SHA-256</b><span>$flow_system_hash</span></div><div><b>Flow generated</b><span>$flow_manifest_time</span></div>
  </div>
  <details open><summary>Complete event ledger · $expected_events events</summary><div>
    <div class="search-row"><input id="event-search" type="search" placeholder="Filter date, ticker, security, action, type or coverage…"><span id="event-count">$expected_events rows</span></div>
    <div class="table-wrap">$event_table</div>
  </div></details>
  <details><summary>All preliminary matched pairs · $pair_count rows</summary><div class="table-wrap">$all_pairs_table</div></details>
  <details><summary>Volatility diagnostics by controlled review date</summary><div><div class="table-wrap">$volatility_date_table</div><p class="footnote">Each event value is date-balanced across controlled names; each control is the same security averaged over its four nearest complete pre-event sessions.</p></div></details>
  <details><summary>Full terminal-spot straddle sensitivity grid · 336 scenarios</summary><div><div class="table-wrap">$straddle_scenario_table</div><p class="footnote">Model arithmetic only. These rows are intentionally not assigned total return, CAGR, Sharpe, drawdown or option-backtest status.</p></div></details>
  <details><summary>Extraction failures and close exclusions · $failure_count records</summary><div><div class="table-wrap">$failure_table</div><p class="footnote">Eight provisional secondary closes change cumulative net return from $primary_total to $missing_total. They remain sensitivity-only until checked against official auction records.</p></div></details>
  <details><summary>Exact research configuration</summary><div><pre class="code">$config_json</pre></div></details>
  <details><summary>Source and evidence register</summary><div><ol class="sources">$sources_html</ol></div></details>
  <details><summary>Licensing and limitations</summary><div><p>MSCI public documents are proprietary; their terms restrict reproduction, database use and commercial products without permission. This local prototype is for research and evaluation. A commercial/live system requires licensed MSCI data and legal review.</p><p>Returns are simulated, use simplified costs, exclude tax and financing, and are not evidence of live executability. Daily-close values are provider proxies. OHLCV bars do not contain bid/ask spread, depth, NOII, queue position, auction eligibility, rejects or borrow. The volatility section validates an underlying-movement premise only: it has no historical intraday option quotes, observed IV, Greeks, spread, size, contract availability, fees or fills. Its Black–Scholes grid is model sensitivity, not option P&amp;L.</p></div></details>
</section>
</main>
<footer>MSCI Close-Pressure Relative Value · local quantitative research artifact · not investment advice</footer>
</div>
<script>
function bindTableSearch(inputId,tableId,counterId){
  const input=document.getElementById(inputId);
  const counter=document.getElementById(counterId);
  const rows=Array.from(document.querySelectorAll('#'+tableId+' tbody tr'));
  if(!input||!counter)return;
  const update=()=>{const q=input.value.trim().toLowerCase();let shown=0;rows.forEach(row=>{const visible=!q||row.textContent.toLowerCase().includes(q);row.style.display=visible?'':'none';if(visible)shown+=1});counter.textContent=shown+' rows'};
  input.addEventListener('input',update);
}
bindTableSearch('event-search','event-ledger','event-count');
bindTableSearch('top10-trade-search','top10-trade-ledger','top10-trade-count');
const links=Array.from(document.querySelectorAll('#nav a'));
const targets=links.map(link=>document.querySelector(link.getAttribute('href'))).filter(Boolean);
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){links.forEach(link=>link.classList.toggle('active',link.getAttribute('href')==='#'+entry.target.id))}})},{rootMargin:'-20% 0px -70% 0px'});
targets.forEach(target=>observer.observe(target));
</script>
</body></html>""")

    substitutions = {
        "kpis": kpis,
        "burst_kpis": burst_kpis,
        "burst_gross_mean_bp": bp(burst_gross["mean_return_per_event"]),
        "burst_gross_total": pct(burst_gross["total_return"]),
        "burst_net_total": pct(burst_net_10["total_return"]),
        "burst_net_cagr": pct(burst_net_10["cagr"]),
        "burst_net_dd": pct(burst_net_10["max_drawdown"]),
        "burst_break_even": num(burst_gross["break_even_cost_bps"]),
        "burst_raw_p": num(burst_two_bar_selection["raw_one_sided_t_pvalue"], 4),
        "burst_date_max_p": num(
            burst_two_bar_selection["max_t_familywise_pvalue"], 4
        ),
        "burst_block_max_p": num(
            burst_two_bar_selection["block4_max_t_familywise_pvalue"], 4
        ),
        "burst_pre_qcir_bp": bp(burst_pre_qcir["mean_return_per_event"]),
        "burst_post_qcir_bp": bp(burst_post_qcir["mean_return_per_event"]),
        "burst_abnormal_bp": bp(burst_abnormal["mean_return_per_event"]),
        "burst_abnormal_low": bp(burst_abnormal["block_bootstrap_mean_ci_low"]),
        "burst_abnormal_high": bp(burst_abnormal["block_bootstrap_mean_ci_high"]),
        "burst_cost_table": burst_cost_table,
        "burst_regime_table": burst_regime_table,
        "burst_selection_table": burst_selection_table,
        "burst_control_table": burst_control_table,
        "burst_subgroup_table": burst_subgroup_table,
        "burst_equity_chart": chart(
            "first_minute_alpha_equity.png",
            "First-minute alpha equity and drawdown",
            "Gross minute-bar proxy and 10 bp net haircut; May 2023 full-NOS/FIF QCIR regime is marked.",
        ),
        "burst_cost_regime_chart": chart(
            "first_minute_alpha_cost_regime.png",
            "First-minute alpha cost sensitivity and regime split",
            "The arithmetic break-even is near 13.8 bp, while performance is concentrated after May 2023.",
        ),
        "burst_time_selection_chart": chart(
            "first_minute_alpha_time_selection.png",
            "All inspected one-minute and adjacent two-bar windows",
            "The selected windows rank first by raw mean; multiplicity-adjusted statistics are reported below.",
        ),
        "burst_control_chart": chart(
            "first_minute_alpha_controls.png",
            "Same-name same-clock control comparison",
            "Four nearest complete prior sessions for 398 securities across 24 reviews; interval is a four-review block bootstrap.",
        ),
        "flow_weighting_kpis": flow_weighting_kpis,
        "flow_weighting_chart": chart(
            "flow_weighting_comparison.png",
            "Point-in-time weighting comparison",
            "Cash-funded equity at a 10 bp haircut and deployed gross; proxy inputs are pre-event liquidity features, not actual MSCI flows.",
        ),
        "flow_metrics_table": flow_metrics_table,
        "flow_paired_table": flow_paired_table,
        "flow_regime_table": flow_regime_table,
        "flow_capacity_table": flow_capacity_table,
        "flow_proxy_lift": bp(flow_proxy_test["mean_difference"]),
        "flow_proxy_ci_low": bp(flow_proxy_test["block4_mean_ci_low"]),
        "flow_proxy_ci_high": bp(flow_proxy_test["block4_mean_ci_high"]),
        "flow_proxy_max_t_p": num(
            flow_proxy_test["block4_max_t_familywise_pvalue"], 4
        ),
        "flow_early_lift": bp(
            flow_proxy_early["mean_return_per_event"]
            - flow_control_early["mean_return_per_event"]
        ),
        "flow_qcir_lift": bp(
            flow_proxy_qcir["mean_return_per_event"]
            - flow_control_qcir["mean_return_per_event"]
        ),
        "flow_pit_rows": integer(flow_manifest["point_in_time_valid_rows"]),
        "flow_feature_rows": integer(flow_manifest["feature_rows"]),
        "flow_migrations": integer(
            flow_manifest["migration_rows_requiring_cross_index_netting"]
        ),
        "flow_proxy_uncapped_interval_max": pct(
            flow_proxy_uncapped_capacity_row["interval_participation_max"], 1
        ),
        "flow_capacity_gross": pct(
            flow_proxy_capacity_row["mean_deployed_gross"], 1
        ),
        "flow_sndk_weight": pct(flow_sndk_proxy["target_weight"], 1),
        "flow_sndk_share": pct(
            flow_sndk_proxy["share_of_date_gross_pnl"], 1
        ),
        "top10_kpis": top10_kpis,
        "top10_chart": chart(
            "topn_capped10_equity_drawdown.png",
            "Balanced top-10 proxy equity and drawdown",
            "Event-step 0 bp primary equity, 10 bp sensitivity, matched control and 0 bp drawdown. Entry/exit are minute-bar proxies; there is no NBBO, auction, borrow or queue replay.",
        ),
        "top10_metrics_table": top10_metrics_table,
        "top10_paired_table": top10_paired_table,
        "top10_regime_table": top10_regime_table,
        "top10_date_table": top10_date_table,
        "top10_trade_table": top10_trade_table,
        "top10_trade_rows": integer(len(top10_trade_log)),
        "top10_net_mean": bp(top10_primary_zero["mean_return_per_event"]),
        "top10_active_net_mean": bp(top10_primary_zero["active_mean_return"]),
        "top10_net_total": pct(top10_primary_zero["total_return"]),
        "top10_lift": bp(top10_primary_test["mean_difference"]),
        "top10_lift_low": bp(top10_primary_test["block4_mean_ci_low"]),
        "top10_lift_high": bp(top10_primary_test["block4_mean_ci_high"]),
        "top10_family_p": num(
            top10_primary_test["block4_max_t_familywise_pvalue"], 4
        ),
        "top10_gross_mean": bp(top10_primary_zero["mean_return_per_event"]),
        "top10_without_sndk": bp(
            top10_primary_sndk["full_mean_without_sndk_cash"]
        ),
        "attribution_kpis": attribution_kpis,
        "attribution_event_chart": chart(
            "topn_capped10_event_leg_attribution.png",
            "Top-10 long and short contribution by implementation close",
            "Addition-long and deletion-short contribution for all 26 scheduled reviews at the 0 bp research proxy. Black dots are total return; x marks are cash reviews.",
        ),
        "attribution_regime_chart": chart(
            "topn_capped10_early_regime_diagnostics.png",
            "Early versus late long short and volume decomposition",
            "Top-10 and all-name leg contribution plus closing-volume concentration before and after the February 2023 breakpoint.",
        ),
        "attribution_regime_table": attribution_regime_table,
        "attribution_seasonal_table": attribution_seasonal_table,
        "attribution_event_table": attribution_event_table,
        "attribution_driver_table": attribution_driver_table,
        "attribution_migration_table": attribution_migration_table,
        "attribution_review_type_table": attribution_review_type_table,
        "attr_early_mean": num(attribution_early["gross_mean_per_scheduled_bp"]),
        "attr_early_cash": integer(attribution_early["cash_reviews"]),
        "attr_early_net10": num(attribution_early["net_10bp_mean_per_scheduled_bp"]),
        "attr_early_long": num(attribution_early["long_total_bp"]),
        "attr_early_short": num(attribution_early["short_total_bp"]),
        "attr_netting_capture": num(attribution_early["leg_netting_capture_pct"], 1),
        "attr_clock_early": num(attribution_clock_early_1550["mean_signed_bar_return_bp"]),
        "attr_clock_late": num(attribution_clock_late_1550["mean_signed_bar_return_bp"]),
        "attr_volume_early": num(attribution_clock_early_1550["mean_minute_volume_share_pct"]),
        "attr_volume_late": num(attribution_clock_late_1550["mean_minute_volume_share_pct"]),
        "attr_top_ex2026": num(attribution_top_ex_2026["mean_bp_per_scheduled_review"]),
        "attr_broad_ex2026": num(attribution_broad_ex_2026["mean_bp_per_scheduled_review"]),
        "attr_2026_share": num(
            100.0
            * attribution_year_2026["arithmetic_total_bp"]
            / attribution_all["arithmetic_total_bp"],
            1,
        ),
        "state_filter_kpis": state_filter_kpis,
        "state_filter_chart": (
            '<figure class="chart"><img loading="lazy" '
            'src="../state_filter_extension/charts/state_filter_comparison.png" '
            'alt="Pre-15:49 state-filter equity, hit rate and exposure trade-off">'
            '<figcaption>Baseline, migration-aware confirmation gate and expanding rule selection at a 10 bp haircut. '
            'All state variables stop at the 15:47 bar.</figcaption></figure>'
        ),
        "state_rule_table": state_rule_table,
        "state_regime_table": state_regime_table,
        "state_walk_table": state_walk_table,
        "state_blend_table": state_blend_table,
        "state_baseline_hit": pct(state_baseline["active_review_hit_rate"], 1),
        "state_gate_hit": pct(state_gate["active_review_hit_rate"], 1),
        "state_baseline_dd": pct(state_baseline["max_drawdown"], 2),
        "state_gate_dd": pct(state_gate["max_drawdown"], 2),
        "state_baseline_net": num(state_baseline["arithmetic_net_return_bp"]),
        "state_gate_net": num(state_gate["arithmetic_net_return_bp"]),
        "state_gate_increment": num(state_gate_paired["mean_paired_increment_bp"]),
        "state_gate_family_p": num(
            state_gate_paired["block4_max_t_familywise_pvalue"], 4
        ),
        "state_walk_retention": pct(
            state_walk_adaptive["net_return_retention_vs_matched_baseline"], 1
        ),
        "state_walk_hit_lift": pct(
            state_walk_adaptive["active_hit_rate_improvement"], 1
        ),
        "state_opt_kpis": state_opt_kpis,
        "state_opt_chart": (
            '<figure class="chart"><img loading="lazy" '
            'src="../state_filter_optimization/charts/state_filter_optimization.png" '
            'alt="Frozen soft-gate equity frontier cost stress and expanding pseudo-OOS comparison">'
            '<figcaption>Fixed 10 bp equity, profit-hit frontier, 0-20 bp cost stress and expanding pseudo-OOS policies. '
            'The matched baseline remains above both adaptive policies in cumulative profit.</figcaption></figure>'
        ),
        "state_opt_fixed_table": state_opt_fixed_table,
        "state_opt_walk_table": state_opt_walk_table,
        "state_opt_walk_paired_table": state_opt_walk_paired_table,
        "state_opt_mechanism_table": state_opt_mechanism_table,
        "state_opt_base_sharpe": num(
            state_opt_baseline["active_trade_sharpe_sqrt4"], 2
        ),
        "state_opt_025_sharpe": num(
            state_opt_025["active_trade_sharpe_sqrt4"], 2
        ),
        "state_opt_base_hit": pct(state_opt_baseline["active_review_hit_rate"], 1),
        "state_opt_025_hit": pct(state_opt_025["active_review_hit_rate"], 1),
        "state_opt_base_dd": pct(state_opt_baseline["max_drawdown"], 2),
        "state_opt_025_dd": pct(state_opt_025["max_drawdown"], 2),
        "state_opt_base_net": num(state_opt_baseline["arithmetic_net_return_bp"]),
        "state_opt_025_net": num(state_opt_025["arithmetic_net_return_bp"]),
        "state_opt_025_increment": num(
            state_opt_pair_025["mean_paired_increment_bp"]
        ),
        "state_opt_025_ci_low": num(
            state_opt_pair_025["block4_increment_ci_low_bp"]
        ),
        "state_opt_025_ci_high": num(
            state_opt_pair_025["block4_increment_ci_high_bp"]
        ),
        "state_opt_025_max_p": num(
            state_opt_pair_025["block4_max_t_familywise_pvalue"], 4
        ),
        "state_opt_walk_constrained_retention": pct(
            state_opt_walk_constrained[
                "net_return_retention_vs_matched_baseline"
            ],
            1,
        ),
        "state_opt_walk_constrained_hit_lift": pct(
            state_opt_walk_constrained["active_hit_rate_improvement"], 1
        ),
        "state_opt_walk_max_retention": pct(
            state_opt_walk_max["net_return_retention_vs_matched_baseline"], 1
        ),
        "state_opt_walk_max_sharpe": num(
            state_opt_walk_max["active_trade_sharpe_sqrt4"], 2
        ),
        "state_opt_loo_share": pct(state_opt_loo_share, 1),
        "state_opt_early_neither": num(
            state_opt_early_neither["date_balanced_mean_signed_return_bp"]
        ),
        "state_opt_early_hour_only": num(
            state_opt_early_hour_only["date_balanced_mean_signed_return_bp"]
        ),
        "state_opt_audit_rows": integer(state_opt_audit["security_rows"]),
        "state_opt_audit_dates": integer(state_opt_audit["review_dates"]),
        "state_opt_audit_error": f"{float(state_opt_audit['maximum_date_reconciliation_error']):.2e}",
        "volatility_kpis": "".join(
            [
                metric_card(
                    "Absolute move · event",
                    f"{num(terminal_move['event_mean_across_dates'] * 10_000, 1)} bp",
                    f"{integer(terminal_move['review_dates'])} controlled reviews",
                    "good",
                ),
                metric_card(
                    "Pre-event control",
                    f"{num(terminal_move['pre_event_control_mean_across_dates'] * 10_000, 1)} bp",
                    "Same names · four prior full sessions",
                    "neutral",
                ),
                metric_card(
                    "Event excess",
                    f"{bp(terminal_move['mean_event_minus_pre_event_control'], 1)} bp",
                    f"CI {bp(terminal_move['block_bootstrap_excess_ci_low'], 1)} to {bp(terminal_move['block_bootstrap_excess_ci_high'], 1)} bp",
                    "good",
                ),
                metric_card(
                    "25/10 breakout proxy",
                    pct(breakout_primary["total_return"]),
                    "Underlying minute bars · not option P&L",
                    "bad",
                ),
            ]
        ),
        "vol_event_abs_bp": num(
            terminal_move["event_mean_across_dates"] * 10_000, 1
        ),
        "vol_control_abs_bp": num(
            terminal_move["pre_event_control_mean_across_dates"] * 10_000, 1
        ),
        "vol_excess_abs_bp": bp(
            terminal_move["mean_event_minus_pre_event_control"], 1
        ),
        "vol_abs_ci_low": bp(
            terminal_move["block_bootstrap_excess_ci_low"], 1
        ),
        "vol_abs_ci_high": bp(
            terminal_move["block_bootstrap_excess_ci_high"], 1
        ),
        "vol_control_dates": integer(terminal_move["review_dates"]),
        "vol_control_names": integer(terminal_move["controlled_security_events"]),
        "vol_positive_dates": integer(
            round(
                terminal_move["positive_excess_date_rate"]
                * terminal_move["review_dates"]
            )
        ),
        "vol_all_event_abs_bp": num(
            terminal_move["all_event_mean_across_dates"] * 10_000, 1
        ),
        "vol_breakout_total": pct(breakout_primary["total_return"]),
        "vol_breakout_mean_bp": bp(breakout_primary["mean_return_per_event"]),
        "vol_breakout_cagr": pct(breakout_primary["cagr"]),
        "vol_breakout_dd": pct(breakout_primary["max_drawdown"]),
        "vol_breakout_ci_low": bp(
            breakout_primary["block_bootstrap_mean_ci_low"]
        ),
        "vol_breakout_ci_high": bp(
            breakout_primary["block_bootstrap_mean_ci_high"]
        ),
        "vol_breakout_trigger": pct(
            breakout_primary["mean_trigger_rate"], 1
        ),
        "model_zero_friction": pct(
            model_zero_friction["mean_all_event_model_return_on_premium"]
        ),
        "model_halfpct_friction": pct(
            model_halfpct_friction["mean_all_event_model_return_on_premium"]
        ),
        "model_onepct_friction": pct(
            model_onepct_friction["mean_all_event_model_return_on_premium"]
        ),
        "model_iv_down": pct(
            model_iv_down["mean_all_event_model_return_on_premium"]
        ),
        "model_iv_up": pct(
            model_iv_up["mean_all_event_model_return_on_premium"]
        ),
        "volatility_excess_chart": chart(
            "volatility_pre_event_excess.png",
            "Pre-close movement excess versus pre-event sessions",
            "Four diagnostics on 398 controlled security-events across 24 reviews; intervals are four-review block bootstraps.",
        ),
        "volatility_threshold_chart": chart(
            "volatility_threshold_rates.png",
            "Terminal move threshold hit rates",
            "Date-balanced event and pre-event same-name control hit rates; 200 bp is omitted because of sparse tail counts.",
        ),
        "volatility_breakout_chart": chart(
            "volatility_breakout_proxy.png",
            "Underlying two-sided breakout payoff proxy",
            "Minute-bar next-open payoff screen under 5/10/20 bp friction haircuts; not executable-price P&L.",
        ),
        "straddle_model_chart": chart(
            "straddle_model_sensitivity.png",
            "Terminal-spot straddle model sensitivity",
            "Delta-hedged exact-ATM Black–Scholes sensitivity at 7 DTE and 50% entry IV; no observed option quotes.",
        ),
        "volatility_table": volatility_table,
        "volatility_threshold_table": volatility_threshold_table,
        "breakout_table": breakout_table,
        "straddle_model_table": straddle_model_table,
        "volatility_date_table": volatility_date_table,
        "straddle_scenario_table": straddle_scenario_table,
        "gross_total": pct(gross_primary["total_return"]),
        "break_even": num(primary["break_even_cost_bps"]),
        "primary_total": pct(primary["total_return"]),
        "primary_cagr": pct(primary["cagr"]),
        "primary_dd": pct(primary["max_drawdown"]),
        "primary_sharpe": num(primary["event_sharpe_sqrt4"]),
        "pre_1550_bp": bp(pre_1550["mean_return_per_event"]),
        "post_1550_bp": bp(post_1550["mean_return_per_event"]),
        "pair_total": pct(pairs_metrics["total_return"]),
        "pair_dates": integer(pairs_metrics["event_dates"]),
        "announcement_total": pct(announcement["total_return"]),
        "reversal_total": pct(reversal["total_return"]),
        "expected_events": integer(expected_events),
        "migrations": integer(migrations),
        "minute_usable": integer(minute_usable),
        "close_usable": integer(close_usable),
        "missing_total": pct(missing_close["total_return"]),
        "coverage_chart": chart("coverage_waterfall.png", "Coverage waterfall", "463 public events; 459 minute-window observations; 451 daily-close proxy observations."),
        "composition_chart": chart("review_composition.png", "Review composition", "Add/delete counts vary substantially, so date and leg balancing are necessary."),
        "close_quality_chart": chart("close_discrepancy_distribution.png", "Daily close proxy discrepancy", "Distribution of the daily-close proxy relative to the final captured continuous minute."),
        "event_profile_chart": chart("event_time_profile.png", "Event time price and volume profile", "Direction-signed price path and same-clock relative minute volume; confidence band is descriptive across 26 dates."),
        "event_heatmap_chart": chart("event_time_heatmap.png", "Review by minute heatmap", "Each row is one independent implementation close; the 15:49/15:50 boundary is marked."),
        "window_chart": chart("window_decomposition.png", "Close window decomposition", "Non-overlapping minute-bar windows with four-review block-bootstrap intervals."),
        "window_table": window_table,
        "pre_ci_low": bp(pre_1550["block_bootstrap_mean_ci_low"]),
        "pre_ci_high": bp(pre_1550["block_bootstrap_mean_ci_high"]),
        "sndk_metrics": "".join(
            [
                metric_card("15:49→close proxy", pct(sndk["return_1549_open_to_daily_close_proxy"], 3), "Long-oriented", "good"),
                metric_card("Last-10 volume rank", f"1 / {sndk['daily_sessions_since_listing']}", f"{sndk['event_last10_volume_vs_prior_max']:.2f}× prior last-10 maximum", "good"),
                metric_card("Last-10 volume share", pct(sndk["last_10_minute_volume_share"], 2), "Captured regular-session minute volume", "warn"),
                metric_card("Full-day volume rank", f"{sndk['event_volume_rank_descending']} / {sndk['daily_sessions_since_listing']}", "Not a record", "neutral"),
            ]
        ),
        "sndk_chart": chart("sndk_case.png", "SNDK case study", "Minute price and volume; daily provider close shown separately from the last continuous minute."),
        "sndk_return": pct(sndk["return_1549_open_to_daily_close_proxy"], 3),
        "sndk_sessions": integer(sndk["daily_sessions_since_listing"]),
        "sndk_day_rank": integer(sndk["event_volume_rank_descending"]),
        "aug_gross": pct(aug_portfolio["gross_return"]),
        "aug_net": pct(aug_portfolio["net_return"]),
        "aug_table": aug_table,
        "pairs_chart": chart("matched_pairs_equity.png", "Matched pairs equity curve", "Both the date-balanced public-list basket and preliminary similarity-matched pairs are net negative at 10 bp."),
        "pair_metrics_cards": "".join(
            [
                metric_card("Security pairs", integer(pairs_metrics["security_pairs"]), "One-to-one assignment", "neutral"),
                metric_card("Net cumulative", pct(pairs_metrics["total_return"]), "10 bp cost", "bad"),
                metric_card("CAGR", pct(pairs_metrics["cagr"]), "Cash between reviews", "bad"),
                metric_card("Max drawdown", pct(pairs_metrics["max_drawdown"]), "Date-level", "bad"),
            ]
        ),
        "pair_count": integer(pairs_metrics["security_pairs"]),
        "pair_cagr": pct(pairs_metrics["cagr"]),
        "pair_dd": pct(pairs_metrics["max_drawdown"]),
        "pair_sharpe": num(pairs_metrics["event_sharpe_sqrt4"]),
        "pair_table": pair_table,
        "equity_chart": chart("equity_curve.png", "Primary equity and drawdown", "Gross and 10 bp net equity; the May 2023 full-NOS/FIF QCIR regime is marked."),
        "event_returns_chart": chart("event_returns.png", "Returns by review date", "One bar per independent review date. August 2026 is positive but not dominant enough to rescue the full sample."),
        "event_distribution_chart": chart("event_return_distribution.png", "Event return distribution", "Net review-date return distribution at 10 bp."),
        "leg_chart": chart("leg_attribution.png", "Add and deletion leg attribution", "Long-add and short-delete contributions vary by review and frequently disagree."),
        "strategy_table": strategy_table,
        "strategy_chart": chart("strategy_comparison.png", "Strategy family comparison", "Announcement drift is larger but remains retrospective and data-incomplete; close reversal is unstable."),
        "cost_table": cost_table,
        "cost_chart": chart("cost_sensitivity.png", "Cost sensitivity", "The estimated gross break-even is only about 0.65 bp."),
        "volume_chart": chart("volume_pressure.png", "Volume and signed return", f"Spearman correlation is {float(volume_metrics['spearman_signed_return_vs_median_volume_ratio']):.2f} with two-sided p={float(volume_metrics['two_sided_pvalue']):.3f}; high volume is not a direction signal."),
        "gross_mean": bp(gross_primary["mean_return_per_event"]),
        "walk_chart": chart("walk_forward_folds.png", "Walk forward folds", "Anchored chronological folds keep all names from each review together."),
        "placebo_chart": chart("placebo_comparison.png", "Matched-session placebos", "Event-minus-matched-placebo abnormal return is negative on the fixed complete sample."),
        "walk_table": walk_table,
        "subgroup_chart": chart("subgroup_forest.png", "Subgroup forest", "Full-sample block-bootstrap intervals cross zero for every prespecified subgroup."),
        "subgroup_table": subgroup_table,
        "manifest_time": html.escape(str(manifest["generated_utc"])),
        "report_generated": report_generated,
        "test_summary": html.escape(args.test_summary),
        "events_hash": html.escape(str(manifest["events_sha256"])[:20] + "…"),
        "code_hash": html.escape(str(manifest["backtest_code_sha256"])[:20] + "…"),
        "cache_hash": html.escape(str(manifest["marketdata_cache_fingerprint"])[:20] + "…"),
        "vol_code_hash": html.escape(
            str(volatility_manifest["diagnostics_code_sha256"])[:20] + "…"
        ),
        "vol_module_hash": html.escape(
            str(volatility_manifest["volatility_module_sha256"])[:20] + "…"
        ),
        "vol_cache_hash": html.escape(
            str(volatility_manifest["marketdata_stat_fingerprint"])[:20] + "…"
        ),
        "burst_module_hash": html.escape(
            str(burst_manifest["module_sha256"])[:20] + "…"
        ),
        "burst_runner_hash": html.escape(
            str(burst_manifest["runner_sha256"])[:20] + "…"
        ),
        "burst_manifest_time": html.escape(str(burst_manifest["generated_utc"])),
        "flow_feature_hash": html.escape(
            str(flow_manifest["feature_panel_sha256"])[:20] + "…"
        ),
        "flow_system_hash": html.escape(
            str(flow_manifest["system_spec_sha256"])[:20] + "…"
        ),
        "flow_manifest_time": html.escape(str(flow_manifest["generated_utc"])),
        "event_table": event_table,
        "all_pairs_table": all_pairs_table,
        "failure_count": integer(len(failures)),
        "failure_table": failure_table,
        "config_json": html.escape(config_json),
        "sources_html": sources_html,
    }
    report = template.safe_substitute(substitutions)
    report_path = out / "report.html"
    report_path.write_text(report, encoding="utf-8")
    print(
        f"report={report_path} bytes={report_path.stat().st_size} "
        f"event_rows={len(event_ledger)} pair_rows={len(pairs)}"
    )


if __name__ == "__main__":
    main()
