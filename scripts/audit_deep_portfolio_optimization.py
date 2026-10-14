#!/usr/bin/env python3
"""Independent accounting and clock audit for deep portfolio optimization outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "baseline_top10_cap10_dollar_neutral"
EXACT_BETA = {
    "top10_stock_beta_neutral",
    "top10_long2_short1_spy_hedged",
    "top10_long_only_spy_hedged",
}
DOLLAR_NEUTRAL = {
    BASELINE,
    "top10_full_equal_dollar_neutral",
    "top6_full_equal_dollar_neutral",
    "top20_full_equal_dollar_neutral",
    "top6_impact_weighted_dollar_neutral",
    "top10_impact_weighted_dollar_neutral",
    "top10_inverse_vol_dollar_neutral",
    "dynamic80_impact_weighted_dollar_neutral",
    "top10_down_to_small_softgate25",
    "top10_relative_hour_reversal",
    "top10_quiet_hour",
    "top10_reversal_or_quiet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("configs/deep_portfolio_optimization_spec.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deep_portfolio_optimization"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.output_dir
    weights = pd.read_csv(root / "deep_portfolio_security_weights.csv", parse_dates=["effective_close_date"])
    trades = pd.read_csv(
        root / "deep_portfolio_trade_log.csv",
        parse_dates=["effective_close_date"],
        low_memory=False,
    )
    dates = pd.read_csv(root / "deep_portfolio_date_returns.csv", parse_dates=["effective_close_date"])
    metrics = pd.read_csv(root / "deep_portfolio_metrics.csv")
    selections = pd.read_csv(
        root / "deep_portfolio_walk_forward_selection.csv",
        parse_dates=["test_date", "selection_asof"],
    )
    walk = pd.read_csv(root / "deep_portfolio_walk_forward_returns.csv", parse_dates=["effective_close_date"])
    frozen = pd.read_csv(spec["inputs"]["frozen_top10_ledger"], parse_dates=["effective_close_date"])

    tolerance = 1e-12
    candidate_order = list(spec["candidate_order"])
    identity_duplicates = int(weights.duplicated(["strategy", "effective_close_date", "ticker"]).sum())
    candidate_set_match = set(weights["strategy"]) == set(candidate_order)
    nonfinite_selected_returns = int(
        (~np.isfinite(pd.to_numeric(weights["underlying_return"], errors="coerce")))
        .where(weights["target_weight"].abs().gt(0), False)
        .sum()
    )

    baseline = weights.loc[weights["strategy"].eq(BASELINE)].copy()
    key = ["effective_close_date", "ticker"]
    base_compare = baseline[key + ["target_weight"]].merge(
        frozen[key + ["target_weight"]],
        on=key,
        suffixes=("_new", "_frozen"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    baseline_membership_error = int(base_compare["_merge"].ne("both").sum())
    baseline_weight_error = float(
        (base_compare["target_weight_new"] - base_compare["target_weight_frozen"])
        .abs()
        .max()
    )

    zero = trades.loc[trades["cost_bps"].eq(0.0)].copy()
    price_rows = zero.loc[zero["instrument_role"].eq("event_stock")].copy()
    manual_underlying = price_rows["exit_close_proxy"] / price_rows["entry_open_proxy"] - 1.0
    price_return_error = float((manual_underlying - price_rows["underlying_return"]).abs().max())

    manual = trades.assign(
        manual_gross=lambda x: x["target_weight"] * x["underlying_return"],
        manual_cost=lambda x: x["target_weight"].abs() * x["cost_bps"] / 10_000.0,
    ).assign(manual_net=lambda x: x["manual_gross"] - x["manual_cost"])
    grouped = manual.groupby(["strategy", "cost_bps", "effective_close_date"], as_index=False).agg(
        manual_gross=("manual_gross", "sum"),
        manual_net=("manual_net", "sum"),
        manual_exposure=("target_weight", lambda x: float(x.abs().sum())),
    )
    compared = dates.merge(
        grouped,
        on=["strategy", "cost_bps", "effective_close_date"],
        how="left",
        validate="many_to_one",
    )
    for column in ("manual_gross", "manual_net", "manual_exposure"):
        compared[column] = compared[column].fillna(0.0)
    date_reconciliation_error = float(
        max(
            (compared["gross_return"] - compared["manual_gross"]).abs().max(),
            (compared["net_return"] - compared["manual_net"]).abs().max(),
            (compared["gross_exposure"] - compared["manual_exposure"]).abs().max(),
        )
    )

    sharpe_errors: list[float] = []
    for row in metrics.itertuples(index=False):
        sample = dates.loc[
            dates["strategy"].eq(row.strategy) & dates["cost_bps"].eq(row.cost_bps)
        ]
        scheduled = sample["net_return"]
        active = sample.loc[sample["gross_exposure"].gt(0), "net_return"]
        scheduled_sharpe = (
            float(np.sqrt(4.0) * scheduled.mean() / scheduled.std(ddof=1))
            if len(scheduled) > 1 and scheduled.std(ddof=1) > 0
            else np.nan
        )
        active_sharpe = (
            float(np.sqrt(4.0) * active.mean() / active.std(ddof=1))
            if len(active) > 1 and active.std(ddof=1) > 0
            else np.nan
        )
        if np.isfinite(scheduled_sharpe):
            sharpe_errors.append(abs(scheduled_sharpe - float(row.event_sharpe_sqrt4)))
        if np.isfinite(active_sharpe):
            sharpe_errors.append(abs(active_sharpe - float(row.active_trade_sharpe_sqrt4)))
    maximum_sharpe_error = max(sharpe_errors)

    beta_by_date = (
        (weights["target_weight"] * weights["beta_used"])
        .groupby([weights["strategy"], weights["effective_close_date"]])
        .sum()
    )
    exact_beta_error = float(
        beta_by_date.loc[beta_by_date.index.get_level_values(0).isin(EXACT_BETA)].abs().max()
    )
    soft_beta = beta_by_date.loc[
        beta_by_date.index.get_level_values(0)
        == "top10_equal_soft_beta_band_spy_hedged"
    ]
    soft_beta_excess = float((soft_beta.abs() - float(spec["portfolio_controls"]["soft_beta_band"])).clip(lower=0).max())
    net_by_date = weights.groupby(["strategy", "effective_close_date"])["target_weight"].sum()
    dollar_neutral_error = float(
        net_by_date.loc[net_by_date.index.get_level_values(0).isin(DOLLAR_NEUTRAL)].abs().max()
    )

    walk_clock_error = int(selections["selection_asof"].ge(selections["test_date"]).sum())
    expected_walk = dates.loc[dates["cost_bps"].eq(float(spec["primary_cost_bps"]))].merge(
        selections[["test_date", "selected_strategy"]],
        left_on=["effective_close_date", "strategy"],
        right_on=["test_date", "selected_strategy"],
        how="inner",
        validate="one_to_one",
    )
    walk_compare = walk.merge(
        expected_walk[["effective_close_date", "net_return"]],
        on="effective_close_date",
        suffixes=("_reported", "_manual"),
        validate="one_to_one",
    )
    walk_return_error = float(
        (walk_compare["net_return_reported"] - walk_compare["net_return_manual"]).abs().max()
    )

    checks = {
        "candidate_set_match": bool(candidate_set_match),
        "identity_duplicates": identity_duplicates,
        "nonfinite_selected_returns": nonfinite_selected_returns,
        "baseline_membership_error": baseline_membership_error,
        "baseline_weight_error": baseline_weight_error,
        "minute_price_return_reconstruction_error": price_return_error,
        "date_accounting_reconciliation_error": date_reconciliation_error,
        "maximum_sharpe_reconstruction_error": maximum_sharpe_error,
        "dollar_neutral_max_abs_net_exposure": dollar_neutral_error,
        "exact_beta_hedge_max_abs_beta_exposure": exact_beta_error,
        "soft_beta_band_max_excess": soft_beta_excess,
        "walk_forward_clock_errors": walk_clock_error,
        "walk_forward_return_reconciliation_error": walk_return_error,
    }
    numeric_errors = [
        baseline_weight_error,
        price_return_error,
        date_reconciliation_error,
        maximum_sharpe_error,
        dollar_neutral_error,
        exact_beta_error,
        soft_beta_excess,
        walk_return_error,
    ]
    pass_status = bool(
        candidate_set_match
        and identity_duplicates == 0
        and nonfinite_selected_returns == 0
        and baseline_membership_error == 0
        and walk_clock_error == 0
        and max(numeric_errors) <= tolerance
    )
    audit_trades = manual.loc[manual["cost_bps"].eq(float(spec["primary_cost_bps"]))].copy()
    sample_parts = [
        audit_trades.nlargest(1, "manual_net"),
        audit_trades.nsmallest(1, "manual_net"),
        audit_trades.loc[audit_trades["action"].eq("add")].head(1),
        audit_trades.loc[audit_trades["action"].eq("delete")].head(1),
        audit_trades.loc[audit_trades["instrument_role"].eq("market_hedge")].head(1),
        audit_trades.loc[
            audit_trades["strategy"].eq(BASELINE) & audit_trades["manual_net"].lt(0)
        ].nsmallest(1, "manual_net"),
    ]
    sample_frame = pd.concat(sample_parts, ignore_index=True).drop_duplicates(
        ["strategy", "effective_close_date", "ticker"]
    )
    samples = sample_frame[
        [
            "strategy",
            "effective_close_date",
            "ticker",
            "action",
            "instrument_role",
            "target_weight",
            "entry_open_proxy",
            "exit_close_proxy",
            "underlying_return",
            "manual_gross",
            "manual_cost",
            "manual_net",
        ]
    ].to_dict("records")
    audit = {
        "status": "pass" if pass_status else "fail",
        "tolerance": tolerance,
        "checks": checks,
        "manual_sample_trades": samples,
        "scope": "portfolio weights, minute-proxy returns, costs, metrics, constraints, and walk-forward clock",
        "excluded_scope": "quote-level execution and future out-of-sample performance",
    }
    (root / "deep_portfolio_manual_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    print(json.dumps(audit, indent=2, default=str))
    if not pass_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
