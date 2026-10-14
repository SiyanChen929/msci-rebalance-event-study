#!/usr/bin/env python3
"""Independent arithmetic audit of the portfolio-constraint comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOLLAR = "dollar_neutral_equal"
BETA = "beta_neutral_equal"
SIGNAL = "signal_weighted_unneutralized"
LONG = "long_only_dollar_leg"
SHORT = "short_only_dollar_leg"


def _expected_weights(base: pd.DataFrame, strategy: str) -> pd.Series:
    base_weight = pd.to_numeric(base["base_target_weight"], errors="raise")
    if strategy == DOLLAR:
        return base_weight
    if strategy == LONG:
        return base_weight.clip(lower=0.0)
    if strategy == SHORT:
        return base_weight.clip(upper=0.0)
    if strategy == SIGNAL:
        score = pd.to_numeric(base["proxy_flow_pressure"], errors="coerce")
        score = score.where(np.isfinite(score) & score.gt(0), 0.0)
        expected = pd.Series(0.0, index=base.index)
        for _, index in base.groupby("effective_close_date", sort=True).groups.items():
            index = list(index)
            total = float(score.loc[index].sum())
            if total <= 0:
                continue
            absolute = (score.loc[index] / total).clip(upper=0.10)
            direction = base.loc[index, "action"].map({"add": 1.0, "delete": -1.0})
            expected.loc[index] = absolute.to_numpy(float) * direction.to_numpy(float)
        return expected
    if strategy == BETA:
        used_beta = pd.to_numeric(base["beta_used"], errors="raise")
        expected = pd.Series(0.0, index=base.index)
        for _, index in base.groupby("effective_close_date", sort=True).groups.items():
            group = base.loc[list(index)]
            longs = group.index[group["action"].eq("add")].to_numpy(int)
            shorts = group.index[group["action"].eq("delete")].to_numpy(int)
            long_beta = float(used_beta.loc[longs].mean())
            short_beta = float(used_beta.loc[shorts].mean())
            base_long = short_beta / (long_beta + short_beta)
            base_short = long_beta / (long_beta + short_beta)
            scale = min(
                1.0,
                len(longs) * 0.10 / base_long,
                len(shorts) * 0.10 / base_short,
            )
            expected.loc[longs] = base_long * scale / len(longs)
            expected.loc[shorts] = -base_short * scale / len(shorts)
        return expected
    raise ValueError(strategy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-trades",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_trade_log.csv"),
    )
    parser.add_argument(
        "--variant-trades",
        type=Path,
        default=Path("outputs/portfolio_variants/portfolio_variant_trade_log.csv"),
    )
    parser.add_argument(
        "--date-portfolios",
        type=Path,
        default=Path("outputs/portfolio_variants/portfolio_variant_date_portfolios.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/portfolio_variants/portfolio_variant_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/portfolio_variants/portfolio_variant_manual_audit.json"),
    )
    args = parser.parse_args()

    base = pd.read_csv(args.base_trades, parse_dates=["effective_close_date"])
    base = base.rename(columns={"target_weight": "base_target_weight"})
    trades = pd.read_csv(args.variant_trades, parse_dates=["effective_close_date"])
    dates = pd.read_csv(args.date_portfolios, parse_dates=["effective_close_date"])
    metrics = pd.read_csv(args.metrics)
    zero = trades.loc[trades["cost_bps"].eq(0.0)].copy()
    key = ["effective_close_date", "ticker"]
    reference_columns = key + [
        "base_target_weight",
        "action",
        "underlying_return",
        "proxy_flow_pressure",
    ]
    merged = zero.drop(columns=["action", "underlying_return", "proxy_flow_pressure"]).merge(
        base[reference_columns], on=key, how="left", validate="many_to_one"
    )
    if merged["base_target_weight"].isna().any():
        raise AssertionError("variant rows do not map one-to-one to base trades")

    weight_errors: dict[str, float] = {}
    beta_errors: dict[str, float] = {}
    for strategy, group in merged.groupby("strategy", sort=True):
        expected = _expected_weights(group, strategy)
        weight_errors[strategy] = float(
            np.max(np.abs(expected.to_numpy(float) - group["target_weight"].to_numpy(float)))
        )
        beta_by_date = (
            (group["target_weight"] * group["beta_used"])
            .groupby(group["effective_close_date"])
            .sum()
        )
        beta_errors[strategy] = float(beta_by_date.abs().max())

    reconstructed = (
        trades.assign(
            manual_gross=lambda x: x["target_weight"] * x["underlying_return"],
            manual_cost=lambda x: x["target_weight"].abs() * x["cost_bps"] / 10_000.0,
        )
        .assign(manual_net=lambda x: x["manual_gross"] - x["manual_cost"])
        .groupby(["strategy", "cost_bps", "effective_close_date"], as_index=False)
        .agg(
            manual_gross_return=("manual_gross", "sum"),
            manual_net_return=("manual_net", "sum"),
            manual_gross_exposure=("target_weight", lambda x: float(x.abs().sum())),
        )
    )
    comparison = dates.merge(
        reconstructed,
        on=["strategy", "cost_bps", "effective_close_date"],
        how="left",
        validate="many_to_one",
    )
    for column in (
        "manual_gross_return",
        "manual_net_return",
        "manual_gross_exposure",
    ):
        comparison[column] = comparison[column].fillna(0.0)
    date_error = float(
        max(
            (comparison["gross_return"] - comparison["manual_gross_return"]).abs().max(),
            (comparison["net_return"] - comparison["manual_net_return"]).abs().max(),
            (comparison["gross_exposure"] - comparison["manual_gross_exposure"]).abs().max(),
        )
    )

    sharpe_errors: dict[str, float] = {}
    for row in metrics.itertuples(index=False):
        portfolio = dates.loc[
            dates["strategy"].eq(row.strategy) & dates["cost_bps"].eq(row.cost_bps)
        ]
        active = portfolio.loc[portfolio["gross_exposure"].gt(0), "net_return"]
        manual = (
            float(np.sqrt(4.0) * active.mean() / active.std(ddof=1))
            if len(active) > 1 and float(active.std(ddof=1)) > 0
            else np.nan
        )
        error = abs(manual - float(row.active_trade_sharpe_sqrt4))
        sharpe_errors[f"{row.strategy}:{row.cost_bps:g}"] = float(error)

    tolerance = 1e-12
    status = "pass" if max(
        max(weight_errors.values()), date_error, max(sharpe_errors.values())
    ) <= tolerance else "fail"
    if beta_errors[BETA] > tolerance:
        status = "fail"
    audit = {
        "status": status,
        "base_security_rows": int(len(base)),
        "variant_security_cost_rows": int(len(trades)),
        "date_portfolio_rows": int(len(dates)),
        "maximum_weight_reconstruction_error": max(weight_errors.values()),
        "weight_error_by_strategy": weight_errors,
        "maximum_date_reconciliation_error": date_error,
        "maximum_active_sharpe_error": max(sharpe_errors.values()),
        "beta_neutral_max_abs_beta_exposure": beta_errors[BETA],
        "tolerance": tolerance,
        "execution_status": "minute_bar_proxy_not_broker_fill",
    }
    args.output.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
