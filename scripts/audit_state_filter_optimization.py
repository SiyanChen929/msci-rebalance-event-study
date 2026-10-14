#!/usr/bin/env python3
"""Independent raw-bar reconstruction of the selected state-filter candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _session(path: Path, event_date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("America/New_York")
    else:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    frame["_timestamp"] = timestamp
    frame["_date"] = timestamp.dt.tz_localize(None).dt.normalize()
    frame["_clock"] = timestamp.dt.strftime("%H:%M")
    return frame.loc[frame["_date"].eq(pd.Timestamp(event_date).normalize())].copy()


def _bar_values(frame: pd.DataFrame) -> dict[str, float] | None:
    values: dict[str, float] = {}
    requests = {
        "hour_open": ("14:48", "open"),
        "feature_close": ("15:47", "close"),
        "entry_open": ("15:49", "open"),
        "exit_close": ("15:50", "close"),
    }
    for key, (clock, field) in requests.items():
        row = frame.loc[frame["_clock"].eq(clock)]
        if row.empty:
            return None
        value = float(pd.to_numeric(row.iloc[-1][field], errors="coerce"))
        if not np.isfinite(value) or value <= 0:
            return None
        values[key] = value
    return values


def _retention(rule: str) -> float:
    if rule == "down_to_small_hour_confirmation":
        return 0.0
    mapping = {
        "down_to_small_hour_soft_075": 0.75,
        "down_to_small_hour_soft_050": 0.50,
        "down_to_small_hour_soft_025": 0.25,
    }
    if rule not in mapping:
        raise ValueError(f"manual reconstruction does not support rule: {rule}")
    return mapping[rule]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/state_filter_optimization/state_optimization_manifest.json"),
    )
    parser.add_argument(
        "--base-ledger",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_trade_log.csv"),
    )
    parser.add_argument(
        "--selected-ledger",
        type=Path,
        default=Path("outputs/state_filter_optimization/state_optimization_selected_trade_log.csv"),
    )
    parser.add_argument(
        "--date-portfolios",
        type=Path,
        default=Path("outputs/state_filter_optimization/state_optimization_date_portfolios.csv"),
    )
    parser.add_argument(
        "--marketdata-dir", type=Path, default=Path("data/marketdata/minute")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/state_filter_optimization")
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rule = str(manifest["final_full_sample_constrained_selection"])
    retention = _retention(rule)
    base = pd.read_csv(args.base_ledger, parse_dates=["effective_close_date"])
    selected = pd.read_csv(args.selected_ledger, parse_dates=["effective_close_date"])
    date_portfolios = pd.read_csv(args.date_portfolios, parse_dates=["effective_close_date"])
    selected = selected.set_index(["effective_close_date", "ticker"], drop=False)
    spy_cache: dict[pd.Timestamp, dict[str, float] | None] = {}
    rows: list[dict[str, object]] = []
    for record in base.itertuples(index=False):
        event_date = pd.Timestamp(record.effective_close_date).normalize()
        if event_date not in spy_cache:
            spy_path = args.marketdata_dir / f"{event_date.date()}_SPY.parquet"
            spy_cache[event_date] = (
                _bar_values(_session(spy_path, event_date)) if spy_path.exists() else None
            )
        stock_path = args.marketdata_dir / (
            f"{event_date.date()}_{_safe_ticker(str(record.ticker))}.parquet"
        )
        stock = (
            _bar_values(_session(stock_path, event_date)) if stock_path.exists() else None
        )
        spy = spy_cache[event_date]
        valid = stock is not None and spy is not None
        direction = 1.0 if record.action == "add" else -1.0
        signed_relative_hour = np.nan
        multiplier = 0.0
        raw_underlying_return = np.nan
        if valid:
            stock_hour = stock["feature_close"] / stock["hour_open"] - 1.0
            spy_hour = spy["feature_close"] / spy["hour_open"] - 1.0
            signed_relative_hour = direction * (stock_hour - spy_hour)
            down_to_small = (
                record.action == "delete" and record.change_type == "down_to_small"
            )
            multiplier = (
                retention
                if down_to_small and signed_relative_hour <= 0
                else 1.0
            )
            raw_underlying_return = stock["exit_close"] / stock["entry_open"] - 1.0
        rows.append(
            {
                "effective_close_date": event_date,
                "ticker": record.ticker,
                "action": record.action,
                "change_type": record.change_type,
                "raw_bar_complete": valid,
                "raw_hour_open": stock["hour_open"] if stock else np.nan,
                "raw_feature_close": stock["feature_close"] if stock else np.nan,
                "raw_entry_open": stock["entry_open"] if stock else np.nan,
                "raw_exit_close": stock["exit_close"] if stock else np.nan,
                "manual_signed_relative_hour": signed_relative_hour,
                "manual_multiplier": multiplier,
                "base_target_weight": float(record.target_weight),
                "provisional_weight": float(record.target_weight) * multiplier,
                "raw_underlying_return": raw_underlying_return,
                "ledger_entry_open": float(record.entry_open_proxy),
                "ledger_exit_close": float(record.exit_close_proxy),
                "ledger_underlying_return": float(record.underlying_return),
            }
        )
    audit = pd.DataFrame(rows)
    audit["manual_filtered_weight"] = audit["provisional_weight"]
    for _, index in audit.groupby("effective_close_date", sort=True).groups.items():
        index = list(index)
        weight = audit.loc[index, "manual_filtered_weight"]
        long_total = float(weight.clip(lower=0).sum())
        short_total = float(-weight.clip(upper=0).sum())
        balanced = min(long_total, short_total)
        if balanced <= 0:
            audit.loc[index, "manual_filtered_weight"] = 0.0
            continue
        long_index = [row for row in index if audit.at[row, "manual_filtered_weight"] > 0]
        short_index = [row for row in index if audit.at[row, "manual_filtered_weight"] < 0]
        audit.loc[long_index, "manual_filtered_weight"] *= balanced / long_total
        audit.loc[short_index, "manual_filtered_weight"] *= balanced / short_total
    selected_weight = selected["filtered_target_weight"]
    audit["ledger_filtered_weight"] = [
        float(selected_weight.loc[(row.effective_close_date, row.ticker)])
        for row in audit.itertuples(index=False)
    ]
    audit["manual_gross_contribution"] = (
        audit["manual_filtered_weight"] * audit["raw_underlying_return"]
    ).fillna(0.0)
    audit["manual_cost_contribution"] = (
        audit["manual_filtered_weight"].abs() * args.cost_bps / 10_000.0
    )
    audit["manual_net_contribution"] = (
        audit["manual_gross_contribution"] - audit["manual_cost_contribution"]
    )
    audit["weight_error"] = (
        audit["manual_filtered_weight"] - audit["ledger_filtered_weight"]
    )
    audit["entry_error"] = audit["raw_entry_open"] - audit["ledger_entry_open"]
    audit["exit_error"] = audit["raw_exit_close"] - audit["ledger_exit_close"]
    audit["return_error"] = (
        audit["raw_underlying_return"] - audit["ledger_underlying_return"]
    )

    date_audit = (
        audit.groupby("effective_close_date", as_index=False)
        .agg(
            manual_gross_return=("manual_gross_contribution", "sum"),
            manual_cost=("manual_cost_contribution", "sum"),
            manual_net_return=("manual_net_contribution", "sum"),
            manual_gross_exposure=("manual_filtered_weight", lambda value: value.abs().sum()),
            max_abs_weight_error=("weight_error", lambda value: value.abs().max()),
            max_abs_return_error=("return_error", lambda value: value.abs().max()),
        )
    )
    reported = date_portfolios.loc[
        date_portfolios["filter_rule"].eq(rule)
        & date_portfolios["cost_bps"].eq(args.cost_bps)
    ][
        [
            "effective_close_date",
            "gross_return",
            "net_return",
            "gross_exposure",
        ]
    ].copy()
    reported["modeled_cost"] = reported["gross_return"] - reported["net_return"]
    date_audit = date_audit.merge(reported, on="effective_close_date", how="outer")
    for manual, reported_column, error in (
        ("manual_gross_return", "gross_return", "gross_return_error"),
        ("manual_cost", "modeled_cost", "cost_error"),
        ("manual_net_return", "net_return", "net_return_error"),
        ("manual_gross_exposure", "gross_exposure", "gross_exposure_error"),
    ):
        date_audit[error] = date_audit[manual] - date_audit[reported_column]

    active = audit.loc[audit["manual_filtered_weight"].ne(0)].copy()
    active["manual_signed_net"] = active["manual_net_contribution"]
    sample_index: list[int] = []
    choices = [
        active.loc[active["action"].eq("add")]["manual_signed_net"].idxmax(),
        active.loc[active["action"].eq("delete")]["manual_signed_net"].idxmax(),
        active["manual_signed_net"].idxmin(),
        audit.loc[
            audit["change_type"].eq("down_to_small")
            & audit["manual_multiplier"].eq(retention)
        ].index[0],
    ]
    for index in choices:
        if int(index) not in sample_index:
            sample_index.append(int(index))
    samples = audit.loc[sample_index].copy()
    samples.insert(0, "audit_sample_reason", [
        "long winner",
        "short winner",
        "largest contribution loss",
        "attenuated down-to-small",
    ][: len(samples)])

    error_columns = [
        "weight_error",
        "entry_error",
        "exit_error",
        "return_error",
    ]
    row_error = float(audit[error_columns].abs().max().max())
    date_error_columns = [
        "gross_return_error",
        "cost_error",
        "net_return_error",
        "gross_exposure_error",
    ]
    date_error = float(date_audit[date_error_columns].abs().max().max())
    status = "pass" if max(row_error, date_error) <= args.tolerance else "fail"
    summary = {
        "status": status,
        "selected_rule": rule,
        "cost_bps": args.cost_bps,
        "security_rows": len(audit),
        "review_dates": len(date_audit),
        "raw_bar_incomplete_rows": int((~audit["raw_bar_complete"]).sum()),
        "maximum_security_reconstruction_error": row_error,
        "maximum_date_reconciliation_error": date_error,
        "tolerance": args.tolerance,
        "execution_status": "minute_bar_proxy_not_broker_fill",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "state_optimization_manual_security_reconstruction.csv", index=False)
    date_audit.to_csv(args.output_dir / "state_optimization_manual_date_reconciliation.csv", index=False)
    samples.to_csv(args.output_dir / "state_optimization_manual_trade_samples.csv", index=False)
    (args.output_dir / "state_optimization_manual_audit.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
