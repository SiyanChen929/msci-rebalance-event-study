#!/usr/bin/env python3
"""Test the final-11-minute long-vol premise without claiming option fills."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from msci_rebalance.backtest import moving_block_bootstrap_mean, performance_metrics
from msci_rebalance.volatility import (
    breakout_proxy,
    hypothetical_straddle_return,
    window_statistics,
)


PRE_EVENT_CONTROL_SESSION_COUNT = 4
PATH_METRICS = (
    "absolute_terminal_move",
    "sqrt_realized_variance_11m",
    "high_low_log_range",
    "max_absolute_excursion",
)
BARRIERS_BPS = (15.0, 25.0, 50.0)
BREAKOUT_COSTS_BPS = (5.0, 10.0, 20.0)
MOVE_THRESHOLDS_BPS = (25.0, 50.0, 100.0, 200.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def session_dates(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def normalized_timestamp(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp


def summarize_paired_dates(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    usable = frame.dropna(subset=["event_value", "control_value", "excess_value"])
    if usable.empty:
        return {
            "metric": metric,
            "review_dates": 0,
            "controlled_security_events": 0,
        }
    excess = usable["excess_value"].to_numpy(float)
    ci_low, ci_high = moving_block_bootstrap_mean(excess)
    paired = stats.ttest_rel(usable["event_value"], usable["control_value"])
    sign = stats.binomtest(
        int((excess > 0).sum()),
        len(excess),
        0.5,
        alternative="greater",
    )
    return {
        "metric": metric,
        "review_dates": len(usable),
        "controlled_security_events": int(usable["controlled_names"].sum()),
        "event_mean_across_dates": float(usable["event_value"].mean()),
        "pre_event_control_mean_across_dates": float(
            usable["control_value"].mean()
        ),
        "mean_event_minus_pre_event_control": float(
            usable["excess_value"].mean()
        ),
        "median_event_minus_pre_event_control": float(
            usable["excess_value"].median()
        ),
        "positive_excess_date_rate": float((usable["excess_value"] > 0).mean()),
        "block_bootstrap_excess_ci_low": ci_low,
        "block_bootstrap_excess_ci_high": ci_high,
        "paired_t_stat": float(paired.statistic),
        "paired_t_two_sided_pvalue": float(paired.pvalue),
        "one_sided_sign_pvalue": float(sign.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv(args.ledger)
    ledger["effective_close_date"] = pd.to_datetime(ledger["effective_close_date"])
    security_rows: list[dict[str, object]] = []
    control_security_rows: list[dict[str, object]] = []
    breakout_rows: list[dict[str, object]] = []
    used_marketdata_paths: list[Path] = []

    for number, row in enumerate(ledger.itertuples(index=False), start=1):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        path = args.marketdata_dir / "minute" / (
            f"{event_date.date()}_{safe_ticker(str(row.ticker))}.parquet"
        )
        if not path.exists():
            continue
        used_marketdata_paths.append(path)
        minute = pd.read_parquet(path).copy()
        timestamp = normalized_timestamp(minute)
        minute["session_date"] = timestamp.dt.tz_localize(None).dt.normalize()
        minute["clock"] = timestamp.dt.strftime("%H:%M")
        dates = sorted(minute["session_date"].drop_duplicates())
        if event_date not in dates:
            continue
        session_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        stats_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        for target_date in dates:
            session = minute.loc[minute["session_date"].eq(target_date)]
            window = window_statistics(session)
            if window is not None:
                session_by_date[target_date] = session
                stats_by_date[target_date] = window
        if event_date not in stats_by_date:
            continue

        # Use the four nearest complete pre-event sessions. This avoids both
        # post-event reversal contamination and the 13:00 Black Friday close,
        # which has no 15:49--15:59 window.
        before = sorted(
            (date for date in stats_by_date if date < event_date), reverse=True
        )
        control_dates = before[:PRE_EVENT_CONTROL_SESSION_COUNT]
        event_session = session_by_date[event_date]
        event_stats = stats_by_date[event_date]
        control_sessions = [session_by_date[date] for date in control_dates]
        control_stats = [stats_by_date[date] for date in control_dates]
        controls_complete = len(control_stats) == PRE_EVENT_CONTROL_SESSION_COUNT
        record: dict[str, object] = {
            "effective_close_date": event_date,
            "ticker": row.ticker,
            "security_name": row.security_name,
            "action": row.action,
            "change_type": row.change_type,
            "controls_complete": controls_complete,
            "available_control_sessions": len(control_stats),
            "pre_event_control_session_dates": "|".join(
                str(date.date()) for date in control_dates
            ),
            "terminal_log_return": event_stats["terminal_log_return"],
        }
        for metric in PATH_METRICS:
            event_value = event_stats[metric]
            control_values = [item[metric] for item in control_stats]
            control_mean = float(np.mean(control_values)) if controls_complete else np.nan
            record[f"event_{metric}"] = event_value
            record[f"pre_event_control_{metric}"] = control_mean
            record[f"event_minus_pre_event_control_{metric}"] = (
                event_value - control_mean if controls_complete else np.nan
            )
        for threshold_bps in MOVE_THRESHOLDS_BPS:
            threshold = math.log1p(threshold_bps / 10_000.0)
            threshold_label = int(threshold_bps)
            event_hit = float(
                event_stats["absolute_terminal_move"] > threshold
            )
            control_hits = [
                float(item["absolute_terminal_move"] > threshold)
                for item in control_stats
            ]
            record[f"event_terminal_move_hit_{threshold_label}bps"] = event_hit
            record[
                f"pre_event_control_terminal_move_hit_rate_{threshold_label}bps"
            ] = float(np.mean(control_hits)) if controls_complete else np.nan
        security_rows.append(record)

        for control_date, item in zip(control_dates, control_stats, strict=True):
            control_security_rows.append(
                {
                    "effective_close_date": event_date,
                    "ticker": row.ticker,
                    "security_name": row.security_name,
                    "action": row.action,
                    "control_session_date": control_date,
                    "terminal_log_return": item["terminal_log_return"],
                }
            )

        for barrier_bps in BARRIERS_BPS:
            event_breakout = breakout_proxy(
                event_session,
                barrier_bps=barrier_bps,
                cost_bps=0.0,
            )
            if event_breakout is None:
                continue
            control_breakouts = [
                breakout_proxy(
                    session,
                    barrier_bps=barrier_bps,
                    cost_bps=0.0,
                )
                for session in control_sessions
            ]
            valid_controls = [item for item in control_breakouts if item is not None]
            complete_breakouts = (
                controls_complete
                and len(valid_controls) == PRE_EVENT_CONTROL_SESSION_COUNT
            )
            for cost_bps in BREAKOUT_COSTS_BPS:
                event_net_return = (
                    float(event_breakout["gross_return"]) - cost_bps / 10_000.0
                    if event_breakout["triggered"]
                    else 0.0
                )
                control_net_returns = [
                    (
                        float(item["gross_return"]) - cost_bps / 10_000.0
                        if item["triggered"]
                        else 0.0
                    )
                    for item in valid_controls
                ]
                breakout_rows.append(
                    {
                        "effective_close_date": event_date,
                        "ticker": row.ticker,
                        "security_name": row.security_name,
                        "action": row.action,
                        "barrier_bps": barrier_bps,
                        "cost_bps": cost_bps,
                        "event_triggered": event_breakout["triggered"],
                        "event_direction": event_breakout["direction"],
                        "event_ambiguous_same_bar": event_breakout["ambiguous_same_bar"],
                        "event_trigger_clock": event_breakout["trigger_clock"],
                        "event_fill_clock": event_breakout["fill_clock"],
                        "event_gross_return": event_breakout["gross_return"],
                        "event_net_return": event_net_return,
                        "controls_complete": complete_breakouts,
                        "pre_event_control_trigger_rate": (
                            float(np.mean([item["triggered"] for item in valid_controls]))
                            if complete_breakouts
                            else np.nan
                        ),
                        "pre_event_control_net_return": (
                            float(np.mean(control_net_returns))
                            if complete_breakouts
                            else np.nan
                        ),
                        "event_minus_pre_event_control_net_return": (
                            event_net_return - float(np.mean(control_net_returns))
                            if complete_breakouts
                            else np.nan
                        ),
                    }
                )
        if number % 75 == 0:
            print(f"processed={number}/{len(ledger)}", flush=True)

    security = pd.DataFrame(security_rows)
    control_security = pd.DataFrame(control_security_rows)
    breakouts = pd.DataFrame(breakout_rows)
    security.to_csv(args.output_dir / "volatility_security_diagnostics.csv", index=False)
    control_security.to_csv(
        args.output_dir / "volatility_pre_event_control_windows.csv", index=False
    )
    breakouts.to_csv(args.output_dir / "breakout_proxy_security_trades.csv", index=False)

    date_frames: list[pd.DataFrame] = []
    paired_summaries: list[dict[str, object]] = []
    complete = security.loc[security["controls_complete"]].copy()
    for metric in PATH_METRICS:
        date_frame = (
            complete.groupby("effective_close_date")
            .agg(
                controlled_names=("ticker", "count"),
                event_value=(f"event_{metric}", "mean"),
                control_value=(f"pre_event_control_{metric}", "mean"),
                excess_value=(f"event_minus_pre_event_control_{metric}", "mean"),
            )
            .reset_index()
        )
        date_frame.insert(1, "metric", metric)
        date_frames.append(date_frame)
        summary = summarize_paired_dates(date_frame, metric)
        all_event_dates = security.groupby("effective_close_date")[
            f"event_{metric}"
        ].mean()
        summary.update(
            {
                "all_event_review_dates": int(len(all_event_dates)),
                "all_event_security_events": int(len(security)),
                "all_event_mean_across_dates": float(all_event_dates.mean()),
                "all_event_median_across_dates": float(all_event_dates.median()),
            }
        )
        paired_summaries.append(summary)
    volatility_dates = pd.concat(date_frames, ignore_index=True)
    volatility_dates.to_csv(args.output_dir / "volatility_date_diagnostics.csv", index=False)
    pd.DataFrame(paired_summaries).to_csv(
        args.output_dir / "volatility_diagnostic_metrics.csv", index=False
    )

    threshold_date_frames: list[pd.DataFrame] = []
    threshold_summaries: list[dict[str, object]] = []
    for threshold_bps in MOVE_THRESHOLDS_BPS:
        threshold_label = int(threshold_bps)
        threshold_frame = complete.assign(
            event_value=complete[f"event_terminal_move_hit_{threshold_label}bps"],
            control_value=complete[
                f"pre_event_control_terminal_move_hit_rate_{threshold_label}bps"
            ],
        )
        threshold_frame["excess_value"] = (
            threshold_frame["event_value"] - threshold_frame["control_value"]
        )
        date_frame = (
            threshold_frame.groupby("effective_close_date")
            .agg(
                controlled_names=("ticker", "count"),
                event_value=("event_value", "mean"),
                control_value=("control_value", "mean"),
                excess_value=("excess_value", "mean"),
            )
            .reset_index()
        )
        date_frame.insert(1, "threshold_bps", threshold_bps)
        threshold_date_frames.append(date_frame)
        threshold_summary = {
            "threshold_bps": threshold_bps,
            **summarize_paired_dates(date_frame, "terminal_move_hit_rate"),
        }
        all_event_dates = security.groupby("effective_close_date")[
            f"event_terminal_move_hit_{threshold_label}bps"
        ].mean()
        threshold_summary.update(
            {
                "all_event_review_dates": int(len(all_event_dates)),
                "all_event_security_events": int(len(security)),
                "all_event_mean_across_dates": float(all_event_dates.mean()),
            }
        )
        threshold_summaries.append(threshold_summary)
    pd.concat(threshold_date_frames, ignore_index=True).to_csv(
        args.output_dir / "volatility_threshold_date_rates.csv", index=False
    )
    pd.DataFrame(threshold_summaries).to_csv(
        args.output_dir / "volatility_threshold_metrics.csv", index=False
    )

    breakout_date = (
        breakouts.groupby(["barrier_bps", "cost_bps", "effective_close_date"])
        .agg(
            eligible_names=("ticker", "count"),
            event_trigger_rate=("event_triggered", "mean"),
            event_ambiguous_rate=("event_ambiguous_same_bar", "mean"),
            event_gross_return=("event_gross_return", "mean"),
            event_net_return=("event_net_return", "mean"),
            complete_control_names=("controls_complete", "sum"),
            pre_event_control_trigger_rate=("pre_event_control_trigger_rate", "mean"),
            pre_event_control_net_return=("pre_event_control_net_return", "mean"),
            event_minus_pre_event_control_net_return=(
                "event_minus_pre_event_control_net_return",
                "mean",
            ),
        )
        .reset_index()
    )
    breakout_date.to_csv(args.output_dir / "breakout_proxy_date_portfolio.csv", index=False)
    breakout_metrics: list[dict[str, object]] = []
    for (barrier_bps, cost_bps), group in breakout_date.groupby(["barrier_bps", "cost_bps"]):
        event_portfolio = group.rename(
            columns={
                "event_net_return": "net_return",
                "event_gross_return": "gross_return",
            }
        ).copy()
        event_portfolio["gross_exposure"] = event_portfolio["event_trigger_rate"]
        event_metrics = performance_metrics(event_portfolio)
        breakout_metrics.append(
            {
                "scope": "event_minute_bar_breakout_payoff_proxy",
                "barrier_bps": barrier_bps,
                "cost_bps": cost_bps,
                "mean_trigger_rate": float(group["event_trigger_rate"].mean()),
                "mean_ambiguous_rate": float(group["event_ambiguous_rate"].mean()),
                **event_metrics,
            }
        )
        excess = group.dropna(
            subset=["event_minus_pre_event_control_net_return"]
        ).copy()
        excess_values = excess[
            "event_minus_pre_event_control_net_return"
        ].to_numpy(float)
        ci_low, ci_high = moving_block_bootstrap_mean(excess_values)
        t_result = stats.ttest_1samp(excess_values, 0.0, alternative="greater")
        sign_result = stats.binomtest(
            int((excess_values > 0).sum()),
            len(excess_values),
            0.5,
            alternative="greater",
        )
        breakout_metrics.append(
            {
                "scope": "event_minus_pre_event_control_payoff_proxy",
                "barrier_bps": barrier_bps,
                "cost_bps": cost_bps,
                "event_dates": int(len(excess_values)),
                "mean_trigger_rate": float(excess["event_trigger_rate"].mean()),
                "mean_ambiguous_rate": float(excess["event_ambiguous_rate"].mean()),
                "mean_return_per_event": float(excess_values.mean()),
                "median_return_per_event": float(np.median(excess_values)),
                "hit_rate": float((excess_values > 0).mean()),
                "block_bootstrap_mean_ci_low": ci_low,
                "block_bootstrap_mean_ci_high": ci_high,
                "t_stat": float(t_result.statistic),
                "one_sided_t_pvalue": float(t_result.pvalue),
                "sign_test_pvalue": float(sign_result.pvalue),
            }
        )
    pd.DataFrame(breakout_metrics).to_csv(
        args.output_dir / "breakout_proxy_metrics.csv", index=False
    )

    # Transparent terminal-spot Black--Scholes sensitivity grid. It uses the
    # observed stock move but hypothetical IV, IV change, DTE and friction. It
    # is not an executable option backtest, so no CAGR, Sharpe or compounded
    # return is calculated.
    straddle_date_frames: list[pd.DataFrame] = []
    straddle_metrics: list[dict[str, object]] = []
    complete_keys = complete[["effective_close_date", "ticker"]].drop_duplicates()
    complete_control_security = control_security.merge(
        complete_keys,
        on=["effective_close_date", "ticker"],
        how="inner",
    )
    for dte in (2, 7, 14, 30):
        for entry_iv in (0.30, 0.50, 0.80):
            for iv_change in (-0.10, -0.05, 0.0, 0.05):
                event_scenario = security[
                    ["effective_close_date", "ticker", "terminal_log_return"]
                ].copy()
                event_scenario["base_model_return_on_premium"] = event_scenario[
                    "terminal_log_return"
                ].map(
                    lambda value: hypothetical_straddle_return(
                        value,
                        dte=dte,
                        entry_volatility=entry_iv,
                        exit_volatility_change=iv_change,
                        friction_fraction_of_premium=0.0,
                    )
                )
                all_event_date = (
                    event_scenario.groupby("effective_close_date")
                    .agg(
                        all_event_security_events=("ticker", "count"),
                        all_event_base_model_return_on_premium=(
                            "base_model_return_on_premium",
                            "mean",
                        ),
                    )
                    .reset_index()
                )
                controlled_event_date = (
                    event_scenario.merge(
                        complete_keys,
                        on=["effective_close_date", "ticker"],
                        how="inner",
                    )
                    .groupby("effective_close_date")
                    .agg(
                        controlled_security_events=("ticker", "count"),
                        controlled_event_base_model_return_on_premium=(
                            "base_model_return_on_premium",
                            "mean",
                        ),
                    )
                    .reset_index()
                )
                control_scenario = complete_control_security.copy()
                control_scenario["base_model_return_on_premium"] = control_scenario[
                    "terminal_log_return"
                ].map(
                    lambda value: hypothetical_straddle_return(
                        value,
                        dte=dte,
                        entry_volatility=entry_iv,
                        exit_volatility_change=iv_change,
                        friction_fraction_of_premium=0.0,
                    )
                )
                pre_event_control_date = (
                    control_scenario.groupby(
                        ["effective_close_date", "ticker"], as_index=False
                    )["base_model_return_on_premium"]
                    .mean()
                    .groupby("effective_close_date", as_index=False)
                    .agg(
                        pre_event_control_base_model_return_on_premium=(
                            "base_model_return_on_premium",
                            "mean",
                        )
                    )
                )
                base_date = all_event_date.merge(
                    controlled_event_date,
                    on="effective_close_date",
                    how="left",
                ).merge(
                    pre_event_control_date,
                    on="effective_close_date",
                    how="left",
                )
                for friction in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
                    date_frame = base_date.copy()
                    date_frame["all_event_model_return_on_premium"] = (
                        date_frame["all_event_base_model_return_on_premium"]
                        - friction
                    )
                    date_frame["controlled_event_model_return_on_premium"] = (
                        date_frame[
                            "controlled_event_base_model_return_on_premium"
                        ]
                        - friction
                    )
                    date_frame["pre_event_control_model_return_on_premium"] = (
                        date_frame[
                            "pre_event_control_base_model_return_on_premium"
                        ]
                        - friction
                    )
                    date_frame[
                        "event_minus_pre_event_control_model_return_on_premium"
                    ] = (
                        date_frame["controlled_event_model_return_on_premium"]
                        - date_frame["pre_event_control_model_return_on_premium"]
                    )
                    date_frame = date_frame.drop(
                        columns=[
                            "all_event_base_model_return_on_premium",
                            "controlled_event_base_model_return_on_premium",
                            "pre_event_control_base_model_return_on_premium",
                        ]
                    )
                    date_frame.insert(0, "dte", dte)
                    date_frame.insert(1, "entry_iv", entry_iv)
                    date_frame.insert(2, "iv_change", iv_change)
                    date_frame.insert(3, "friction_fraction_of_premium", friction)
                    date_frame.insert(4, "delta_hedged", True)
                    straddle_date_frames.append(date_frame)
                    all_event_values = date_frame[
                        "all_event_model_return_on_premium"
                    ].dropna().to_numpy(float)
                    controlled_dates = date_frame.dropna(
                        subset=[
                            "controlled_event_model_return_on_premium",
                            "pre_event_control_model_return_on_premium",
                            "event_minus_pre_event_control_model_return_on_premium",
                        ]
                    )
                    excess_values = controlled_dates[
                        "event_minus_pre_event_control_model_return_on_premium"
                    ].to_numpy(float)
                    ci_low, ci_high = moving_block_bootstrap_mean(excess_values)
                    straddle_metrics.append(
                        {
                            "dte": dte,
                            "entry_iv": entry_iv,
                            "iv_change": iv_change,
                            "friction_fraction_of_premium": friction,
                            "delta_hedged": True,
                            "all_event_review_dates": int(len(all_event_values)),
                            "controlled_review_dates": int(len(controlled_dates)),
                            "mean_all_event_model_return_on_premium": float(
                                all_event_values.mean()
                            ),
                            "median_all_event_model_return_on_premium": float(
                                np.median(all_event_values)
                            ),
                            "all_event_positive_date_rate": float(
                                (all_event_values > 0).mean()
                            ),
                            "all_event_p05": float(np.quantile(all_event_values, 0.05)),
                            "all_event_p95": float(np.quantile(all_event_values, 0.95)),
                            "mean_controlled_event_model_return_on_premium": float(
                                controlled_dates[
                                    "controlled_event_model_return_on_premium"
                                ].mean()
                            ),
                            "mean_pre_event_control_model_return_on_premium": float(
                                controlled_dates[
                                    "pre_event_control_model_return_on_premium"
                                ].mean()
                            ),
                            "mean_event_minus_pre_event_control": float(
                                excess_values.mean()
                            ),
                            "median_event_minus_pre_event_control": float(
                                np.median(excess_values)
                            ),
                            "positive_excess_date_rate": float(
                                (excess_values > 0).mean()
                            ),
                            "block_bootstrap_excess_ci_low": ci_low,
                            "block_bootstrap_excess_ci_high": ci_high,
                        }
                    )
    pd.concat(straddle_date_frames, ignore_index=True).to_csv(
        args.output_dir / "hypothetical_straddle_date_scenarios.csv", index=False
    )
    pd.DataFrame(straddle_metrics).to_csv(
        args.output_dir / "hypothetical_straddle_scenario_metrics.csv", index=False
    )

    cache_digest = hashlib.sha256()
    for path in sorted(set(used_marketdata_paths)):
        stat = path.stat()
        cache_digest.update(
            f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    project_root = Path(__file__).resolve().parents[1]
    volatility_manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256_file(args.ledger),
        "diagnostics_code_sha256": sha256_file(Path(__file__).resolve()),
        "volatility_module_sha256": sha256_file(
            project_root / "src" / "msci_rebalance" / "volatility.py"
        ),
        "marketdata_file_count": len(set(used_marketdata_paths)),
        "marketdata_stat_fingerprint": cache_digest.hexdigest(),
        "event_security_windows": len(security),
        "complete_pre_event_control_security_events": len(complete),
        "event_review_dates": int(security["effective_close_date"].nunique()),
        "controlled_review_dates": int(
            volatility_dates["effective_close_date"].nunique()
        ),
        "path_metrics": list(PATH_METRICS),
        "move_thresholds_bps": list(MOVE_THRESHOLDS_BPS),
        "breakout_barriers_bps": list(BARRIERS_BPS),
        "breakout_friction_bps": list(BREAKOUT_COSTS_BPS),
        "pre_event_control_sessions": PRE_EVENT_CONTROL_SESSION_COUNT,
        "option_pnl_status": "not tested; historical intraday OPRA unavailable",
        "terminal_spot_model_scenarios": len(straddle_metrics),
    }
    (args.output_dir / "volatility_run_manifest.json").write_text(
        json.dumps(volatility_manifest, indent=2), encoding="utf-8"
    )

    print(
        f"event_security_windows={len(security)} "
        f"complete_pre_event_controls={len(complete)} "
        f"review_dates={security['effective_close_date'].nunique()} "
        f"breakout_rows={len(breakouts)} model_scenarios={len(straddle_metrics)}"
    )


if __name__ == "__main__":
    main()
