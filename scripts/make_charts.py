#!/usr/bin/env python3
"""Create report charts from the audited backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PRIMARY = "flow_1549_to_daily_close_proxy"
TOP10_PRIMARY_COST_BPS = 0.0
TOP10_SENSITIVITY_COST_BPS = 10.0


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    args = parser.parse_args()
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.output_dir / "portfolio_equity_curves.csv")
    curves["effective_close_date"] = pd.to_datetime(curves["effective_close_date"])
    primary = curves.loc[curves["strategy"].eq(PRIMARY)].copy()
    primary["gross_equity"] = (1.0 + primary["gross_return"]).cumprod()

    fig, (ax, dd_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax.plot(primary["effective_close_date"], primary["gross_equity"], label="Gross (0 bp)", linewidth=2)
    ax.plot(primary["effective_close_date"], primary["equity"], label="Net (10 bp all-in)", linewidth=2)
    ax.axhline(1.0, color="0.45", linewidth=1)
    ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle="--", linewidth=1)
    ax.text(pd.Timestamp("2023-06-20"), ax.get_ylim()[1], "Full NOS/FIF QCIR regime", va="top", fontsize=9)
    ax.set_title("MSCI USA Standard: 15:49 to daily-close proxy portfolio")
    ax.set_ylabel("Equity (start = 1.00)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    dd_ax.fill_between(
        primary["effective_close_date"],
        primary["drawdown"] * 100,
        0,
        alpha=0.45,
    )
    dd_ax.set_ylabel("Drawdown, %")
    dd_ax.set_xlabel("Implementation close")
    dd_ax.grid(axis="y", alpha=0.25)
    dd_ax.xaxis.set_major_locator(mdates.YearLocator())
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, chart_dir / "equity_curve.png")

    colors = ["#277a53" if value >= 0 else "#b54a4a" for value in primary["net_return"]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(primary["effective_close_date"], primary["net_return"] * 10_000, width=35, color=colors)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle="--", linewidth=1)
    ax.set_title("Net return by independent review date (10 bp all-in)")
    ax.set_ylabel("Basis points")
    ax.set_xlabel("Implementation close")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    last_bar = bars[-1]
    ax.annotate(
        "Aug 2026: +18.0 bp basket\n(SNDK alone: +4.30%)",
        xy=(last_bar.get_x() + last_bar.get_width() / 2, last_bar.get_height()),
        xytext=(-115, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.35"},
        fontsize=9,
    )
    _save(fig, chart_dir / "event_returns.png")

    sensitivity = pd.read_csv(args.output_dir / "cost_sensitivity.csv")
    sensitivity = sensitivity.loc[sensitivity["strategy"].eq(PRIMARY)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(sensitivity["cost_bps"], sensitivity["total_return"] * 100, marker="o", linewidth=2)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axvline(0.653, color="0.45", linestyle="--", linewidth=1)
    ax.text(1.4, ax.get_ylim()[1] * 0.88, "Estimated break-even ≈ 0.65 bp", fontsize=9)
    ax.set_title("Cost sensitivity of the 15:49 strategy")
    ax.set_xlabel("Assumed all-in round-trip cost (bp of gross exposure)")
    ax.set_ylabel("Cumulative return, 2020–2026 (%)")
    ax.grid(alpha=0.25)
    _save(fig, chart_dir / "cost_sensitivity.png")

    minute = pd.read_parquet(args.marketdata_dir / "minute" / "2026-08-31_SNDK.parquet")
    minute["clock"] = pd.to_datetime(minute["timestamp"]).dt.strftime("%H:%M")
    tail = minute.loc[minute["clock"].between("15:30", "15:59")].copy()
    case = pd.read_json(args.output_dir / "sndk_case_study.json", typ="series")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    volume_ax = ax.twinx()
    volume_ax.bar(tail["clock"], tail["volume"] / 1_000, alpha=0.22, color="#5279a5", label="Minute volume")
    ax.plot(tail["clock"], tail["close"], marker="o", markersize=3, linewidth=2, color="#9d4b38", label="Continuous close")
    ax.axhline(float(case["daily_close_proxy"]), color="#277a53", linestyle="--", linewidth=1.5, label="Daily close proxy")
    entry_index = tail.index[tail["clock"].eq("15:49")][0]
    ax.scatter("15:49", float(minute.loc[entry_index, "open"]), color="#222222", zorder=4, label="15:49 entry")
    ax.set_title("SNDK on the August 31, 2026 implementation close")
    ax.set_xlabel("Eastern time")
    ax.set_ylabel("Price")
    volume_ax.set_ylabel("Minute volume (thousand shares)")
    tick_positions = list(range(0, len(tail), 5)) + [len(tail) - 1]
    ax.set_xticks(tick_positions, [tail["clock"].iloc[i] for i in tick_positions], rotation=0)
    ax.grid(axis="y", alpha=0.2)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, frameon=False, loc="upper left")
    _save(fig, chart_dir / "sndk_case.png")

    strategies = {
        "flow_1549_to_daily_close_proxy": "15:49 → close proxy",
        "auction_reversal_to_next_open": "Close → next open reversal",
        "announcement_to_implementation": "Announcement → close",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for key, label in strategies.items():
        frame = curves.loc[curves["strategy"].eq(key)]
        target = axes[1] if key == "announcement_to_implementation" else axes[0]
        target.plot(frame["effective_close_date"], frame["equity"], linewidth=2, label=label)
    for axis in axes:
        axis.axhline(1.0, color="0.45", linewidth=1)
        axis.grid(axis="y", alpha=0.25)
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.legend(frameon=False)
        axis.set_xlabel("Implementation close")
    axes[0].set_title("Implementation-window hypotheses")
    axes[1].set_title("Announcement-drift research candidate")
    axes[0].set_ylabel("Net equity (start = 1.00)")
    _save(fig, chart_dir / "strategy_comparison.png")

    # Non-overlapping close-window anatomy.  These are descriptive one-minute
    # bars; the daily close is a provider proxy rather than a verified auction
    # print.
    windows = pd.read_csv(args.output_dir / "window_decomposition_metrics.csv")
    window_order = [
        "15:49_open_to_15:50_open",
        "15:50_open_to_15:59_close",
        "15:59_close_to_daily_close_proxy",
        "15:49_open_to_daily_close_proxy",
    ]
    window_labels = ["15:49→15:50", "15:50→15:59", "15:59→close proxy", "15:49→close proxy"]
    gross_windows = windows.loc[windows["cost_bps"].eq(0.0)].set_index("window").loc[window_order]
    estimates = gross_windows["mean_return_per_event"].to_numpy() * 10_000
    ci_low = gross_windows["block_bootstrap_mean_ci_low"].to_numpy() * 10_000
    ci_high = gross_windows["block_bootstrap_mean_ci_high"].to_numpy() * 10_000
    y = np.arange(len(window_order))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(y, estimates, color=["#277a53" if value >= 0 else "#b54a4a" for value in estimates])
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - ci_low, ci_high - estimates]),
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
    )
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y, window_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Gross mean return per review date (bp); 95% four-review block-bootstrap CI")
    ax.set_title("Close-window decomposition: the positive component is concentrated before 15:50")
    ax.grid(axis="x", alpha=0.22)
    for index, value in enumerate(estimates):
        ax.text(value + (0.18 if value >= 0 else -0.18), index, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    _save(fig, chart_dir / "window_decomposition.png")

    pairs = pd.read_csv(args.output_dir / "matched_pairs_portfolio.csv")
    pairs["effective_close_date"] = pd.to_datetime(pairs["effective_close_date"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(primary["effective_close_date"], primary["equity"], linewidth=2, label="Date-balanced public-list basket")
    ax.plot(pairs["effective_close_date"], pairs["equity"], linewidth=2, label="Pre-announcement similarity-matched pairs")
    ax.axhline(1.0, color="0.4", linewidth=1)
    ax.set_title("Matching public additions to deletions does not rescue the 15:49 rule")
    ax.set_ylabel("Net equity at 10 bp (start = 1.00)")
    ax.set_xlabel("Implementation close")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    _save(fig, chart_dir / "matched_pairs_equity.png")

    volume_dates = pd.read_csv(args.output_dir / "volume_diagnostics_by_date.csv")
    volume_dates["effective_close_date"] = pd.to_datetime(volume_dates["effective_close_date"])
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    scatter = ax.scatter(
        volume_dates["median_daily_volume_ratio"],
        volume_dates["mean_signed_return"] * 10_000,
        c=volume_dates["effective_close_date"].dt.year,
        cmap="viridis",
        s=55,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Median event-day volume / trailing 20-day median (log scale)")
    ax.set_ylabel("Mean direction-signed security return, 15:49→close proxy (bp)")
    ax.set_title("Higher event-day volume is not sufficient for reliable signed alpha")
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Review year")
    aug = volume_dates.loc[volume_dates["effective_close_date"].eq(pd.Timestamp("2026-08-31"))]
    if not aug.empty:
        ax.annotate(
            "Aug 2026",
            xy=(aug.iloc[0]["median_daily_volume_ratio"], aug.iloc[0]["mean_signed_return"] * 10_000),
            xytext=(12, 15),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "0.35"},
            fontsize=9,
        )
    _save(fig, chart_dir / "volume_pressure.png")

    coverage = pd.read_csv(args.output_dir / "data_quality_coverage.csv")
    expected = int(coverage["expected_security_events"].sum())
    minute_usable = int(coverage["minute_entry_usable"].sum())
    close_usable = int(coverage["daily_close_proxy_usable"].sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    counts = [expected, minute_usable, close_usable]
    labels = ["Public-list events", "15:49 minute usable", "Daily-close proxy usable"]
    bars = ax.bar(labels, counts, color=["#5279a5", "#4d8f78", "#9b7653"])
    ax.set_ylim(0, expected * 1.12)
    ax.set_ylabel("Security events")
    ax.set_title("Data coverage waterfall")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + expected * 0.018, f"{value}\n({value / expected:.1%})", ha="center", fontsize=9)
    _save(fig, chart_dir / "coverage_waterfall.png")

    subgroup = pd.read_csv(args.output_dir / "subgroup_metrics.csv")
    subgroup = subgroup.loc[
        subgroup["sample"].eq("full")
        & subgroup["subgroup"].isin(
            ["size_migrations", "non_migration_entries_exits", "additions_only", "deletions_only"]
        )
    ].copy()
    subgroup_labels = {
        "size_migrations": "Standard↔Small migrations",
        "non_migration_entries_exits": "New entries / full exits",
        "additions_only": "Additions",
        "deletions_only": "Deletions (short-oriented)",
    }
    subgroup["label"] = subgroup["subgroup"].map(subgroup_labels)
    subgroup["estimate"] = subgroup["mean_return_per_event"] * 10_000
    subgroup["low"] = subgroup["block_bootstrap_mean_ci_low"] * 10_000
    subgroup["high"] = subgroup["block_bootstrap_mean_ci_high"] * 10_000
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sy = np.arange(len(subgroup))
    ax.errorbar(
        subgroup["estimate"], sy,
        xerr=np.vstack([subgroup["estimate"] - subgroup["low"], subgroup["high"] - subgroup["estimate"]]),
        fmt="o", color="#315f85", ecolor="#5279a5", capsize=4, markersize=7,
    )
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(sy, subgroup["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Net mean return per review date at 10 bp (bp); block-bootstrap CI")
    ax.set_title("No full-sample subgroup has a reliably positive 15:49 edge")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, chart_dir / "subgroup_forest.png")

    placebo = pd.read_csv(args.output_dir / "placebo_metrics.csv")
    paired_placebo = pd.read_csv(args.output_dir / "paired_placebo_metrics.csv").iloc[0]
    placebo_labels = [f"Session {int(offset):+d}" for offset in placebo["session_offset"]] + ["Event minus\nmatched-placebo mean"]
    placebo_est = (placebo["mean_return_per_event"] * 10_000).tolist() + [float(paired_placebo["mean_return_per_event"]) * 10_000]
    placebo_low = (placebo["block_bootstrap_mean_ci_low"] * 10_000).tolist() + [float(paired_placebo["block_bootstrap_mean_ci_low"]) * 10_000]
    placebo_high = (placebo["block_bootstrap_mean_ci_high"] * 10_000).tolist() + [float(paired_placebo["block_bootstrap_mean_ci_high"]) * 10_000]
    px = np.arange(len(placebo_labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.errorbar(
        px, placebo_est,
        yerr=np.vstack([np.array(placebo_est) - np.array(placebo_low), np.array(placebo_high) - np.array(placebo_est)]),
        fmt="o", color="#315f85", ecolor="#5279a5", capsize=4, markersize=7,
    )
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xticks(px, placebo_labels)
    ax.set_ylabel("Gross mean per review date (bp); block-bootstrap CI")
    ax.set_title("Matched-session controls do not isolate positive implementation alpha")
    ax.grid(axis="y", alpha=0.22)
    _save(fig, chart_dir / "placebo_comparison.png")

    walk = pd.read_csv(args.output_dir / "walk_forward_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    values = walk["test_total_return"] * 100
    bars = ax.bar(walk["fold_test_year"].astype(str), values, color=["#277a53" if value >= 0 else "#b54a4a" for value in values])
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xlabel("Chronological test year (all names from each review kept together)")
    ax.set_ylabel("Net fold return (%)")
    ax.set_title("Anchored walk-forward results are regime-dependent")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.08 if value >= 0 else -0.08), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    _save(fig, chart_dir / "walk_forward_folds.png")

    event_time = pd.read_csv(args.output_dir / "event_time_date_panel.csv")
    clock_order = [f"15:{minute:02d}" for minute in range(30, 60)]
    profile = event_time.groupby("clock").agg(
        mean_signed_return=("signed_return_from_1530_open", "mean"),
        sem_signed_return=("signed_return_from_1530_open", "sem"),
        median_relative_volume=("median_minute_volume_vs_prior_same_clock", "median"),
    ).reindex(clock_order)
    profile["ci"] = profile["sem_signed_return"] * 2.06
    x = np.arange(len(profile))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    volume_ax = ax.twinx()
    volume_ax.bar(x, profile["median_relative_volume"], color="#5279a5", alpha=0.16, width=0.85, label="Median volume / same-clock normal")
    ax.plot(x, profile["mean_signed_return"] * 10_000, color="#9d4b38", linewidth=2.2, label="Mean direction-signed cumulative return")
    ax.fill_between(
        x,
        (profile["mean_signed_return"] - profile["ci"]) * 10_000,
        (profile["mean_signed_return"] + profile["ci"]) * 10_000,
        color="#9d4b38",
        alpha=0.15,
    )
    for label, color in [("15:49", "0.35"), ("15:50", "#b54a4a"), ("15:55", "#6f4c8e")]:
        ax.axvline(clock_order.index(label), color=color, linestyle="--", linewidth=1)
    ax.axhline(0, color="0.4", linewidth=1)
    ticks = [0, 5, 10, 15, 20, 25, 29]
    ax.set_xticks(ticks, [clock_order[i] for i in ticks])
    ax.set_xlabel("Eastern time (minute-bar close)")
    ax.set_ylabel("Date-balanced signed return from 15:30 open (bp)")
    volume_ax.set_ylabel("Median relative minute volume (× normal same-clock minute)")
    ax.set_title("Event-time anatomy: repricing and closing-volume acceleration")
    ax.grid(axis="y", alpha=0.2)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, frameon=False, loc="upper left")
    _save(fig, chart_dir / "event_time_profile.png")

    heat = event_time.pivot(index="effective_close_date", columns="clock", values="signed_bar_return").reindex(columns=clock_order)
    heat.index = pd.to_datetime(heat.index)
    heat = heat.sort_index()
    limit = float(np.nanquantile(np.abs(heat.to_numpy() * 10_000), 0.95))
    fig, ax = plt.subplots(figsize=(12, 7.0))
    image_handle = ax.imshow(heat.to_numpy() * 10_000, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    xticks = [0, 10, 20, 25, 29]
    ax.set_xticks(xticks, [clock_order[i] for i in xticks])
    ystep = max(1, len(heat) // 9)
    yticks = list(range(0, len(heat), ystep))
    ax.set_yticks(yticks, [heat.index[i].strftime("%Y-%m-%d") for i in yticks])
    ax.axvline(19.5, color="black", linewidth=1.2)
    ax.set_xlabel("Eastern time (minute bar)")
    ax.set_ylabel("Implementation close")
    ax.set_title("Review-date × minute signed-return heatmap")
    colorbar = fig.colorbar(image_handle, ax=ax, pad=0.015)
    colorbar.set_label("Direction-signed minute return (bp; clipped at 95th percentile)")
    _save(fig, chart_dir / "event_time_heatmap.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 14
    ax.bar(primary["effective_close_date"] - pd.to_timedelta(width / 2, unit="D"), primary["add_leg_return"] * 10_000, width=width, label="Additions, long-oriented", color="#277a53", alpha=0.78)
    ax.bar(primary["effective_close_date"] + pd.to_timedelta(width / 2, unit="D"), primary["delete_leg_return"] * 10_000, width=width, label="Deletions, short-oriented", color="#b54a4a", alpha=0.78)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_title("Long and short legs vary sharply across independent review dates")
    ax.set_ylabel("Gross leg return, 15:49→close proxy (bp)")
    ax.set_xlabel("Implementation close")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    _save(fig, chart_dir / "leg_attribution.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    event_bp = primary["net_return"] * 10_000
    ax.hist(event_bp, bins=10, color="#5279a5", alpha=0.78, edgecolor="white")
    ax.axvline(event_bp.mean(), color="#b54a4a", linewidth=2, label=f"Mean {event_bp.mean():+.2f} bp")
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_title("Distribution of net review-date returns")
    ax.set_xlabel("Net return at 10 bp (bp)")
    ax.set_ylabel("Review dates")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    _save(fig, chart_dir / "event_return_distribution.png")

    ledger = pd.read_csv(args.output_dir / "trade_ledger.csv")
    ledger["effective_close_date"] = pd.to_datetime(ledger["effective_close_date"])
    composition = ledger.groupby(["effective_close_date", "action"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(composition.index, composition.get("add", 0), width=35, label="Additions", color="#277a53")
    ax.bar(composition.index, composition.get("delete", 0), bottom=composition.get("add", 0), width=35, label="Deletions", color="#b54a4a")
    ax.set_title("Review composition is highly uneven across dates")
    ax.set_ylabel("Security events")
    ax.set_xlabel("Implementation close")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _save(fig, chart_dir / "review_composition.png")

    discrepancy = ledger.loc[ledger["close_discrepancy_bps"].notna(), "close_discrepancy_bps"]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    bound = float(np.nanquantile(np.abs(discrepancy), 0.99))
    clipped = discrepancy.clip(-bound, bound)
    ax.hist(clipped, bins=35, color="#9b7653", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_title("Daily-close proxy minus last continuous minute: data-quality distribution")
    ax.set_xlabel(f"Difference (bp; 1st/99th tails clipped at ±{bound:.1f} bp)")
    ax.set_ylabel("Security events")
    ax.grid(axis="y", alpha=0.18)
    _save(fig, chart_dir / "close_discrepancy_distribution.png")

    vol_metrics = pd.read_csv(args.output_dir / "volatility_diagnostic_metrics.csv")
    vol_order = [
        "absolute_terminal_move",
        "sqrt_realized_variance_11m",
        "high_low_log_range",
        "max_absolute_excursion",
    ]
    vol_labels = {
        "absolute_terminal_move": "Absolute terminal move",
        "sqrt_realized_variance_11m": "√ realized variance (11m)",
        "high_low_log_range": "High–low log range",
        "max_absolute_excursion": "Maximum excursion",
    }
    vol_plot = vol_metrics.set_index("metric").loc[vol_order]
    estimates = vol_plot["mean_event_minus_pre_event_control"].to_numpy() * 10_000
    lows = vol_plot["block_bootstrap_excess_ci_low"].to_numpy() * 10_000
    highs = vol_plot["block_bootstrap_excess_ci_high"].to_numpy() * 10_000
    y = np.arange(len(vol_plot))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color="#315f85",
        ecolor="#5279a5",
        capsize=4,
        markersize=7,
    )
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y, [vol_labels[item] for item in vol_order])
    ax.invert_yaxis()
    ax.set_xlabel("Event minus four pre-event sessions (bp); four-review block-bootstrap CI")
    ax.set_title("Implementation names exhibit substantially larger pre-close movement")
    ax.grid(axis="x", alpha=0.22)
    for index, value in enumerate(estimates):
        ax.text(value + 1.5, index, f"{value:+.1f}", va="center", fontsize=9)
    _save(fig, chart_dir / "volatility_pre_event_excess.png")

    threshold_metrics = pd.read_csv(args.output_dir / "volatility_threshold_metrics.csv")
    threshold_metrics = threshold_metrics.loc[threshold_metrics["threshold_bps"].le(100)].copy()
    x = np.arange(len(threshold_metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    event_rates = threshold_metrics["event_mean_across_dates"] * 100
    control_rates = threshold_metrics["pre_event_control_mean_across_dates"] * 100
    ax.bar(x - width / 2, event_rates, width, label="MSCI implementation", color="#315f85")
    ax.bar(x + width / 2, control_rates, width, label="Four pre-event sessions", color="#9b7653")
    ax.set_xticks(x, [f">{int(value)} bp" for value in threshold_metrics["threshold_bps"]])
    ax.set_ylabel("Date-balanced share of securities (%)")
    ax.set_title("Large terminal moves occur more often on implementation dates")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for index, row in threshold_metrics.reset_index(drop=True).iterrows():
        excess = row["mean_event_minus_pre_event_control"] * 100
        ax.text(index, max(event_rates.iloc[index], control_rates.iloc[index]) + 2.0, f"+{excess:.1f} pp", ha="center", fontsize=9)
    _save(fig, chart_dir / "volatility_threshold_rates.png")

    breakout = pd.read_csv(args.output_dir / "breakout_proxy_metrics.csv")
    breakout = breakout.loc[
        breakout["scope"].eq("event_minute_bar_breakout_payoff_proxy")
    ].copy()
    fig, ax = plt.subplots(figsize=(9.3, 4.9))
    offsets = np.linspace(-0.22, 0.22, breakout["cost_bps"].nunique())
    for offset, (cost_bps, group) in zip(
        offsets,
        breakout.groupby("cost_bps"),
        strict=True,
    ):
        group = group.sort_values("barrier_bps")
        means = group["mean_return_per_event"].to_numpy() * 10_000
        lows = group["block_bootstrap_mean_ci_low"].to_numpy() * 10_000
        highs = group["block_bootstrap_mean_ci_high"].to_numpy() * 10_000
        bx = np.arange(len(group)) + offset
        ax.errorbar(
            bx,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="o",
            capsize=3,
            label=f"{int(cost_bps)} bp friction",
        )
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xticks(np.arange(3), ["15 bp", "25 bp", "50 bp"])
    ax.set_xlabel("Two-sided barrier from 15:49 reference")
    ax.set_ylabel("Cash-funded minute-bar payoff proxy (bp/review)")
    ax.set_title("The underlying breakout proxy is not profitable outright")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, chart_dir / "volatility_breakout_proxy.png")

    model = pd.read_csv(args.output_dir / "hypothetical_straddle_scenario_metrics.csv")
    model = model.loc[model["dte"].eq(7) & model["entry_iv"].eq(0.50)].copy()
    heat = model.pivot(
        index="iv_change",
        columns="friction_fraction_of_premium",
        values="mean_all_event_model_return_on_premium",
    ).sort_index()
    values = heat.to_numpy() * 100
    limit = float(np.nanmax(np.abs(values)))
    fig, ax = plt.subplots(figsize=(10.0, 5.1))
    handle = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(
        np.arange(len(heat.columns)),
        [f"{value:.1%}" for value in heat.columns],
    )
    ax.set_yticks(
        np.arange(len(heat.index)),
        [f"{value * 100:+.0f} vol pts" for value in heat.index],
    )
    ax.set_xlabel("Hypothetical round-trip friction / entry premium")
    ax.set_ylabel("Hypothetical IV change")
    ax.set_title("7-DTE, 50% IV terminal-spot model sensitivity — not an option backtest")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > limit * 0.48 else "#172739"
            ax.text(column, row, f"{values[row, column]:+.1f}%", ha="center", va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(handle, ax=ax, pad=0.02)
    colorbar.set_label("Mean modeled return on premium (%)")
    _save(fig, chart_dir / "straddle_model_sensitivity.png")

    # Frozen 15:49-open through 15:50-bar-close research candidate.  The
    # minute OHLC endpoints remain descriptive price proxies, not NBBO fills.
    burst_portfolios = pd.read_csv(args.output_dir / "first_minute_alpha_portfolio.csv")
    burst_portfolios["effective_close_date"] = pd.to_datetime(
        burst_portfolios["effective_close_date"]
    )
    burst_gross = burst_portfolios.loc[burst_portfolios["cost_bps"].eq(0.0)].copy()
    burst_net = burst_portfolios.loc[burst_portfolios["cost_bps"].eq(10.0)].copy()
    fig, (ax, dd_ax) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax.plot(
        burst_gross["effective_close_date"],
        burst_gross["equity"],
        linewidth=2,
        label="Gross bar proxy",
    )
    ax.plot(
        burst_net["effective_close_date"],
        burst_net["equity"],
        linewidth=2,
        label="Net after 10 bp haircut",
    )
    ax.axhline(1.0, color="0.4", linewidth=1)
    ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle="--", linewidth=1)
    ax.set_title("15:49 open → 15:50 bar close: frozen minute-bar proxy")
    ax.set_ylabel("Equity (start = 1.00)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    dd_ax.fill_between(
        burst_net["effective_close_date"], burst_net["drawdown"] * 100, 0, alpha=0.45
    )
    dd_ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle="--", linewidth=1)
    dd_ax.set_ylabel("DD, %")
    dd_ax.set_xlabel("Implementation close")
    dd_ax.grid(axis="y", alpha=0.22)
    dd_ax.xaxis.set_major_locator(mdates.YearLocator())
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, chart_dir / "first_minute_alpha_equity.png")

    burst_cost = pd.read_csv(args.output_dir / "first_minute_alpha_cost_sensitivity.csv")
    burst_regime = pd.read_csv(args.output_dir / "first_minute_alpha_regime_metrics.csv")
    regime_order = [
        "2020_through_2022",
        "2023_through_2026",
        "pre_full_qcir_before_may_2023",
        "full_qcir_from_may_2023",
        "retrospective_2024_2026",
    ]
    regime_labels = ["2020–22", "2023–26", "Pre-May 2023", "May 2023+", "2024–26"]
    regime = (
        burst_regime.loc[burst_regime["cost_bps"].eq(0.0)]
        .set_index("sample")
        .loc[regime_order]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].plot(
        burst_cost["cost_bps"], burst_cost["total_return"] * 100, marker="o", linewidth=2
    )
    axes[0].axhline(0, color="0.35", linewidth=1)
    axes[0].axvline(13.8, color="0.45", linestyle="--", linewidth=1)
    axes[0].set_title("Cost sensitivity")
    axes[0].set_xlabel("All-in round-trip haircut (bp)")
    axes[0].set_ylabel("Cumulative return (%)")
    axes[0].grid(alpha=0.22)
    regime_values = regime["mean_return_per_event"] * 10_000
    axes[1].bar(
        regime_labels,
        regime_values,
        color=["#b54a4a" if value < 0 else "#277a53" for value in regime_values],
    )
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].set_title("Strong structural break")
    axes[1].set_ylabel("Gross mean per review (bp)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.22)
    _save(fig, chart_dir / "first_minute_alpha_cost_regime.png")

    time_metrics = pd.read_csv(args.output_dir / "first_minute_alpha_time_window_metrics.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), sharex=False)
    for axis, analysis, selected, title in [
        (axes[0], "one_minute_bar", "15:50", "All one-minute bars"),
        (
            axes[1],
            "adjacent_two_bar_window",
            "15:49_open_to_15:50_close",
            "All adjacent two-bar holding windows",
        ),
    ]:
        frame = time_metrics.loc[time_metrics["analysis"].eq(analysis)].copy()
        if analysis == "adjacent_two_bar_window":
            frame["clock"] = frame["window"].str.slice(0, 5)
        else:
            frame["clock"] = frame["window"]
        frame = frame.sort_values("clock")
        x = np.arange(len(frame))
        colors = ["#b68b4b" if item == selected else "#6e8ba3" for item in frame["window"]]
        axis.bar(x, frame["mean_return_per_event"] * 10_000, color=colors)
        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("Gross mean (bp)")
        ticks = list(range(0, len(frame), 5)) + [len(frame) - 1]
        ticks = sorted(set(ticks))
        axis.set_xticks(ticks, [frame["clock"].iloc[index] for index in ticks])
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xlabel("Start-stamped Eastern-time minute")
    _save(fig, chart_dir / "first_minute_alpha_time_selection.png")

    control_metrics = pd.read_csv(args.output_dir / "first_minute_alpha_control_metrics.csv")
    control_order = [
        "event_on_complete_control_sample",
        "matched_control_mean",
        "event_minus_matched_control",
    ]
    control_labels = ["MSCI event", "Four prior sessions", "Event − control"]
    control = control_metrics.set_index("analysis").loc[control_order]
    estimates = control["mean_return_per_event"].to_numpy() * 10_000
    lows = control["block_bootstrap_mean_ci_low"].to_numpy() * 10_000
    highs = control["block_bootstrap_mean_ci_high"].to_numpy() * 10_000
    y = np.arange(len(control))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color="#315f85",
        ecolor="#5279a5",
        capsize=4,
        markersize=7,
    )
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y, control_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Gross mean per review (bp); four-review block-bootstrap CI")
    ax.set_title("Same-name, same-clock controls: suggestive, not block-robust")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, chart_dir / "first_minute_alpha_controls.png")

    topn = pd.read_csv(args.output_dir / "topn_capped10_date_portfolios.csv")
    topn["effective_close_date"] = pd.to_datetime(topn["effective_close_date"])
    primary_zero = topn.loc[
        topn["strategy"].eq("impact_proxy_top10")
        & topn["cost_bps"].eq(TOP10_PRIMARY_COST_BPS)
    ].sort_values("effective_close_date")
    primary_10bp = topn.loc[
        topn["strategy"].eq("impact_proxy_top10")
        & topn["cost_bps"].eq(TOP10_SENSITIVITY_COST_BPS)
    ].sort_values("effective_close_date")
    control_zero = topn.loc[
        topn["strategy"].eq("capped_full_valid_universe")
        & topn["cost_bps"].eq(TOP10_PRIMARY_COST_BPS)
    ].sort_values("effective_close_date")
    topn_metrics = pd.read_csv(args.output_dir / "topn_capped10_metrics.csv")
    zero_sharpe = topn_metrics.loc[
        topn_metrics["strategy"].eq("impact_proxy_top10")
        & topn_metrics["cost_bps"].eq(TOP10_PRIMARY_COST_BPS),
        "event_sharpe_sqrt4",
    ].iloc[0]
    sensitivity_sharpe = topn_metrics.loc[
        topn_metrics["strategy"].eq("impact_proxy_top10")
        & topn_metrics["cost_bps"].eq(TOP10_SENSITIVITY_COST_BPS),
        "event_sharpe_sqrt4",
    ].iloc[0]

    initial_date = primary_zero["effective_close_date"].iloc[0] - pd.Timedelta(days=1)

    def event_step(frame: pd.DataFrame, column: str) -> tuple[pd.Series, np.ndarray]:
        dates = pd.concat(
            [pd.Series([initial_date]), frame["effective_close_date"].reset_index(drop=True)],
            ignore_index=True,
        )
        values = np.r_[1.0 if column == "equity" else 0.0, frame[column].to_numpy(float)]
        return dates, values

    fig, (ax, dd_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    dates, values = event_step(primary_zero, "equity")
    ax.step(
        dates,
        values,
        where="post",
        linewidth=2.2,
        label="Impact up to 5+5 · primary 0 bp",
        color="#277a53",
    )
    dates, values = event_step(primary_10bp, "equity")
    ax.step(
        dates,
        values,
        where="post",
        linewidth=2.2,
        label="Impact up to 5+5 · sensitivity 10 bp",
        color="#315f85",
    )
    dates, values = event_step(control_zero, "equity")
    ax.step(
        dates,
        values,
        where="post",
        linewidth=1.7,
        linestyle="--",
        label="Matched full basket · 0 bp",
        color="#8a6b38",
    )
    ax.axhline(1.0, color="0.45", linewidth=1)
    ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle=":", linewidth=1)
    ax.set_title(
        "Balanced top-10 equity curve · annualized event Sharpe "
        f"{zero_sharpe:.2f} at 0 bp / {sensitivity_sharpe:.2f} at 10 bp"
    )
    ax.set_ylabel("Cash-funded equity (start = 1.00)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    cash = primary_zero.loc[primary_zero["gross_exposure"].eq(0.0)]
    ax.scatter(
        cash["effective_close_date"],
        cash["equity"],
        facecolors="white",
        edgecolors="#6f7f8c",
        linewidths=1.1,
        s=28,
        zorder=4,
        label="Cash review",
    )
    ax.legend(frameon=False, loc="upper left")
    dd_dates, dd_values = event_step(primary_zero, "drawdown")
    dd_ax.fill_between(
        dd_dates,
        dd_values * 100,
        0,
        step="post",
        color="#b54a4a",
        alpha=0.42,
    )
    dd_ax.axvline(pd.Timestamp("2023-05-31"), color="0.45", linestyle=":", linewidth=1)
    dd_ax.set_ylabel("Primary DD, %\n(0 bp cost)")
    dd_ax.set_xlabel("Implementation close")
    dd_ax.grid(axis="y", alpha=0.22)
    dd_ax.xaxis.set_major_locator(mdates.YearLocator())
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, chart_dir / "topn_capped10_equity_drawdown.png")

    print(f"charts={len(list(chart_dir.glob('*.png')))} output={chart_dir}")


if __name__ == "__main__":
    main()
