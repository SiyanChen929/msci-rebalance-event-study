#!/usr/bin/env python3
"""Build auditable long/short attribution and regime diagnostics for the frozen Top-10 ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_STRATEGY = "impact_proxy_top10"
PRIMARY_COST_BPS = 0.0
BREAK_DATE = pd.Timestamp("2023-02-01")
LONG_COLOR = "#2a7f62"
SHORT_COLOR = "#b44d52"
TOTAL_COLOR = "#222222"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(np.abs(denominator) > 1e-15, 100.0 * numerator / denominator, np.nan),
        index=numerator.index,
    )


def regime_name(date: pd.Timestamp) -> str:
    return "2020-2022" if date < BREAK_DATE else "2023+"


def make_event_ledger(trades: pd.DataFrame, date_shell: pd.DataFrame) -> pd.DataFrame:
    contribution = (
        trades.pivot_table(
            index="effective_close_date",
            columns="action",
            values="security_pnl_contribution",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns={"add": "long_contribution", "delete": "short_contribution"})
        .reset_index()
    )
    sensitivity = (
        trades.pivot_table(
            index="effective_close_date",
            columns="action",
            values="sensitivity_net_pnl_contribution_10bp",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns={"add": "long_net_10bp", "delete": "short_net_10bp"})
        .reset_index()
    )
    counts = (
        trades.pivot_table(
            index="effective_close_date",
            columns="action",
            values="ticker",
            aggfunc="size",
            fill_value=0,
        )
        .rename(columns={"add": "n_long", "delete": "n_short"})
        .reset_index()
    )
    exposures = (
        trades.assign(abs_weight=trades["target_weight"].abs())
        .pivot_table(
            index="effective_close_date",
            columns="action",
            values="abs_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns={"add": "long_exposure", "delete": "short_exposure"})
        .reset_index()
    )

    top_rows = trades.loc[
        trades.groupby("effective_close_date")["security_pnl_contribution"].apply(
            lambda values: values.abs().idxmax()
        )
    ].copy()
    top_rows["top_contributor"] = (
        top_rows["ticker"].astype(str)
        + " "
        + (10_000.0 * top_rows["security_pnl_contribution"]).map(lambda value: f"{value:+.2f}bp")
    )

    keep = [
        "effective_close_date",
        "trade_status",
        "eligible_adds",
        "eligible_deletes",
        "gross_exposure",
        "long_exposure",
        "short_exposure",
        "n_names",
        "gross_return",
        "equity",
        "drawdown",
    ]
    result = date_shell[keep].copy()
    result = result.merge(contribution, on="effective_close_date", how="left")
    result = result.merge(sensitivity, on="effective_close_date", how="left")
    result = result.merge(counts, on="effective_close_date", how="left")
    result = result.drop(columns=["long_exposure", "short_exposure"])
    result = result.merge(exposures, on="effective_close_date", how="left")
    result = result.merge(
        top_rows[["effective_close_date", "top_contributor"]],
        on="effective_close_date",
        how="left",
    )

    fill_zero = [
        "long_contribution",
        "short_contribution",
        "long_net_10bp",
        "short_net_10bp",
        "n_long",
        "n_short",
        "long_exposure",
        "short_exposure",
    ]
    result[fill_zero] = result[fill_zero].fillna(0.0)
    result[["n_long", "n_short"]] = result[["n_long", "n_short"]].astype(int)
    result["effective_close_date"] = pd.to_datetime(result["effective_close_date"])
    result["quarter"] = result["effective_close_date"].dt.to_period("Q").astype(str)
    result["year"] = result["effective_close_date"].dt.year
    result["quarter_number"] = result["effective_close_date"].dt.quarter
    result["review_class"] = np.where(
        result["effective_close_date"].dt.month.isin([2, 8]), "February/August", "May/November"
    )
    result["regime"] = result["effective_close_date"].map(regime_name)
    result["status"] = np.where(result["trade_status"].eq("Traded"), "Traded", "Cash")
    result["long_contribution_bp"] = 10_000.0 * result["long_contribution"]
    result["short_contribution_bp"] = 10_000.0 * result["short_contribution"]
    result["gross_return_bp"] = 10_000.0 * result["gross_return"]
    result["net_10bp_bp"] = 10_000.0 * (result["long_net_10bp"] + result["short_net_10bp"])
    result["signed_long_share_pct"] = safe_share(
        result["long_contribution"], result["gross_return"]
    )
    result["signed_short_share_pct"] = safe_share(
        result["short_contribution"], result["gross_return"]
    )
    absolute_sum = result["long_contribution"].abs() + result["short_contribution"].abs()
    result["absolute_long_share_pct"] = safe_share(
        result["long_contribution"].abs(), absolute_sum
    )
    result["absolute_short_share_pct"] = safe_share(
        result["short_contribution"].abs(), absolute_sum
    )
    result["leg_netting_efficiency_pct"] = safe_share(result["gross_return"], absolute_sum)

    max_reconciliation_error = float(
        np.max(
            np.abs(
                result["long_contribution"]
                + result["short_contribution"]
                - result["gross_return"]
            )
        )
    )
    if max_reconciliation_error > 1e-12:
        raise AssertionError(f"Date-level leg reconciliation failed: {max_reconciliation_error}")
    if len(result) != 26 or int(result["status"].eq("Traded").sum()) != 17:
        raise AssertionError("Unexpected review-date shell")

    ordered = [
        "quarter",
        "effective_close_date",
        "regime",
        "review_class",
        "status",
        "eligible_adds",
        "eligible_deletes",
        "n_long",
        "n_short",
        "long_exposure",
        "short_exposure",
        "long_contribution_bp",
        "short_contribution_bp",
        "gross_return_bp",
        "net_10bp_bp",
        "signed_long_share_pct",
        "signed_short_share_pct",
        "absolute_long_share_pct",
        "absolute_short_share_pct",
        "leg_netting_efficiency_pct",
        "top_contributor",
        "equity",
        "drawdown",
    ]
    return result[ordered].sort_values("effective_close_date").reset_index(drop=True)


def summarize_period(label: str, rows: pd.DataFrame) -> dict[str, object]:
    active = rows.loc[rows["status"].eq("Traded")].copy()
    long_total = float(rows["long_contribution_bp"].sum())
    short_total = float(rows["short_contribution_bp"].sum())
    total = long_total + short_total
    absolute_aggregate = abs(long_total) + abs(short_total)
    date_leg_absolute = float(
        (rows["long_contribution_bp"].abs() + rows["short_contribution_bp"].abs()).sum()
    )
    both_positive = int(
        ((active["long_contribution_bp"] > 0) & (active["short_contribution_bp"] > 0)).sum()
    )
    both_negative = int(
        ((active["long_contribution_bp"] < 0) & (active["short_contribution_bp"] < 0)).sum()
    )
    disagreement = int(len(active) - both_positive - both_negative)
    return {
        "period": label,
        "scheduled_reviews": int(len(rows)),
        "active_reviews": int(len(active)),
        "cash_reviews": int(len(rows) - len(active)),
        "long_total_bp": long_total,
        "short_total_bp": short_total,
        "arithmetic_total_bp": total,
        "long_mean_per_scheduled_bp": long_total / len(rows),
        "short_mean_per_scheduled_bp": short_total / len(rows),
        "gross_mean_per_scheduled_bp": total / len(rows),
        "gross_mean_per_active_bp": total / len(active) if len(active) else np.nan,
        "long_signed_share_pct": 100.0 * long_total / total if abs(total) > 1e-15 else np.nan,
        "short_signed_share_pct": 100.0 * short_total / total if abs(total) > 1e-15 else np.nan,
        "long_absolute_aggregate_share_pct": (
            100.0 * abs(long_total) / absolute_aggregate if absolute_aggregate else np.nan
        ),
        "short_absolute_aggregate_share_pct": (
            100.0 * abs(short_total) / absolute_aggregate if absolute_aggregate else np.nan
        ),
        "leg_netting_capture_pct": 100.0 * total / date_leg_absolute if date_leg_absolute else np.nan,
        "long_active_hit_rate": float((active["long_contribution_bp"] > 0).mean()),
        "short_active_hit_rate": float((active["short_contribution_bp"] > 0).mean()),
        "both_legs_positive": both_positive,
        "both_legs_negative": both_negative,
        "leg_sign_disagreement": disagreement,
        "active_leg_correlation": float(
            active[["long_contribution_bp", "short_contribution_bp"]].corr().iloc[0, 1]
        ),
        "net_10bp_total_bp": float(rows["net_10bp_bp"].sum()),
        "net_10bp_mean_per_scheduled_bp": float(rows["net_10bp_bp"].mean()),
    }


def make_period_summaries(event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(event["effective_close_date"])
    regimes = pd.DataFrame(
        [
            summarize_period("All", event),
            summarize_period("2020-2022", event.loc[dates < BREAK_DATE]),
            summarize_period("2023+", event.loc[dates >= BREAK_DATE]),
            summarize_period("May-2023+", event.loc[dates >= pd.Timestamp("2023-05-01")]),
        ]
    )

    seasonal_rows: list[dict[str, object]] = []
    for quarter_number, rows in event.groupby(pd.to_datetime(event["effective_close_date"]).dt.quarter):
        seasonal_rows.append(summarize_period(f"Q{quarter_number}", rows))
    seasonal = pd.DataFrame(seasonal_rows)

    yearly_rows: list[dict[str, object]] = []
    for year, rows in event.groupby(pd.to_datetime(event["effective_close_date"]).dt.year):
        yearly_rows.append(summarize_period(str(year), rows))
    yearly = pd.DataFrame(yearly_rows)
    return regimes, seasonal, yearly


def make_broad_diagnostics(
    broad: pd.DataFrame, volume: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    broad = broad.loc[np.isclose(broad["cost_bps"], PRIMARY_COST_BPS)].copy()
    broad["effective_close_date"] = pd.to_datetime(broad["effective_close_date"])
    broad["regime"] = np.where(
        broad["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    broad["review_class"] = np.where(
        broad["effective_close_date"].dt.month.isin([2, 8]),
        "February/August",
        "May/November",
    )
    broad["long_portfolio_contribution_bp"] = 5_000.0 * broad["add_leg_return"]
    broad["short_portfolio_contribution_bp"] = 5_000.0 * broad["delete_leg_return"]
    broad["gross_return_bp"] = 10_000.0 * broad["gross_return"]
    review_type = (
        broad.groupby(["regime", "review_class"], as_index=False)
        .agg(
            review_dates=("effective_close_date", "size"),
            security_events=("n_names", "sum"),
            mean_security_events=("n_names", "mean"),
            mean_long_contribution_bp=("long_portfolio_contribution_bp", "mean"),
            mean_short_contribution_bp=("short_portfolio_contribution_bp", "mean"),
            mean_gross_return_bp=("gross_return_bp", "mean"),
            hit_rate=("gross_return", lambda values: float((values > 0).mean())),
        )
        .sort_values(["regime", "review_class"])
    )

    broad_regime = (
        broad.groupby("regime", as_index=False)
        .agg(
            review_dates=("effective_close_date", "size"),
            security_events=("n_names", "sum"),
            mean_security_events=("n_names", "mean"),
            mean_long_contribution_bp=("long_portfolio_contribution_bp", "mean"),
            mean_short_contribution_bp=("short_portfolio_contribution_bp", "mean"),
            mean_gross_return_bp=("gross_return_bp", "mean"),
            hit_rate=("gross_return", lambda values: float((values > 0).mean())),
        )
        .set_index("regime")
    )

    volume = volume.copy()
    volume["effective_close_date"] = pd.to_datetime(volume["effective_close_date"])
    volume["regime"] = np.where(
        volume["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    volume_regime = (
        volume.groupby("regime", as_index=True)
        .agg(
            mean_event_daily_volume_ratio=("median_daily_volume_ratio", "mean"),
            median_event_daily_volume_ratio=("median_daily_volume_ratio", "median"),
            mean_last10_volume_share=("median_last10_volume_share", "mean"),
            median_last10_volume_share=("median_last10_volume_share", "median"),
            mean_last_minute_volume_share=("median_last_minute_volume_share", "mean"),
            median_last_minute_volume_share=("median_last_minute_volume_share", "median"),
        )
    )
    combined = broad_regime.join(volume_regime).reset_index()
    return combined, review_type


def make_change_type_diagnostics(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = trades.copy()
    work["effective_close_date"] = pd.to_datetime(work["effective_close_date"])
    work["regime"] = np.where(
        work["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    summary = (
        work.groupby(["regime", "action", "change_type"], as_index=False)
        .agg(
            selected_security_events=("ticker", "size"),
            total_contribution_bp=("security_pnl_contribution", lambda values: 10_000.0 * values.sum()),
            mean_directional_security_return_bp=(
                "directional_security_return",
                lambda values: 10_000.0 * values.mean(),
            ),
            median_directional_security_return_bp=(
                "directional_security_return",
                lambda values: 10_000.0 * values.median(),
            ),
            security_hit_rate=("directional_security_return", lambda values: float((values > 0).mean())),
            mean_exit_bar_volume_share=("exit_bar_volume_share", "mean"),
        )
        .sort_values(["regime", "action", "change_type"])
    )
    worst_early_deletions = (
        work.loc[(work["regime"].eq("2020-2022")) & (work["action"].eq("delete"))]
        .sort_values("security_pnl_contribution")
        .head(12)[
            [
                "effective_close_date",
                "ticker",
                "change_type",
                "target_weight",
                "directional_security_return",
                "security_pnl_contribution",
                "exit_bar_volume_share",
            ]
        ]
        .copy()
    )
    worst_early_deletions["directional_security_return_bp"] = (
        10_000.0 * worst_early_deletions["directional_security_return"]
    )
    worst_early_deletions["contribution_bp"] = (
        10_000.0 * worst_early_deletions["security_pnl_contribution"]
    )
    return summary, worst_early_deletions


def make_all_name_change_type_diagnostics(security: pd.DataFrame) -> pd.DataFrame:
    """Show whether the public add/delete sign works across actual migration types."""
    work = security.copy()
    work["effective_close_date"] = pd.to_datetime(work["effective_close_date"])
    work["regime"] = np.where(
        work["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    return (
        work.groupby(["regime", "action", "change_type"], as_index=False)
        .agg(
            security_events=("ticker", "size"),
            mean_directional_return_bp=("gross_return", lambda values: 10_000.0 * values.mean()),
            median_directional_return_bp=("gross_return", lambda values: 10_000.0 * values.median()),
            security_hit_rate=("gross_return", lambda values: float((values > 0).mean())),
            mean_exit_bar_volume_share=("exit_bar_volume_share", "mean"),
        )
        .sort_values(["regime", "action", "change_type"])
        .reset_index(drop=True)
    )


def make_date_balanced_change_type_diagnostics(security: pd.DataFrame) -> pd.DataFrame:
    """Aggregate migration-type returns by review date before comparing regimes."""
    work = security.copy()
    work["effective_close_date"] = pd.to_datetime(work["effective_close_date"])
    work["regime"] = np.where(
        work["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    by_date = (
        work.groupby(
            ["regime", "effective_close_date", "action", "change_type"],
            as_index=False,
        )
        .agg(
            security_events=("ticker", "size"),
            date_mean_directional_return=("gross_return", "mean"),
        )
    )
    return (
        by_date.groupby(["regime", "action", "change_type"], as_index=False)
        .agg(
            review_dates=("effective_close_date", "nunique"),
            security_events=("security_events", "sum"),
            mean_date_directional_return_bp=(
                "date_mean_directional_return",
                lambda values: 10_000.0 * values.mean(),
            ),
            median_date_directional_return_bp=(
                "date_mean_directional_return",
                lambda values: 10_000.0 * values.median(),
            ),
            date_hit_rate=(
                "date_mean_directional_return",
                lambda values: float((values > 0).mean()),
            ),
        )
        .sort_values(["regime", "action", "change_type"])
        .reset_index(drop=True)
    )
def make_regime_clock_profile(event_time: pd.DataFrame) -> pd.DataFrame:
    """Aggregate date-level signed minute returns without treating names as independent."""
    work = event_time.copy()
    work["effective_close_date"] = pd.to_datetime(work["effective_close_date"])
    work["regime"] = np.where(
        work["effective_close_date"] < BREAK_DATE, "2020-2022", "2023+"
    )
    return (
        work.groupby(["regime", "clock"], as_index=False)
        .agg(
            review_dates=("effective_close_date", "nunique"),
            mean_signed_bar_return_bp=("signed_bar_return", lambda values: 10_000.0 * values.mean()),
            median_signed_bar_return_bp=("signed_bar_return", lambda values: 10_000.0 * values.median()),
            mean_minute_volume_share_pct=("mean_minute_volume_share", lambda values: 100.0 * values.mean()),
            median_same_clock_volume_ratio=("median_minute_volume_vs_prior_same_clock", "median"),
        )
        .sort_values(["regime", "clock"])
        .reset_index(drop=True)
    )


def make_influence_diagnostics(event: pd.DataFrame, broad: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate simple means after removing obvious one-off or recent-event clusters."""
    broad_work = broad.loc[np.isclose(broad["cost_bps"], PRIMARY_COST_BPS)].copy()
    broad_work["effective_close_date"] = pd.to_datetime(broad_work["effective_close_date"])
    broad_work["gross_return_bp"] = 10_000.0 * broad_work["gross_return"]
    top_work = event.copy()
    top_work["effective_close_date"] = pd.to_datetime(top_work["effective_close_date"])

    samples = {
        "2020-2022": lambda dates: dates < BREAK_DATE,
        "2020-2022 ex May-2020": lambda dates: (dates < BREAK_DATE)
        & (dates != pd.Timestamp("2020-05-29")),
        "2023+": lambda dates: dates >= BREAK_DATE,
        "2023+ ex May-2023": lambda dates: (dates >= BREAK_DATE)
        & (dates != pd.Timestamp("2023-05-31")),
        "2023-2025": lambda dates: (dates >= BREAK_DATE) & (dates.dt.year <= 2025),
        "2023-2024": lambda dates: dates.dt.year.isin([2023, 2024]),
        "2025-2026": lambda dates: dates.dt.year.isin([2025, 2026]),
    }
    rows: list[dict[str, object]] = []
    for portfolio, frame in [("Impact top-10", top_work), ("All-name", broad_work)]:
        dates = frame["effective_close_date"]
        for sample, selector in samples.items():
            subset = frame.loc[selector(dates)].copy()
            if portfolio == "Impact top-10":
                active = subset["status"].eq("Traded")
            else:
                active = subset["gross_return_bp"].notna()
            rows.append(
                {
                    "portfolio": portfolio,
                    "sample": sample,
                    "scheduled_reviews": int(len(subset)),
                    "active_reviews": int(active.sum()),
                    "mean_bp_per_scheduled_review": float(subset["gross_return_bp"].mean()),
                    "active_hit_rate": float((subset.loc[active, "gross_return_bp"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_driver_table(
    regime: pd.DataFrame, broad: pd.DataFrame, review_type: pd.DataFrame
) -> pd.DataFrame:
    top = regime.set_index("period")
    all_names = broad.set_index("regime")
    review = review_type.set_index(["regime", "review_class"])
    rows = [
        {
            "metric": "Top-10 mean gross return per scheduled review",
            "2020_2022": top.loc["2020-2022", "gross_mean_per_scheduled_bp"],
            "2023_plus": top.loc["2023+", "gross_mean_per_scheduled_bp"],
            "unit": "bp",
            "interpretation": "Weak early total; materially larger late-period estimate",
        },
        {
            "metric": "Top-10 long contribution per scheduled review",
            "2020_2022": top.loc["2020-2022", "long_mean_per_scheduled_bp"],
            "2023_plus": top.loc["2023+", "long_mean_per_scheduled_bp"],
            "unit": "bp",
            "interpretation": "Long additions are positive in both periods",
        },
        {
            "metric": "Top-10 short contribution per scheduled review",
            "2020_2022": top.loc["2020-2022", "short_mean_per_scheduled_bp"],
            "2023_plus": top.loc["2023+", "short_mean_per_scheduled_bp"],
            "unit": "bp",
            "interpretation": "Deletion short leg is the main early-period failure",
        },
        {
            "metric": "Top-10 leg netting capture",
            "2020_2022": top.loc["2020-2022", "leg_netting_capture_pct"],
            "2023_plus": top.loc["2023+", "leg_netting_capture_pct"],
            "unit": "%",
            "interpretation": "Early legs frequently offset; late legs usually align",
        },
        {
            "metric": "All-name mean gross return per review",
            "2020_2022": all_names.loc["2020-2022", "mean_gross_return_bp"],
            "2023_plus": all_names.loc["2023+", "mean_gross_return_bp"],
            "unit": "bp",
            "interpretation": "Regime break is not unique to Top-10 selection",
        },
        {
            "metric": "All-name February/August events per review",
            "2020_2022": review.loc[("2020-2022", "February/August"), "mean_security_events"],
            "2023_plus": review.loc[("2023+", "February/August"), "mean_security_events"],
            "unit": "names",
            "interpretation": "QCIR substantially broadened the formerly light reviews",
        },
        {
            "metric": "Mean median last-10-minute volume share",
            "2020_2022": all_names.loc["2020-2022", "mean_last10_volume_share"] * 100.0,
            "2023_plus": all_names.loc["2023+", "mean_last10_volume_share"] * 100.0,
            "unit": "% day volume",
            "interpretation": "Closing volume became more concentrated after 2023",
        },
        {
            "metric": "Mean median last-minute volume share",
            "2020_2022": all_names.loc["2020-2022", "mean_last_minute_volume_share"] * 100.0,
            "2023_plus": all_names.loc["2023+", "mean_last_minute_volume_share"] * 100.0,
            "unit": "% day volume",
            "interpretation": "Terminal-minute concentration is also higher late",
        },
    ]
    result = pd.DataFrame(rows)
    result["difference"] = result["2023_plus"] - result["2020_2022"]
    return result[
        ["metric", "2020_2022", "2023_plus", "difference", "unit", "interpretation"]
    ]


def plot_event_legs(event: pd.DataFrame, destination: Path) -> None:
    labels = event["quarter"].tolist()
    x = np.arange(len(event))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.6, 5.3))
    ax.axvspan(-0.6, 10.5, color="#eadfca", alpha=0.35, label="2020-2022")
    ax.bar(
        x - width / 2,
        event["long_contribution_bp"],
        width,
        color=LONG_COLOR,
        label="Long additions",
    )
    ax.bar(
        x + width / 2,
        event["short_contribution_bp"],
        width,
        color=SHORT_COLOR,
        label="Short deletions",
    )
    ax.scatter(x, event["gross_return_bp"], color=TOTAL_COLOR, s=18, zorder=4, label="Total")
    cash = event["status"].eq("Cash")
    ax.scatter(x[cash], np.zeros(int(cash.sum())), marker="x", s=30, color="#6b7280", zorder=5)
    ax.axvline(10.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Portfolio contribution (bp of capital)")
    ax.set_title("Top-10 long/short attribution by MSCI implementation close - primary 0 bp proxy")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regime_diagnostics(
    regime: pd.DataFrame, broad: pd.DataFrame, destination: Path
) -> None:
    top = regime.set_index("period").loc[["2020-2022", "2023+"]]
    all_names = broad.set_index("regime").loc[["2020-2022", "2023+"]]
    labels = ["2020-2022", "2023+"]
    x = np.arange(2)
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    axes[0].bar(
        x - width / 2,
        top["long_mean_per_scheduled_bp"],
        width,
        color=LONG_COLOR,
        label="Long",
    )
    axes[0].bar(
        x + width / 2,
        top["short_mean_per_scheduled_bp"],
        width,
        color=SHORT_COLOR,
        label="Short",
    )
    axes[0].set_title("Top-10 contribution")
    axes[0].set_ylabel("Mean bp / scheduled review")

    axes[1].bar(
        x - width / 2,
        all_names["mean_long_contribution_bp"],
        width,
        color=LONG_COLOR,
        label="Long",
    )
    axes[1].bar(
        x + width / 2,
        all_names["mean_short_contribution_bp"],
        width,
        color=SHORT_COLOR,
        label="Short",
    )
    axes[1].set_title("All-name contribution")
    axes[1].set_ylabel("Mean bp / review")

    axes[2].bar(
        x - width / 2,
        100.0 * all_names["mean_last10_volume_share"],
        width,
        color="#4c78a8",
        label="Last 10 min",
    )
    axes[2].bar(
        x + width / 2,
        100.0 * all_names["mean_last_minute_volume_share"],
        width,
        color="#f2a65a",
        label="Last minute",
    )
    axes[2].set_title("Closing-volume concentration")
    axes[2].set_ylabel("Mean of date medians (% day volume)")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("The early-period weakness is concentrated in the deletion leg", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    output_dir = args.output_dir
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "trade_log": output_dir / "topn_capped10_trade_log.csv",
        "date_log": output_dir / "topn_capped10_date_trade_log.csv",
        "broad_portfolio": output_dir / "first_minute_alpha_portfolio.csv",
        "broad_security": output_dir / "first_minute_alpha_security_ledger.csv",
        "event_time_date": output_dir / "event_time_date_panel.csv",
        "volume_by_date": output_dir / "volume_diagnostics_by_date.csv",
    }
    trades_all = pd.read_csv(paths["trade_log"])
    date_all = pd.read_csv(paths["date_log"])
    broad = pd.read_csv(paths["broad_portfolio"])
    broad_security = pd.read_csv(paths["broad_security"])
    event_time_date = pd.read_csv(paths["event_time_date"])
    volume = pd.read_csv(paths["volume_by_date"])

    trades = trades_all.loc[
        trades_all["strategy"].eq(PRIMARY_STRATEGY)
        & np.isclose(trades_all["modeled_cost_bps"], PRIMARY_COST_BPS)
    ].copy()
    date_shell = date_all.loc[
        date_all["strategy"].eq(PRIMARY_STRATEGY)
        & np.isclose(date_all["cost_bps"], PRIMARY_COST_BPS)
    ].copy()
    if len(trades) != 161:
        raise AssertionError(f"Expected 161 primary security rows, found {len(trades)}")

    event = make_event_ledger(trades, date_shell)
    regime, seasonal, yearly = make_period_summaries(event)
    broad_regime, review_type = make_broad_diagnostics(broad, volume)
    change_type, worst_early = make_change_type_diagnostics(trades)
    broad_change_type = make_all_name_change_type_diagnostics(broad_security)
    date_balanced_change_type = make_date_balanced_change_type_diagnostics(
        broad_security
    )
    regime_clock = make_regime_clock_profile(event_time_date)
    influence = make_influence_diagnostics(event, broad)
    drivers = build_driver_table(regime, broad_regime, review_type)

    outputs = {
        "event": output_dir / "topn_capped10_quarterly_event_attribution.csv",
        "regime": output_dir / "topn_capped10_leg_regime_attribution.csv",
        "seasonal": output_dir / "topn_capped10_quarter_seasonality.csv",
        "yearly": output_dir / "topn_capped10_yearly_leg_attribution.csv",
        "broad_regime": output_dir / "first_minute_alpha_regime_leg_diagnostics.csv",
        "review_type": output_dir / "first_minute_alpha_review_type_diagnostics.csv",
        "change_type": output_dir / "topn_capped10_change_type_diagnostics.csv",
        "broad_change_type": output_dir / "first_minute_alpha_change_type_diagnostics.csv",
        "date_balanced_change_type": output_dir / "first_minute_alpha_change_type_date_balanced.csv",
        "regime_clock": output_dir / "first_minute_alpha_regime_clock_profile.csv",
        "influence": output_dir / "topn_capped10_regime_influence_diagnostics.csv",
        "worst_early": output_dir / "topn_capped10_early_deletion_tail.csv",
        "drivers": output_dir / "topn_capped10_early_regime_drivers.csv",
        "event_chart": chart_dir / "topn_capped10_event_leg_attribution.png",
        "regime_chart": chart_dir / "topn_capped10_early_regime_diagnostics.png",
        "manifest": output_dir / "topn_capped10_attribution_manifest.json",
    }
    event.to_csv(outputs["event"], index=False)
    regime.to_csv(outputs["regime"], index=False)
    seasonal.to_csv(outputs["seasonal"], index=False)
    yearly.to_csv(outputs["yearly"], index=False)
    broad_regime.to_csv(outputs["broad_regime"], index=False)
    review_type.to_csv(outputs["review_type"], index=False)
    change_type.to_csv(outputs["change_type"], index=False)
    broad_change_type.to_csv(outputs["broad_change_type"], index=False)
    date_balanced_change_type.to_csv(
        outputs["date_balanced_change_type"], index=False
    )
    regime_clock.to_csv(outputs["regime_clock"], index=False)
    influence.to_csv(outputs["influence"], index=False)
    worst_early.to_csv(outputs["worst_early"], index=False)
    drivers.to_csv(outputs["drivers"], index=False)
    plot_event_legs(event, outputs["event_chart"])
    plot_regime_diagnostics(regime, broad_regime, outputs["regime_chart"])

    arithmetic_total = float(trades["security_pnl_contribution"].sum())
    if not np.isclose(arithmetic_total, event["gross_return_bp"].sum() / 10_000.0, atol=1e-12):
        raise AssertionError("Security and date arithmetic totals do not reconcile")
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": PRIMARY_STRATEGY,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "information_status": "research_proxy_not_broker_fill",
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "scheduled_reviews": int(len(event)),
        "active_reviews": int(event["status"].eq("Traded").sum()),
        "security_rows": int(len(trades)),
        "arithmetic_total_return": arithmetic_total,
        "output_rows": {
            name: int(len(frame))
            for name, frame in {
                "event": event,
                "regime": regime,
                "seasonal": seasonal,
                "yearly": yearly,
                "broad_regime": broad_regime,
                "review_type": review_type,
                "change_type": change_type,
                "broad_change_type": broad_change_type,
                "date_balanced_change_type": date_balanced_change_type,
                "regime_clock": regime_clock,
                "influence": influence,
                "worst_early": worst_early,
                "drivers": drivers,
            }.items()
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"event_rows={len(event)} security_rows={len(trades)} "
        f"arithmetic_total={arithmetic_total:.12f} manifest={outputs['manifest']}"
    )


if __name__ == "__main__":
    main()
