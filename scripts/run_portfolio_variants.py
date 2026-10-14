#!/usr/bin/env python3
"""Compare neutral and unneutralized portfolios on the frozen MSCI top-10 panel."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msci_rebalance.backtest import (
    active_trade_metrics,
    add_equity_columns,
    moving_block_bootstrap_mean,
    performance_metrics,
)
from msci_rebalance.portfolio_variants import (
    build_beta_neutral_equal_weights,
    build_unneutralized_signal_weights,
    estimate_pre_event_beta,
    split_dollar_neutral_legs,
)
from msci_rebalance.weighted_backtest import weighted_date_portfolio


PRIMARY_INPUT = "impact_proxy_top10"
DOLLAR = "dollar_neutral_equal"
BETA = "beta_neutral_equal"
SIGNAL = "signal_weighted_unneutralized"
LONG = "long_only_dollar_leg"
SHORT = "short_only_dollar_leg"
STRATEGY_ORDER = [DOLLAR, BETA, SIGNAL, LONG, SHORT]
COSTS = (0.0, 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _add_betas(ledger: pd.DataFrame, marketdata_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in ledger.itertuples(index=False):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        cutoff = pd.Timestamp(row.feature_cutoff_date).normalize()
        prefix = f"{event_date.date()}_{_safe_ticker(str(row.ticker))}.parquet"
        beta = estimate_pre_event_beta(
            marketdata_root / "daily" / prefix,
            marketdata_root / "daily" / f"{event_date.date()}_SPY.parquet",
            cutoff,
        )
        if pd.notna(beta["beta_feature_asof"]) and pd.Timestamp(
            beta["beta_feature_asof"]
        ).normalize() > cutoff:
            raise AssertionError("beta feature uses data after its declared cutoff")
        rows.append(beta)
    output = pd.concat([ledger.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if (~np.isfinite(pd.to_numeric(output["beta_used"], errors="coerce"))).any():
        raise ValueError("beta construction produced non-finite exposures")
    return output


def _build_ledgers(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dollar = base.copy()
    dollar["strategy"] = DOLLAR

    beta = build_beta_neutral_equal_weights(base)
    beta["strategy"] = BETA

    signal = build_unneutralized_signal_weights(base, "proxy_flow_pressure")
    signal["strategy"] = SIGNAL

    long_only, short_only = split_dollar_neutral_legs(base)
    long_only["strategy"] = LONG
    short_only["strategy"] = SHORT
    return {
        DOLLAR: dollar,
        BETA: beta,
        SIGNAL: signal,
        LONG: long_only,
        SHORT: short_only,
    }


def _spy_window_return(path: Path, event_date: pd.Timestamp) -> float:
    if not path.exists():
        return np.nan
    frame = pd.read_parquet(path).copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
        session = timestamp.dt.tz_localize(None).dt.normalize()
    else:
        session = timestamp.dt.normalize()
    clock = timestamp.dt.strftime("%H:%M")
    day = frame.loc[session.eq(event_date)].copy()
    day["clock"] = clock.loc[session.eq(event_date)].to_numpy()
    entry = day.loc[day["clock"].eq("15:49")]
    exit_ = day.loc[day["clock"].eq("15:50")]
    if entry.empty or exit_.empty:
        return np.nan
    entry_price = float(entry.iloc[0]["open"])
    exit_price = float(exit_.iloc[0]["close"])
    if min(entry_price, exit_price) <= 0:
        return np.nan
    return exit_price / entry_price - 1.0


def _evaluate(
    ledgers: dict[str, pd.DataFrame],
    scheduled_dates: list[pd.Timestamp],
    marketdata_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spy_returns = {
        date: _spy_window_return(
            marketdata_root / "minute" / f"{date.date()}_SPY.parquet", date
        )
        for date in scheduled_dates
    }
    portfolio_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    trade_rows: list[pd.DataFrame] = []
    for strategy in STRATEGY_ORDER:
        ledger = ledgers[strategy].copy()
        ledger["estimated_beta_contribution"] = (
            ledger["target_weight"] * ledger["beta_used"]
        )
        exposure = (
            ledger.groupby("effective_close_date", as_index=False)
            .agg(
                estimated_beta_exposure=("estimated_beta_contribution", "sum"),
                beta_long_exposure=(
                    "estimated_beta_contribution",
                    lambda x: float(x.clip(lower=0).sum()),
                ),
                beta_short_exposure=(
                    "estimated_beta_contribution",
                    lambda x: float(-x.clip(upper=0).sum()),
                ),
            )
        )
        for cost in COSTS:
            portfolio = weighted_date_portfolio(
                ledger, cost_bps=cost, scheduled_dates=scheduled_dates
            ).merge(exposure, on="effective_close_date", how="left", validate="one_to_one")
            for column in (
                "estimated_beta_exposure",
                "beta_long_exposure",
                "beta_short_exposure",
            ):
                portfolio[column] = portfolio[column].fillna(0.0)
            portfolio["spy_window_return"] = portfolio["effective_close_date"].map(spy_returns)
            portfolio["beta_return_proxy"] = (
                portfolio["estimated_beta_exposure"] * portfolio["spy_window_return"]
            )
            portfolio["residual_return_proxy"] = (
                portfolio["gross_return"] - portfolio["beta_return_proxy"].fillna(0.0)
            )
            portfolio["net_return_per_unit_gross"] = np.where(
                portfolio["gross_exposure"].gt(0),
                portfolio["net_return"] / portfolio["gross_exposure"],
                np.nan,
            )
            portfolio.insert(0, "strategy", strategy)
            portfolio = add_equity_columns(portfolio)
            portfolio_rows.append(portfolio)

            active = portfolio.loc[portfolio["gross_exposure"].gt(0)].copy()
            unit = active_trade_metrics(
                portfolio,
                return_column="net_return_per_unit_gross",
                exposure_column="gross_exposure",
            )
            metric_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost,
                    **performance_metrics(portfolio, "net_return"),
                    **active_trade_metrics(portfolio, "net_return"),
                    "active_unit_gross_mean_return": unit["active_mean_return"],
                    "active_unit_gross_sharpe_sqrt4": unit[
                        "active_trade_sharpe_sqrt4"
                    ],
                    "mean_active_gross_exposure": float(active["gross_exposure"].mean()),
                    "mean_active_net_exposure": float(active["net_exposure"].mean()),
                    "mean_active_abs_net_exposure": float(active["net_exposure"].abs().mean()),
                    "mean_active_estimated_beta_exposure": float(
                        active["estimated_beta_exposure"].mean()
                    ),
                    "mean_active_abs_estimated_beta_exposure": float(
                        active["estimated_beta_exposure"].abs().mean()
                    ),
                    "active_mean_beta_return_proxy": float(active["beta_return_proxy"].mean()),
                    "active_mean_residual_return_proxy": float(
                        active["residual_return_proxy"].mean()
                    ),
                    "beta_fallback_names": int(ledger["beta_source"].eq("unit_beta_fallback").sum()),
                }
            )

            trades = ledger.copy()
            trades["cost_bps"] = cost
            trades["gross_pnl_contribution"] = (
                trades["target_weight"] * trades["underlying_return"]
            )
            trades["cost_contribution"] = (
                trades["target_weight"].abs() * cost / 10_000.0
            )
            trades["net_pnl_contribution"] = (
                trades["gross_pnl_contribution"] - trades["cost_contribution"]
            )
            trade_rows.append(trades)
    return (
        pd.concat(portfolio_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(trade_rows, ignore_index=True),
    )


def _paired_tests(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost in COSTS:
        pivot = portfolios.loc[portfolios["cost_bps"].eq(cost)].pivot(
            index="effective_close_date", columns="strategy", values="net_return"
        )
        for strategy in STRATEGY_ORDER[1:]:
            difference = (pivot[strategy] - pivot[DOLLAR]).dropna()
            low, high = moving_block_bootstrap_mean(
                difference.to_numpy(float), block_size=4
            )
            rows.append(
                {
                    "strategy": strategy,
                    "comparator": DOLLAR,
                    "cost_bps": cost,
                    "review_dates": len(difference),
                    "mean_difference": float(difference.mean()),
                    "positive_difference_rate": float(difference.gt(0).mean()),
                    "block4_mean_ci_low": low,
                    "block4_mean_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _chart(portfolios: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    labels = {
        DOLLAR: "Dollar neutral",
        BETA: "Beta neutral",
        SIGNAL: "Signal weighted",
        LONG: "Long only",
        SHORT: "Short only",
    }
    zero = portfolios.loc[portfolios["cost_bps"].eq(0.0)].copy()
    summary = metrics.loc[metrics["cost_bps"].eq(0.0)].set_index("strategy")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for strategy in STRATEGY_ORDER:
        part = zero.loc[zero["strategy"].eq(strategy)]
        axes[0, 0].plot(part["effective_close_date"], part["equity"], label=labels[strategy])
        axes[0, 1].plot(part["effective_close_date"], part["drawdown"], label=labels[strategy])
    axes[0, 0].set_title("0 bp event-step equity")
    axes[0, 1].set_title("0 bp event-step drawdown")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    x = np.arange(len(STRATEGY_ORDER))
    axes[1, 0].bar(
        x - 0.18,
        summary.loc[STRATEGY_ORDER, "mean_active_gross_exposure"],
        width=0.36,
        label="Gross",
    )
    axes[1, 0].bar(
        x + 0.18,
        summary.loc[STRATEGY_ORDER, "mean_active_net_exposure"],
        width=0.36,
        label="Net dollar",
    )
    axes[1, 0].set_title("Mean active exposure")
    axes[1, 0].set_xticks(x, [labels[s] for s in STRATEGY_ORDER], rotation=18, ha="right")
    axes[1, 0].legend(frameon=False)
    beta_bp = summary.loc[STRATEGY_ORDER, "active_mean_beta_return_proxy"] * 10_000
    residual_bp = summary.loc[STRATEGY_ORDER, "active_mean_residual_return_proxy"] * 10_000
    axes[1, 1].bar(x, beta_bp, label="SPY beta proxy")
    axes[1, 1].bar(x, residual_bp, bottom=beta_bp, label="Residual proxy")
    axes[1, 1].set_title("Mean active gross return decomposition")
    axes[1, 1].set_ylabel("bp per active review")
    axes[1, 1].set_xticks(x, [labels[s] for s in STRATEGY_ORDER], rotation=18, ha="right")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _pct(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:+.{digits}f}%"


def _bp(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 10_000:+.{digits}f}"


def _num(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _html_table(frame: pd.DataFrame) -> str:
    labels = {
        DOLLAR: "Dollar neutral",
        BETA: "Beta neutral",
        SIGNAL: "Signal-weighted / unneutralized",
        LONG: "Long-only contribution",
        SHORT: "Short-only contribution",
    }
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[str(row.strategy)])}</td>"
            f"<td>{float(row.cost_bps):.0f}</td>"
            f"<td>{int(row.active_trade_dates)}</td>"
            f"<td>{_bp(row.active_mean_return)}</td>"
            f"<td>{_bp(row.active_unit_gross_mean_return)}</td>"
            f"<td>{_num(row.active_trade_sharpe_sqrt4)}</td>"
            f"<td>{_pct(row.active_trade_hit_rate, 1)}</td>"
            f"<td>{_pct(row.total_return)}</td>"
            f"<td>{_pct(row.max_drawdown)}</td>"
            f"<td>{_pct(row.mean_active_gross_exposure, 1)}</td>"
            f"<td>{_pct(row.mean_active_net_exposure, 1)}</td>"
            f"<td>{_num(row.mean_active_abs_estimated_beta_exposure, 3)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _report(metrics: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    ordered = metrics.assign(
        _strategy_order=metrics["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)})
    ).sort_values(["cost_bps", "_strategy_order"])
    zero = ordered.loc[ordered["cost_bps"].eq(0.0)]
    dollar = zero.loc[zero["strategy"].eq(DOLLAR)].iloc[0]
    long = zero.loc[zero["strategy"].eq(LONG)].iloc[0]
    short = zero.loc[zero["strategy"].eq(SHORT)].iloc[0]
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSCI closing-window portfolio constraint comparison</title>
<style>
body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#172033}}main{{max-width:1280px;margin:0 auto;padding:32px}}h1{{font-size:34px;margin-bottom:8px}}h2{{margin-top:34px}}.lede{{font-size:18px;color:#4b5870}}.callout{{padding:18px 20px;border-left:5px solid #1d6f50;background:white;border-radius:8px;margin:18px 0}}.warn{{border-color:#b07800}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.kpi{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 4px #0001}}.kpi b{{font-size:24px;display:block;margin-top:8px}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dfe4eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e8edf4}}img{{width:100%;background:white;border-radius:10px}}code{{background:#e8edf4;padding:2px 5px;border-radius:4px}}a{{color:#145ea8}}.small{{font-size:13px;color:#5d687b}}
</style></head><body><main>
<h1>MSCI 15:49–15:50 portfolio constraint comparison</h1>
<p class="lede">Same universe, same signal direction, same execution proxy; only the portfolio constraints change. Dollar neutral is the research control; the production candidate allows net exposure.</p>
<div class="callout"><b>Why a neutral control?</b><br>The MSCI add/delete direction is always kept. Neutralization removes only the common same-window market-beta component, testing whether returns come from mechanical index flow. If the beta-neutral portfolio stays profitable, the cross-sectional MSCI explanation is stronger; if returns vanish with beta, they depend on the market's direction.</div>
<div class="grid">
<div class="kpi">Dollar-neutral 0 bp active Sharpe<b>{_num(dollar['active_trade_sharpe_sqrt4'])}</b></div>
<div class="kpi">Dollar-neutral active mean<b>{_bp(dollar['active_mean_return'])} bp</b></div>
<div class="kpi">Long-leg active mean<b>{_bp(long['active_mean_return'])} bp</b></div>
<div class="kpi">Short-leg active mean<b>{_bp(short['active_mean_return'])} bp</b></div>
</div>
<h2>Five auditable portfolios</h2>
<table><thead><tr><th>Portfolio</th><th>Cost bp</th><th>Active</th><th>Active mean bp</th><th>Mean / gross bp</th><th>Active Sharpe</th><th>Active hit</th><th>Cumulative</th><th>Max DD</th><th>Gross</th><th>Net $</th><th>|Beta|</th></tr></thead><tbody>{_html_table(ordered)}</tbody></table>
<p class="small">Active means gross exposure &gt; 0. Long-only and short-only keep the baseline's ~50% leg capital for contribution analysis; Mean/gross divides return by realized gross exposure. 0 bp is the base case; 10 bp is the stress test.</p>
<h2>Equity, drawdown and exposure</h2><img src="charts/portfolio_variant_comparison.png" alt="portfolio variant comparison">
<h2>Weighting rules</h2>
<ul><li><b>Dollar neutral:</b> equal capital in add and delete legs, 10% single-name cap.</li><li><b>Beta neutral:</b> equal weight within each leg; leg notionals solved so ex-ante beta exposure is zero; net dollar exposure allowed.</li><li><b>Signal weighted:</b> allocated by <code>proxy_flow_pressure</code> across both sides without dollar/beta neutralization; weight above the 10% cap stays in cash so redistribution does not collapse to equal weight.</li><li><b>Long/short only:</b> keep only the add or delete leg of the dollar-neutral baseline to measure each leg's contribution.</li></ul>
<h2>Information timing</h2><p>Beta uses only <code>feature_cutoff_date</code> and up to 40 aligned daily returns before it, shrunk 50% toward 1 and clipped to [0.25, 2.50]; with fewer than 20 returns beta falls back to 1. Implementation-day SPY returns are used only for attribution, never for weights. Flow pressure is a proxy built from public data.</p>
<div class="callout warn"><b>Execution model</b><br>The start-stamped minute OHLCV bar is the execution proxy; NBBO, order book, auction/NOII, borrow and market impact are modeled separately. Results are a retrospective comparison.</div>
<p><a href="portfolio_variant_metrics.csv">Metrics CSV</a> · <a href="portfolio_variant_date_portfolios.csv">26-review ledger</a> · <a href="portfolio_variant_trade_log.csv">security trade log</a> · <a href="portfolio_variant_paired_tests.csv">paired comparisons</a> · <a href="portfolio_variant_manual_audit.json">manual audit</a></p>
<p class="small">Generated {datetime.now(UTC).isoformat()} · Paired comparison rows: {len(paired)}</p>
</main></body></html>"""
    output.write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trade-log",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_trade_log.csv"),
    )
    parser.add_argument(
        "--date-log",
        type=Path,
        default=Path("outputs/baseline/topn_capped10_date_trade_log.csv"),
    )
    parser.add_argument(
        "--marketdata-root", type=Path, default=Path("data/marketdata")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/portfolio_variants")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(
        args.trade_log,
        parse_dates=["effective_close_date", "feature_cutoff_date"],
    )
    if set(base["strategy"]) != {PRIMARY_INPUT}:
        raise ValueError(f"expected only the frozen {PRIMARY_INPUT} ledger")
    if not base["feature_point_in_time_valid"].fillna(False).all():
        raise ValueError("base selection contains non-point-in-time features")
    schedule = pd.read_csv(args.date_log, parse_dates=["effective_close_date"])
    scheduled_dates = sorted(schedule["effective_close_date"].dt.normalize().unique())
    base = _add_betas(base, args.marketdata_root)
    ledgers = _build_ledgers(base)
    portfolios, metrics, trades = _evaluate(
        ledgers, scheduled_dates, args.marketdata_root
    )
    paired = _paired_tests(portfolios)

    portfolios.to_csv(args.output_dir / "portfolio_variant_date_portfolios.csv", index=False)
    metrics.to_csv(args.output_dir / "portfolio_variant_metrics.csv", index=False)
    trades.to_csv(args.output_dir / "portfolio_variant_trade_log.csv", index=False)
    paired.to_csv(args.output_dir / "portfolio_variant_paired_tests.csv", index=False)
    beta_columns = [
        "effective_close_date",
        "ticker",
        "feature_cutoff_date",
        "beta_feature_asof",
        "beta_observations",
        "beta_raw",
        "beta_used",
        "beta_source",
    ]
    base[beta_columns].to_csv(
        args.output_dir / "portfolio_variant_beta_features.csv", index=False
    )
    _chart(portfolios, metrics, chart_dir / "portfolio_variant_comparison.png")

    manifest = {
        "version": "portfolio-constraint-comparison-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_trade_log": str(args.trade_log),
        "input_trade_log_sha256": _sha256(args.trade_log),
        "input_date_log": str(args.date_log),
        "input_date_log_sha256": _sha256(args.date_log),
        "research_clock": {
            "information_cutoff": "15:48:59.999 America/New_York",
            "entry_proxy": "open of start-stamped 15:49 minute bar",
            "exit_proxy": "close of start-stamped 15:50 minute bar (~15:50:59)",
            "beta_cutoff": "feature_cutoff_date, normally prior session close",
        },
        "weighting": {
            "universe": "frozen impact_proxy_top10 selection",
            "gross_target": 1.0,
            "max_name_weight": 0.10,
            "signal_score": "proxy_flow_pressure",
            "beta": "40 aligned daily returns, minimum 20, 50% shrink to 1, clip [0.25,2.50], unit fallback",
        },
        "cost_bps": list(COSTS),
        "strategies": STRATEGY_ORDER,
        "independent_unit": "MSCI implementation review date",
        "status": "retrospective_minute_bar_proxy_not_executable",
    }
    (args.output_dir / "portfolio_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    _report(metrics, paired, args.output_dir / "report.html")
    print(
        f"strategies={len(STRATEGY_ORDER)} costs={len(COSTS)} "
        f"reviews={len(scheduled_dates)} trades={len(trades)}"
    )


if __name__ == "__main__":
    main()
