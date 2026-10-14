#!/usr/bin/env python3
"""Build and validate point-in-time weights for the 15:50 MSCI burst.

The runner supports licensed net-flow inputs when supplied.  Without them it
produces clearly labelled liquidity proxies; those outputs must not be read as
estimates of actual MSCI tracker flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from msci_rebalance.backtest import add_equity_columns, moving_block_bootstrap_mean, performance_metrics
from msci_rebalance.first_minute import block_max_t_sign_flip_pvalue
from msci_rebalance.flow_features import build_pre_event_feature_panel
from msci_rebalance.weighted_backtest import build_capped_leg_weights, weighted_date_portfolio
from msci_rebalance.weighting import WeightingConfig, build_target_weights


COST_GRID_BPS = (0.0, 2.5, 5.0, 10.0, 15.0, 20.0)
BASELINE = "equal_leg_uncapped"
EQUAL_CAPPED = "equal_leg_10pct_cap"
EQUAL_PIT_CAPPED = "equal_pit_universe_10pct_cap"
EQUAL_PIT_CAPACITY = "equal_pit_universe_capacity"
SQRT_ADV = "sqrt_adv_10pct_cap"
SQRT_ADV_CAPACITY = "sqrt_adv_capacity"
PROXY_PRESSURE = "turnover_pressure_proxy_10pct_cap"
PROXY_CAPACITY = "turnover_pressure_proxy_capacity"
UNAMBIGUOUS_EQUAL = "unambiguous_new_exit_equal_capacity"
UNAMBIGUOUS = "unambiguous_new_exit_proxy_capacity"
LICENSED = "licensed_net_flow_pressure"
LICENSED_EQUAL = "licensed_universe_equal_capacity"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marketdata_stat_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.parquet")):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _validate_reused_features(features: pd.DataFrame) -> None:
    required = {
        "effective_close_date",
        "ticker",
        "feature_cutoff_date",
        "liquidity_asof",
        "median_daily_dollar_volume_20d",
        "expected_interval_dollar_volume",
        "daily_log_volatility_20d",
        "feature_point_in_time_valid",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"cached feature panel lacks fields: {sorted(missing)}")
    event_date = pd.to_datetime(features["effective_close_date"]).dt.normalize()
    cutoff = pd.to_datetime(features["feature_cutoff_date"], errors="coerce")
    liquidity_asof = pd.to_datetime(features["liquidity_asof"], errors="coerce")
    recomputed = (
        cutoff.lt(event_date)
        & liquidity_asof.lt(event_date)
        & pd.to_numeric(
            features["median_daily_dollar_volume_20d"], errors="coerce"
        ).gt(0)
        & pd.to_numeric(features["expected_interval_dollar_volume"], errors="coerce").gt(0)
        & pd.to_numeric(features["daily_log_volatility_20d"], errors="coerce").gt(0)
    )
    stored = features["feature_point_in_time_valid"].fillna(False).astype(bool)
    if not recomputed.equals(stored):
        raise ValueError("cached point-in-time validity flags do not match recomputation")
    if features.duplicated(["effective_close_date", "ticker"]).any():
        raise ValueError("cached feature identities are not unique")


def _template(path: Path) -> None:
    columns = [
        "effective_close_date",
        "security_id",
        "ticker",
        "product_id",
        "overlap_group",
        "benchmark",
        "signal_asof",
        "pre_weight",
        "post_weight",
        "delta_weight",
        "tracker_aum_usd",
        "replication_fraction",
        "predicted_net_flow_usd",
        "expected_interval_dollar_volume",
        "expected_interval_asof",
        "borrow_available",
    ]
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _licensed_security_flows(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {
        "effective_close_date",
        "security_id",
        "ticker",
        "signal_asof",
        "borrow_available",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"licensed flow input lacks fields: {sorted(missing)}")
    if raw.empty:
        raise ValueError("licensed flow input is empty")
    if raw[["security_id", "ticker", "signal_asof", "borrow_available"]].isna().any().any():
        raise ValueError("licensed security_id, ticker, signal_asof and borrow_available cannot be missing")
    raw["effective_close_date"] = pd.to_datetime(raw["effective_close_date"]).dt.normalize()
    raw["signal_asof"] = pd.to_datetime(raw["signal_asof"], errors="raise")
    if "expected_interval_asof" in raw:
        raw["expected_interval_asof"] = pd.to_datetime(
            raw["expected_interval_asof"], errors="coerce"
        )
    for column in (
        "pre_weight",
        "post_weight",
        "delta_weight",
        "tracker_aum_usd",
        "replication_fraction",
        "predicted_net_flow_usd",
        "expected_interval_dollar_volume",
    ):
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    if {"pre_weight", "post_weight"}.issubset(raw.columns):
        reconstructed = raw["post_weight"] - raw["pre_weight"]
        if "delta_weight" in raw:
            conflict = raw["delta_weight"].notna() & ~np.isclose(
                raw["delta_weight"], reconstructed, atol=1e-12, rtol=1e-8
            )
            if conflict.any():
                raise ValueError("delta_weight conflicts with post_weight - pre_weight")
            raw["delta_weight"] = raw["delta_weight"].fillna(reconstructed)
        else:
            raw["delta_weight"] = reconstructed

    predicted = (
        raw["predicted_net_flow_usd"].copy()
        if "predicted_net_flow_usd" in raw
        else pd.Series(np.nan, index=raw.index, dtype=float)
    )
    if {"delta_weight", "tracker_aum_usd"}.issubset(raw.columns):
        replication = (
            raw["replication_fraction"].fillna(1.0)
            if "replication_fraction" in raw
            else 1.0
        )
        derived = raw["tracker_aum_usd"] * replication * raw["delta_weight"]
        predicted = predicted.fillna(derived)
    if predicted.isna().any():
        raise ValueError(
            "every licensed row needs predicted_net_flow_usd or delta_weight × tracker_aum_usd"
        )
    raw["predicted_net_flow_usd"] = predicted

    if "product_id" in raw and raw["product_id"].notna().any():
        product_keys = ["effective_close_date", "security_id", "product_id"]
        if "benchmark" in raw:
            product_keys.append("benchmark")
        if raw.duplicated(product_keys).any():
            raise ValueError("duplicate licensed security/product/benchmark exposure rows")

    def parse_borrow(value: object) -> bool:
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "available"}:
            return True
        if normalized in {"false", "0", "no", "n", "unavailable"}:
            return False
        raise ValueError(f"unrecognized borrow_available value: {value}")

    raw["borrow_available"] = raw["borrow_available"].map(parse_borrow)
    group_keys = ["effective_close_date", "security_id", "ticker"]
    if "expected_interval_dollar_volume" in raw:
        interval_unique = raw.groupby(group_keys)["expected_interval_dollar_volume"].nunique(
            dropna=True
        )
        if interval_unique.gt(1).any():
            raise ValueError("licensed interval liquidity conflicts within a security/date")
        supplied_interval = raw["expected_interval_dollar_volume"].notna()
        if supplied_interval.any() and (
            "expected_interval_asof" not in raw
            or raw.loc[supplied_interval, "expected_interval_asof"].isna().any()
        ):
            raise ValueError("licensed interval liquidity requires expected_interval_asof")

    aggregations: dict[str, object] = {
        "predicted_net_flow_usd": lambda values: values.sum(min_count=1),
        "signal_asof": "max",
        "borrow_available": "min",
    }
    if "expected_interval_asof" in raw:
        aggregations["expected_interval_asof"] = "max"
    if "expected_interval_dollar_volume" in raw:
        aggregations["expected_interval_dollar_volume"] = "first"
    return raw.groupby(group_keys, as_index=False, dropna=False).agg(aggregations)


def _base_proxy_candidates(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    direction = np.where(frame["action"].eq("add"), 1.0, -1.0)
    adv = pd.to_numeric(frame["median_daily_dollar_volume_20d"], errors="coerce")
    frame["predicted_net_flow_usd"] = direction * adv
    frame["pre_event_adv20_usd"] = adv
    frame["pre_event_interval_dollar_volume"] = frame["expected_interval_dollar_volume"]
    frame["signal_asof"] = frame["liquidity_asof"]
    frame["expected_interval_asof"] = frame["liquidity_asof"]
    frame["flow_input_type"] = "daily_turnover_proxy_not_msci_flow"
    frame["signal_kind"] = "inverse_clock_volume_share_proxy"
    return frame


def _module_weights(
    candidates: pd.DataFrame,
    *,
    capital_usd: float,
    max_name_weight: float,
    participation: float,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["signal_freeze_ts"] = (
        pd.to_datetime(candidates["effective_close_date"]).dt.normalize()
        + pd.Timedelta(hours=15, minutes=48, seconds=30)
    )
    result = build_target_weights(
        candidates,
        WeightingConfig(
            capital_usd=capital_usd,
            gross_target=1.0,
            max_name_weight=max_name_weight,
            max_interval_participation=participation,
            missing_liquidity_max_weight=min(0.01, max_name_weight),
            require_both_legs=True,
            require_asof_for_true_inputs=True,
        ),
    )
    weights = result.weights.copy()
    weights.insert(0, "strategy", strategy)
    diagnostics = result.diagnostics.copy()
    diagnostics.insert(0, "strategy", strategy)
    return weights, diagnostics


def _scored_weights(
    features: pd.DataFrame,
    score: pd.Series | None,
    strategy: str,
    max_name_weight: float,
    require_both_legs: bool,
    cap: pd.Series | None = None,
) -> pd.DataFrame:
    frame = features.copy()
    score_column = None
    if score is not None:
        frame["allocation_input"] = pd.to_numeric(score, errors="coerce")
        score_column = "allocation_input"
    cap_column = None
    if cap is not None:
        frame["allocation_cap"] = pd.to_numeric(cap, errors="coerce")
        cap_column = "allocation_cap"
    weights = build_capped_leg_weights(
        frame,
        score_column,
        max_name_weight=max_name_weight,
        cap_column=cap_column,
        require_both_legs=require_both_legs,
    )
    weights.insert(0, "strategy", strategy)
    return weights


def _evaluate(
    ledgers: list[pd.DataFrame],
    costs: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scheduled_dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for ledger in ledgers
            for value in ledger["effective_close_date"].dropna().unique()
        }
    )
    portfolio_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for ledger in ledgers:
        strategy = str(ledger["strategy"].iloc[0])
        for cost in costs:
            portfolio = weighted_date_portfolio(
                ledger,
                cost_bps=cost,
                scheduled_dates=scheduled_dates,
            )
            portfolio.insert(0, "strategy", strategy)
            portfolio = add_equity_columns(portfolio)
            portfolio_rows.append(portfolio)
            metrics = performance_metrics(portfolio, "net_return")
            metric_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost,
                    **metrics,
                    "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
                    "mean_max_abs_name_weight": float(portfolio["max_abs_name_weight"].mean()),
                    "mean_effective_names": float(portfolio["effective_names"].mean()),
                }
            )
    return pd.concat(portfolio_rows, ignore_index=True), pd.DataFrame(metric_rows)


def _paired_tests(portfolios: pd.DataFrame) -> pd.DataFrame:
    gross = portfolios.loc[portfolios["cost_bps"].eq(0.0)].pivot(
        index="effective_close_date", columns="strategy", values="gross_return"
    )
    rows: list[dict[str, object]] = []
    capped_alternatives: list[str] = []
    difference_long: list[pd.DataFrame] = []
    for strategy in gross.columns:
        if strategy == BASELINE:
            continue
        comparator_map = {
            EQUAL_CAPPED: BASELINE,
            EQUAL_PIT_CAPPED: EQUAL_CAPPED,
            EQUAL_PIT_CAPACITY: EQUAL_PIT_CAPPED,
            SQRT_ADV: EQUAL_PIT_CAPPED,
            PROXY_PRESSURE: EQUAL_PIT_CAPPED,
            SQRT_ADV_CAPACITY: EQUAL_PIT_CAPACITY,
            PROXY_CAPACITY: EQUAL_PIT_CAPACITY,
            UNAMBIGUOUS_EQUAL: EQUAL_PIT_CAPACITY,
            UNAMBIGUOUS: UNAMBIGUOUS_EQUAL,
            LICENSED_EQUAL: EQUAL_PIT_CAPACITY,
            LICENSED: LICENSED_EQUAL,
        }
        comparator = comparator_map.get(strategy, BASELINE)
        if comparator not in gross:
            comparator = BASELINE
        pair = gross[[strategy, comparator]].dropna()
        difference = pair[strategy] - pair[comparator]
        low, high = moving_block_bootstrap_mean(difference.to_numpy(float), block_size=4)
        test = stats.ttest_1samp(difference, 0.0, alternative="greater") if len(difference) > 1 else None
        rows.append(
            {
                "strategy": strategy,
                "comparator": comparator,
                "review_dates": len(difference),
                "mean_difference": float(difference.mean()),
                "median_difference": float(difference.median()),
                "positive_difference_rate": float(difference.gt(0).mean()),
                "one_sided_t_pvalue": float(test.pvalue) if test else np.nan,
                "block4_mean_ci_low": low,
                "block4_mean_ci_high": high,
            }
        )
        if strategy not in {
            EQUAL_CAPPED,
            EQUAL_PIT_CAPPED,
            EQUAL_PIT_CAPACITY,
            UNAMBIGUOUS_EQUAL,
            LICENSED_EQUAL,
        }:
            capped_alternatives.append(strategy)
            difference_long.append(
                pd.DataFrame(
                    {
                        "effective_close_date": difference.index,
                        "window": strategy,
                        "gross_return": difference.to_numpy(float),
                    }
                )
            )
    output = pd.DataFrame(rows)
    if difference_long:
        family = pd.concat(difference_long, ignore_index=True)
        for strategy in capped_alternatives:
            pvalue, dates, candidates = block_max_t_sign_flip_pvalue(
                family, strategy, block_size=4
            )
            output.loc[output["strategy"].eq(strategy), "block4_max_t_familywise_pvalue"] = pvalue
            output.loc[output["strategy"].eq(strategy), "family_complete_dates"] = dates
            output.loc[output["strategy"].eq(strategy), "family_candidates"] = candidates
    return output


def _regime_metrics(portfolios: pd.DataFrame) -> pd.DataFrame:
    frame = portfolios.copy()
    frame["effective_close_date"] = pd.to_datetime(frame["effective_close_date"])
    regimes = {
        "full": pd.Series(True, index=frame.index),
        "2020_through_2022": frame["effective_close_date"].lt("2023-01-01"),
        "2023_through_2026": frame["effective_close_date"].ge("2023-01-01"),
        "pre_full_qcir_before_may_2023": frame["effective_close_date"].lt("2023-05-01"),
        "full_qcir_from_may_2023": frame["effective_close_date"].ge("2023-05-01"),
        "excluding_aug_2026_discovery": ~frame["effective_close_date"].eq("2026-08-31"),
    }
    rows: list[dict[str, object]] = []
    for (strategy, cost), group in frame.groupby(["strategy", "cost_bps"], sort=True):
        for label in regimes:
            mask = regimes[label].loc[group.index]
            sample = group.loc[mask].copy()
            if sample.empty:
                continue
            metrics = performance_metrics(sample, "net_return")
            rows.append(
                {
                    "strategy": strategy,
                    "sample": label,
                    "cost_bps": cost,
                    **metrics,
                    "mean_gross_exposure": float(sample["gross_exposure"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _loo_stability(portfolios: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    gross = portfolios.loc[portfolios["cost_bps"].eq(0.0)].pivot(
        index="effective_close_date", columns="strategy", values="gross_return"
    )
    rows: list[dict[str, object]] = []
    for comparison in paired.itertuples(index=False):
        pair = gross[[comparison.strategy, comparison.comparator]].dropna()
        difference = pair[comparison.strategy] - pair[comparison.comparator]
        loo = pd.Series(
            [difference.drop(index=date).mean() for date in difference.index],
            index=difference.index,
        )
        rows.append(
            {
                "strategy": comparison.strategy,
                "comparator": comparison.comparator,
                "review_dates": len(difference),
                "full_mean_difference": float(difference.mean()),
                "loo_mean_difference_min": float(loo.min()),
                "loo_mean_difference_max": float(loo.max()),
                "loo_mean_difference_median": float(loo.median()),
                "loo_negative_count": int(loo.lt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _name_concentration(
    ledgers: list[pd.DataFrame],
    ticker: str,
    event_date: str,
) -> pd.DataFrame:
    target_date = pd.Timestamp(event_date).normalize()
    scheduled_date_count = len(
        {
            pd.Timestamp(value).normalize()
            for ledger in ledgers
            for value in ledger["effective_close_date"].dropna().unique()
        }
    )
    rows: list[dict[str, object]] = []
    for ledger in ledgers:
        strategy = str(ledger["strategy"].iloc[0])
        date_group = ledger.loc[
            pd.to_datetime(ledger["effective_close_date"]).dt.normalize().eq(target_date)
        ].copy()
        name = date_group.loc[date_group["ticker"].eq(ticker)]
        if name.empty:
            continue
        contribution = float(
            name["target_weight"].iloc[0] * name["underlying_return"].iloc[0]
        )
        date_return = float(
            (date_group["target_weight"] * date_group["underlying_return"]).sum()
        )
        all_contributions = ledger["target_weight"] * ledger["underlying_return"]
        rows.append(
            {
                "strategy": strategy,
                "effective_close_date": target_date,
                "ticker": ticker,
                "target_weight": float(name["target_weight"].iloc[0]),
                "pnl_contribution": contribution,
                "date_gross_return": date_return,
                "share_of_date_gross_pnl": contribution / date_return
                if date_return != 0
                else np.nan,
                "full_sample_mean_without_name_no_renormalization": float(
                    (all_contributions.sum() - contribution) / scheduled_date_count
                ),
            }
        )
    return pd.DataFrame(rows)


def _capacity_summary(
    ledgers: list[pd.DataFrame],
    capital_usd: float,
    max_interval_participation: float,
) -> pd.DataFrame:
    scheduled_dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for ledger in ledgers
            for value in ledger["effective_close_date"].dropna().unique()
        }
    )
    rows: list[dict[str, object]] = []
    for ledger in ledgers:
        strategy = str(ledger["strategy"].iloc[0])
        frame = ledger.copy()
        frame["active"] = frame["target_weight"].abs().gt(0)
        frame["position_usd"] = frame["target_weight"].abs() * capital_usd
        interval_column = (
            "expected_interval_dollar_volume_used"
            if "expected_interval_dollar_volume_used" in frame
            else "expected_interval_dollar_volume"
        )
        interval = pd.to_numeric(frame[interval_column], errors="coerce")
        adv = pd.to_numeric(frame["median_daily_dollar_volume_20d"], errors="coerce")
        frame["interval_participation"] = frame["position_usd"] / interval
        frame["adv_participation"] = frame["position_usd"] / adv
        active = frame.loc[frame["active"]].copy()
        active["_interval_used"] = interval.loc[active.index]
        date_capacity: list[float] = []
        for _, group in active.groupby("effective_close_date"):
            ratio = (
                max_interval_participation
                * pd.to_numeric(group[interval_column], errors="coerce")
                / group["target_weight"].abs()
            )
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                date_capacity.append(float(ratio.min()))
        interval_values = active["interval_participation"].replace([np.inf, -np.inf], np.nan).dropna()
        adv_values = active["adv_participation"].replace([np.inf, -np.inf], np.nan).dropna()
        gross_by_date = active.groupby("effective_close_date")["target_weight"].apply(
            lambda values: float(values.abs().sum())
        ).reindex(scheduled_dates, fill_value=0.0)
        missing_interval = ~np.isfinite(active["_interval_used"]) | active[
            "_interval_used"
        ].le(0)
        rows.append(
            {
                "strategy": strategy,
                "active_positions": len(active),
                "mean_deployed_gross": float(gross_by_date.mean()) if len(gross_by_date) else 0.0,
                "scheduled_review_dates": len(scheduled_dates),
                "interval_participation_limit": max_interval_participation,
                "unknown_interval_active_names": int(missing_interval.sum()),
                "unknown_interval_target_weight": float(
                    active.loc[missing_interval, "target_weight"].abs().sum()
                ),
                "interval_participation_p50": float(interval_values.quantile(0.50)) if len(interval_values) else np.nan,
                "interval_participation_p95": float(interval_values.quantile(0.95)) if len(interval_values) else np.nan,
                "interval_participation_max": float(interval_values.max()) if len(interval_values) else np.nan,
                "adv_participation_p95": float(adv_values.quantile(0.95)) if len(adv_values) else np.nan,
                "adv_participation_max": float(adv_values.max()) if len(adv_values) else np.nan,
                "median_date_capacity_at_5pct_interval_usd": float(np.median(date_capacity))
                if date_capacity
                else np.nan,
                "minimum_date_capacity_at_5pct_interval_usd": float(np.min(date_capacity))
                if date_capacity
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _chart(portfolios: pd.DataFrame, output: Path) -> None:
    selected = portfolios.loc[portfolios["cost_bps"].eq(10.0)].copy()
    selected = selected.loc[
        ~selected["strategy"].isin([UNAMBIGUOUS_EQUAL, UNAMBIGUOUS])
    ]
    labels = {
        BASELINE: "Equal baseline",
        EQUAL_CAPPED: "Equal + 10% cap",
        EQUAL_PIT_CAPPED: "Equal + cap (PIT universe)",
        EQUAL_PIT_CAPACITY: "Equal + 5% interval cap",
        SQRT_ADV: "√ADV + 10% cap",
        SQRT_ADV_CAPACITY: "√ADV + 5% interval cap",
        PROXY_PRESSURE: "Inverse clock-volume-share proxy",
        PROXY_CAPACITY: "Clock-share proxy + 5% interval cap",
        UNAMBIGUOUS: "New/exit only + capacity",
        UNAMBIGUOUS_EQUAL: "New/exit equal + capacity",
        LICENSED: "Licensed net flow",
        LICENSED_EQUAL: "Licensed universe equal + capacity",
    }
    fig, (ax, exposure_ax) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for strategy, group in selected.groupby("strategy"):
        ax.plot(
            pd.to_datetime(group["effective_close_date"]),
            group["equity"],
            linewidth=2,
            label=labels.get(strategy, strategy),
        )
        if strategy in {BASELINE, EQUAL_PIT_CAPACITY, PROXY_CAPACITY}:
            exposure_ax.plot(
                pd.to_datetime(group["effective_close_date"]),
                group["gross_exposure"],
                linewidth=1.8,
                label=labels.get(strategy, strategy),
            )
    ax.axhline(1.0, color="0.4", linewidth=1)
    ax.set_title("MSCI 15:50 burst: point-in-time weighting comparisons")
    ax.set_ylabel("Equity at 10 bp cost")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    exposure_ax.set_ylabel("Gross")
    exposure_ax.set_xlabel("Implementation close")
    exposure_ax.set_ylim(-0.02, 1.05)
    exposure_ax.grid(axis="y", alpha=0.25)
    exposure_ax.legend(frameon=False, fontsize=8)
    exposure_ax.xaxis.set_major_locator(mdates.YearLocator())
    exposure_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--security-ledger", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--licensed-flow-input", type=Path)
    parser.add_argument("--capital-usd", type=float, default=1_000_000.0)
    parser.add_argument("--max-name-weight", type=float, default=0.10)
    parser.add_argument("--max-interval-participation", type=float, default=0.05)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    template_path = args.output_dir / "licensed_flow_input_template.csv"
    _template(template_path)

    security = pd.read_csv(args.security_ledger)
    security["effective_close_date"] = pd.to_datetime(security["effective_close_date"])
    feature_path = args.output_dir / "flow_weighting_feature_panel.csv"
    feature_metadata_path = args.output_dir / "flow_weighting_feature_metadata.json"
    feature_code_path = Path(__file__).parents[1] / "src/msci_rebalance/flow_features.py"
    expected_feature_metadata = {
        "schema_version": 1,
        "security_ledger_sha256": _sha256(args.security_ledger),
        "feature_code_sha256": _sha256(feature_code_path),
        "marketdata_stat_fingerprint": _marketdata_stat_fingerprint(args.marketdata_dir),
        "information_cutoff": "strictly before each implementation session",
    }
    if args.reuse_features and feature_path.exists():
        if not feature_metadata_path.exists():
            raise ValueError("--reuse-features requires flow_weighting_feature_metadata.json")
        stored_metadata = json.loads(feature_metadata_path.read_text())
        for key, expected in expected_feature_metadata.items():
            if stored_metadata.get(key) != expected:
                raise ValueError(f"stale feature cache: {key} changed")
        features = pd.read_csv(feature_path, parse_dates=["effective_close_date"])
        _validate_reused_features(features)
    else:
        features = build_pre_event_feature_panel(security, args.marketdata_dir)
        _validate_reused_features(features)
        features.to_csv(feature_path, index=False)
        feature_metadata_path.write_text(
            json.dumps(
                {
                    **expected_feature_metadata,
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "feature_rows": len(features),
                },
                indent=2,
            )
        )

    valid = features["feature_point_in_time_valid"].fillna(False)
    adv = pd.to_numeric(features["median_daily_dollar_volume_20d"], errors="coerce")
    equal = _scored_weights(features, None, BASELINE, 0.50, False)
    equal_cap = _scored_weights(features, None, EQUAL_CAPPED, args.max_name_weight, True)
    equal_pit_cap = _scored_weights(
        features.loc[valid].copy(),
        None,
        EQUAL_PIT_CAPPED,
        args.max_name_weight,
        True,
    )
    sqrt_adv = _scored_weights(
        features,
        np.sqrt(adv.where(valid)),
        SQRT_ADV,
        args.max_name_weight,
        True,
    )
    interval_cap = (
        pd.to_numeric(features["expected_interval_dollar_volume"], errors="coerce")
        * args.max_interval_participation
        / args.capital_usd
    ).where(valid)
    equal_pit_capacity = _scored_weights(
        features,
        valid.astype(float).where(valid),
        EQUAL_PIT_CAPACITY,
        args.max_name_weight,
        True,
        cap=interval_cap,
    )
    sqrt_adv_capacity = _scored_weights(
        features,
        np.sqrt(adv.where(valid)),
        SQRT_ADV_CAPACITY,
        args.max_name_weight,
        True,
        cap=interval_cap,
    )

    proxy = _base_proxy_candidates(features.loc[valid].copy())
    proxy_pressure, proxy_pressure_diag = _module_weights(
        proxy,
        capital_usd=1.0,
        max_name_weight=args.max_name_weight,
        participation=1e12,
        strategy=PROXY_PRESSURE,
    )
    proxy_capacity, proxy_capacity_diag = _module_weights(
        proxy,
        capital_usd=args.capital_usd,
        max_name_weight=args.max_name_weight,
        participation=args.max_interval_participation,
        strategy=PROXY_CAPACITY,
    )
    unambiguous_proxy = proxy.loc[
        proxy["change_type"].isin(["new_to_standard", "exit_standard"])
    ].copy()
    unambiguous_cap = (
        pd.to_numeric(
            unambiguous_proxy["expected_interval_dollar_volume"], errors="coerce"
        )
        * args.max_interval_participation
        / args.capital_usd
    )
    unambiguous_equal = _scored_weights(
        unambiguous_proxy,
        None,
        UNAMBIGUOUS_EQUAL,
        args.max_name_weight,
        True,
        cap=unambiguous_cap,
    )
    unambiguous, unambiguous_diag = _module_weights(
        unambiguous_proxy,
        capital_usd=args.capital_usd,
        max_name_weight=args.max_name_weight,
        participation=args.max_interval_participation,
        strategy=UNAMBIGUOUS,
    )

    ledgers = [
        equal,
        equal_cap,
        equal_pit_cap,
        equal_pit_capacity,
        sqrt_adv,
        sqrt_adv_capacity,
        proxy_pressure,
        proxy_capacity,
        unambiguous_equal,
        unambiguous,
    ]
    diagnostics = [proxy_pressure_diag, proxy_capacity_diag, unambiguous_diag]
    licensed_status = "not_supplied"
    if args.licensed_flow_input:
        licensed_input = _licensed_security_flows(args.licensed_flow_input)
        licensed_candidates = features.merge(
            licensed_input,
            on=["effective_close_date", "ticker"],
            how="inner",
            suffixes=("", "_licensed"),
        )
        licensed_candidates["safe_pre_event_interval_dollar_volume"] = licensed_candidates[
            "expected_interval_dollar_volume"
        ]
        licensed_candidates["safe_pre_event_interval_asof"] = licensed_candidates[
            "liquidity_asof"
        ]
        for column in (
            "predicted_net_flow_usd",
            "signal_asof",
            "expected_interval_dollar_volume",
            "expected_interval_asof",
        ):
            licensed_column = f"{column}_licensed"
            if licensed_column in licensed_candidates:
                licensed_candidates[column] = licensed_candidates[licensed_column].combine_first(
                    licensed_candidates.get(column)
                )
        licensed_candidates["pre_event_adv20_usd"] = licensed_candidates[
            "median_daily_dollar_volume_20d"
        ]
        licensed_candidates["pre_event_interval_dollar_volume"] = licensed_candidates[
            "safe_pre_event_interval_dollar_volume"
        ]
        licensed_candidates["liquidity_asof"] = licensed_candidates[
            "safe_pre_event_interval_asof"
        ]
        licensed_candidates["signal_kind"] = "licensed_net_flow"
        licensed_cap = (
            pd.to_numeric(
                licensed_candidates["pre_event_interval_dollar_volume"], errors="coerce"
            )
            * args.max_interval_participation
            / args.capital_usd
        )
        licensed_equal = _scored_weights(
            licensed_candidates,
            None,
            LICENSED_EQUAL,
            args.max_name_weight,
            True,
            cap=licensed_cap,
        )
        licensed, licensed_diag = _module_weights(
            licensed_candidates,
            capital_usd=args.capital_usd,
            max_name_weight=args.max_name_weight,
            participation=args.max_interval_participation,
            strategy=LICENSED,
        )
        ledgers.append(licensed_equal)
        ledgers.append(licensed)
        diagnostics.append(licensed_diag)
        licensed_status = "supplied_and_evaluated"

    all_weights = pd.concat(ledgers, ignore_index=True, sort=False)
    all_weights.to_csv(args.output_dir / "flow_weighting_security_weights.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        args.output_dir / "flow_weighting_allocation_diagnostics.csv", index=False
    )
    portfolios, metrics = _evaluate(ledgers, COST_GRID_BPS)
    portfolios.to_csv(args.output_dir / "flow_weighting_date_portfolios.csv", index=False)
    metrics.to_csv(args.output_dir / "flow_weighting_metrics.csv", index=False)
    paired = _paired_tests(portfolios)
    paired.to_csv(args.output_dir / "flow_weighting_paired_tests.csv", index=False)
    regimes = _regime_metrics(portfolios)
    regimes.to_csv(args.output_dir / "flow_weighting_regime_metrics.csv", index=False)
    loo = _loo_stability(portfolios, paired)
    loo.to_csv(args.output_dir / "flow_weighting_loo_stability.csv", index=False)
    concentration = _name_concentration(ledgers, "SNDK", "2026-08-31")
    concentration.to_csv(
        args.output_dir / "flow_weighting_sndk_concentration.csv", index=False
    )
    capacity = _capacity_summary(
        ledgers,
        args.capital_usd,
        args.max_interval_participation,
    )
    capacity.to_csv(args.output_dir / "flow_weighting_capacity_summary.csv", index=False)
    _chart(portfolios, chart_dir / "flow_weighting_comparison.png")

    proxy_test = paired.loc[paired["strategy"].eq(PROXY_PRESSURE)].iloc[0]
    system_spec = {
        "version": "flow-weighting-v1",
        "status": "research_control_plus_shadow_challenger",
        "selection_decision": (
            "Retain exact-universe capped equal weight as the research control; "
            "shadow the inverse-clock-volume-share proxy; select no proxy for live use."
        ),
        "signal_freeze": "15:48:30 America/New_York on the implementation-close session",
        "research_control": {
            "strategy": EQUAL_PIT_CAPPED,
            "direction": "published add=long, published delete=short",
            "allocation": "equal within each leg",
            "max_name_weight": args.max_name_weight,
            "leg_policy": "50% long and 50% short targets; scale both to the constrained leg",
        },
        "shadow_proxy": {
            "strategy": PROXY_PRESSURE,
            "definition": (
                "ADV20 / expected normal 15:49-15:50 dollar volume; this is inverse "
                "clock-volume share, not estimated MSCI flow"
            ),
            "paired_mean_improvement": float(proxy_test["mean_difference"]),
            "block4_ci": [
                float(proxy_test["block4_mean_ci_low"]),
                float(proxy_test["block4_mean_ci_high"]),
            ],
            "current_five_scheme_block_max_t_pvalue": float(
                proxy_test["block4_max_t_familywise_pvalue"]
            ),
            "warning": (
                "post-hoc family does not cover all research choices; incremental return "
                "is concentrated before the May-2023 full-QCIR regime"
            ),
        },
        "licensed_production_rule": {
            "net_flow_formula": (
                "F_i = sum_product AUM_product * replication_fraction_product * "
                "(post_weight_i_product - pre_weight_i_product)"
            ),
            "cross_index_policy": (
                "net Standard, Small Cap and overlapping tracker flows before taking direction"
            ),
            "direction": "sign(F_i)",
            "allocation_score": "abs(F_i) / expected pre-event interval dollar volume",
            "max_name_weight": args.max_name_weight,
            "max_interval_participation": args.max_interval_participation,
            "borrow_policy": "known unavailable shorts receive zero capacity; unknown borrow is flagged",
            "cash_policy": "leave residual cash when name, liquidity or sparse-leg caps bind",
        },
        "execution_window": {
            "entry_proxy": "start-stamped 15:49 bar open",
            "exit_proxy": "start-stamped 15:50 bar close",
            "production_requirement": "ask-in/bid-out NBBO replay with latency, size, impact and borrow",
        },
        "validation": {
            "independent_unit": "implementation review date",
            "current_reviews": 26,
            "in_sample_status": "all weighting comparisons are retrospective",
            "inference": "paired four-review blocks plus max-t across frozen candidates",
            "regime_gate": "report before and after May 2023 separately",
            "future_gate": "freeze before the next review and accumulate 8 future quarterly reviews",
            "purged_kfold": "not used; quarterly review labels do not overlap and date count is too small",
        },
        "required_inputs": [
            "licensed point-in-time MSCI pre/post weights, NOS, FIF and amendments",
            "tracker product AUM and replication fractions with overlap de-duplication",
            "pre-event same-clock executable liquidity forecast",
            "point-in-time primary venue, NBBO depth, halts and auction messages",
            "short locate availability and borrow cost",
        ],
    }
    system_spec_path = args.output_dir / "flow_weighting_system_spec.json"
    system_spec_path.write_text(json.dumps(system_spec, indent=2))

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "proxy_weighting_research; licensed_flow_required_for_production_claim",
        "information_cutoff": "strictly before implementation session for all proxy features",
        "capital_usd": args.capital_usd,
        "max_name_weight": args.max_name_weight,
        "max_interval_participation": args.max_interval_participation,
        "licensed_flow_status": licensed_status,
        "strategies": [str(frame["strategy"].iloc[0]) for frame in ledgers],
        "feature_rows": len(features),
        "point_in_time_valid_rows": int(valid.sum()),
        "migration_rows_requiring_cross_index_netting": int(
            features["change_type"].isin(["up_from_small", "down_to_small"]).sum()
        ),
        "security_ledger_sha256": _sha256(args.security_ledger),
        "feature_panel_sha256": _sha256(feature_path),
        "feature_metadata_sha256": _sha256(feature_metadata_path),
        "marketdata_stat_fingerprint": expected_feature_metadata[
            "marketdata_stat_fingerprint"
        ],
        "licensed_template": str(template_path),
        "system_spec": str(system_spec_path),
        "system_spec_sha256": _sha256(system_spec_path),
    }
    (args.output_dir / "flow_weighting_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(
        f"features={len(features)} pit_valid={int(valid.sum())} "
        f"strategies={len(ledgers)} licensed={licensed_status}"
    )


if __name__ == "__main__":
    main()
