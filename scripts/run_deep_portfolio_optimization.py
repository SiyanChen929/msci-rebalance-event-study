#!/usr/bin/env python3
"""Run a bounded, point-in-time portfolio optimization on the MSCI burst proxy."""

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
from scipy import stats

from msci_rebalance.backtest import (
    active_trade_metrics,
    add_equity_columns,
    moving_block_bootstrap_mean,
    performance_metrics,
)
from msci_rebalance.deep_portfolio import (
    append_spy_beta_hedge,
    build_asymmetric_leg_weights,
    expanding_portfolio_selection,
    select_dynamic_score_coverage,
    select_top_k_per_side,
)
from msci_rebalance.first_minute import block_max_t_sign_flip_pvalue
from msci_rebalance.portfolio_variants import (
    build_beta_neutral_equal_weights,
    estimate_pre_event_beta,
)
from msci_rebalance.ranking import (
    add_liquidity_volatility_proxy_scores,
    validate_pre_event_features,
)
from msci_rebalance.state_filters import apply_filter_and_balance, validate_state_features
from msci_rebalance.weighted_backtest import build_capped_leg_weights, weighted_date_portfolio


BASELINE = "baseline_top10_cap10_dollar_neutral"
PRIMARY_COST = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _spy_window_return(path: Path, event_date: pd.Timestamp) -> float:
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
    start = float(entry.iloc[0]["open"])
    end = float(exit_.iloc[0]["close"])
    return end / start - 1.0 if min(start, end) > 0 else np.nan


def _attach_betas(
    frame: pd.DataFrame,
    cached: pd.DataFrame,
    marketdata_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["effective_close_date", "ticker"]
    beta_columns = key + [
        "beta_raw",
        "beta_used",
        "beta_observations",
        "beta_feature_asof",
        "beta_source",
    ]
    cache = cached[beta_columns].copy()
    cache["effective_close_date"] = pd.to_datetime(cache["effective_close_date"]).dt.normalize()
    output = frame.merge(cache, on=key, how="left", validate="one_to_one")
    missing = output["beta_used"].isna()
    additions: list[dict[str, object]] = []
    for row in output.loc[missing].itertuples(index=False):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        cutoff = pd.Timestamp(row.feature_cutoff_date).normalize()
        stem = f"{event_date.date()}_{_safe_ticker(str(row.ticker))}.parquet"
        beta = estimate_pre_event_beta(
            marketdata_root / "daily" / stem,
            marketdata_root / "daily" / f"{event_date.date()}_SPY.parquet",
            cutoff,
        )
        additions.append({"effective_close_date": event_date, "ticker": row.ticker, **beta})
    if additions:
        added = pd.DataFrame(additions)
        for column in beta_columns[2:]:
            mapping = added.set_index(key)[column]
            output.loc[missing, column] = pd.MultiIndex.from_frame(
                output.loc[missing, key]
            ).map(mapping)
        cache = pd.concat([cache, added], ignore_index=True).drop_duplicates(key, keep="last")
    if output["beta_used"].isna().any():
        raise ValueError("beta attachment left missing selected securities")
    feature_asof = pd.to_datetime(output["beta_feature_asof"], errors="coerce")
    cutoff = pd.to_datetime(output["feature_cutoff_date"]).dt.normalize()
    if (feature_asof.notna() & feature_asof.dt.normalize().gt(cutoff)).any():
        raise ValueError("beta feature timestamp exceeds its point-in-time cutoff")
    return output, cache.sort_values(key).reset_index(drop=True)


def _label(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    output = frame.copy()
    output["strategy"] = strategy
    if "instrument_role" not in output:
        output["instrument_role"] = "event_stock"
    output["target_abs_weight"] = output["target_weight"].abs()
    return output


def _build_candidates(
    feature_panel: pd.DataFrame,
    frozen_top10: pd.DataFrame,
    beta_cache: pd.DataFrame,
    state_panel: pd.DataFrame,
    marketdata_root: Path,
    spec: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    controls = spec["portfolio_controls"]
    minimum = int(spec["minimum_names_per_side"])
    scored = add_liquidity_volatility_proxy_scores(feature_panel)
    scored["impact_proxy"] = scored["volatility_scaled_pressure_proxy"]
    scored["inverse_vol_score"] = 1.0 / pd.to_numeric(
        scored["daily_log_volatility_20d"], errors="coerce"
    ).clip(lower=0.005)

    top6, d6 = select_top_k_per_side(
        scored, "impact_proxy", k_per_side=3, minimum_per_side=minimum
    )
    top10, d10 = select_top_k_per_side(
        scored, "impact_proxy", k_per_side=5, minimum_per_side=minimum
    )
    top20, d20 = select_top_k_per_side(
        scored,
        "impact_proxy",
        k_per_side=int(controls["dynamic_max_names_per_side"]),
        minimum_per_side=minimum,
    )
    dynamic, ddynamic = select_dynamic_score_coverage(
        scored,
        "impact_proxy",
        coverage=float(controls["dynamic_score_coverage"]),
        minimum_per_side=minimum,
        maximum_per_side=int(controls["dynamic_max_names_per_side"]),
    )

    key = ["effective_close_date", "ticker"]
    frozen = frozen_top10.copy()
    frozen["effective_close_date"] = pd.to_datetime(frozen["effective_close_date"]).dt.normalize()
    selected_keys = set(map(tuple, top10[key].to_numpy()))
    frozen_keys = set(map(tuple, frozen[key].to_numpy()))
    if selected_keys != frozen_keys:
        raise AssertionError(
            f"rebuilt top10 membership differs from frozen ledger: {len(selected_keys ^ frozen_keys)}"
        )

    top20_beta, expanded_beta = _attach_betas(top20, beta_cache, marketdata_root)
    beta_columns = key + [
        "beta_raw",
        "beta_used",
        "beta_observations",
        "beta_feature_asof",
        "beta_source",
    ]

    def with_beta(panel: pd.DataFrame) -> pd.DataFrame:
        return panel.merge(
            top20_beta[beta_columns], on=key, how="left", validate="one_to_one"
        )

    top6 = with_beta(top6)
    top10 = with_beta(top10)
    top20 = top20_beta
    dynamic = with_beta(dynamic)

    spy_returns = {
        pd.Timestamp(date).normalize(): _spy_window_return(
            marketdata_root / "minute" / f"{pd.Timestamp(date).date()}_SPY.parquet",
            pd.Timestamp(date).normalize(),
        )
        for date in sorted(top10["effective_close_date"].unique())
    }
    if not all(np.isfinite(value) for value in spy_returns.values()):
        raise ValueError("selected review is missing its SPY window return")

    optimized_cap = float(controls["optimized_max_name_weight"])
    asymmetric_cap = float(controls["asymmetric_max_name_weight"])
    candidates: dict[str, pd.DataFrame] = {}

    baseline = top10.merge(
        frozen[key + ["target_weight"]], on=key, how="left", validate="one_to_one"
    )
    candidates[BASELINE] = _label(baseline, BASELINE)

    full_equal = build_capped_leg_weights(
        top10, None, gross_target=1.0, max_name_weight=optimized_cap
    )
    candidates["top10_full_equal_dollar_neutral"] = _label(
        full_equal, "top10_full_equal_dollar_neutral"
    )
    candidates["top6_full_equal_dollar_neutral"] = _label(
        build_capped_leg_weights(top6, None, gross_target=1.0, max_name_weight=optimized_cap),
        "top6_full_equal_dollar_neutral",
    )
    candidates["top20_full_equal_dollar_neutral"] = _label(
        build_capped_leg_weights(top20, None, gross_target=1.0, max_name_weight=optimized_cap),
        "top20_full_equal_dollar_neutral",
    )
    candidates["top6_impact_weighted_dollar_neutral"] = _label(
        build_capped_leg_weights(
            top6, "impact_proxy", gross_target=1.0, max_name_weight=optimized_cap
        ),
        "top6_impact_weighted_dollar_neutral",
    )
    candidates["top10_impact_weighted_dollar_neutral"] = _label(
        build_capped_leg_weights(
            top10, "impact_proxy", gross_target=1.0, max_name_weight=optimized_cap
        ),
        "top10_impact_weighted_dollar_neutral",
    )
    candidates["top10_inverse_vol_dollar_neutral"] = _label(
        build_capped_leg_weights(
            top10, "inverse_vol_score", gross_target=1.0, max_name_weight=optimized_cap
        ),
        "top10_inverse_vol_dollar_neutral",
    )
    candidates["dynamic80_impact_weighted_dollar_neutral"] = _label(
        build_capped_leg_weights(
            dynamic, "impact_proxy", gross_target=1.0, max_name_weight=optimized_cap
        ),
        "dynamic80_impact_weighted_dollar_neutral",
    )
    candidates["top10_stock_beta_neutral"] = _label(
        build_beta_neutral_equal_weights(
            top10, gross_target=1.0, max_name_weight=optimized_cap
        ),
        "top10_stock_beta_neutral",
    )

    long_heavy = build_asymmetric_leg_weights(
        top10,
        long_fraction=float(controls["long_heavy_long_fraction"]),
        gross_target=1.0,
        max_name_weight=asymmetric_cap,
    )
    candidates["top10_long2_short1_unhedged"] = _label(
        long_heavy, "top10_long2_short1_unhedged"
    )
    candidates["top10_long2_short1_spy_hedged"] = _label(
        append_spy_beta_hedge(long_heavy, spy_returns, beta_band=0.0),
        "top10_long2_short1_spy_hedged",
    )
    long_only = build_asymmetric_leg_weights(
        top10,
        long_fraction=1.0,
        gross_target=1.0,
        max_name_weight=asymmetric_cap,
    )
    candidates["top10_long_only_spy_hedged"] = _label(
        append_spy_beta_hedge(long_only, spy_returns, beta_band=0.0),
        "top10_long_only_spy_hedged",
    )
    candidates["top10_equal_soft_beta_band_spy_hedged"] = _label(
        append_spy_beta_hedge(
            full_equal,
            spy_returns,
            beta_band=float(controls["soft_beta_band"]),
        ),
        "top10_equal_soft_beta_band_spy_hedged",
    )

    state = state_panel.copy()
    state["effective_close_date"] = pd.to_datetime(state["effective_close_date"]).dt.normalize()
    validate_state_features(state)
    state_columns = [
        "effective_close_date",
        "ticker",
        "change_type",
        "state_feature_point_in_time_valid",
        "signed_relative_hour_return_pre",
        "last_hour_volume_share_side_percentile",
        "stock_feature_asof_timestamp",
        "spy_feature_asof_timestamp",
        "stock_daily_feature_asof",
        "spy_daily_feature_asof",
    ]
    state = state[state_columns].drop_duplicates(key)
    state_top10 = full_equal.drop(columns=["change_type"]).merge(
        state, on=key, how="left", validate="one_to_one"
    )
    if state_top10["state_feature_point_in_time_valid"].isna().any():
        raise ValueError("state panel does not cover the frozen top10 membership")
    softened = apply_filter_and_balance(state_top10, "down_to_small_hour_soft_025")
    softened["target_weight"] = softened["filtered_target_weight"]
    candidates["top10_down_to_small_softgate25"] = _label(
        softened, "top10_down_to_small_softgate25"
    )

    valid_state = state_top10["state_feature_point_in_time_valid"].fillna(False).astype(bool)
    reversal = valid_state & pd.to_numeric(
        state_top10["signed_relative_hour_return_pre"], errors="coerce"
    ).le(0)
    quiet = valid_state & pd.to_numeric(
        state_top10["last_hour_volume_share_side_percentile"], errors="coerce"
    ).le(0.50)
    for strategy, keep in (
        ("top10_relative_hour_reversal", reversal),
        ("top10_quiet_hour", quiet),
        ("top10_reversal_or_quiet", reversal | quiet),
    ):
        subset = state_top10.loc[keep].copy()
        weights = build_capped_leg_weights(
            subset,
            None,
            gross_target=1.0,
            max_name_weight=asymmetric_cap,
            require_both_legs=True,
        )
        candidates[strategy] = _label(weights, strategy)

    expected = list(spec["candidate_order"])
    if list(candidates) != expected:
        raise AssertionError(
            f"candidate order differs from frozen specification: {list(candidates)}"
        )
    selection_diagnostics = pd.concat(
        [
            d6.assign(candidate_universe="top6"),
            d10.assign(candidate_universe="top10"),
            d20.assign(candidate_universe="top20"),
            ddynamic.assign(candidate_universe="dynamic80"),
        ],
        ignore_index=True,
    )
    return candidates, expanded_beta, selection_diagnostics


def _evaluate(
    candidates: dict[str, pd.DataFrame],
    schedule: list[pd.Timestamp],
    costs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    trade_rows: list[pd.DataFrame] = []
    attribution_rows: list[pd.DataFrame] = []
    for strategy, raw in candidates.items():
        weights = raw.copy()
        weights["estimated_beta_contribution"] = (
            pd.to_numeric(weights["target_weight"], errors="raise")
            * pd.to_numeric(weights["beta_used"], errors="raise")
        )
        weights["stock_abs_weight"] = np.where(
            weights["instrument_role"].eq("event_stock"), weights["target_weight"].abs(), 0.0
        )
        weights["hedge_abs_weight"] = np.where(
            weights["instrument_role"].eq("market_hedge"), weights["target_weight"].abs(), 0.0
        )
        exposure = weights.groupby("effective_close_date", as_index=False).agg(
            estimated_beta_exposure=("estimated_beta_contribution", "sum"),
            stock_gross_exposure=("stock_abs_weight", "sum"),
            hedge_gross_exposure=("hedge_abs_weight", "sum"),
        )
        for cost in costs:
            portfolio = weighted_date_portfolio(
                weights, cost_bps=float(cost), scheduled_dates=schedule
            ).merge(exposure, on="effective_close_date", how="left", validate="one_to_one")
            for column in (
                "estimated_beta_exposure",
                "stock_gross_exposure",
                "hedge_gross_exposure",
            ):
                portfolio[column] = portfolio[column].fillna(0.0)
            portfolio["net_return_per_unit_gross"] = np.where(
                portfolio["gross_exposure"].gt(0),
                portfolio["net_return"] / portfolio["gross_exposure"],
                np.nan,
            )
            portfolio.insert(0, "strategy", strategy)
            portfolio = add_equity_columns(portfolio)
            date_rows.append(portfolio)
            active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
            unit = active_trade_metrics(
                portfolio,
                return_column="net_return_per_unit_gross",
                exposure_column="gross_exposure",
            )
            gross_sum = float(portfolio["gross_return"].sum())
            exposure_sum = float(portfolio["gross_exposure"].sum())
            metric_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": float(cost),
                    **performance_metrics(portfolio, "net_return"),
                    **active_trade_metrics(portfolio, "net_return"),
                    "active_unit_gross_mean_return": unit["active_mean_return"],
                    "active_unit_gross_sharpe_sqrt4": unit["active_trade_sharpe_sqrt4"],
                    "arithmetic_gross_return_bp": gross_sum * 10_000.0,
                    "arithmetic_net_return_bp": float(portfolio["net_return"].sum() * 10_000.0),
                    "break_even_cost_bps": gross_sum / exposure_sum * 10_000.0
                    if exposure_sum > 0
                    else np.nan,
                    "active_review_dates": int(len(active)),
                    "cash_review_dates": int(len(portfolio) - len(active)),
                    "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
                    "mean_active_gross_exposure": float(active["gross_exposure"].mean())
                    if len(active)
                    else np.nan,
                    "mean_active_stock_gross": float(active["stock_gross_exposure"].mean())
                    if len(active)
                    else np.nan,
                    "mean_active_hedge_gross": float(active["hedge_gross_exposure"].mean())
                    if len(active)
                    else np.nan,
                    "mean_active_abs_beta_exposure": float(
                        active["estimated_beta_exposure"].abs().mean()
                    )
                    if len(active)
                    else np.nan,
                    "mean_active_names": float(active["n_names"].mean()) if len(active) else np.nan,
                    "mean_active_effective_names": float(active["effective_names"].mean())
                    if len(active)
                    else np.nan,
                    "mean_active_top_name_abs_pnl_share": float(
                        active["top_name_share_of_absolute_pnl"].mean()
                    )
                    if len(active)
                    else np.nan,
                }
            )
            trades = weights.copy()
            trades["cost_bps"] = float(cost)
            trades["gross_pnl_contribution"] = (
                trades["target_weight"] * trades["underlying_return"]
            )
            trades["cost_contribution"] = (
                trades["target_weight"].abs() * float(cost) / 10_000.0
            )
            trades["net_pnl_contribution"] = (
                trades["gross_pnl_contribution"] - trades["cost_contribution"]
            )
            trade_rows.append(trades)
            attribution_rows.append(
                trades.groupby(["strategy", "cost_bps", "instrument_role", "action"], as_index=False)
                .agg(
                    security_rows=("ticker", "size"),
                    gross_exposure_sum=("target_abs_weight", "sum"),
                    gross_pnl_sum=("gross_pnl_contribution", "sum"),
                    net_pnl_sum=("net_pnl_contribution", "sum"),
                )
            )
    return (
        pd.concat(date_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(trade_rows, ignore_index=True),
        pd.concat(attribution_rows, ignore_index=True),
    )


def _paired_tests(
    portfolios: pd.DataFrame,
    candidate_order: list[str],
    primary_cost: float,
) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost)]
    wide = sample.pivot(index="effective_close_date", columns="strategy", values="net_return")
    rows: list[dict[str, object]] = []
    family: list[pd.DataFrame] = []
    for strategy in candidate_order[1:]:
        difference = (wide[strategy] - wide[BASELINE]).dropna()
        low, high = moving_block_bootstrap_mean(
            difference.to_numpy(float), block_size=4, seed=20260902
        )
        test = stats.ttest_1samp(difference, 0.0, alternative="greater")
        rows.append(
            {
                "strategy": strategy,
                "comparator": BASELINE,
                "cost_bps": primary_cost,
                "review_dates": len(difference),
                "mean_increment_bp": float(difference.mean() * 10_000.0),
                "median_increment_bp": float(difference.median() * 10_000.0),
                "positive_increment_rate": float(difference.gt(0).mean()),
                "block4_ci_low_bp": low * 10_000.0,
                "block4_ci_high_bp": high * 10_000.0,
                "one_sided_t_pvalue": float(test.pvalue),
                "bonferroni_pvalue": min(1.0, float(test.pvalue) * (len(candidate_order) - 1)),
            }
        )
        family.append(
            pd.DataFrame(
                {
                    "effective_close_date": difference.index,
                    "window": strategy,
                    "gross_return": difference.to_numpy(float),
                }
            )
        )
    family_frame = pd.concat(family, ignore_index=True)
    for row in rows:
        pvalue, dates, candidates = block_max_t_sign_flip_pvalue(
            family_frame, str(row["strategy"]), block_size=4
        )
        row["block4_max_t_familywise_pvalue"] = pvalue
        row["family_complete_dates"] = dates
        row["family_candidates"] = candidates
    return pd.DataFrame(rows)


def _regime_metrics(
    portfolios: pd.DataFrame,
    candidate_order: list[str],
    primary_cost: float,
) -> pd.DataFrame:
    definitions = {
        "2020-2022": lambda x: x.lt("2023-01-01"),
        "2023-plus": lambda x: x.ge("2023-01-01"),
        "excluding-2026": lambda x: x.lt("2026-01-01"),
        "2024-plus": lambda x: x.ge("2024-01-01"),
    }
    rows: list[dict[str, object]] = []
    for strategy in candidate_order:
        sample = portfolios.loc[
            portfolios["strategy"].eq(strategy) & portfolios["cost_bps"].eq(primary_cost)
        ].copy()
        date = pd.to_datetime(sample["effective_close_date"])
        for regime, condition in definitions.items():
            subset = add_equity_columns(sample.loc[condition(date)].copy())
            active = subset.loc[subset["gross_exposure"].gt(0)]
            summary = performance_metrics(subset, "net_return")
            rows.append(
                {
                    "strategy": strategy,
                    "regime": regime,
                    "cost_bps": primary_cost,
                    "review_dates": len(subset),
                    "active_review_dates": len(active),
                    "mean_net_bp_per_scheduled_review": summary["mean_return_per_event"] * 10_000.0,
                    "scheduled_event_sharpe": summary["event_sharpe_sqrt4"],
                    "active_mean_net_bp": float(active["net_return"].mean() * 10_000.0)
                    if len(active)
                    else np.nan,
                    "active_hit_rate": float(active["net_return"].gt(0).mean())
                    if len(active)
                    else np.nan,
                    "arithmetic_net_bp": float(subset["net_return"].sum() * 10_000.0),
                    "max_drawdown": summary["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def _leave_one_out(portfolios: pd.DataFrame, candidate_order: list[str]) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(PRIMARY_COST)].copy()
    dates = sorted(sample["effective_close_date"].unique())
    rows: list[dict[str, object]] = []
    for excluded in dates:
        training = sample.loc[~sample["effective_close_date"].eq(excluded)]
        means = training.groupby("strategy")["net_return"].mean().reindex(candidate_order)
        ordered = means.sort_values(ascending=False, kind="stable")
        rows.append(
            {
                "excluded_review": excluded,
                "winner": ordered.index[0],
                "winner_training_mean_bp": ordered.iloc[0] * 10_000.0,
                "baseline_rank": int(ordered.index.get_loc(BASELINE) + 1),
            }
        )
    return pd.DataFrame(rows)


def _robustness_tests(
    portfolios: pd.DataFrame,
    candidate_order: list[str],
    primary_cost: float,
) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost)].pivot(
        index="effective_close_date", columns="strategy", values="net_return"
    ).sort_index()
    baseline = sample[BASELINE]
    rows: list[dict[str, object]] = []
    for strategy in candidate_order[1:]:
        masks = {
            "all": pd.Series(True, index=sample.index),
            "excluding_2026": pd.Series(sample.index < pd.Timestamp("2026-01-01"), index=sample.index),
            "2020-2022": pd.Series(sample.index < pd.Timestamp("2023-01-01"), index=sample.index),
            "2023-plus": pd.Series(sample.index >= pd.Timestamp("2023-01-01"), index=sample.index),
            "excluding_candidate_best_review": pd.Series(
                sample.index != sample[strategy].idxmax(), index=sample.index
            ),
        }
        for robustness_sample, mask in masks.items():
            difference = (sample.loc[mask, strategy] - baseline.loc[mask]).dropna()
            low, high = moving_block_bootstrap_mean(
                difference.to_numpy(float), block_size=4, seed=20260902
            )
            rows.append(
                {
                    "strategy": strategy,
                    "comparator": BASELINE,
                    "sample": robustness_sample,
                    "review_dates": len(difference),
                    "mean_increment_bp": float(difference.mean() * 10_000.0),
                    "positive_increment_rate": float(difference.gt(0).mean()),
                    "block4_ci_low_bp": low * 10_000.0,
                    "block4_ci_high_bp": high * 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def _increment_decomposition(portfolios: pd.DataFrame, primary_cost: float) -> pd.DataFrame:
    sample = portfolios.loc[portfolios["cost_bps"].eq(primary_cost)].pivot(
        index="effective_close_date", columns="strategy", values="net_return"
    ).sort_index()
    comparisons = [
        (
            "capital_utilization",
            "top10_full_equal_dollar_neutral",
            BASELINE,
        ),
        (
            "impact_weighting_conditional_on_full_gross",
            "top10_impact_weighted_dollar_neutral",
            "top10_full_equal_dollar_neutral",
        ),
        (
            "top10_total_improvement",
            "top10_impact_weighted_dollar_neutral",
            BASELINE,
        ),
        (
            "top6_impact_weighting_conditional_on_breadth",
            "top6_impact_weighted_dollar_neutral",
            "top6_full_equal_dollar_neutral",
        ),
    ]
    rows: list[dict[str, object]] = []
    for component, strategy, comparator in comparisons:
        difference = sample[strategy] - sample[comparator]
        low, high = moving_block_bootstrap_mean(
            difference.to_numpy(float), block_size=4, seed=20260902
        )
        rows.append(
            {
                "component": component,
                "strategy": strategy,
                "comparator": comparator,
                "review_dates": len(difference),
                "mean_increment_bp": float(difference.mean() * 10_000.0),
                "block4_ci_low_bp": low * 10_000.0,
                "block4_ci_high_bp": high * 10_000.0,
                "positive_increment_rate": float(difference.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _walk_metrics(walk: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, raw in (("walk_forward_selector", walk), ("matched_baseline", baseline)):
        portfolio = add_equity_columns(raw.sort_values("effective_close_date").copy())
        active = portfolio.loc[portfolio["gross_exposure"].gt(0)]
        rows.append(
            {
                "strategy": label,
                **performance_metrics(portfolio, "net_return"),
                **active_trade_metrics(portfolio, "net_return"),
                "arithmetic_net_bp": float(portfolio["net_return"].sum() * 10_000.0),
                "active_review_dates": len(active),
                "mean_gross_exposure": float(portfolio["gross_exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _walk_paired_test(walk: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    first = walk.set_index("effective_close_date")["net_return"]
    second = baseline.set_index("effective_close_date")["net_return"]
    difference = (first - second).dropna()
    low, high = moving_block_bootstrap_mean(
        difference.to_numpy(float), block_size=4, seed=20260902
    )
    return pd.DataFrame(
        [
            {
                "strategy": "walk_forward_selector",
                "comparator": "matched_baseline",
                "review_dates": len(difference),
                "mean_increment_bp": float(difference.mean() * 10_000.0),
                "positive_increment_rate": float(difference.gt(0).mean()),
                "block4_ci_low_bp": low * 10_000.0,
                "block4_ci_high_bp": high * 10_000.0,
                "warning": "retrospective pseudo-OOS; strategy family was motivated by the full history",
            }
        ]
    )


def _charts(
    portfolios: pd.DataFrame,
    metrics: pd.DataFrame,
    regimes: pd.DataFrame,
    walk: pd.DataFrame,
    matched: pd.DataFrame,
    output: Path,
) -> None:
    primary = metrics.loc[metrics["cost_bps"].eq(PRIMARY_COST)].sort_values(
        "mean_return_per_event", ascending=False
    )
    shown = list(dict.fromkeys([BASELINE, *primary.head(5)["strategy"].tolist()]))
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    date_sample = portfolios.loc[
        portfolios["cost_bps"].eq(PRIMARY_COST) & portfolios["strategy"].isin(shown)
    ]
    for strategy, group in date_sample.groupby("strategy", sort=False):
        axes[0, 0].plot(
            pd.to_datetime(group["effective_close_date"]),
            group["equity"],
            linewidth=1.8,
            label=strategy.replace("_", " "),
        )
    axes[0, 0].set_title("10 bp event-step equity: baseline and top retrospective variants")
    axes[0, 0].legend(frameon=False, fontsize=7)

    axes[0, 1].scatter(
        primary["mean_return_per_event"] * 10_000.0,
        primary["event_sharpe_sqrt4"],
        s=45 + primary["mean_gross_exposure"] * 80,
        alpha=0.75,
    )
    for row in primary.head(5).itertuples(index=False):
        axes[0, 1].annotate(
            str(row.strategy).replace("top10_", "").replace("_", " "),
            (row.mean_return_per_event * 10_000.0, row.event_sharpe_sqrt4),
            fontsize=7,
        )
    axes[0, 1].set_title("Scheduled return versus Sharpe at 10 bp")
    axes[0, 1].set_xlabel("Mean net bp / scheduled review")
    axes[0, 1].set_ylabel("Scheduled event Sharpe")

    cost_shown = list(dict.fromkeys([BASELINE, *primary.head(3)["strategy"].tolist()]))
    for strategy in cost_shown:
        group = metrics.loc[metrics["strategy"].eq(strategy)].sort_values("cost_bps")
        axes[1, 0].plot(
            group["cost_bps"],
            group["mean_return_per_event"] * 10_000.0,
            marker="o",
            label=strategy.replace("_", " "),
        )
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].set_title("Cost sensitivity")
    axes[1, 0].set_xlabel("Modeled cost (bp on gross)")
    axes[1, 0].set_ylabel("Mean net bp / scheduled review")
    axes[1, 0].legend(frameon=False, fontsize=7)

    walk_equity = add_equity_columns(walk.sort_values("effective_close_date").copy())
    base_equity = add_equity_columns(matched.sort_values("effective_close_date").copy())
    axes[1, 1].plot(
        walk_equity["effective_close_date"], walk_equity["equity"], label="Walk-forward selector"
    )
    axes[1, 1].plot(
        base_equity["effective_close_date"], base_equity["equity"], label="Matched baseline"
    )
    axes[1, 1].set_title("Retrospective expanding walk-forward at 10 bp")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)

    regime_order = ["2020-2022", "2023-plus", "excluding-2026", "2024-plus"]
    selected = list(dict.fromkeys([BASELINE, *primary.head(5)["strategy"].tolist()]))
    table = regimes.loc[regimes["strategy"].isin(selected)].pivot(
        index="strategy", columns="regime", values="mean_net_bp_per_scheduled_review"
    ).reindex(index=selected, columns=regime_order)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    image = ax.imshow(table.to_numpy(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), table.columns)
    ax.set_yticks(np.arange(len(table.index)), [x.replace("_", " ") for x in table.index])
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            ax.text(column, row, f"{table.iloc[row, column]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("10 bp mean net return by regime (bp / scheduled review)")
    fig.colorbar(image, ax=ax, label="bp")
    fig.tight_layout()
    fig.savefig(output.with_name("deep_portfolio_regime_heatmap.png"), dpi=190)
    plt.close(fig)


def _bp(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 10_000:+.2f}"


def _num(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _pct(value: object, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def _report(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    regimes: pd.DataFrame,
    loo: pd.DataFrame,
    walk_metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    decomposition: pd.DataFrame,
    walk_paired: pd.DataFrame,
    output: Path,
) -> None:
    primary = metrics.loc[metrics["cost_bps"].eq(PRIMARY_COST)].sort_values(
        "mean_return_per_event", ascending=False
    )
    pair_map = paired.set_index("strategy")
    rows = []
    for row in primary.itertuples(index=False):
        paired_row = pair_map.loc[row.strategy] if row.strategy in pair_map.index else None
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.strategy))}</td>"
            f"<td>{_bp(row.mean_return_per_event)}</td>"
            f"<td>{_num(row.event_sharpe_sqrt4)}</td>"
            f"<td>{_bp(row.active_mean_return)}</td>"
            f"<td>{_num(row.active_trade_sharpe_sqrt4)}</td>"
            f"<td>{_pct(row.active_trade_hit_rate)}</td>"
            f"<td>{_pct(row.mean_gross_exposure)}</td>"
            f"<td>{_num(row.mean_active_names, 1)}</td>"
            f"<td>{_pct(row.max_drawdown, 2)}</td>"
            f"<td>{'—' if paired_row is None else f'{float(paired_row.mean_increment_bp):+.2f}'}</td>"
            f"<td>{'—' if paired_row is None else f'{float(paired_row.block4_ci_low_bp):+.2f} to {float(paired_row.block4_ci_high_bp):+.2f}'}</td>"
            f"<td>{'—' if paired_row is None else f'{float(paired_row.block4_max_t_familywise_pvalue):.3f}'}</td>"
            "</tr>"
        )
    winner = primary.iloc[0]
    baseline = primary.loc[primary["strategy"].eq(BASELINE)].iloc[0]
    stable = loo["winner"].value_counts()
    stable_name = str(stable.index[0])
    stable_count = int(stable.iloc[0])
    walk = walk_metrics.set_index("strategy")
    evidence_pool = paired.loc[
        paired["mean_increment_bp"].gt(0) & paired["block4_ci_low_bp"].gt(0)
    ].sort_values(
        ["block4_max_t_familywise_pvalue", "mean_increment_bp"],
        ascending=[True, False],
    )
    evidence_name = str(evidence_pool.iloc[0]["strategy"])
    evidence_metric = primary.loc[primary["strategy"].eq(evidence_name)].iloc[0]
    evidence_pair = paired.loc[paired["strategy"].eq(evidence_name)].iloc[0]
    robustness_row = robustness.loc[
        robustness["strategy"].eq(evidence_name)
        & robustness["sample"].eq("excluding_2026")
    ].iloc[0]
    decomposition_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.component))}</td>"
        f"<td>{html.escape(str(row.strategy))}</td>"
        f"<td>{html.escape(str(row.comparator))}</td>"
        f"<td>{float(row.mean_increment_bp):+.2f}</td>"
        f"<td>{float(row.block4_ci_low_bp):+.2f} to {float(row.block4_ci_high_bp):+.2f}</td>"
        "</tr>"
        for row in decomposition.itertuples(index=False)
    )
    walk_pair = walk_paired.iloc[0]
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSCI burst deep portfolio optimization</title><style>
body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f3f6f9;color:#172033}}main{{max-width:1380px;margin:auto;padding:32px}}h1{{font-size:34px;margin-bottom:8px}}h2{{margin-top:34px}}.lede{{font-size:18px;color:#4d5d73}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.kpi,.callout{{background:white;padding:17px;border-radius:10px;box-shadow:0 1px 4px #0001}}.kpi b{{display:block;font-size:24px;margin-top:8px}}.callout{{margin:18px 0;border-left:5px solid #b07800}}table{{width:100%;border-collapse:collapse;background:white;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #dfe5ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e6ecf3;position:sticky;top:0}}img{{width:100%;background:white;border-radius:10px}}code{{background:#e6ecf3;padding:2px 5px;border-radius:4px}}.small{{font-size:13px;color:#607086}}a{{color:#145ea8}}
</style></head><body><main><h1>MSCI 15:49–15:50 Deep Portfolio Optimization</h1>
<p class="lede">The minute-bar return proxy is held fixed; only breadth, weighting, long/short budget, beta hedge, state overlay and concentration are optimized. All 17 candidates are reported; the primary cost is 10 bp.</p>
<div class="callout"><b>Scope</b><br>This is retrospective portfolio research. It identifies which constructions are most robust across the 26 observed review dates; fill probabilities are modeled separately.</div>
<div class="grid"><div class="kpi">Evidence-ranked candidate<b>{html.escape(evidence_name)}</b></div><div class="kpi">Candidate mean, 10 bp<b>{_bp(evidence_metric.mean_return_per_event)} bp</b></div><div class="kpi">Increment vs baseline<b>{float(evidence_pair.mean_increment_bp):+.2f} bp</b></div><div class="kpi">Family-wise max-t p<b>{float(evidence_pair.block4_max_t_familywise_pvalue):.3f}</b></div></div>
<div class="callout"><b>The highest return is not the strongest evidence.</b><br>The full-sample leader is <code>{html.escape(str(winner.strategy))}</code>, but it is negative in 2020–2022 and its family-wise p is weak. The evidence ranking selects <code>{html.escape(evidence_name)}</code>: full-sample block CI {float(evidence_pair.block4_ci_low_bp):+.2f} to {float(evidence_pair.block4_ci_high_bp):+.2f} bp; excluding 2026 the increment is still {float(robustness_row.mean_increment_bp):+.2f} bp. The ranking was formed after seeing history, so it is a candidate for a future holdout.</div>
<h2>All constructions, 10 bp</h2><table><thead><tr><th>Strategy</th><th>Scheduled mean bp</th><th>Scheduled Sharpe</th><th>Active mean bp</th><th>Active Sharpe</th><th>Active hit</th><th>Mean gross</th><th>Names</th><th>Max DD</th><th>Increment bp</th><th>Block CI bp</th><th>Max-t p</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="small">Increment, CI and max-t p are relative to the frozen baseline; the max-t family contains all 16 challengers. Active means gross exposure &gt; 0; Scheduled means all 26 review dates.</p>
<h2>Equity, risk, cost and walk-forward</h2><img src="charts/deep_portfolio_overview.png" alt="deep portfolio overview">
<h2>Regime stability</h2><img src="charts/deep_portfolio_regime_heatmap.png" alt="deep portfolio regime heatmap">
<h2>Increment decomposition</h2><table><thead><tr><th>Component</th><th>Strategy</th><th>Comparator</th><th>Mean increment bp</th><th>Block CI bp</th></tr></thead><tbody>{decomposition_rows}</tbody></table>
<p class="small">The increment of top-10 impact weighting over full-gross equal weighting is listed separately to separate capital utilization from allocation.</p>
<h2>Expanding pseudo-OOS</h2><p>At each test review the selector uses only earlier dates; the rule is training scheduled mean minus 0.5 standard error. Selector arithmetic net P&amp;L is <b>{float(walk.loc['walk_forward_selector','arithmetic_net_bp']):+.1f} bp</b>; matched baseline <b>{float(walk.loc['matched_baseline','arithmetic_net_bp']):+.1f} bp</b>; paired increment <b>{float(walk_pair.mean_increment_bp):+.2f} bp/review</b>, block CI {float(walk_pair.block4_ci_low_bp):+.2f} to {float(walk_pair.block4_ci_high_bp):+.2f} bp. Because the hypothesis was prompted by 2026 observations, this is a retrospective pseudo-OOS.</p>
<h2>Result files</h2><p><a href="deep_portfolio_metrics.csv">Metrics</a> · <a href="deep_portfolio_date_returns.csv">Review ledger</a> · <a href="deep_portfolio_trade_log.csv">Trade log</a> · <a href="deep_portfolio_paired_tests.csv">Paired tests</a> · <a href="deep_portfolio_robustness_tests.csv">Robustness</a> · <a href="deep_portfolio_increment_decomposition.csv">Increment decomposition</a> · <a href="deep_portfolio_regime_metrics.csv">Regimes</a> · <a href="deep_portfolio_walk_forward_selection.csv">Walk-forward selection</a> · <a href="deep_portfolio_manual_audit.json">Manual audit</a></p>
<p class="small">Generated {datetime.now(UTC).isoformat()}</p></main></body></html>"""
    output.write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("configs/deep_portfolio_optimization_spec.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deep_portfolio_optimization"))
    parser.add_argument("--marketdata-root", type=Path, default=Path("data/marketdata"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    inputs = {name: Path(path) for name, path in spec["inputs"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = args.output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(inputs["feature_panel"], parse_dates=["effective_close_date"])
    validate_pre_event_features(features)
    features = features.loc[features["feature_point_in_time_valid"].fillna(False)].copy()
    frozen = pd.read_csv(inputs["frozen_top10_ledger"], parse_dates=["effective_close_date"])
    beta = pd.read_csv(inputs["beta_features"], parse_dates=["effective_close_date"])
    state = pd.read_csv(inputs["state_features"], parse_dates=["effective_close_date"])
    schedule = sorted(
        pd.read_csv(inputs["review_schedule"], parse_dates=["effective_close_date"])[
            "effective_close_date"
        ].dt.normalize().unique()
    )

    candidates, expanded_beta, diagnostics = _build_candidates(
        features, frozen, beta, state, args.marketdata_root, spec
    )
    costs = [float(value) for value in spec["costs_bps"]]
    portfolios, metrics, trades, attribution = _evaluate(candidates, schedule, costs)
    candidate_order = list(spec["candidate_order"])
    paired = _paired_tests(portfolios, candidate_order, float(spec["primary_cost_bps"]))
    regimes = _regime_metrics(portfolios, candidate_order, float(spec["primary_cost_bps"]))
    loo = _leave_one_out(portfolios, candidate_order)
    robustness = _robustness_tests(
        portfolios, candidate_order, float(spec["primary_cost_bps"])
    )
    decomposition = _increment_decomposition(
        portfolios, float(spec["primary_cost_bps"])
    )

    walk_spec = spec["walk_forward"]
    selection, walk_scores, walk_returns = expanding_portfolio_selection(
        portfolios,
        candidate_order,
        cost_bps=float(spec["primary_cost_bps"]),
        minimum_training_reviews=int(walk_spec["minimum_training_reviews"]),
        minimum_active_reviews=int(walk_spec["minimum_active_training_reviews"]),
        minimum_exposure_ratio=float(walk_spec["minimum_exposure_ratio_to_baseline"]),
        fallback=str(walk_spec["fallback"]),
    )
    test_dates = set(pd.to_datetime(walk_returns["effective_close_date"]).dt.normalize())
    matched = portfolios.loc[
        portfolios["strategy"].eq(BASELINE)
        & portfolios["cost_bps"].eq(float(spec["primary_cost_bps"]))
        & pd.to_datetime(portfolios["effective_close_date"]).dt.normalize().isin(test_dates)
    ].copy()
    walk_metrics = _walk_metrics(walk_returns, matched)
    walk_paired = _walk_paired_test(walk_returns, matched)
    selected_pairs = set(
        zip(pd.to_datetime(selection["test_date"]).dt.normalize(), selection["selected_strategy"])
    )
    walk_trades = trades.loc[
        trades["cost_bps"].eq(float(spec["primary_cost_bps"]))
        & pd.Series(
            list(zip(pd.to_datetime(trades["effective_close_date"]).dt.normalize(), trades["strategy"])),
            index=trades.index,
        ).isin(selected_pairs)
    ].copy()

    outputs = {
        "deep_portfolio_security_weights.csv": pd.concat(candidates.values(), ignore_index=True),
        "deep_portfolio_date_returns.csv": portfolios,
        "deep_portfolio_metrics.csv": metrics,
        "deep_portfolio_trade_log.csv": trades,
        "deep_portfolio_attribution.csv": attribution,
        "deep_portfolio_paired_tests.csv": paired,
        "deep_portfolio_regime_metrics.csv": regimes,
        "deep_portfolio_leave_one_out.csv": loo,
        "deep_portfolio_robustness_tests.csv": robustness,
        "deep_portfolio_increment_decomposition.csv": decomposition,
        "deep_portfolio_walk_forward_selection.csv": selection,
        "deep_portfolio_walk_forward_scores.csv": walk_scores,
        "deep_portfolio_walk_forward_returns.csv": walk_returns,
        "deep_portfolio_walk_forward_metrics.csv": walk_metrics,
        "deep_portfolio_walk_forward_paired_test.csv": walk_paired,
        "deep_portfolio_walk_forward_trade_log.csv": walk_trades,
        "deep_portfolio_beta_features.csv": expanded_beta,
        "deep_portfolio_selection_diagnostics.csv": diagnostics,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    _charts(
        portfolios,
        metrics,
        regimes,
        walk_returns,
        matched,
        chart_dir / "deep_portfolio_overview.png",
    )

    primary = metrics.loc[metrics["cost_bps"].eq(float(spec["primary_cost_bps"]))]
    leader = primary.sort_values("mean_return_per_event", ascending=False).iloc[0]
    manifest = {
        "version": "deep-portfolio-optimization-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "spec": str(args.spec),
        "spec_sha256": _sha256(args.spec),
        "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
        "code_sha256": {
            "runner": _sha256(Path(__file__)),
            "module": _sha256(Path("src/msci_rebalance/deep_portfolio.py")),
        },
        "candidate_count": len(candidate_order),
        "scheduled_reviews": len(schedule),
        "valid_feature_rows": len(features),
        "top10_membership_reconstruction_error": 0,
        "primary_cost_bps": float(spec["primary_cost_bps"]),
        "retrospective_leader": str(leader["strategy"]),
        "retrospective_leader_mean_net_bp": float(leader["mean_return_per_event"] * 10_000.0),
        "research_status": "retrospective_minute_bar_proxy_optimized_not_future_validated",
        "execution_scope": "held fixed and explicitly out of scope by user request",
    }
    (args.output_dir / "deep_portfolio_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    _report(
        metrics,
        paired,
        regimes,
        loo,
        walk_metrics,
        robustness,
        decomposition,
        walk_paired,
        args.output_dir / "report.html",
    )
    print(
        f"candidates={len(candidate_order)} reviews={len(schedule)} "
        f"leader={leader['strategy']} mean10bp={leader['mean_return_per_event'] * 10_000.0:.2f}bp"
    )


if __name__ == "__main__":
    main()
