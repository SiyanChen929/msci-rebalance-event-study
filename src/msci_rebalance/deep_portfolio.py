"""Bounded portfolio-construction tools for the MSCI closing-window study.

The functions in this module do not inspect outcome returns when selecting
securities or assigning weights.  Cross-sectional inputs must already satisfy
the project's point-in-time validation rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _dates(values: pd.Series | Sequence[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def _capped_allocation(
    scores: np.ndarray,
    target: float,
    cap: float,
) -> np.ndarray:
    """Allocate ``target`` proportionally, redistributing weights at the cap."""

    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    result = np.zeros(len(scores), dtype=float)
    active = np.flatnonzero(scores > 0)
    remaining = float(target)
    while len(active) and remaining > 1e-15:
        proposed = remaining * scores[active] / float(scores[active].sum())
        room = float(cap) - result[active]
        binding = proposed >= room - 1e-15
        if not binding.any():
            result[active] += proposed
            break
        bound = active[binding]
        result[bound] += room[binding]
        remaining = float(target) - float(result.sum())
        active = active[~binding]
    return result


def select_top_k_per_side(
    frame: pd.DataFrame,
    score_column: str,
    *,
    k_per_side: int,
    minimum_per_side: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select up to ``k_per_side`` independently on additions and deletions."""

    required = {"effective_close_date", "ticker", "action", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selection panel lacks fields: {sorted(missing)}")
    if k_per_side < minimum_per_side or minimum_per_side <= 0:
        raise ValueError("k_per_side must be at least minimum_per_side")
    working = frame.copy()
    working["effective_close_date"] = _dates(working["effective_close_date"])
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.loc[
        working["action"].isin(["add", "delete"])
        & np.isfinite(working[score_column])
        & working[score_column].gt(0)
    ].copy()
    selected: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for event_date, group in working.groupby("effective_close_date", sort=True):
        sides = {
            side: group.loc[group["action"].eq(side)].sort_values(
                [score_column, "ticker"], ascending=[False, True], kind="stable"
            )
            for side in ("add", "delete")
        }
        eligible = min(len(sides["add"]), len(sides["delete"])) >= minimum_per_side
        diagnostics.append(
            {
                "effective_close_date": event_date,
                "selection": f"top_{k_per_side}_per_side",
                "available_adds": len(sides["add"]),
                "available_deletes": len(sides["delete"]),
                "selected_adds": min(k_per_side, len(sides["add"])) if eligible else 0,
                "selected_deletes": min(k_per_side, len(sides["delete"])) if eligible else 0,
                "review_eligible": bool(eligible),
            }
        )
        if not eligible:
            continue
        for side in ("add", "delete"):
            chosen = sides[side].head(k_per_side).copy()
            chosen["side_rank"] = np.arange(1, len(chosen) + 1)
            selected.append(chosen)
    output = pd.concat(selected, ignore_index=True) if selected else working.iloc[0:0].copy()
    return output, pd.DataFrame(diagnostics)


def select_dynamic_score_coverage(
    frame: pd.DataFrame,
    score_column: str,
    *,
    coverage: float = 0.80,
    minimum_per_side: int = 3,
    maximum_per_side: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the smallest side breadth covering a fixed share of top scores."""

    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    ranked, base_diagnostics = select_top_k_per_side(
        frame,
        score_column,
        k_per_side=maximum_per_side,
        minimum_per_side=minimum_per_side,
    )
    selected: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (event_date, side), group in ranked.groupby(
        ["effective_close_date", "action"], sort=True
    ):
        ordered = group.sort_values(
            [score_column, "ticker"], ascending=[False, True], kind="stable"
        ).copy()
        score = pd.to_numeric(ordered[score_column], errors="raise").to_numpy(float)
        total = float(score.sum())
        cumulative = np.cumsum(score) / total
        threshold_count = int(np.searchsorted(cumulative, coverage, side="left") + 1)
        take = min(len(ordered), max(minimum_per_side, threshold_count))
        chosen = ordered.head(take).copy()
        chosen["dynamic_side_count"] = take
        selected.append(chosen)
        rows.append(
            {
                "effective_close_date": event_date,
                "action": side,
                "selection": "dynamic_score_coverage",
                "available_top_scores": len(ordered),
                "selected_names": take,
                "target_coverage": coverage,
                "realized_score_coverage": float(cumulative[take - 1]),
            }
        )
    output = pd.concat(selected, ignore_index=True) if selected else ranked.iloc[0:0].copy()
    diagnostics = base_diagnostics.merge(
        pd.DataFrame(rows).groupby("effective_close_date", as_index=False).agg(
            dynamic_selected_adds=("selected_names", "first"),
            dynamic_selected_deletes=("selected_names", "last"),
            mean_realized_score_coverage=("realized_score_coverage", "mean"),
        ),
        on="effective_close_date",
        how="left",
    )
    return output, diagnostics


def build_asymmetric_leg_weights(
    securities: pd.DataFrame,
    *,
    long_fraction: float,
    score_column: str | None = None,
    gross_target: float = 1.0,
    max_name_weight: float = 0.25,
    require_both_legs: bool = True,
) -> pd.DataFrame:
    """Allocate a fixed gross fraction to additions and the remainder to deletions."""

    required = {"effective_close_date", "ticker", "action"}
    missing = required.difference(securities.columns)
    if missing:
        raise ValueError(f"asymmetric panel lacks fields: {sorted(missing)}")
    if score_column is not None and score_column not in securities:
        raise ValueError(f"score column not found: {score_column}")
    if not 0 <= long_fraction <= 1 or gross_target <= 0 or max_name_weight <= 0:
        raise ValueError("invalid asymmetric weighting control")
    frame = securities.copy().reset_index(drop=True)
    frame["effective_close_date"] = _dates(frame["effective_close_date"])
    score = (
        pd.to_numeric(frame[score_column], errors="coerce")
        if score_column is not None
        else pd.Series(1.0, index=frame.index)
    )
    score = score.where(np.isfinite(score) & score.gt(0), 0.0)
    frame["target_weight"] = 0.0
    targets = {"add": gross_target * long_fraction, "delete": gross_target * (1 - long_fraction)}
    signs = {"add": 1.0, "delete": -1.0}
    for _, positions in frame.groupby("effective_close_date", sort=True).groups.items():
        group = frame.loc[list(positions)]
        if require_both_legs and not {"add", "delete"}.issubset(set(group["action"])):
            continue
        for side in ("add", "delete"):
            side_positions = group.index[group["action"].eq(side)].to_numpy(int)
            if not len(side_positions) or targets[side] <= 0:
                continue
            allocation = _capped_allocation(
                score.loc[side_positions].to_numpy(float),
                targets[side],
                max_name_weight,
            )
            frame.loc[side_positions, "target_weight"] = signs[side] * allocation
    frame["target_abs_weight"] = frame["target_weight"].abs()
    return frame


def append_spy_beta_hedge(
    stock_weights: pd.DataFrame,
    spy_returns: Mapping[pd.Timestamp, float],
    *,
    beta_band: float = 0.0,
    gross_target: float = 1.0,
) -> pd.DataFrame:
    """Add a synthetic SPY position and rescale total stock-plus-hedge gross."""

    required = {
        "effective_close_date",
        "ticker",
        "target_weight",
        "beta_used",
        "underlying_return",
    }
    missing = required.difference(stock_weights.columns)
    if missing:
        raise ValueError(f"hedge panel lacks fields: {sorted(missing)}")
    if beta_band < 0 or gross_target <= 0:
        raise ValueError("beta band must be nonnegative and gross target positive")
    frame = stock_weights.copy().reset_index(drop=True)
    frame["effective_close_date"] = _dates(frame["effective_close_date"])
    frame["instrument_role"] = "event_stock"
    outputs: list[pd.DataFrame] = []
    for event_date, group in frame.groupby("effective_close_date", sort=True):
        group = group.copy()
        stock_beta = float(
            (pd.to_numeric(group["target_weight"], errors="raise")
             * pd.to_numeric(group["beta_used"], errors="raise")).sum()
        )
        if stock_beta > beta_band:
            raw_hedge = -(stock_beta - beta_band)
        elif stock_beta < -beta_band:
            raw_hedge = -(stock_beta + beta_band)
        else:
            raw_hedge = 0.0
        spy_return = float(spy_returns.get(pd.Timestamp(event_date).normalize(), np.nan))
        if raw_hedge != 0.0 and not np.isfinite(spy_return):
            raise ValueError(f"missing SPY window return for {event_date.date()}")
        if raw_hedge != 0.0:
            hedge = {column: np.nan for column in group.columns}
            hedge.update(
                {
                    "effective_close_date": event_date,
                    "ticker": "SPY_HEDGE",
                    "security_name": "SPDR S&P 500 ETF hedge",
                    "action": "hedge",
                    "change_type": "market_hedge",
                    "target_weight": raw_hedge,
                    "target_abs_weight": abs(raw_hedge),
                    "beta_used": 1.0,
                    "underlying_return": spy_return,
                    "instrument_role": "market_hedge",
                }
            )
            group = pd.concat([group, pd.DataFrame([hedge])], ignore_index=True)
        gross = float(group["target_weight"].abs().sum())
        if gross > 0:
            group["target_weight"] *= gross_target / gross
        group["target_abs_weight"] = group["target_weight"].abs()
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True) if outputs else frame


def expanding_portfolio_selection(
    date_portfolios: pd.DataFrame,
    candidate_order: Sequence[str],
    *,
    cost_bps: float,
    minimum_training_reviews: int,
    minimum_active_reviews: int,
    minimum_exposure_ratio: float,
    fallback: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a portfolio using only strictly earlier review-date returns."""

    sample = date_portfolios.loc[date_portfolios["cost_bps"].eq(float(cost_bps))].copy()
    sample["effective_close_date"] = _dates(sample["effective_close_date"])
    dates = sorted(sample["effective_close_date"].unique())
    order_rank = {name: index for index, name in enumerate(candidate_order)}
    selections: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    realized_rows: list[dict[str, object]] = []
    for index in range(minimum_training_reviews, len(dates)):
        test_date = pd.Timestamp(dates[index]).normalize()
        train_dates = [pd.Timestamp(value).normalize() for value in dates[:index]]
        training = sample.loc[sample["effective_close_date"].isin(train_dates)]
        baseline = training.loc[training["strategy"].eq(fallback)]
        baseline_exposure = float(baseline["gross_exposure"].mean())
        candidates: list[dict[str, object]] = []
        for strategy in candidate_order:
            group = training.loc[training["strategy"].eq(strategy)].sort_values(
                "effective_close_date"
            )
            if len(group) != len(train_dates):
                raise ValueError(f"incomplete training schedule for {strategy}")
            returns = group["net_return"].to_numpy(float)
            mean = float(np.mean(returns))
            std = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
            standard_error = std / np.sqrt(len(returns)) if np.isfinite(std) else np.nan
            active = int(group["gross_exposure"].gt(0).sum())
            exposure_ratio = (
                float(group["gross_exposure"].mean()) / baseline_exposure
                if baseline_exposure > 0
                else 0.0
            )
            eligible = bool(
                strategy == fallback
                or (
                    active >= minimum_active_reviews
                    and exposure_ratio >= minimum_exposure_ratio
                    and mean > 0
                )
            )
            score = mean - 0.5 * standard_error if np.isfinite(standard_error) else -np.inf
            row = {
                "test_date": test_date,
                "train_start": min(train_dates),
                "train_end": max(train_dates),
                "training_reviews": len(train_dates),
                "strategy": strategy,
                "training_mean_net_bp": mean * 10_000.0,
                "training_standard_error_bp": standard_error * 10_000.0,
                "training_selection_score_bp": score * 10_000.0,
                "training_active_reviews": active,
                "training_exposure_ratio_to_baseline": exposure_ratio,
                "eligible": eligible,
                "candidate_order": order_rank[strategy],
            }
            candidates.append(row)
            score_rows.append(row)
        eligible_rows = [row for row in candidates if row["eligible"]]
        chosen = max(
            eligible_rows,
            key=lambda row: (row["training_selection_score_bp"], -row["candidate_order"]),
        )["strategy"]
        selections.append(
            {
                "test_date": test_date,
                "selection_asof": max(train_dates),
                "selected_strategy": chosen,
                "training_reviews": len(train_dates),
            }
        )
        realized = sample.loc[
            sample["effective_close_date"].eq(test_date)
            & sample["strategy"].eq(chosen)
        ]
        if len(realized) != 1:
            raise ValueError(f"expected one realized portfolio for {test_date.date()}:{chosen}")
        realized_rows.append(
            {
                "selected_strategy": chosen,
                **realized.iloc[0].to_dict(),
            }
        )
    return (
        pd.DataFrame(selections),
        pd.DataFrame(score_rows),
        pd.DataFrame(realized_rows).sort_values("effective_close_date"),
    )
