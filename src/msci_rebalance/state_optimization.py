"""Chronological selection policies for the MSCI state-filter optimization.

All selection statistics are computed on scheduled review dates strictly before
the test review.  The module consumes already constructed date portfolios and
does not inspect security-level outcome fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _normalize_dates(values: pd.Series | Sequence[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def training_scores(
    portfolios: pd.DataFrame,
    train_dates: Sequence[object],
    candidate_rules: Sequence[str],
    intervention_rank: Mapping[str, int],
    *,
    minimum_active_reviews: int,
    minimum_return_retention: float,
    minimum_hit_improvement: float,
    minimum_exposure_ratio: float,
) -> pd.DataFrame:
    """Score one frozen candidate family using only the supplied train dates."""

    required = {
        "effective_close_date",
        "filter_rule",
        "net_return",
        "gross_exposure",
    }
    missing = required.difference(portfolios.columns)
    if missing:
        raise ValueError(f"optimization portfolio lacks fields: {sorted(missing)}")
    dates = _normalize_dates(train_dates)
    working = portfolios.copy()
    working["effective_close_date"] = _normalize_dates(
        working["effective_close_date"]
    )
    working = working.loc[working["effective_close_date"].isin(dates)]
    baseline = (
        working.loc[working["filter_rule"].eq("baseline_all")]
        .set_index("effective_close_date")
        .reindex(dates)
    )
    if baseline["net_return"].isna().any():
        raise ValueError("baseline does not cover every training review")
    baseline_net = float(baseline["net_return"].sum())
    baseline_active = baseline.loc[baseline["gross_exposure"].gt(0)]
    baseline_hit = (
        float(baseline_active["net_return"].gt(0).mean())
        if len(baseline_active)
        else np.nan
    )
    baseline_exposure = float(baseline["gross_exposure"].mean())
    rows: list[dict[str, object]] = []
    for rule in candidate_rules:
        sample = (
            working.loc[working["filter_rule"].eq(rule)]
            .set_index("effective_close_date")
            .reindex(dates)
        )
        if sample["net_return"].isna().any():
            raise ValueError(f"candidate {rule} does not cover every training review")
        active = sample.loc[sample["gross_exposure"].gt(0)]
        candidate_net = float(sample["net_return"].sum())
        candidate_hit = (
            float(active["net_return"].gt(0).mean()) if len(active) else np.nan
        )
        exposure_ratio = (
            float(sample["gross_exposure"].mean()) / baseline_exposure
            if baseline_exposure > 0
            else 0.0
        )
        if baseline_net > 0:
            return_gate = candidate_net >= minimum_return_retention * baseline_net
            retention = candidate_net / baseline_net
        else:
            return_gate = candidate_net > 0 and candidate_net >= baseline_net
            retention = np.nan
        hit_improvement = (
            candidate_hit - baseline_hit
            if np.isfinite(candidate_hit) and np.isfinite(baseline_hit)
            else np.nan
        )
        eligible = bool(
            rule != "baseline_all"
            and len(active) >= minimum_active_reviews
            and exposure_ratio >= minimum_exposure_ratio
            and return_gate
            and np.isfinite(hit_improvement)
            and hit_improvement >= minimum_hit_improvement
        )
        rows.append(
            {
                "filter_rule": rule,
                "training_reviews": int(len(dates)),
                "training_active_reviews": int(len(active)),
                "training_net_bp": candidate_net * 10_000.0,
                "training_mean_net_bp_per_scheduled": float(
                    sample["net_return"].mean() * 10_000.0
                ),
                "training_active_hit_rate": candidate_hit,
                "training_hit_improvement": hit_improvement,
                "training_exposure_ratio_to_baseline": exposure_ratio,
                "training_net_retention_vs_baseline": retention,
                "return_gate_pass": bool(return_gate),
                "eligible_constrained": eligible,
                "intervention_rank": int(intervention_rank[rule]),
            }
        )
    return pd.DataFrame(rows)


def select_rule(scores: pd.DataFrame, policy: str) -> str:
    """Select a rule from training-only scores under one frozen policy."""

    if policy == "constrained_minimum_intervention":
        eligible = scores.loc[scores["eligible_constrained"]].sort_values(
            ["intervention_rank", "filter_rule"], kind="stable"
        )
        return str(eligible.iloc[0]["filter_rule"]) if len(eligible) else "baseline_all"
    if policy == "maximum_training_mean":
        eligible = scores.loc[
            scores["training_active_reviews"].ge(3)
            & scores["training_exposure_ratio_to_baseline"].ge(0.35)
        ].sort_values(
            ["training_mean_net_bp_per_scheduled", "training_active_hit_rate", "filter_rule"],
            ascending=[False, False, True],
            kind="stable",
        )
        return str(eligible.iloc[0]["filter_rule"]) if len(eligible) else "baseline_all"
    raise ValueError(f"unknown walk-forward selection policy: {policy}")


def expanding_walk_forward(
    portfolios: pd.DataFrame,
    candidate_rules: Sequence[str],
    intervention_rank: Mapping[str, int],
    *,
    minimum_training_reviews: int,
    minimum_active_reviews: int,
    minimum_return_retention: float,
    minimum_hit_improvement: float,
    minimum_exposure_ratio: float,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a strictly chronological expanding review-date selector."""

    working = portfolios.copy()
    working["effective_close_date"] = _normalize_dates(
        working["effective_close_date"]
    )
    dates = sorted(working["effective_close_date"].unique())
    selections: list[dict[str, object]] = []
    all_scores: list[pd.DataFrame] = []
    realized_rows: list[dict[str, object]] = []
    for index in range(minimum_training_reviews, len(dates)):
        test_date = pd.Timestamp(dates[index]).normalize()
        train_dates = [pd.Timestamp(value).normalize() for value in dates[:index]]
        scores = training_scores(
            working,
            train_dates,
            candidate_rules,
            intervention_rank,
            minimum_active_reviews=minimum_active_reviews,
            minimum_return_retention=minimum_return_retention,
            minimum_hit_improvement=minimum_hit_improvement,
            minimum_exposure_ratio=minimum_exposure_ratio,
        )
        scores.insert(0, "selection_policy", policy)
        scores.insert(1, "test_date", test_date)
        scores.insert(2, "train_start", min(train_dates))
        scores.insert(3, "train_end", max(train_dates))
        all_scores.append(scores)
        chosen = select_rule(scores, policy)
        chosen_score = scores.loc[scores["filter_rule"].eq(chosen)].iloc[0]
        selections.append(
            {
                "selection_policy": policy,
                "test_date": test_date,
                "selection_asof": max(train_dates),
                "selected_filter_rule": chosen,
                "training_reviews": len(train_dates),
                "training_net_bp": chosen_score["training_net_bp"],
                "training_active_hit_rate": chosen_score["training_active_hit_rate"],
                "training_net_retention_vs_baseline": chosen_score[
                    "training_net_retention_vs_baseline"
                ],
                "training_hit_improvement": chosen_score[
                    "training_hit_improvement"
                ],
                "training_exposure_ratio_to_baseline": chosen_score[
                    "training_exposure_ratio_to_baseline"
                ],
            }
        )
        realized = working.loc[
            working["effective_close_date"].eq(test_date)
            & working["filter_rule"].eq(chosen)
        ]
        if len(realized) != 1:
            raise ValueError(f"expected one realized row for {test_date.date()}:{chosen}")
        realized_rows.append(
            {
                "selection_policy": policy,
                "selected_filter_rule": chosen,
                **realized.iloc[0].to_dict(),
            }
        )
    return (
        pd.DataFrame(selections),
        pd.concat(all_scores, ignore_index=True),
        pd.DataFrame(realized_rows).sort_values("effective_close_date"),
    )

