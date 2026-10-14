from __future__ import annotations

import numpy as np
import pandas as pd

from msci_rebalance.deep_portfolio import (
    append_spy_beta_hedge,
    build_asymmetric_leg_weights,
    expanding_portfolio_selection,
    select_dynamic_score_coverage,
    select_top_k_per_side,
)


def _cross_section(names: int = 6) -> pd.DataFrame:
    rows = []
    for side in ("add", "delete"):
        for index in range(names):
            rows.append(
                {
                    "effective_close_date": "2026-08-31",
                    "ticker": f"{side[0]}{index}",
                    "action": side,
                    "score": float(names - index),
                    "underlying_return": 0.001,
                    "beta_used": 1.0 + 0.1 * index,
                }
            )
    return pd.DataFrame(rows)


def test_top_k_selection_keeps_independent_side_breadth() -> None:
    panel = _cross_section()
    selected, diagnostics = select_top_k_per_side(
        panel, "score", k_per_side=5, minimum_per_side=3
    )
    assert selected.groupby("action").size().to_dict() == {"add": 5, "delete": 5}
    assert diagnostics["review_eligible"].all()


def test_dynamic_coverage_uses_no_outcome_column() -> None:
    panel = _cross_section()
    selected, _ = select_dynamic_score_coverage(
        panel, "score", coverage=0.70, minimum_per_side=3, maximum_per_side=6
    )
    changed = panel.copy()
    changed["underlying_return"] = np.arange(len(changed)) * 100.0
    second, _ = select_dynamic_score_coverage(
        changed, "score", coverage=0.70, minimum_per_side=3, maximum_per_side=6
    )
    assert set(zip(selected["action"], selected["ticker"])) == set(
        zip(second["action"], second["ticker"])
    )


def test_asymmetric_weights_hit_fixed_leg_targets() -> None:
    weights = build_asymmetric_leg_weights(
        _cross_section(), long_fraction=2 / 3, gross_target=1.0, max_name_weight=0.25
    )
    assert np.isclose(weights.loc[weights["action"].eq("add"), "target_weight"].sum(), 2 / 3)
    assert np.isclose(-weights.loc[weights["action"].eq("delete"), "target_weight"].sum(), 1 / 3)
    assert np.isclose(weights["target_weight"].abs().sum(), 1.0)


def test_spy_hedge_zeroes_beta_and_preserves_total_gross() -> None:
    stocks = build_asymmetric_leg_weights(
        _cross_section(), long_fraction=2 / 3, gross_target=1.0, max_name_weight=0.25
    )
    date = pd.Timestamp("2026-08-31")
    hedged = append_spy_beta_hedge(stocks, {date: 0.002}, beta_band=0.0)
    beta = float((hedged["target_weight"] * hedged["beta_used"]).sum())
    assert abs(beta) < 1e-12
    assert np.isclose(hedged["target_weight"].abs().sum(), 1.0)
    assert hedged["ticker"].eq("SPY_HEDGE").sum() == 1


def _date_portfolios() -> pd.DataFrame:
    dates = pd.date_range("2020-05-29", periods=10, freq="90D")
    rows = []
    for index, date in enumerate(dates):
        for strategy, value in (("baseline", 0.001), ("challenger", 0.002)):
            rows.append(
                {
                    "strategy": strategy,
                    "effective_close_date": date,
                    "cost_bps": 10.0,
                    "net_return": value if index < 9 else -value,
                    "gross_exposure": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_selection_cannot_read_future_return() -> None:
    panel = _date_portfolios()
    kwargs = dict(
        cost_bps=10.0,
        minimum_training_reviews=8,
        minimum_active_reviews=3,
        minimum_exposure_ratio=0.5,
        fallback="baseline",
    )
    first = expanding_portfolio_selection(
        panel, ["baseline", "challenger"], **kwargs
    )[0]
    changed = panel.copy()
    final = changed["effective_close_date"].max()
    changed.loc[changed["effective_close_date"].eq(final), "net_return"] *= -1000
    second = expanding_portfolio_selection(
        changed, ["baseline", "challenger"], **kwargs
    )[0]
    assert first.iloc[0]["selected_strategy"] == second.iloc[0]["selected_strategy"]
