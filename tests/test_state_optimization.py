import pandas as pd

from msci_rebalance.state_optimization import (
    expanding_walk_forward,
    select_rule,
    training_scores,
)


def _portfolio_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-05-31", periods=10, freq="90D")
    rows = []
    for index, date in enumerate(dates):
        for rule, net, exposure in (
            ("baseline_all", 0.001 if index < 8 else -0.01, 0.60),
            ("down_to_small_hour_soft_075", 0.0011 if index < 8 else 0.05, 0.55),
            ("down_to_small_hour_confirmation", 0.0009 if index < 8 else 0.08, 0.40),
        ):
            rows.append(
                {
                    "effective_close_date": date,
                    "filter_rule": rule,
                    "net_return": net,
                    "gross_exposure": exposure,
                }
            )
    return pd.DataFrame(rows)


def test_constrained_selector_uses_lowest_intervention() -> None:
    scores = pd.DataFrame(
        {
            "filter_rule": [
                "baseline_all",
                "down_to_small_hour_soft_075",
                "down_to_small_hour_confirmation",
            ],
            "eligible_constrained": [False, True, True],
            "intervention_rank": [0, 1, 4],
        }
    )
    assert (
        select_rule(scores, "constrained_minimum_intervention")
        == "down_to_small_hour_soft_075"
    )


def test_training_scores_require_hit_and_return_constraints() -> None:
    panel = _portfolio_panel()
    dates = sorted(panel["effective_close_date"].unique())[:8]
    scores = training_scores(
        panel,
        dates,
        [
            "baseline_all",
            "down_to_small_hour_soft_075",
            "down_to_small_hour_confirmation",
        ],
        {
            "baseline_all": 0,
            "down_to_small_hour_soft_075": 1,
            "down_to_small_hour_confirmation": 4,
        },
        minimum_active_reviews=3,
        minimum_return_retention=0.8,
        minimum_hit_improvement=0.05,
        minimum_exposure_ratio=0.35,
    )
    assert not scores["eligible_constrained"].any()


def test_walk_forward_does_not_use_future_returns_for_selection() -> None:
    panel = _portfolio_panel()
    rules = [
        "baseline_all",
        "down_to_small_hour_soft_075",
        "down_to_small_hour_confirmation",
    ]
    ranks = {
        "baseline_all": 0,
        "down_to_small_hour_soft_075": 1,
        "down_to_small_hour_confirmation": 4,
    }
    kwargs = dict(
        minimum_training_reviews=8,
        minimum_active_reviews=3,
        minimum_return_retention=0.8,
        minimum_hit_improvement=0.05,
        minimum_exposure_ratio=0.35,
        policy="maximum_training_mean",
    )
    first = expanding_walk_forward(panel, rules, ranks, **kwargs)[0]
    changed = panel.copy()
    final_date = changed["effective_close_date"].max()
    changed.loc[changed["effective_close_date"].eq(final_date), "net_return"] *= -100
    second = expanding_walk_forward(changed, rules, ranks, **kwargs)[0]
    assert first.iloc[0]["selected_filter_rule"] == second.iloc[0][
        "selected_filter_rule"
    ]

