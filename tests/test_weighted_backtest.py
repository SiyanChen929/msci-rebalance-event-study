import numpy as np
import pandas as pd
import pytest

from msci_rebalance.weighted_backtest import (
    build_capped_leg_weights,
    weighted_date_portfolio,
)


def test_capped_weights_balance_sparse_legs_and_leave_cash() -> None:
    securities = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "add", "delete"],
            "score": [3.0, 2.0, 1.0, 1.0],
        }
    )
    result = build_capped_leg_weights(
        securities,
        "score",
        max_name_weight=0.10,
    )

    assert result.loc[result.action.eq("delete"), "target_weight"].iloc[0] == pytest.approx(-0.10)
    assert result["target_weight"].clip(lower=0).sum() == pytest.approx(0.10)
    assert result["target_weight"].abs().sum() == pytest.approx(0.20)


def test_weighted_portfolio_uses_signed_weights_and_exposure_scaled_cost() -> None:
    weights = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 2,
            "ticker": ["ADD", "DELETE"],
            "target_weight": [0.25, -0.25],
            "underlying_return": [0.02, -0.01],
        }
    )
    portfolio = weighted_date_portfolio(weights, cost_bps=10.0)
    row = portfolio.iloc[0]

    assert row["gross_return"] == pytest.approx(0.0075)
    assert row["gross_exposure"] == pytest.approx(0.50)
    assert row["net_return"] == pytest.approx(0.0070)
    assert row["net_exposure"] == pytest.approx(0.0)
    assert row["effective_names"] == pytest.approx(2.0)
    assert np.isfinite(row["weight_hhi"])


def test_security_specific_capacity_caps_bind_and_keep_legs_balanced() -> None:
    securities = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "delete", "delete"],
            "score": [4.0, 1.0, 4.0, 1.0],
            "capacity_cap": [0.03, 0.10, 0.03, 0.10],
        }
    )
    result = build_capped_leg_weights(
        securities,
        "score",
        max_name_weight=0.10,
        cap_column="capacity_cap",
    )

    assert result.loc[result.ticker.eq("A"), "target_weight"].iloc[0] == pytest.approx(0.03)
    assert result.loc[result.ticker.eq("C"), "target_weight"].iloc[0] == pytest.approx(-0.03)
    assert result["target_weight"].clip(lower=0).sum() == pytest.approx(0.13)
    assert -result["target_weight"].clip(upper=0).sum() == pytest.approx(0.13)
    assert (result["target_weight"].abs() <= result["effective_cap_weight"] + 1e-12).all()


def test_missing_selected_return_fails_instead_of_erasing_position() -> None:
    weights = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"],
            "ticker": ["ADD"],
            "target_weight": [0.10],
            "underlying_return": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="missing realized returns"):
        weighted_date_portfolio(weights)


def test_scheduled_no_trade_review_is_recorded_as_zero() -> None:
    weights = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"],
            "ticker": ["ADD"],
            "target_weight": [0.10],
            "underlying_return": [0.01],
        }
    )
    result = weighted_date_portfolio(
        weights,
        scheduled_dates=["2026-05-29", "2026-08-31"],
    )

    assert len(result) == 2
    assert result.loc[result["effective_close_date"].eq("2026-05-29"), "gross_return"].iloc[0] == 0
