import numpy as np
import pandas as pd
import pytest

from msci_rebalance.capacity_research import cap_frozen_portfolio


def fixture():
    return pd.DataFrame({"effective_close_date": ["2024-05-31"] * 4,
                         "ticker": ["A", "B", "C", "D"],
                         "target_weight": [0.3, 0.2, -0.25, -0.25],
                         "expected_interval_dollar_volume": [1e6, 10e6, 10e6, 10e6],
                         "liquidity_asof": ["2024-05-30"] * 4})


def test_round_trip_volume_budget_and_neutrality():
    result = cap_frozen_portfolio(fixture(), capital_usd=1e6, round_trip_participation=0.05)
    np.testing.assert_allclose(result.target_weight, [0.025, 0.2, -0.1125, -0.1125])
    assert result.target_weight.sum() == pytest.approx(0)
    assert result.round_trip_participation.max() <= 0.05 + 1e-12
    assert (result.target_weight.abs() <= result.original_target_weight.abs()).all()


def test_post_event_volume_and_return_cannot_change_capacity():
    frame = fixture()
    reference = cap_frozen_portfolio(frame, capital_usd=1e6, round_trip_participation=0.05)
    frame["exit_bar_volume"] = [1, 1e20, 0, np.nan]
    frame["underlying_return"] = [100, -1, 20, 0]
    result = cap_frozen_portfolio(frame, capital_usd=1e6, round_trip_participation=0.05)
    np.testing.assert_allclose(result.target_weight, reference.target_weight)
    frame.loc[0, "liquidity_asof"] = "2024-05-31"
    with pytest.raises(ValueError, match="strictly before"):
        cap_frozen_portfolio(frame, capital_usd=1e6, round_trip_participation=0.05)


def test_missing_liquidity_and_duplicates_fail_closed():
    frame = fixture()
    frame.loc[0, "expected_interval_dollar_volume"] = np.nan
    with pytest.raises(ValueError, match="positive prior liquidity"):
        cap_frozen_portfolio(frame, capital_usd=1e6, round_trip_participation=0.05)
    frame = pd.concat([fixture(), fixture().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        cap_frozen_portfolio(frame, capital_usd=1e6, round_trip_participation=0.05)
