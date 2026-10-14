from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msci_rebalance.weighting import (
    WeightingConfig,
    build_target_weights,
    estimate_pre_event_liquidity,
)


def test_pre_event_liquidity_ignores_event_and_future_sessions() -> None:
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01"]
            ),
            "close": [10.0, 10.0, 10.0, 10.0],
            "volume": [100, 300, 10_000_000, 20_000_000],
        }
    )

    result = estimate_pre_event_liquidity(
        daily,
        "2026-08-31",
        lookback_sessions=2,
        interval_share_of_adv=0.01,
    )

    assert result["pre_event_adv20_usd"] == 2_000.0
    assert result["pre_event_interval_dollar_volume"] == 20.0
    assert result["liquidity_asof"] == pd.Timestamp("2026-08-28")
    assert result["liquidity_observations"] == 2


def test_true_flow_pressure_drives_weights_and_balances_legs() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 4,
            "ticker": ["B", "A", "D", "C"],
            "action": ["add", "add", "delete", "delete"],
            "predicted_net_flow_usd": [20.0, 10.0, -20.0, -10.0],
            "expected_interval_dollar_volume": [100.0] * 4,
        }
    )
    config = WeightingConfig(
        capital_usd=1.0,
        max_name_weight=0.40,
        max_interval_participation=1.0,
        missing_liquidity_max_weight=0.01,
    )

    result = build_target_weights(candidates, config)
    weights = result.weights.set_index("ticker")

    assert weights.loc["B", "target_weight"] == pytest.approx(1 / 3)
    assert weights.loc["A", "target_weight"] == pytest.approx(1 / 6)
    assert weights.loc["D", "target_weight"] == pytest.approx(-1 / 3)
    assert weights.loc["C", "target_weight"] == pytest.approx(-1 / 6)
    assert result.diagnostics.loc[0, "gross_weight"] == pytest.approx(1.0)
    assert result.diagnostics.loc[0, "net_weight"] == pytest.approx(0.0)


def test_delta_weight_controls_direction_when_flow_is_missing() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["UP", "DOWN"],
            "action": ["delete", "add"],
            "delta_weight": [0.001, -0.002],
            "expected_interval_dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )
    config = WeightingConfig(
        capital_usd=1_000_000,
        max_name_weight=0.5,
        max_interval_participation=1.0,
    )

    weights = build_target_weights(candidates, config).weights.set_index("ticker")

    assert weights.loc["UP", "target_weight"] == pytest.approx(0.5)
    assert weights.loc["DOWN", "target_weight"] == pytest.approx(-0.5)
    assert weights.loc["UP", "flow_direction_conflicts_with_action"]
    assert weights.loc["DOWN", "flow_direction_conflicts_with_action"]


def test_proxy_uses_pre_event_liquidity_and_applies_participation_cap() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "delete", "delete"],
            "pre_event_interval_dollar_volume": [100.0, 400.0, 100.0, 400.0],
            # Deliberately extreme event-day fields: the implementation must ignore them.
            "day_volume": [1e12, 1.0, 1e12, 1.0],
            "last10_volume": [1e12, 1.0, 1e12, 1.0],
        }
    )
    config = WeightingConfig(
        capital_usd=100.0,
        max_name_weight=0.5,
        max_interval_participation=0.1,
    )

    result = build_target_weights(candidates, config)
    weights = result.weights.set_index("ticker")

    assert weights.loc["A", "effective_cap_weight"] == pytest.approx(0.1)
    assert weights.loc["B", "effective_cap_weight"] == pytest.approx(0.4)
    assert weights.loc["A", "target_abs_weight"] <= 0.1
    assert weights.loc["C", "target_abs_weight"] <= 0.1
    assert result.diagnostics.loc[0, "long_weight"] == pytest.approx(
        result.diagnostics.loc[0, "short_weight"]
    )
    assert set(weights["signal_source"]) == {"action_liquidity_proxy"}


def test_missing_liquidity_falls_back_to_small_name_cap() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["ADD", "DELETE"],
            "action": ["add", "delete"],
        }
    )
    config = WeightingConfig(
        missing_liquidity_max_weight=0.02,
        max_name_weight=0.10,
    )

    result = build_target_weights(candidates, config)

    assert np.allclose(result.weights["target_abs_weight"], 0.02)
    assert result.diagnostics.loc[0, "gross_weight"] == pytest.approx(0.04)
    assert result.diagnostics.loc[0, "missing_liquidity_names"] == 2
    assert result.weights["fallback_reason"].str.contains("missing_pre_event_liquidity").all()


def test_event_day_liquidity_is_rejected_without_using_unsafe_columns() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["ADD", "DELETE"],
            "action": ["add", "delete"],
            "pre_event_interval_dollar_volume": [1_000_000.0, 1_000_000.0],
            "liquidity_asof": ["2026-08-31 15:50", "2026-08-31 15:50"],
        }
    )
    config = WeightingConfig(missing_liquidity_max_weight=0.015)

    result = build_target_weights(candidates, config)

    assert np.allclose(result.weights["target_abs_weight"], 0.015)
    assert result.weights["liquidity_lookahead_rejected"].all()
    assert result.diagnostics.loc[0, "liquidity_lookahead_rejected_names"] == 2


def test_signal_after_freeze_is_rejected_and_falls_back_to_action() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["ADD", "DELETE"],
            "action": ["add", "delete"],
            "predicted_net_flow_usd": [-100.0, 100.0],
            "signal_asof": ["2026-08-31 15:49:01", "2026-08-31 15:49:01"],
            "pre_event_interval_dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )

    result = build_target_weights(candidates, freeze_time="2026-08-31 15:48:30")
    weights = result.weights.set_index("ticker")

    assert weights.loc["ADD", "target_weight"] > 0
    assert weights.loc["DELETE", "target_weight"] < 0
    assert weights["signal_lookahead_rejected"].all()
    assert set(weights["signal_source"]) == {"action_liquidity_proxy"}


def test_missing_leg_produces_cash_under_production_default() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["A", "B"],
            "action": ["add", "add"],
            "pre_event_interval_dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )

    result = build_target_weights(candidates)

    assert np.allclose(result.weights["target_weight"], 0.0)
    assert result.diagnostics.loc[0, "cash_weight"] == pytest.approx(1.0)
    assert not result.diagnostics.loc[0, "both_legs_present"]


def test_name_and_asymmetric_capacity_caps_preserve_leg_balance() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 3,
            "ticker": ["LARGE_LONG", "OTHER_LONG", "SMALL_SHORT"],
            "action": ["add", "add", "delete"],
            "predicted_net_flow_usd": [100.0, 1.0, -100.0],
            "expected_interval_dollar_volume": [10_000.0, 10_000.0, 100.0],
        }
    )
    config = WeightingConfig(
        capital_usd=100.0,
        max_name_weight=0.25,
        max_interval_participation=0.10,
    )

    result = build_target_weights(candidates, config)
    weights = result.weights.set_index("ticker")

    assert weights["target_abs_weight"].max() <= 0.25
    assert weights.loc["SMALL_SHORT", "target_abs_weight"] == pytest.approx(0.10)
    assert result.diagnostics.loc[0, "long_weight"] == pytest.approx(0.10)
    assert result.diagnostics.loc[0, "short_weight"] == pytest.approx(0.10)


def test_output_is_deterministic_under_input_shuffle() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"] * 4,
            "ticker": ["D", "B", "A", "C"],
            "action": ["delete", "add", "add", "delete"],
            "predicted_net_flow_usd": [-20.0, 20.0, 10.0, -10.0],
            "expected_interval_dollar_volume": [100.0] * 4,
        }
    )
    config = WeightingConfig(
        capital_usd=1.0,
        max_name_weight=0.5,
        max_interval_participation=1.0,
    )

    first = build_target_weights(candidates, config).weights
    second = build_target_weights(candidates.sample(frac=1.0, random_state=17), config).weights

    pd.testing.assert_frame_equal(first, second)


def test_duplicate_identity_is_rejected() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["A", "A"],
            "action": ["add", "add"],
        }
    )

    with pytest.raises(ValueError, match="Candidate identity must be unique"):
        build_target_weights(candidates)


def test_required_asof_mode_rejects_missing_freeze_cutoff() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["ADD", "DELETE"],
            "action": ["add", "delete"],
            "predicted_net_flow_usd": [100.0, -100.0],
            "signal_asof": ["2026-08-20", "2026-08-20"],
        }
    )

    with pytest.raises(ValueError, match="requires signal_freeze_ts"):
        build_target_weights(
            candidates,
            WeightingConfig(require_asof_for_true_inputs=True),
        )


def test_unavailable_short_borrow_zeros_short_and_balanced_long_leg() -> None:
    candidates = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "ticker": ["ADD", "DELETE"],
            "action": ["add", "delete"],
            "pre_event_interval_dollar_volume": [1_000_000.0, 1_000_000.0],
            "borrow_available": [True, False],
        }
    )

    result = build_target_weights(candidates)

    assert np.allclose(result.weights["target_weight"], 0.0)
    assert result.weights.loc[
        result.weights["ticker"].eq("DELETE"), "borrow_unavailable_rejected"
    ].iloc[0]
    assert result.diagnostics.loc[0, "borrow_unavailable_rejected_names"] == 1
