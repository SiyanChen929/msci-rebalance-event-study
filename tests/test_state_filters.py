import numpy as np
import pandas as pd
import pytest

from msci_rebalance.state_filters import (
    _intraday_state,
    apply_filter_and_balance,
    rule_mask,
    rule_multiplier,
    validate_state_features,
)


def _minute_session() -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-31 09:30",
        "2026-08-31 15:50",
        freq="min",
        tz="America/New_York",
    )
    close = 100.0 + np.arange(len(timestamp)) * 0.01
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 0.005,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": 1_000,
        }
    )


def test_safe_state_does_not_read_1548_bar() -> None:
    base = _minute_session()
    changed = base.copy()
    target = changed["timestamp"].dt.strftime("%H:%M").eq("15:48")
    changed.loc[target, ["open", "high", "low", "close"]] = 1_000.0
    changed.loc[target, "volume"] = 9_000_000
    safe_base = _intraday_state(base, pd.Timestamp("2026-08-31"), "15:47")
    safe_changed = _intraday_state(changed, pd.Timestamp("2026-08-31"), "15:47")
    tight_base = _intraday_state(base, pd.Timestamp("2026-08-31"), "15:48")
    tight_changed = _intraday_state(changed, pd.Timestamp("2026-08-31"), "15:48")
    assert safe_base is not None and safe_changed is not None
    assert safe_base["hour_return_pre"] == safe_changed["hour_return_pre"]
    assert safe_base["pre_day_dollar_volume"] == safe_changed["pre_day_dollar_volume"]
    assert tight_base is not None and tight_changed is not None
    assert tight_base["hour_return_pre"] != tight_changed["hour_return_pre"]


def test_direction_signed_relative_hour_rule() -> None:
    panel = pd.DataFrame(
        {
            "state_feature_point_in_time_valid": [True, True, True, False],
            "signed_relative_hour_return_pre": [0.01, -0.01, 0.02, 0.03],
        }
    )
    assert rule_mask(panel, "relative_hour_momentum").tolist() == [True, False, True, False]
    assert rule_mask(panel, "relative_hour_reversal").tolist() == [False, True, False, False]


def test_filter_balances_legs_without_raising_name_cap() -> None:
    panel = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2026-08-31"] * 4),
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "delete", "delete"],
            "target_weight": [0.1, 0.1, -0.1, -0.1],
            "state_feature_point_in_time_valid": [True] * 4,
            "signed_relative_hour_return_pre": [0.02, 0.01, 0.03, -0.01],
        }
    )
    filtered = apply_filter_and_balance(panel, "relative_hour_momentum")
    weights = filtered.set_index("ticker")["filtered_target_weight"]
    assert weights["A"] == pytest.approx(0.05)
    assert weights["B"] == pytest.approx(0.05)
    assert weights["C"] == pytest.approx(-0.1)
    assert weights["D"] == 0.0
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().max() <= 0.1


def test_validation_rejects_1548_state_bar() -> None:
    panel = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2026-08-31"]),
            "ticker": ["A"],
            "stock_feature_asof_timestamp": pd.to_datetime(
                ["2026-08-31 15:47:00-04:00"]
            ),
            "spy_feature_asof_timestamp": pd.to_datetime(
                ["2026-08-31 15:47:00-04:00"]
            ),
            "stock_daily_feature_asof": pd.to_datetime(["2026-08-28"]),
            "spy_daily_feature_asof": pd.to_datetime(["2026-08-28"]),
            "state_feature_point_in_time_valid": [True],
        }
    )
    validate_state_features(panel)
    panel.loc[0, "stock_feature_asof_timestamp"] = pd.Timestamp(
        "2026-08-31 15:48:00-04:00"
    )
    with pytest.raises(ValueError, match="reaches the event cutoff"):
        validate_state_features(panel)


def test_same_clock_rvol_uses_only_prior_sessions_and_safe_window() -> None:
    sessions = []
    for date, volume in (
        ("2026-08-26", 1_000),
        ("2026-08-27", 1_000),
        ("2026-08-28", 1_000),
        ("2026-08-31", 2_000),
    ):
        frame = _minute_session().copy()
        frame["timestamp"] = pd.date_range(
            f"{date} 09:30", f"{date} 15:50", freq="min", tz="America/New_York"
        )
        frame["volume"] = volume
        sessions.append(frame)
    combined = pd.concat(sessions, ignore_index=True)
    after_cutoff = (
        combined["timestamp"].dt.normalize().eq(pd.Timestamp("2026-08-31", tz="America/New_York"))
        & combined["timestamp"].dt.strftime("%H:%M").ge("15:48")
    )
    combined.loc[after_cutoff, "volume"] = 99_000_000
    state = _intraday_state(combined, pd.Timestamp("2026-08-31"), "15:47")
    assert state is not None
    assert state["prior_same_clock_sessions"] == 3
    assert state["same_clock_day_rvol"] == pytest.approx(2.0)
    assert state["same_clock_hour_rvol"] == pytest.approx(2.0)


def test_migration_gate_only_vetoes_unconfirmed_down_to_small() -> None:
    panel = pd.DataFrame(
        {
            "state_feature_point_in_time_valid": [True, True, True, True],
            "action": ["add", "delete", "delete", "delete"],
            "change_type": [
                "new_to_standard",
                "down_to_small",
                "down_to_small",
                "exit_standard",
            ],
            "signed_relative_hour_return_pre": [-0.02, -0.01, 0.01, -0.03],
        }
    )
    assert rule_mask(panel, "down_to_small_hour_confirmation").tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_soft_migration_gate_attenuates_only_unconfirmed_down_to_small() -> None:
    panel = pd.DataFrame(
        {
            "state_feature_point_in_time_valid": [True, True, True, True, False],
            "action": ["add", "delete", "delete", "delete", "delete"],
            "change_type": [
                "new_to_standard",
                "down_to_small",
                "down_to_small",
                "exit_standard",
                "down_to_small",
            ],
            "signed_relative_hour_return_pre": [-0.02, -0.01, 0.01, -0.03, 0.04],
        }
    )
    assert rule_multiplier(panel, "down_to_small_hour_soft_050").tolist() == [
        1.0,
        0.5,
        1.0,
        1.0,
        0.0,
    ]


def test_hour_day_confirmation_logic_is_predeclared() -> None:
    panel = pd.DataFrame(
        {
            "state_feature_point_in_time_valid": [True] * 4,
            "action": ["delete"] * 4,
            "change_type": ["down_to_small"] * 4,
            "signed_relative_hour_return_pre": [0.01, 0.01, -0.01, -0.01],
            "signed_relative_day_return_pre": [0.01, -0.01, 0.01, -0.01],
        }
    )
    assert rule_mask(panel, "down_to_small_hour_or_day_confirmation").tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert rule_mask(panel, "down_to_small_hour_and_day_confirmation").tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_soft_gate_balancing_never_raises_original_cap() -> None:
    panel = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2026-08-31"] * 4),
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "delete", "delete"],
            "change_type": [
                "new_to_standard",
                "new_to_standard",
                "down_to_small",
                "down_to_small",
            ],
            "target_weight": [0.1, 0.1, -0.1, -0.1],
            "state_feature_point_in_time_valid": [True] * 4,
            "signed_relative_hour_return_pre": [0.0, 0.0, 0.01, -0.01],
        }
    )
    filtered = apply_filter_and_balance(panel, "down_to_small_hour_soft_050")
    weights = filtered.set_index("ticker")["filtered_target_weight"]
    assert weights["A"] == pytest.approx(0.075)
    assert weights["B"] == pytest.approx(0.075)
    assert weights["C"] == pytest.approx(-0.1)
    assert weights["D"] == pytest.approx(-0.05)
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().max() <= 0.1
