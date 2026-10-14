from __future__ import annotations

import pandas as pd
import pytest

from msci_rebalance.volatility import (
    breakout_proxy,
    hypothetical_straddle_return,
    window_statistics,
)


def _session(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    clocks = [f"2026-08-31 15:{minute:02d}:00" for minute in range(49, 60)]
    opens = [100.0, *closes[:-1]]
    highs = highs or [max(open_, close) for open_, close in zip(opens, closes, strict=True)]
    lows = lows or [min(open_, close) for open_, close in zip(opens, closes, strict=True)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(clocks),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_window_statistics_uses_full_11_minute_path() -> None:
    session = _session([100.1, 100.2, 100.1, 100.3, 100.4, 100.5, 100.4, 100.6, 100.7, 100.8, 101.0])
    result = window_statistics(session)
    assert result is not None
    assert result["absolute_terminal_move"] == pytest.approx(abs(__import__("math").log(1.01)))
    assert result["sqrt_realized_variance_11m"] > 0


def test_breakout_waits_until_next_bar_and_charges_cost() -> None:
    closes = [100.2, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.3]
    opens = [100.0, *closes[:-1]]
    highs = [100.30] + [max(open_, close) + 0.05 for open_, close in zip(opens[1:], closes[1:], strict=True)]
    lows = [99.95] + [min(open_, close) - 0.05 for open_, close in zip(opens[1:], closes[1:], strict=True)]
    session = _session(closes, highs, lows)
    result = breakout_proxy(session, barrier_bps=25, cost_bps=10)
    assert result is not None and result["triggered"]
    assert result["direction"] == 1
    assert result["fill_clock"] == "15:50"
    assert result["fill_price"] == pytest.approx(100.25)
    assert result["net_return"] == pytest.approx(101.3 / 100.25 - 1 - 0.001)


def test_ambiguous_bar_assigns_worse_direction() -> None:
    highs = [100.4] + [100.2] * 10
    lows = [99.6] + [99.8] * 10
    session = _session([100.0] * 11, highs, lows)
    result = breakout_proxy(session, barrier_bps=25, cost_bps=0)
    assert result is not None and result["ambiguous_same_bar"]
    assert result["gross_return"] <= 0


def test_incoherent_ohlc_window_is_rejected() -> None:
    session = _session([100.0] * 11)
    session.loc[3, "high"] = 99.0
    assert window_statistics(session) is None


def test_hypothetical_straddle_rewards_large_move_but_is_not_free() -> None:
    flat = hypothetical_straddle_return(0.0, dte=7, entry_volatility=0.5)
    moved = hypothetical_straddle_return(0.03, dte=7, entry_volatility=0.5)
    moved_with_friction = hypothetical_straddle_return(
        0.03,
        dte=7,
        entry_volatility=0.5,
        friction_fraction_of_premium=0.10,
    )
    assert moved > flat
    assert moved_with_friction == pytest.approx(moved - 0.10)
