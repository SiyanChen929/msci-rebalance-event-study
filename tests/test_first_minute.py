from __future__ import annotations

import pandas as pd
import pytest

from msci_rebalance.first_minute import (
    block_max_t_sign_flip_pvalue,
    build_time_window_portfolios,
    extract_proxy_window,
    max_t_sign_flip_pvalue,
)


def test_exact_window_uses_start_stamped_1550_bar_close() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-31 15:49:00",
                    "2026-08-31 15:50:00",
                    "2026-08-31 15:51:00",
                ]
            ),
            "open": [100.0, 101.0, 105.0],
            "close": [101.0, 104.0, 106.0],
            "volume": [1_000, 4_000, 2_000],
        }
    )

    result = extract_proxy_window(frame, pd.Timestamp("2026-08-31"))

    assert result is not None
    assert result["entry_open_proxy"] == 100.0
    assert result["exit_close_proxy"] == 104.0
    assert result["underlying_return"] == pytest.approx(0.04)
    assert pd.Timestamp(result["exit_bar_timestamp"]).strftime("%H:%M") == "15:50"


def test_time_window_builder_preserves_add_delete_direction() -> None:
    clocks = [f"15:{minute:02d}" for minute in range(30, 60)]
    records = []
    for action, multiplier in [("add", 1.0), ("delete", -1.0)]:
        for index, clock in enumerate(clocks):
            open_price = 100.0 + index
            close_price = open_price + multiplier
            records.append(
                {
                    "effective_close_date": "2026-08-31",
                    "ticker": action.upper(),
                    "action": action,
                    "clock": clock,
                    "bar_open": open_price,
                    "bar_close": close_price,
                    "signed_bar_return": (1.0 if action == "add" else -1.0)
                    * (close_price / open_price - 1.0),
                }
            )
    _, portfolios = build_time_window_portfolios(pd.DataFrame(records))

    one_minute = portfolios.loc[
        portfolios["analysis"].eq("one_minute_bar")
        & portfolios["window"].eq("15:50")
    ].iloc[0]
    assert one_minute["gross_return"] > 0
    two_bar = portfolios.loc[
        portfolios["analysis"].eq("adjacent_two_bar_window")
        & portfolios["window"].eq("15:49_open_to_15:50_close")
    ].iloc[0]
    assert two_bar["gross_return"] > 0


def test_max_t_randomization_is_deterministic() -> None:
    rows = []
    for date_index, date in enumerate(pd.date_range("2020-01-01", periods=8, freq="90D")):
        for window, shift in [("candidate", 0.002), ("placebo", 0.0)]:
            rows.append(
                {
                    "analysis": "toy",
                    "window": window,
                    "effective_close_date": date,
                    "gross_return": shift + (date_index - 3.5) * 0.0001,
                }
            )
    frame = pd.DataFrame(rows)
    first = max_t_sign_flip_pvalue(frame, "candidate", samples=1_000, seed=7)
    second = max_t_sign_flip_pvalue(frame, "candidate", samples=1_000, seed=7)
    assert first == second
    block_result = block_max_t_sign_flip_pvalue(frame, "candidate", block_size=4)
    assert 0.0 <= block_result[0] <= 1.0
