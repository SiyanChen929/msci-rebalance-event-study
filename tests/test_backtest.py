from __future__ import annotations

import pandas as pd
import pytest

from msci_rebalance.backtest import (
    _intraday_window_return,
    _minute_to_daily_price_scale,
    active_trade_metrics,
    aggregate_date_portfolio,
)


def _minute(prices: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        prices,
        columns=["timestamp", "open", "high", "low", "close"],
    ).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))


def test_price_scale_reconciles_vendor_split_basis() -> None:
    minute = _minute(
        [
            ("2020-05-29 09:30", 106.0, 106.0, 105.0, 105.5),
            ("2020-05-29 15:59", 114.0, 116.0, 113.0, 113.5),
        ]
    )
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-05-29"]),
            "open": [53.0],
            "high": [58.0],
            "low": [52.5],
            "close": [56.75],
        }
    )
    assert _minute_to_daily_price_scale(minute, daily, pd.Timestamp("2020-05-29")) == 0.5


def test_price_scale_does_not_erase_auction_difference() -> None:
    minute = _minute(
        [
            ("2026-08-31 09:30", 100.0, 101.0, 99.0, 100.5),
            ("2026-08-31 15:59", 104.0, 105.0, 103.0, 104.5),
        ]
    )
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-31"]),
            "open": [101.0],
            "high": [106.0],
            "low": [99.0],
            "close": [102.0],
        }
    )
    assert _minute_to_daily_price_scale(minute, daily, pd.Timestamp("2026-08-31")) == 1.0


def test_date_portfolio_balances_add_and_delete_legs() -> None:
    trades = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2026-08-31"] * 3),
            "action": ["add", "add", "delete"],
            "return": [0.02, 0.04, 0.01],
        }
    )
    portfolio = aggregate_date_portfolio(trades, "return", cost_bps=10.0)
    assert portfolio.loc[0, "gross_return"] == 0.02
    assert portfolio.loc[0, "net_return"] == 0.019


def test_single_leg_date_keeps_other_half_in_cash() -> None:
    trades = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(["2025-05-30"] * 2),
            "action": ["delete", "delete"],
            "return": [0.02, 0.04],
        }
    )
    portfolio = aggregate_date_portfolio(trades, "return", cost_bps=10.0)
    assert portfolio.loc[0, "gross_exposure"] == 0.5
    assert portfolio.loc[0, "gross_return"] == 0.015
    assert portfolio.loc[0, "net_return"] == pytest.approx(0.0145)


def test_active_trade_sharpe_excludes_zero_exposure_decisions() -> None:
    portfolio = pd.DataFrame(
        {
            "effective_close_date": pd.to_datetime(
                ["2026-02-27", "2026-05-29", "2026-08-31"]
            ),
            "gross_exposure": [1.0, 1.0, 0.0],
            "net_return": [0.01, -0.005, 0.0],
        }
    )
    metrics = active_trade_metrics(portfolio)
    expected_mean = 0.0025
    expected_std = pd.Series([0.01, -0.005]).std(ddof=1)
    assert metrics["active_trade_dates"] == 2
    assert metrics["active_mean_return"] == pytest.approx(expected_mean)
    assert metrics["active_volatility_per_event"] == pytest.approx(expected_std)
    assert metrics["active_trade_sharpe_sqrt4"] == pytest.approx(
        2.0 * expected_mean / expected_std
    )


def test_intraday_window_decomposes_at_1550_without_overlap() -> None:
    minute = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-31 15:49:00",
                    "2026-08-31 15:50:00",
                    "2026-08-31 15:59:00",
                ]
            ),
            "open": [100.0, 102.0, 104.0],
            "high": [102.0, 104.0, 106.0],
            "low": [99.0, 101.0, 103.0],
            "close": [101.0, 103.0, 105.0],
            "volume": [1_000, 2_000, 7_000],
        }
    )

    result = _intraday_window_return(minute, pd.Timestamp("2026-08-31"))

    assert result is not None
    assert result["underlying_1549_to_1550_return"] == pytest.approx(0.02)
    assert result["underlying_1550_bar_return"] == pytest.approx(103 / 102 - 1)
    assert result["underlying_1549_to_1550_close_return"] == pytest.approx(0.03)
    assert result["underlying_1550_to_last_minute_return"] == pytest.approx(105 / 102 - 1)
    assert (1 + result["underlying_1549_to_1550_return"]) * (
        1 + result["underlying_1550_to_last_minute_return"]
    ) - 1 == pytest.approx(result["underlying_last_minute_return"])
