import numpy as np
import pandas as pd
import pytest

from msci_rebalance.ranking import (
    add_licensed_flow_scores,
    add_liquidity_volatility_proxy_scores,
    select_balanced_top_n,
)


def _panel(names_per_side: int = 6) -> pd.DataFrame:
    rows = []
    for side in ("add", "delete"):
        for index in range(names_per_side):
            rows.append(
                {
                    "effective_close_date": "2026-08-31",
                    "ticker": f"{side[0]}{index}",
                    "action": side,
                    "score": float(index),
                }
            )
    return pd.DataFrame(rows)


def test_balanced_top_ten_selects_five_per_side() -> None:
    selected, diagnostics = select_balanced_top_n(_panel(), "score")
    assert selected.groupby("action").size().to_dict() == {"add": 5, "delete": 5}
    assert selected.groupby("action")["side_rank"].max().eq(5).all()
    assert diagnostics["review_eligible"].all()


def test_balanced_top_ten_skips_sparse_review() -> None:
    selected, diagnostics = select_balanced_top_n(
        _panel(names_per_side=2), "score", minimum_per_side=3
    )
    assert selected.empty
    assert not diagnostics["review_eligible"].any()


def test_proxy_score_is_volatility_times_sqrt_pressure() -> None:
    frame = pd.DataFrame(
        {
            "median_daily_dollar_volume_20d": [100.0],
            "expected_interval_dollar_volume": [4.0],
            "daily_log_volatility_20d": [0.02],
        }
    )
    scored = add_liquidity_volatility_proxy_scores(frame)
    assert scored.loc[0, "proxy_pressure"] == pytest.approx(25.0)
    assert scored.loc[0, "volatility_scaled_pressure_proxy"] == pytest.approx(0.10)


def test_licensed_score_components() -> None:
    frame = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31", "2026-08-31"],
            "predicted_net_flow_usd": [100.0, -50.0],
            "float_adjusted_market_cap_usd": [1000.0, 2000.0],
            "expected_interval_dollar_volume": [25.0, 25.0],
            "daily_log_volatility_20d": [0.02, 0.04],
        }
    )
    scored = add_licensed_flow_scores(frame)
    assert scored["expected_flow_side"].tolist() == [1.0, -1.0]
    assert scored.loc[0, "flow_to_float_market_cap"] == pytest.approx(0.1)
    assert np.isfinite(scored["licensed_composite_score"]).all()
