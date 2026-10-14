from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from msci_rebalance.portfolio_variants import (
    build_beta_neutral_equal_weights,
    build_unneutralized_signal_weights,
    estimate_pre_event_beta,
    split_dollar_neutral_legs,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_close_date": ["2026-05-29"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "action": ["add", "add", "delete", "delete"],
            "beta_used": [1.0, 1.0, 2.0, 2.0],
            "score": [6.0, 2.0, 1.0, 1.0],
            "target_weight": [0.25, 0.25, -0.25, -0.25],
        }
    )


def test_beta_neutral_weights_allow_net_dollar_exposure() -> None:
    weights = build_beta_neutral_equal_weights(
        _panel(), gross_target=1.0, max_name_weight=0.40
    )
    beta_exposure = float((weights["target_weight"] * weights["beta_used"]).sum())
    assert abs(beta_exposure) < 1e-12
    assert np.isclose(weights["target_weight"].abs().sum(), 1.0)
    assert np.isclose(weights["target_weight"].sum(), 1.0 / 3.0)
    assert weights["target_weight"].abs().max() <= 0.40


def test_signal_weighting_does_not_redistribute_name_cap() -> None:
    weights = build_unneutralized_signal_weights(
        _panel(), "score", gross_target=1.0, max_name_weight=0.30
    )
    assert np.allclose(weights["target_weight"], [0.30, 0.20, -0.10, -0.10])
    assert np.isclose(weights["target_weight"].abs().sum(), 0.70)
    assert np.isclose(weights["target_weight"].sum(), 0.30)


def test_single_legs_exactly_reconstruct_base_weights() -> None:
    base = _panel()
    long_only, short_only = split_dollar_neutral_legs(base)
    reconstructed = long_only["target_weight"] + short_only["target_weight"]
    assert np.allclose(reconstructed, base["target_weight"])
    assert long_only["target_weight"].ge(0).all()
    assert short_only["target_weight"].le(0).all()


def test_beta_estimator_ignores_prices_after_cutoff(tmp_path: Path) -> None:
    dates = pd.date_range("2026-01-02", periods=30, freq="B", tz="America/New_York")
    market_return = np.linspace(-0.01, 0.012, len(dates))
    asset_return = 1.5 * market_return
    market_close = 100.0 * np.cumprod(1.0 + market_return)
    asset_close = 50.0 * np.cumprod(1.0 + asset_return)
    asset = pd.DataFrame({"timestamp": dates, "close": asset_close})
    market = pd.DataFrame({"timestamp": dates, "close": market_close})
    asset_path = tmp_path / "asset.parquet"
    market_path = tmp_path / "market.parquet"
    asset.to_parquet(asset_path)
    market.to_parquet(market_path)
    cutoff = dates[24].tz_localize(None)
    before = estimate_pre_event_beta(asset_path, market_path, cutoff)
    asset.loc[25:, "close"] *= 20.0
    asset.to_parquet(asset_path)
    after = estimate_pre_event_beta(asset_path, market_path, cutoff)
    assert np.isclose(before["beta_raw"], after["beta_raw"])
    assert before["beta_feature_asof"] == after["beta_feature_asof"]
    assert pd.Timestamp(before["beta_feature_asof"]) <= pd.Timestamp(cutoff)
