from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_leg_attribution import (
    make_date_balanced_change_type_diagnostics,
    make_event_ledger,
    make_period_summaries,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "baseline"

pytestmark = pytest.mark.skipif(
    not (OUT / "topn_capped10_trade_log.csv").exists(),
    reason="needs frozen pipeline outputs in outputs/baseline (run the pipeline first)",
)


def _frozen_primary_ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(OUT / "topn_capped10_trade_log.csv")
    dates = pd.read_csv(OUT / "topn_capped10_date_trade_log.csv")
    trades = trades.loc[
        trades["strategy"].eq("impact_proxy_top10")
        & np.isclose(trades["modeled_cost_bps"], 0.0)
    ].copy()
    dates = dates.loc[
        dates["strategy"].eq("impact_proxy_top10")
        & np.isclose(dates["cost_bps"], 0.0)
    ].copy()
    return trades, dates


def test_frozen_leg_attribution_reconciles_to_every_review() -> None:
    trades, dates = _frozen_primary_ledgers()
    event = make_event_ledger(trades, dates)

    assert len(event) == 26
    assert int(event["status"].eq("Traded").sum()) == 17
    assert int(event["status"].eq("Cash").sum()) == 9
    assert np.allclose(
        event["long_contribution_bp"] + event["short_contribution_bp"],
        event["gross_return_bp"],
        atol=1e-10,
    )
    assert np.isclose(
        event["gross_return_bp"].sum(),
        10_000.0 * trades["security_pnl_contribution"].sum(),
        atol=1e-10,
    )


def test_frozen_leg_totals_and_regime_signs_are_stable() -> None:
    trades, dates = _frozen_primary_ledgers()
    event = make_event_ledger(trades, dates)
    regime, _, _ = make_period_summaries(event)
    rows = regime.set_index("period")

    assert np.isclose(rows.loc["All", "long_total_bp"], 193.1197237977261)
    assert np.isclose(rows.loc["All", "short_total_bp"], 133.92013138801448)
    assert rows.loc["2020-2022", "long_total_bp"] > 0
    assert rows.loc["2020-2022", "short_total_bp"] < 0
    assert rows.loc["2023+", "long_total_bp"] > 0
    assert rows.loc["2023+", "short_total_bp"] > 0


def test_migration_diagnostic_is_date_balanced_before_regime_comparison() -> None:
    security = pd.read_csv(OUT / "first_minute_alpha_security_ledger.csv")
    diagnostic = make_date_balanced_change_type_diagnostics(security).set_index(
        ["regime", "action", "change_type"]
    )

    early = diagnostic.loc[("2020-2022", "delete", "down_to_small")]
    late = diagnostic.loc[("2023+", "delete", "down_to_small")]
    assert int(early["review_dates"]) == 11
    assert int(early["security_events"]) == 95
    assert early["mean_date_directional_return_bp"] < 0
    assert late["mean_date_directional_return_bp"] > 0
