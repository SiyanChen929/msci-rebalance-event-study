from pathlib import Path

import numpy as np
import pandas as pd

from msci_rebalance.flow_features import (
    build_pre_event_feature_panel,
    extract_pre_event_features,
)


def _write_cache(root: Path, event_date: str, ticker: str) -> None:
    daily_dates = pd.date_range("2026-08-24", periods=6, freq="B", tz="America/New_York")
    daily = pd.DataFrame(
        {
            "ticker": ticker,
            "timestamp": daily_dates,
            "open": [10.0] * 6,
            "high": [10.0] * 6,
            "low": [10.0] * 6,
            "close": [10.0, 10.1, 10.2, 10.3, 10.4, 100.0],
            "volume": [100, 100, 100, 100, 100, 1_000_000],
        }
    )
    minute_rows = []
    for date, scale in (("2026-08-27", 1), ("2026-08-28", 2), (event_date, 1_000)):
        for clock in ("15:49", "15:50"):
            minute_rows.append(
                {
                    "ticker": ticker,
                    "timestamp": pd.Timestamp(f"{date} {clock}", tz="America/New_York"),
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100 * scale,
                }
            )
    minute = pd.DataFrame(minute_rows)
    safe = ticker.replace(".", "_").replace("/", "_")
    (root / "daily").mkdir(parents=True)
    (root / "minute").mkdir(parents=True)
    daily.to_parquet(root / "daily" / f"{event_date}_{safe}.parquet", index=False)
    minute.to_parquet(root / "minute" / f"{event_date}_{safe}.parquet", index=False)


def test_extract_pre_event_features_excludes_event_day(tmp_path: Path) -> None:
    _write_cache(tmp_path, "2026-08-31", "TEST")
    result = extract_pre_event_features(pd.Timestamp("2026-08-31"), "TEST", tmp_path)

    assert result["daily_feature_sessions"] == 5
    assert np.isclose(result["median_daily_dollar_volume_20d"], 1_020.0)
    assert result["interval_feature_sessions"] == 2
    assert result["expected_interval_dollar_volume_direct"] == 3_000.0
    assert result["expected_interval_dollar_volume_direct"] < 1_000_000.0


def test_feature_panel_builds_signed_proxy_and_pit_flag(tmp_path: Path) -> None:
    _write_cache(tmp_path, "2026-08-31", "TEST")
    securities = pd.DataFrame(
        {
            "effective_close_date": ["2026-08-31"],
            "ticker": ["TEST"],
            "action": ["delete"],
            "gross_return": [0.001],
        }
    )
    panel = build_pre_event_feature_panel(
        securities,
        tmp_path,
        minimum_daily_sessions=5,
    )

    row = panel.iloc[0]
    assert row["proxy_predicted_net_flow_usd"] < 0
    assert np.isfinite(row["proxy_flow_pressure"])
    assert bool(row["feature_point_in_time_valid"])
    assert pd.Timestamp(row["feature_cutoff_date"]) < pd.Timestamp(row["effective_close_date"])
