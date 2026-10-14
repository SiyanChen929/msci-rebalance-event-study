#!/usr/bin/env python3
"""Research diagnostics for close-window decomposition and matched pairs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from msci_rebalance.backtest import aggregate_date_portfolio, performance_metrics


WINDOWS = {
    "15:49_open_to_15:50_open": "pre_1550_return_gross",
    "15:49_open_to_15:50_close_proxy": "first_minute_alpha_return_gross",
    "15:50_open_to_15:50_close_proxy": "release_minute_return_gross",
    "15:50_open_to_15:59_close": "post_1550_continuous_return_gross",
    "15:59_close_to_daily_close_proxy": "daily_close_gap_return_gross",
    "15:49_open_to_daily_close_proxy": "flow_return_gross",
}


def safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def session_dates(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def daily_features(
    daily_path: Path,
    spy_path: Path,
    announcement_date: pd.Timestamp,
) -> dict[str, object] | None:
    if not daily_path.exists() or not spy_path.exists():
        return None
    asset = pd.read_parquet(daily_path).copy()
    spy = pd.read_parquet(spy_path).copy()
    asset["date"] = session_dates(asset)
    spy["date"] = session_dates(spy)
    asset = asset.loc[asset["date"].le(announcement_date)].sort_values("date").tail(31)
    spy = spy.loc[spy["date"].le(announcement_date)].sort_values("date").tail(31)
    if len(asset) < 16 or len(spy) < 16:
        return None
    asset["return"] = asset["close"].pct_change()
    spy["spy_return"] = spy["close"].pct_change()
    aligned = asset[["date", "return"]].merge(spy[["date", "spy_return"]], on="date").dropna()
    if len(aligned) < 15 or float(aligned["spy_return"].var()) <= 0:
        return None
    beta = float(aligned["return"].cov(aligned["spy_return"]) / aligned["spy_return"].var())
    volatility = float(aligned["return"].std(ddof=1) * math.sqrt(252))
    momentum = float(asset["close"].iloc[-1] / asset["close"].iloc[0] - 1.0)
    dollar_volume = float((asset["close"] * asset["volume"]).median())
    return {
        "beta": beta,
        "volatility": volatility,
        "momentum": momentum,
        "log_dollar_volume": math.log(max(dollar_volume, 1.0)),
        "returns": aligned.set_index("date")["return"],
        "feature_observations": len(aligned),
    }


def match_event_pairs(group: pd.DataFrame, marketdata_root: Path) -> list[dict[str, object]]:
    event_date = pd.Timestamp(group["effective_close_date"].iloc[0]).normalize()
    announcement_date = pd.Timestamp(group["announcement_date"].iloc[0]).normalize()
    spy_path = marketdata_root / "daily" / f"{event_date.date()}_SPY.parquet"
    securities: list[dict[str, object]] = []
    for row in group.itertuples(index=False):
        daily_path = marketdata_root / "daily" / (
            f"{event_date.date()}_{safe_ticker(str(row.ticker))}.parquet"
        )
        features = daily_features(daily_path, spy_path, announcement_date)
        if features is None or not np.isfinite(row.flow_return_gross):
            continue
        securities.append({**row._asdict(), **features})
    frame = pd.DataFrame(securities)
    if frame.empty or not {"add", "delete"}.issubset(set(frame["action"])):
        return []

    feature_columns = ["beta", "volatility", "momentum", "log_dollar_volume"]
    for column in feature_columns:
        standard_deviation = float(frame[column].std(ddof=0))
        frame[f"z_{column}"] = (
            (frame[column] - frame[column].mean()) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )
    adds = frame.loc[frame["action"].eq("add")].reset_index(drop=True)
    deletes = frame.loc[frame["action"].eq("delete")].reset_index(drop=True)
    costs = np.empty((len(adds), len(deletes)), dtype=float)
    correlations = np.empty_like(costs)
    for add_index, add in adds.iterrows():
        for delete_index, delete in deletes.iterrows():
            overlap = pd.concat(
                [add["returns"].rename("add"), delete["returns"].rename("delete")],
                axis=1,
                join="inner",
            ).dropna()
            correlation = float(overlap.corr().iloc[0, 1]) if len(overlap) >= 15 else 0.0
            if not np.isfinite(correlation):
                correlation = 0.0
            feature_distance = math.sqrt(
                sum(
                    (float(add[f"z_{column}"]) - float(delete[f"z_{column}"])) ** 2
                    for column in feature_columns
                )
            )
            correlations[add_index, delete_index] = correlation
            costs[add_index, delete_index] = feature_distance + 0.5 * (1.0 - correlation)

    add_indices, delete_indices = linear_sum_assignment(costs)
    records: list[dict[str, object]] = []
    for pair_number, (add_index, delete_index) in enumerate(
        zip(add_indices, delete_indices, strict=True),
        start=1,
    ):
        add = adds.iloc[add_index]
        delete = deletes.iloc[delete_index]
        records.append(
            {
                "effective_close_date": event_date,
                "announcement_date": announcement_date,
                "pair_number": pair_number,
                "add_ticker": add["ticker"],
                "add_security_name": add["security_name"],
                "delete_ticker": delete["ticker"],
                "delete_security_name": delete["security_name"],
                "pre_event_return_correlation": correlations[add_index, delete_index],
                "matching_cost": costs[add_index, delete_index],
                "add_beta": add["beta"],
                "delete_beta": delete["beta"],
                "add_volatility": add["volatility"],
                "delete_volatility": delete["volatility"],
                "add_momentum": add["momentum"],
                "delete_momentum": delete["momentum"],
                "add_log_dollar_volume": add["log_dollar_volume"],
                "delete_log_dollar_volume": delete["log_dollar_volume"],
                "feature_observations": min(
                    int(add["feature_observations"]), int(delete["feature_observations"])
                ),
                "add_signed_return": add["flow_return_gross"],
                "delete_signed_return": delete["flow_return_gross"],
                "pair_gross_return": 0.5
                * (float(add["flow_return_gross"]) + float(delete["flow_return_gross"])),
            }
        )
    return records


def event_time_panel(
    trades: pd.DataFrame,
    marketdata_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a date-balanced 15:30–15:59 event-time panel.

    Prices are descriptive minute bars, not executable quotes.  Return signs are
    oriented to the published add/delete direction.  Volume is unsigned and is
    normalized against the same clock minute in prior cached sessions when at
    least three such sessions are available.
    """

    security_rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        path = marketdata_root / "minute" / (
            f"{event_date.date()}_{safe_ticker(str(row.ticker))}.parquet"
        )
        if not path.exists():
            continue
        frame = pd.read_parquet(path).copy()
        frame["date"] = session_dates(frame)
        timestamp = pd.to_datetime(frame["timestamp"])
        if getattr(timestamp.dt, "tz", None) is not None:
            timestamp = timestamp.dt.tz_convert("America/New_York")
        frame["clock"] = timestamp.dt.strftime("%H:%M")
        event = frame.loc[
            frame["date"].eq(event_date) & frame["clock"].between("15:30", "15:59")
        ].sort_values("timestamp")
        if len(event) != 30 or not {"15:30", "15:49", "15:50", "15:59"}.issubset(
            set(event["clock"])
        ):
            continue
        baseline = float(event.iloc[0]["open"])
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        day_volume = float(frame.loc[frame["date"].eq(event_date), "volume"].sum())
        direction = 1.0 if row.action == "add" else -1.0
        prior = frame.loc[frame["date"].lt(event_date)]
        prior_dates = prior["date"].nunique()
        prior_clock_medians = (
            prior.groupby("clock")["volume"].median() if prior_dates >= 3 else pd.Series(dtype=float)
        )
        for bar in event.itertuples(index=False):
            normal_volume = float(prior_clock_medians.get(bar.clock, np.nan))
            security_rows.append(
                {
                    "effective_close_date": event_date,
                    "ticker": row.ticker,
                    "action": row.action,
                    "clock": bar.clock,
                    "bar_open": float(bar.open),
                    "bar_close": float(bar.close),
                    "signed_return_from_1530_open": direction
                    * (float(bar.close) / baseline - 1.0),
                    "signed_bar_return": direction * (float(bar.close) / float(bar.open) - 1.0),
                    "minute_volume_share": float(bar.volume) / day_volume if day_volume > 0 else np.nan,
                    "minute_volume_vs_prior_same_clock": (
                        float(bar.volume) / normal_volume
                        if np.isfinite(normal_volume) and normal_volume > 0
                        else np.nan
                    ),
                }
            )
    security = pd.DataFrame(security_rows)
    records: list[dict[str, object]] = []
    if security.empty:
        return security, pd.DataFrame()
    for (event_date, clock), group in security.groupby(
        ["effective_close_date", "clock"], sort=True
    ):
        adds = group.loc[group["action"].eq("add")]
        deletes = group.loc[group["action"].eq("delete")]

        def balanced(column: str) -> float:
            if not adds.empty and not deletes.empty:
                return 0.5 * float(adds[column].mean()) + 0.5 * float(deletes[column].mean())
            if not adds.empty:
                return 0.5 * float(adds[column].mean())
            if not deletes.empty:
                return 0.5 * float(deletes[column].mean())
            return np.nan

        records.append(
            {
                "effective_close_date": event_date,
                "clock": clock,
                "signed_return_from_1530_open": balanced("signed_return_from_1530_open"),
                "signed_bar_return": balanced("signed_bar_return"),
                "add_signed_return_from_1530_open": float(
                    adds["signed_return_from_1530_open"].mean()
                )
                if not adds.empty
                else np.nan,
                "delete_signed_return_from_1530_open": float(
                    deletes["signed_return_from_1530_open"].mean()
                )
                if not deletes.empty
                else np.nan,
                "mean_minute_volume_share": float(group["minute_volume_share"].mean()),
                "median_minute_volume_vs_prior_same_clock": float(
                    group["minute_volume_vs_prior_same_clock"].median()
                ),
                "names": len(group),
            }
        )
    return security, pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--marketdata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(args.ledger)
    trades["effective_close_date"] = pd.to_datetime(trades["effective_close_date"])
    trades["announcement_date"] = pd.to_datetime(trades["announcement_date"])

    window_portfolios: list[pd.DataFrame] = []
    window_metrics: list[dict[str, object]] = []
    for window, column in WINDOWS.items():
        for cost_bps in (0.0, 10.0):
            portfolio = aggregate_date_portfolio(trades, column, cost_bps=cost_bps)
            portfolio.insert(0, "window", window)
            window_portfolios.append(portfolio)
            window_metrics.append(
                {
                    "window": window,
                    "cost_bps": cost_bps,
                    **performance_metrics(portfolio),
                }
            )
    pd.concat(window_portfolios, ignore_index=True).to_csv(
        args.output_dir / "window_decomposition_portfolios.csv", index=False
    )
    pd.DataFrame(window_metrics).to_csv(
        args.output_dir / "window_decomposition_metrics.csv", index=False
    )

    pair_records: list[dict[str, object]] = []
    for _, group in trades.groupby("effective_close_date", sort=True):
        pair_records.extend(match_event_pairs(group, args.marketdata_dir))
    pairs = pd.DataFrame(pair_records)
    pairs.to_csv(args.output_dir / "matched_pairs.csv", index=False)
    pair_portfolio = (
        pairs.groupby("effective_close_date")
        .agg(
            gross_return=("pair_gross_return", "mean"),
            n_pairs=("pair_number", "count"),
            median_pre_event_correlation=("pre_event_return_correlation", "median"),
            median_matching_cost=("matching_cost", "median"),
        )
        .reset_index()
        .sort_values("effective_close_date")
    )
    pair_portfolio["gross_exposure"] = 1.0
    pair_portfolio["net_return"] = pair_portfolio["gross_return"] - 10.0 / 10_000.0
    pair_portfolio["equity"] = (1.0 + pair_portfolio["net_return"]).cumprod()
    pair_portfolio["peak"] = pair_portfolio["equity"].cummax().clip(lower=1.0)
    pair_portfolio["drawdown"] = pair_portfolio["equity"] / pair_portfolio["peak"] - 1.0
    pair_portfolio.to_csv(args.output_dir / "matched_pairs_portfolio.csv", index=False)
    pd.DataFrame(
        [
            {
                "construction": "announcement_time_similarity_matched_add_delete_pairs",
                "cost_bps": 10.0,
                "security_pairs": len(pairs),
                **performance_metrics(pair_portfolio),
            }
        ]
    ).to_csv(args.output_dir / "matched_pairs_metrics.csv", index=False)

    volume_by_date = (
        trades.groupby("effective_close_date")
        .agg(
            security_events=("ticker", "count"),
            mean_signed_return=("flow_return_gross", "mean"),
            median_daily_volume_ratio=("volume_vs_20d_median", "median"),
            median_last10_volume_share=("last10_volume_share", "median"),
            median_last_minute_volume_share=("last_minute_volume_share", "median"),
        )
        .reset_index()
    )
    usable_volume = volume_by_date.dropna(
        subset=["mean_signed_return", "median_daily_volume_ratio"]
    )
    spearman = stats.spearmanr(
        usable_volume["mean_signed_return"],
        usable_volume["median_daily_volume_ratio"],
    )
    volume_by_date.to_csv(args.output_dir / "volume_diagnostics_by_date.csv", index=False)
    pd.DataFrame(
        [
            {
                "review_dates": len(usable_volume),
                "spearman_signed_return_vs_median_volume_ratio": spearman.statistic,
                "two_sided_pvalue": spearman.pvalue,
                "median_event_daily_volume_vs_20d": usable_volume[
                    "median_daily_volume_ratio"
                ].median(),
                "median_event_last10_share": usable_volume[
                    "median_last10_volume_share"
                ].median(),
            }
        ]
    ).to_csv(args.output_dir / "volume_diagnostics_metrics.csv", index=False)

    event_time_security, event_time_dates = event_time_panel(trades, args.marketdata_dir)
    event_time_security.to_csv(
        args.output_dir / "event_time_security_panel.csv", index=False
    )
    event_time_dates.to_csv(args.output_dir / "event_time_date_panel.csv", index=False)

    print(
        f"window_rows={sum(len(frame) for frame in window_portfolios)} "
        f"matched_pairs={len(pairs)} pair_dates={len(pair_portfolio)} "
        f"event_time_security_rows={len(event_time_security)}"
    )


if __name__ == "__main__":
    main()
