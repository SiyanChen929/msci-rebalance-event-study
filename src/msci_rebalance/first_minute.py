"""Validation helpers for the 15:49-open through 15:50-close proxy.

MarketData minute candles are start-stamped.  The exit candle labelled 15:50
covers trades from 15:50:00 through 15:50:59.  These functions deliberately
call the candle close a proxy: it is neither a quote nor a guaranteed fill.
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from msci_rebalance.backtest import aggregate_date_portfolio, performance_metrics


ENTRY_CLOCK = "15:49"
EXIT_BAR_CLOCK = "15:50"
CONTROL_OFFSETS = (-4, -3, -2, -1)


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def _clock(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp.dt.strftime("%H:%M")


def extract_proxy_window(
    frame: pd.DataFrame,
    session_date: pd.Timestamp,
    entry_clock: str = ENTRY_CLOCK,
    exit_bar_clock: str = EXIT_BAR_CLOCK,
) -> dict[str, object] | None:
    """Extract an open-to-later-bar-close return from one complete session."""

    if {"_date", "_clock"}.issubset(frame.columns):
        working = frame
    else:
        working = frame.copy()
        working["_date"] = _session_dates(working)
        working["_clock"] = _clock(working)
    session = working.loc[working["_date"].eq(pd.Timestamp(session_date).normalize())]
    entry = session.loc[session["_clock"].eq(entry_clock)].sort_values("timestamp")
    exit_bar = session.loc[session["_clock"].eq(exit_bar_clock)].sort_values("timestamp")
    if entry.empty or exit_bar.empty:
        return None
    entry_row = entry.iloc[0]
    exit_row = exit_bar.iloc[-1]
    entry_price = float(entry_row["open"])
    exit_price = float(exit_row["close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or min(entry_price, exit_price) <= 0:
        return None
    day_volume = float(session["volume"].sum()) if "volume" in session else np.nan
    exit_volume = float(exit_row["volume"]) if "volume" in exit_row else np.nan
    return {
        "session_date": pd.Timestamp(session_date).normalize(),
        "entry_bar_timestamp": entry_row["timestamp"],
        "exit_bar_timestamp": exit_row["timestamp"],
        "entry_open_proxy": entry_price,
        "exit_close_proxy": exit_price,
        "underlying_return": exit_price / entry_price - 1.0,
        "exit_bar_volume": exit_volume,
        "day_volume": day_volume,
        "exit_bar_volume_share": exit_volume / day_volume
        if np.isfinite(exit_volume) and np.isfinite(day_volume) and day_volume > 0
        else np.nan,
    }


def build_event_and_control_ledgers(
    trades: pd.DataFrame,
    marketdata_root: str | Path,
    offsets: tuple[int, ...] = CONTROL_OFFSETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact event windows and same-name prior-session controls."""

    root = Path(marketdata_root)
    event_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        event_date = pd.Timestamp(row.effective_close_date).normalize()
        path = root / "minute" / f"{event_date.date()}_{_safe_ticker(str(row.ticker))}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["_date"] = _session_dates(frame)
        frame["_clock"] = _clock(frame)
        frame_dates = frame["_date"]
        session_dates = sorted(frame_dates.drop_duplicates())
        if event_date not in session_dates:
            continue
        direction = 1.0 if row.action == "add" else -1.0
        event_window = extract_proxy_window(frame, event_date)
        if event_window is None:
            continue
        identity = {
            "effective_close_date": event_date,
            "announcement_date": pd.Timestamp(row.announcement_date).normalize(),
            "ticker": row.ticker,
            "security_name": row.security_name,
            "name_key": row.name_key,
            "action": row.action,
            "change_type": row.change_type,
            "direction": direction,
        }
        event_rows.append(
            {
                **identity,
                **event_window,
                "gross_return": direction * float(event_window["underlying_return"]),
                "fill_model": "minute_bar_open_to_later_bar_close_proxy",
                "information_cutoff": "15:48:59.999 America/New_York",
            }
        )
        event_index = session_dates.index(event_date)
        complete_control_number = 0
        for target_index in range(event_index - 1, -1, -1):
            target_date = session_dates[target_index]
            control_window = extract_proxy_window(frame, target_date)
            if control_window is None:
                continue
            complete_control_number += 1
            control_rank = -complete_control_number
            control_rows.append(
                {
                    **identity,
                    "control_offset": control_rank,
                    "source_session_offset": target_index - event_index,
                    "control_session_date": target_date,
                    **control_window,
                    "gross_return": direction * float(control_window["underlying_return"]),
                }
            )
            if complete_control_number == len(offsets):
                break
    return pd.DataFrame(event_rows), pd.DataFrame(control_rows)


def matched_control_analysis(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    offsets: tuple[int, ...] = CONTROL_OFFSETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return complete-pair security rows, portfolios and gross metrics."""

    keys = ["effective_close_date", "ticker", "name_key", "action"]
    control_wide = controls.pivot_table(
        index=keys,
        columns="control_offset",
        values="gross_return",
        aggfunc="first",
    )
    available = [offset for offset in offsets if offset in control_wide.columns]
    if len(available) != len(offsets):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    control_wide = control_wide.dropna(subset=list(offsets)).reset_index()
    control_wide["matched_control_mean"] = control_wide[list(offsets)].mean(axis=1)
    event_sample = events.rename(columns={"gross_return": "event_gross_return"})
    paired = event_sample.merge(control_wide, on=keys, how="inner")
    paired["paired_abnormal_return"] = paired["event_gross_return"] - paired["matched_control_mean"]

    portfolio_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    candidates: list[tuple[str, pd.DataFrame, str]] = [
        ("event_on_complete_control_sample", paired, "event_gross_return"),
        ("matched_control_mean", paired, "matched_control_mean"),
        ("event_minus_matched_control", paired, "paired_abnormal_return"),
    ]
    for offset in offsets:
        candidates.append((f"control_session_{offset:+d}", paired, offset))
    for label, frame, column in candidates:
        portfolio = aggregate_date_portfolio(frame, column, cost_bps=0.0)
        portfolio.insert(0, "analysis", label)
        portfolio_frames.append(portfolio)
        metric_rows.append({"analysis": label, **performance_metrics(portfolio, "gross_return")})
    return paired, pd.concat(portfolio_frames, ignore_index=True), pd.DataFrame(metric_rows)


def build_time_window_portfolios(
    security_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every one-minute bar and every adjacent two-bar proxy window."""

    required = {"bar_open", "bar_close", "signed_bar_return"}
    missing = required.difference(security_panel.columns)
    if missing:
        raise ValueError(f"event-time panel lacks fields: {sorted(missing)}")
    panel = security_panel.copy()
    panel["effective_close_date"] = pd.to_datetime(panel["effective_close_date"])
    clocks = [f"15:{minute:02d}" for minute in range(30, 60)]
    panel["clock"] = pd.Categorical(panel["clock"], categories=clocks, ordered=True)

    security_windows: list[dict[str, object]] = []
    group_keys = ["effective_close_date", "ticker", "action"]
    for keys, group in panel.groupby(group_keys, observed=True, sort=True):
        ordered = group.sort_values("clock")
        if len(ordered) != len(clocks):
            continue
        direction = 1.0 if keys[2] == "add" else -1.0
        for index in range(len(clocks) - 1):
            entry = ordered.iloc[index]
            exit_bar = ordered.iloc[index + 1]
            underlying = float(exit_bar["bar_close"]) / float(entry["bar_open"]) - 1.0
            security_windows.append(
                {
                    "effective_close_date": keys[0],
                    "ticker": keys[1],
                    "action": keys[2],
                    "analysis": "adjacent_two_bar_window",
                    "window": f"{clocks[index]}_open_to_{clocks[index + 1]}_close",
                    "gross_return": direction * underlying,
                }
            )
        for row in ordered.itertuples(index=False):
            security_windows.append(
                {
                    "effective_close_date": keys[0],
                    "ticker": keys[1],
                    "action": keys[2],
                    "analysis": "one_minute_bar",
                    "window": str(row.clock),
                    "gross_return": float(row.signed_bar_return),
                }
            )
    security = pd.DataFrame(security_windows)
    portfolios: list[pd.DataFrame] = []
    for (analysis, window), group in security.groupby(["analysis", "window"], sort=True):
        portfolio = aggregate_date_portfolio(group, "gross_return", cost_bps=0.0)
        portfolio.insert(0, "window", window)
        portfolio.insert(0, "analysis", analysis)
        portfolios.append(portfolio)
    return security, pd.concat(portfolios, ignore_index=True)


def _t_statistic(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    std = float(values.std(ddof=1))
    return float(values.mean() / (std / math.sqrt(len(values)))) if std > 0 else np.nan


def max_t_sign_flip_pvalue(
    portfolio: pd.DataFrame,
    candidate_window: str,
    samples: int = 20_000,
    seed: int = 20260901,
) -> tuple[float, int, int]:
    """Family-wise p-value across all inspected windows using date sign flips."""

    matrix = portfolio.pivot(
        index="effective_close_date", columns="window", values="gross_return"
    ).dropna(axis=0, how="any")
    if candidate_window not in matrix.columns or len(matrix) < 2:
        return np.nan, len(matrix), len(matrix.columns)
    values = matrix.to_numpy(float)
    observed = _t_statistic(matrix[candidate_window].to_numpy(float))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    chunk_size = 500
    while completed < samples:
        size = min(chunk_size, samples - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(matrix), 1))
        randomized = signs * values[None, :, :]
        means = randomized.mean(axis=1)
        stds = randomized.std(axis=1, ddof=1)
        t_stats = np.divide(
            means * math.sqrt(len(matrix)),
            stds,
            out=np.full_like(means, -np.inf),
            where=stds > 0,
        )
        exceed += int((np.max(t_stats, axis=1) >= observed).sum())
        completed += size
    return (exceed + 1) / (samples + 1), len(matrix), len(matrix.columns)


def block_max_t_sign_flip_pvalue(
    portfolio: pd.DataFrame,
    candidate_window: str,
    block_size: int = 4,
) -> tuple[float, int, int]:
    """Exact max-t test with one random sign per contiguous review block."""

    matrix = portfolio.pivot(
        index="effective_close_date", columns="window", values="gross_return"
    ).dropna(axis=0, how="any").sort_index()
    if candidate_window not in matrix.columns or len(matrix) < 2:
        return np.nan, len(matrix), len(matrix.columns)
    values = matrix.to_numpy(float)
    observed = _t_statistic(matrix[candidate_window].to_numpy(float))
    block_count = math.ceil(len(matrix) / block_size)
    exceed = 0
    permutations = 0
    for block_signs in product((-1.0, 1.0), repeat=block_count):
        signs = np.repeat(np.asarray(block_signs), block_size)[: len(matrix), None]
        randomized = signs * values
        means = randomized.mean(axis=0)
        stds = randomized.std(axis=0, ddof=1)
        t_stats = np.divide(
            means * math.sqrt(len(matrix)),
            stds,
            out=np.full_like(means, -np.inf),
            where=stds > 0,
        )
        exceed += int(np.max(t_stats) >= observed)
        permutations += 1
    return exceed / permutations, len(matrix), len(matrix.columns)


def summarize_time_windows(
    portfolios: pd.DataFrame,
    samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metrics and max-stat corrected tests for the selected windows."""

    metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    candidates = {
        "one_minute_bar": "15:50",
        "adjacent_two_bar_window": "15:49_open_to_15:50_close",
    }
    for analysis, group in portfolios.groupby("analysis"):
        for window, portfolio in group.groupby("window"):
            metric_rows.append(
                {
                    "analysis": analysis,
                    "window": window,
                    **performance_metrics(portfolio, "gross_return"),
                }
            )
        candidate = candidates[analysis]
        candidate_portfolio = group.loc[group["window"].eq(candidate)]
        values = candidate_portfolio["gross_return"].to_numpy(float)
        raw_test = stats.ttest_1samp(values, 0.0, alternative="greater")
        adjusted, dates, windows = max_t_sign_flip_pvalue(
            group, candidate, samples=samples, seed=20260901 + len(summary_rows)
        )
        block_adjusted, _, _ = block_max_t_sign_flip_pvalue(group, candidate, block_size=4)
        means = group.groupby("window")["gross_return"].mean().sort_values(ascending=False)
        summary_rows.append(
            {
                "analysis": analysis,
                "candidate_window": candidate,
                "complete_review_dates": dates,
                "windows_examined": windows,
                "candidate_mean_return": float(values.mean()),
                "candidate_t_stat": float(raw_test.statistic),
                "raw_one_sided_t_pvalue": float(raw_test.pvalue),
                "bonferroni_pvalue": min(1.0, float(raw_test.pvalue) * windows),
                "max_t_familywise_pvalue": adjusted,
                "block4_max_t_familywise_pvalue": block_adjusted,
                "rank_by_mean": int(means.index.get_loc(candidate) + 1),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(summary_rows)
