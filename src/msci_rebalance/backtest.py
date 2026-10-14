"""Event extraction, portfolio aggregation, and date-level inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PLACEBO_OFFSETS = (-3, -1, 1, 3)


def _safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def _paths(root: Path, event_date: pd.Timestamp, ticker: str) -> tuple[Path, Path]:
    prefix = f"{event_date.strftime('%Y-%m-%d')}_{_safe_ticker(ticker)}.parquet"
    return root / "minute" / prefix, root / "daily" / prefix


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    if "_session_date" in frame:
        return frame["_session_date"]
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def _clock(frame: pd.DataFrame) -> pd.Series:
    if "_clock" in frame:
        return frame["_clock"]
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp.dt.strftime("%H:%M")


def _intraday_window_return(frame: pd.DataFrame, session_date: pd.Timestamp) -> dict[str, float] | None:
    session = frame.loc[_session_dates(frame).eq(session_date)].copy()
    if session.empty:
        return None
    session["clock"] = _clock(session)
    entry = session.loc[session["clock"].eq("15:49")]
    dissemination = session.loc[session["clock"].eq("15:50")]
    last = session.sort_values("timestamp").tail(1)
    if (
        entry.empty
        or dissemination.empty
        or last.empty
        or str(last["clock"].iloc[0]) != "15:59"
    ):
        return None
    entry_price = float(entry.iloc[0]["open"])
    entry_close = float(entry.iloc[0]["close"])
    dissemination_open = float(dissemination.iloc[0]["open"])
    dissemination_close = float(dissemination.iloc[0]["close"])
    close_price = float(last.iloc[0]["close"])
    prices = (
        entry_price,
        entry_close,
        dissemination_open,
        dissemination_close,
        close_price,
    )
    if not all(np.isfinite(price) for price in prices) or min(prices) <= 0:
        return None
    last10 = session.loc[session["clock"].between("15:50", "15:59")]
    return {
        "entry_1549": entry_price,
        "close_1549": entry_close,
        "entry_1550": dissemination_open,
        "close_1550": dissemination_close,
        "last_minute_close": close_price,
        "underlying_1549_bar_return": entry_close / entry_price - 1.0,
        "underlying_1549_to_1550_return": dissemination_open / entry_price - 1.0,
        "underlying_1550_bar_return": dissemination_close / dissemination_open - 1.0,
        "underlying_1549_to_1550_close_return": dissemination_close / entry_price - 1.0,
        "underlying_1550_to_last_minute_return": close_price / dissemination_open - 1.0,
        "underlying_last_minute_return": close_price / entry_price - 1.0,
        "day_volume": float(session["volume"].sum()),
        "last10_volume": float(last10["volume"].sum()),
        "last_minute_volume": float(last.iloc[0]["volume"]),
        "dissemination_minute_volume": float(dissemination.iloc[0]["volume"]),
        "last10_volume_share": float(last10["volume"].sum() / session["volume"].sum())
        if session["volume"].sum() > 0
        else np.nan,
        "last_minute_volume_share": float(last.iloc[0]["volume"] / session["volume"].sum())
        if session["volume"].sum() > 0
        else np.nan,
    }


def _next_daily_open(frame: pd.DataFrame, event_date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Return the provider's official next-session daily open."""

    if frame.empty or "timestamp" not in frame:
        return None
    dates = sorted(_session_dates(frame).drop_duplicates())
    candidates = [date for date in dates if date > event_date]
    if not candidates:
        return None
    date = candidates[0]
    session = frame.loc[_session_dates(frame).eq(date)].sort_values("timestamp")
    if session.empty:
        return None
    return date, float(session.iloc[0]["open"])


def _announcement_drift(
    daily: pd.DataFrame,
    announcement_date: pd.Timestamp,
    event_date: pd.Timestamp,
) -> tuple[pd.Timestamp, float, float] | None:
    if daily.empty or "timestamp" not in daily:
        return None
    dates = _session_dates(daily)
    post_announcement = sorted(dates.loc[dates.gt(announcement_date)].drop_duplicates())
    event = daily.loc[dates.eq(event_date)].sort_values("timestamp")
    if not post_announcement or event.empty:
        return None
    entry_date = post_announcement[0]
    entry = daily.loc[dates.eq(entry_date)].sort_values("timestamp")
    if entry.empty or entry_date > event_date:
        return None
    return entry_date, float(entry.iloc[0]["open"]), float(event.iloc[-1]["close"])


def _daily_volume_features(daily: pd.DataFrame, event_date: pd.Timestamp) -> dict[str, float]:
    if daily.empty or "timestamp" not in daily:
        return {"daily_event_volume": np.nan, "volume_vs_20d_median": np.nan}
    dates = _session_dates(daily)
    event = daily.loc[dates.eq(event_date)]
    history = daily.loc[dates.lt(event_date)].sort_values("timestamp").tail(20)
    if event.empty or history.empty:
        return {"daily_event_volume": np.nan, "volume_vs_20d_median": np.nan}
    event_volume = float(event.iloc[-1]["volume"])
    median = float(history["volume"].median())
    return {
        "daily_event_volume": event_volume,
        "volume_vs_20d_median": event_volume / median if median > 0 else np.nan,
    }


def _daily_close(daily: pd.DataFrame, event_date: pd.Timestamp) -> float:
    if daily.empty or "timestamp" not in daily:
        return np.nan
    event = daily.loc[_session_dates(daily).eq(event_date)].sort_values("timestamp")
    return float(event.iloc[-1]["close"]) if not event.empty else np.nan


def _minute_to_daily_price_scale(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    event_date: pd.Timestamp,
) -> float:
    """Detect vendor price-basis mismatches without absorbing auction moves.

    Some historical minute files can be on a different split/ADR basis than
    the corresponding daily candle.  The median of open/high/low ratios is
    robust to an auction-only extreme in one field.  A wide 10% gate means
    ordinary opening/closing-auction differences are deliberately left alone.
    """

    if daily.empty:
        return 1.0
    minute_session = minute.loc[_session_dates(minute).eq(event_date)].sort_values("timestamp")
    daily_session = daily.loc[_session_dates(daily).eq(event_date)].sort_values("timestamp")
    if minute_session.empty or daily_session.empty:
        return 1.0
    daily_row = daily_session.iloc[-1]
    candidates = (
        float(daily_row["open"]) / float(minute_session.iloc[0]["open"]),
        float(daily_row["high"]) / float(minute_session["high"].max()),
        float(daily_row["low"]) / float(minute_session["low"].min()),
    )
    valid = [value for value in candidates if np.isfinite(value) and value > 0]
    if len(valid) < 2:
        return 1.0
    candidate = float(np.median(valid))
    return candidate if candidate < 0.9 or candidate > 1.1 else 1.0


def extract_trade_ledger(events: pd.DataFrame, marketdata_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(marketdata_root)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    spy_cache: dict[pd.Timestamp, dict[str, float] | None] = {}
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.effective_close_date).normalize()
        announcement_date = pd.Timestamp(event.announcement_date).normalize()
        minute_path, daily_path = _paths(root, event_date, str(event.ticker))
        if not minute_path.exists():
            failures.append(
                {
                    "effective_close_date": event_date,
                    "ticker": event.ticker,
                    "security_name": event.security_name,
                    "reason": "missing_minute_cache_file",
                }
            )
            continue
        minute = pd.read_parquet(minute_path)
        daily = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
        minute["_session_date"] = _session_dates(minute)
        minute["_clock"] = _clock(minute)
        if not daily.empty:
            daily["_session_date"] = _session_dates(daily)
        window = _intraday_window_return(minute, event_date)
        if window is None:
            failures.append(
                {
                    "effective_close_date": event_date,
                    "ticker": event.ticker,
                    "security_name": event.security_name,
                    "reason": "missing_1549_or_close_bar",
                }
            )
            continue
        direction = 1.0 if event.action == "add" else -1.0
        row = event._asdict()
        row.update(window)
        price_scale = _minute_to_daily_price_scale(minute, daily, event_date)
        row["raw_entry_1549"] = window["entry_1549"]
        row["raw_close_1549"] = window["close_1549"]
        row["raw_entry_1550"] = window["entry_1550"]
        row["raw_close_1550"] = window["close_1550"]
        row["raw_last_minute_close"] = window["last_minute_close"]
        row["minute_to_daily_price_scale"] = price_scale
        row["price_scale_reconciled"] = not np.isclose(price_scale, 1.0)
        row["entry_1549"] = window["entry_1549"] * price_scale
        row["close_1549"] = window["close_1549"] * price_scale
        row["entry_1550"] = window["entry_1550"] * price_scale
        row["close_1550"] = window["close_1550"] * price_scale
        row["last_minute_close"] = window["last_minute_close"] * price_scale
        daily_close = _daily_close(daily, event_date)
        row["daily_close_proxy"] = daily_close
        row["close_source"] = "marketdata_daily_close_proxy" if np.isfinite(daily_close) else "missing"
        if not np.isfinite(daily_close):
            failures.append(
                {
                    "effective_close_date": event_date,
                    "ticker": event.ticker,
                    "security_name": event.security_name,
                    "reason": "missing_daily_close_proxy_strategy_excluded",
                }
            )
        row["close_discrepancy_bps"] = (
            (row["last_minute_close"] / daily_close - 1.0) * 10_000.0
            if np.isfinite(daily_close) and daily_close > 0
            else np.nan
        )
        row["flow_return_gross"] = (
            direction * (daily_close / row["entry_1549"] - 1.0)
            if np.isfinite(daily_close)
            else np.nan
        )
        row["last_minute_flow_return_gross"] = direction * window["underlying_last_minute_return"]
        row["pre_1550_return_gross"] = direction * window["underlying_1549_to_1550_return"]
        row["release_minute_return_gross"] = direction * window["underlying_1550_bar_return"]
        row["first_minute_alpha_return_gross"] = (
            direction * window["underlying_1549_to_1550_close_return"]
        )
        row["post_1550_continuous_return_gross"] = (
            direction * window["underlying_1550_to_last_minute_return"]
        )
        row["daily_close_gap_return_gross"] = (
            direction * (daily_close / row["last_minute_close"] - 1.0)
            if np.isfinite(daily_close) and row["last_minute_close"] > 0
            else np.nan
        )
        row.update(_daily_volume_features(daily, event_date))

        next_open = _next_daily_open(daily, event_date)
        if next_open and np.isfinite(daily_close):
            row["next_session_date"] = next_open[0]
            row["next_open"] = next_open[1]
            overnight_underlying = row["next_open"] / daily_close - 1.0
            row["reversal_return_gross"] = -direction * overnight_underlying
        else:
            row["next_session_date"] = pd.NaT
            row["next_open"] = np.nan
            row["reversal_return_gross"] = np.nan

        drift = _announcement_drift(daily, announcement_date, event_date)
        if drift:
            row["drift_entry_date"] = drift[0]
            row["drift_entry_open"] = drift[1]
            row["announcement_drift_return_gross"] = direction * (drift[2] / drift[1] - 1.0)
        else:
            row["drift_entry_date"] = pd.NaT
            row["drift_entry_open"] = np.nan
            row["announcement_drift_return_gross"] = np.nan

        session_dates = sorted(_session_dates(minute).drop_duplicates())
        if event_date in session_dates:
            event_idx = session_dates.index(event_date)
            for offset in PLACEBO_OFFSETS:
                target_idx = event_idx + offset
                placebo = None
                placebo_close = np.nan
                if 0 <= target_idx < len(session_dates):
                    target_date = session_dates[target_idx]
                    placebo = _intraday_window_return(minute, target_date)
                    placebo_close = _daily_close(daily, target_date)
                    placebo_scale = _minute_to_daily_price_scale(minute, daily, target_date)
                row[f"placebo_{offset:+d}_gross"] = (
                    direction * (placebo_close / (placebo["entry_1549"] * placebo_scale) - 1.0)
                    if placebo and np.isfinite(placebo_close) and placebo_close > 0
                    else np.nan
                )

        if event_date not in spy_cache:
            spy_minute, spy_daily = _paths(root, event_date, "SPY")
            spy_window = (
                _intraday_window_return(pd.read_parquet(spy_minute), event_date)
                if spy_minute.exists()
                else None
            )
            if spy_window and spy_daily.exists():
                spy_daily_frame = pd.read_parquet(spy_daily)
                spy_minute_frame = pd.read_parquet(spy_minute)
                spy_close = _daily_close(spy_daily_frame, event_date)
                spy_scale = _minute_to_daily_price_scale(
                    spy_minute_frame, spy_daily_frame, event_date
                )
                spy_window["underlying_daily_close_return"] = (
                    spy_close / (spy_window["entry_1549"] * spy_scale) - 1.0
                    if np.isfinite(spy_close) and spy_close > 0
                    else np.nan
                )
            spy_cache[event_date] = spy_window
        row["spy_flow_return"] = (
            spy_cache[event_date]["underlying_daily_close_return"]
            if spy_cache[event_date]
            else np.nan
        )
        row["market_adjusted_flow_return_gross"] = (
            row["flow_return_gross"] - direction * row["spy_flow_return"]
            if np.isfinite(row["flow_return_gross"]) and np.isfinite(row["spy_flow_return"])
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(failures)


def aggregate_date_portfolio(
    trades: pd.DataFrame,
    return_column: str,
    cost_bps: float = 0.0,
    single_leg_weight: float = 0.5,
) -> pd.DataFrame:
    """Leg-balance additions/deletions and create one observation per date.

    Each side receives 50% gross weight when both are present.  If a review has
    only one side, that side receives ``single_leg_weight`` and the remainder is
    cash, keeping the exposure convention stable across review compositions.
    """

    records: list[dict[str, object]] = []
    usable = trades.dropna(subset=[return_column]).copy()
    for event_date, group in usable.groupby("effective_close_date"):
        adds = group.loc[group["action"].eq("add"), return_column]
        deletes = group.loc[group["action"].eq("delete"), return_column]
        if not adds.empty and not deletes.empty:
            gross = 0.5 * float(adds.mean()) + 0.5 * float(deletes.mean())
            gross_exposure = 1.0
        elif not adds.empty:
            gross = single_leg_weight * float(adds.mean())
            gross_exposure = single_leg_weight
        elif not deletes.empty:
            gross = single_leg_weight * float(deletes.mean())
            gross_exposure = single_leg_weight
        else:
            continue
        records.append(
            {
                "effective_close_date": pd.Timestamp(event_date),
                "gross_return": gross,
                "net_return": gross - gross_exposure * cost_bps / 10_000.0,
                "cost_bps": cost_bps,
                "gross_exposure": gross_exposure,
                "n_adds": len(adds),
                "n_deletes": len(deletes),
                "n_names": len(group),
                "add_leg_return": float(adds.mean()) if not adds.empty else np.nan,
                "delete_leg_return": float(deletes.mean()) if not deletes.empty else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values("effective_close_date").reset_index(drop=True)


def moving_block_bootstrap_mean(
    returns: np.ndarray,
    block_size: int = 4,
    samples: int = 10_000,
    seed: int = 20260831,
) -> tuple[float, float]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = [np.take(values, np.arange(start, start + block_size) % len(values)) for start in range(len(values))]
    means = np.empty(samples)
    need = math.ceil(len(values) / block_size)
    for i in range(samples):
        chosen = rng.integers(0, len(blocks), size=need)
        sample = np.concatenate([blocks[j] for j in chosen])[: len(values)]
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def performance_metrics(portfolio: pd.DataFrame, return_column: str = "net_return") -> dict[str, object]:
    frame = portfolio.dropna(subset=[return_column]).sort_values("effective_close_date")
    returns = frame[return_column].to_numpy(float)
    if not len(returns):
        return {}
    equity = np.cumprod(1.0 + returns)
    equity_with_initial = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / peaks - 1.0
    years = max((pd.Timestamp(frame.effective_close_date.iloc[-1]) - pd.Timestamp(frame.effective_close_date.iloc[0])).days / 365.25, 1 / 365.25)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    downside = returns[returns < 0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else np.nan
    mean = float(np.mean(returns))
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    ci_low, ci_high = moving_block_bootstrap_mean(returns)
    t_result = stats.ttest_1samp(returns, 0.0, alternative="greater") if len(returns) > 1 else None
    sign_result = stats.binomtest(int((returns > 0).sum()), len(returns), 0.5, alternative="greater")
    total_positive = float(wins.sum())
    best_contribution = float(wins.max() / total_positive) if len(wins) and total_positive > 0 else np.nan
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    max_dd = float(drawdowns.min())
    return {
        "start_date": frame.effective_close_date.iloc[0],
        "end_date": frame.effective_close_date.iloc[-1],
        "event_dates": len(returns),
        "total_return": float(equity[-1] - 1.0),
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "event_sharpe_sqrt4": float(np.sqrt(4) * mean / std) if std and std > 0 else np.nan,
        "event_sortino_sqrt4": float(np.sqrt(4) * mean / downside_dev)
        if downside_dev and downside_dev > 0
        else np.nan,
        "mean_return_per_event": mean,
        "median_return_per_event": float(np.median(returns)),
        "volatility_per_event": std,
        "hit_rate": float((returns > 0).mean()),
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) else np.nan,
        "worst_event": float(returns.min()),
        "best_event": float(returns.max()),
        "expected_shortfall_95": float(np.mean(np.sort(returns)[: max(1, math.ceil(0.05 * len(returns)))])),
        "t_stat": float(t_result.statistic) if t_result else np.nan,
        "one_sided_t_pvalue": float(t_result.pvalue) if t_result else np.nan,
        "sign_test_pvalue": float(sign_result.pvalue),
        "block_bootstrap_mean_ci_low": ci_low,
        "block_bootstrap_mean_ci_high": ci_high,
        "best_date_share_of_positive_pnl": best_contribution,
        "break_even_cost_bps": float(
            frame["gross_return"].mean() / frame["gross_exposure"].mean() * 10_000.0
        )
        if "gross_return" in frame and "gross_exposure" in frame
        else float(frame["gross_return"].mean() * 10_000.0)
        if "gross_return" in frame
        else mean * 10_000.0,
    }


def active_trade_metrics(
    portfolio: pd.DataFrame,
    return_column: str = "net_return",
    *,
    exposure_column: str = "gross_exposure",
    risk_free_return_per_active_event: float = 0.0,
    events_per_year: float = 4.0,
) -> dict[str, object]:
    """Measure conditional trade quality after excluding zero-exposure decisions.

    The optional risk-free return must match the actual holding interval.  It is
    explicit so a quarterly cash yield cannot accidentally be subtracted from a
    one-minute trade return.
    """

    if exposure_column not in portfolio or return_column not in portfolio:
        raise KeyError(f"missing {exposure_column!r} or {return_column!r}")
    exposure = pd.to_numeric(portfolio[exposure_column], errors="coerce")
    active = portfolio.loc[exposure.gt(0)]
    returns = pd.to_numeric(active[return_column], errors="coerce").dropna().to_numpy(float)
    risk_free = float(risk_free_return_per_active_event)
    if not len(returns):
        return {
            "active_trade_dates": 0,
            "active_mean_return": np.nan,
            "active_excess_mean_return": np.nan,
            "active_volatility_per_event": np.nan,
            "active_trade_sharpe_sqrt4": np.nan,
            "active_trade_hit_rate": np.nan,
            "risk_free_return_per_active_event": risk_free,
        }
    excess = returns - risk_free
    volatility = float(np.std(excess, ddof=1)) if len(excess) > 1 else np.nan
    sharpe = (
        float(np.sqrt(events_per_year) * np.mean(excess) / volatility)
        if volatility and volatility > 0
        else np.nan
    )
    return {
        "active_trade_dates": int(len(returns)),
        "active_mean_return": float(np.mean(returns)),
        "active_excess_mean_return": float(np.mean(excess)),
        "active_volatility_per_event": volatility,
        "active_trade_sharpe_sqrt4": sharpe,
        "active_trade_hit_rate": float(np.mean(returns > 0)),
        "risk_free_return_per_active_event": risk_free,
    }


def add_equity_columns(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.sort_values("effective_close_date").copy()
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["peak"] = out["equity"].cummax().clip(lower=1.0)
    out["drawdown"] = out["equity"] / out["peak"] - 1.0
    return out
