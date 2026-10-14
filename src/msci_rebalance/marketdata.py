"""MarketData.app research downloader with local, inspectable parquet caches."""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


@dataclass(frozen=True)
class DownloadJob:
    ticker: str
    event_date: pd.Timestamp
    announcement_date: pd.Timestamp


def load_token(env_file: str | Path | None = None) -> str:
    token = os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
    if token:
        return token
    if env_file:
        for line in Path(env_file).read_text().splitlines():
            if line.strip().startswith(("MARKETDATA_TOKEN=", "MARKETDATA_API_KEY=")):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("MarketData.app token not found in environment or env file")


def _request_candles(
    ticker: str,
    resolution: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    token: str,
    retries: int = 6,
) -> tuple[pd.DataFrame, int, str]:
    symbol = quote(ticker, safe=".-")
    url = f"https://api.marketdata.app/v1/stocks/candles/{resolution}/{symbol}/"
    params = {
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "extended": "false",
        "adjustsplits": "true",
        "adjustdividends": "false",
    }
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=45)
            if response.status_code == 429:
                time.sleep((2**attempt) + random.random())
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("s") == "no_data":
                return pd.DataFrame(), response.status_code, "no_data"
            if payload.get("s") != "ok":
                return pd.DataFrame(), response.status_code, str(payload.get("errmsg", payload))
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert("America/New_York"),
                    "open": pd.to_numeric(payload["o"], errors="coerce"),
                    "high": pd.to_numeric(payload["h"], errors="coerce"),
                    "low": pd.to_numeric(payload["l"], errors="coerce"),
                    "close": pd.to_numeric(payload["c"], errors="coerce"),
                    "volume": pd.to_numeric(payload["v"], errors="coerce"),
                }
            )
            frame.insert(0, "ticker", ticker)
            return frame, response.status_code, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep((attempt + 1) * 0.75)
    return pd.DataFrame(), 0, last_error


def _download_job(job: DownloadJob, root: Path, token: str, overwrite: bool) -> dict[str, object]:
    event_key = job.event_date.strftime("%Y-%m-%d")
    safe_ticker = job.ticker.replace("/", "_").replace(".", "_")
    minute_path = root / "minute" / f"{event_key}_{safe_ticker}.parquet"
    daily_path = root / "daily" / f"{event_key}_{safe_ticker}.parquet"
    minute_start = job.event_date - pd.Timedelta(days=9)
    minute_end = job.event_date + pd.Timedelta(days=6)
    daily_start = job.announcement_date - pd.Timedelta(days=45)
    daily_end = job.event_date + pd.Timedelta(days=6)
    today_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    # The provider can return a partially finalized current session when a
    # multi-day request extends into the future.  Exact-date retrieval produced
    # the complete 390-bar regular session in the audited August 2026 case.
    if job.event_date >= today_et:
        minute_start = job.event_date
        minute_end = job.event_date
        daily_end = job.event_date

    result: dict[str, object] = {
        "ticker": job.ticker,
        "event_date": event_key,
        "minute_path": str(minute_path),
        "daily_path": str(daily_path),
    }
    if minute_path.exists() and not overwrite:
        minute = pd.read_parquet(minute_path)
        result.update(minute_status="cached", minute_rows=len(minute), minute_http="")
    else:
        minute, status, error = _request_candles(job.ticker, "1", minute_start, minute_end, token)
        if not minute.empty:
            minute_path.parent.mkdir(parents=True, exist_ok=True)
            minute.to_parquet(minute_path, index=False)
        result.update(minute_status=error or "ok", minute_rows=len(minute), minute_http=status)

    if daily_path.exists() and not overwrite:
        daily = pd.read_parquet(daily_path)
        result.update(daily_status="cached", daily_rows=len(daily), daily_http="")
    else:
        daily, status, error = _request_candles(job.ticker, "D", daily_start, daily_end, token)
        if not daily.empty:
            daily_path.parent.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(daily_path, index=False)
        result.update(daily_status=error or "ok", daily_rows=len(daily), daily_http=status)
    return result


def download_event_windows(
    events: pd.DataFrame,
    root: str | Path,
    token: str,
    workers: int = 4,
    overwrite: bool = False,
    only_date: str | None = None,
    only_tickers: set[str] | None = None,
) -> pd.DataFrame:
    root = Path(root)
    jobs_frame = (
        events.loc[events["ticker"].fillna("").ne(""), ["ticker", "effective_close_date", "announcement_date"]]
        .drop_duplicates(["ticker", "effective_close_date"])
        .copy()
    )
    jobs_frame["effective_close_date"] = pd.to_datetime(jobs_frame["effective_close_date"])
    jobs_frame["announcement_date"] = pd.to_datetime(jobs_frame["announcement_date"])
    if only_date:
        jobs_frame = jobs_frame.loc[
            jobs_frame["effective_close_date"].eq(pd.Timestamp(only_date).normalize())
        ].copy()
    if only_tickers:
        jobs_frame = jobs_frame.loc[jobs_frame["ticker"].isin(only_tickers)].copy()
    jobs = [DownloadJob(row.ticker, row.effective_close_date, row.announcement_date) for row in jobs_frame.itertuples()]

    # Benchmarks support market-adjusted diagnostics and matched-date controls.
    dates = jobs_frame[["effective_close_date", "announcement_date"]].drop_duplicates()
    jobs.extend(DownloadJob("SPY", row.effective_close_date, row.announcement_date) for row in dates.itertuples())

    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_job, job, root, token, overwrite): job for job in jobs}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                job = futures[future]
                rows.append(
                    {
                        "ticker": job.ticker,
                        "event_date": job.event_date.strftime("%Y-%m-%d"),
                        "minute_status": f"exception: {exc}",
                        "daily_status": "not_attempted",
                    }
                )
            if number % 25 == 0:
                print(f"downloaded={number}/{len(jobs)}", flush=True)
    audit = pd.DataFrame(rows).sort_values(["event_date", "ticker"])
    root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(root / "download_audit.csv", index=False)
    (root / "download_manifest.json").write_text(
        json.dumps(
            {
                "provider": "MarketData.app",
                "resolution": "1 minute plus daily",
                "regular_hours_only": True,
                "split_adjusted": True,
                "dividend_adjusted": False,
                "jobs": len(jobs),
            },
            indent=2,
        )
    )
    return audit
