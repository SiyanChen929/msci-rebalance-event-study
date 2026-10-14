#!/usr/bin/env python3
"""Create an auditable Yahoo symbol-search cross-check for MSCI issuer names."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from msci_rebalance.events import name_similarity


EXPANSIONS = {
    "BKRS": "BROKERS",
    "COMM": "COMMUNICATIONS",
    "COS": "COMPANIES",
    "ENT": "ENTERTAINMENT",
    "ENTMT": "ENTERTAINMENT",
    "FINL": "FINANCIAL",
    "HLDG": "HOLDING",
    "HLDGS": "HOLDINGS",
    "INTL": "INTERNATIONAL",
    "LABS": "LABORATORIES",
    "MGMT": "MANAGEMENT",
    "PPTYS": "PROPERTIES",
    "TECH": "TECHNOLOGY",
    "SVCS": "SERVICES",
}
US_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "PNK", "BTS"}


def expand_query(name: str) -> str:
    value = re.sub(r"\((?:US|NEW)\)", " ", str(name), flags=re.I)
    tokens = re.findall(r"[A-Za-z0-9&.-]+", value)
    tokens = [EXPANSIONS.get(token.upper(), token) for token in tokens]
    if tokens and tokens[-1].upper() in {"A", "B", "C"}:
        tokens.pop()
    return " ".join(tokens)


def query_one(name: str, retries: int = 5) -> dict[str, object]:
    query = expand_query(name)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={quote(query)}&quotesCount=8&newsCount=0"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    error = ""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 429:
                time.sleep((2**attempt) + random.random())
                continue
            response.raise_for_status()
            quotes = response.json().get("quotes", [])
            candidates = []
            for item in quotes:
                if item.get("quoteType") != "EQUITY" or item.get("exchange") not in US_EXCHANGES:
                    continue
                display = item.get("longname") or item.get("shortname") or ""
                score = max(name_similarity(name, display), name_similarity(query, display))
                candidates.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "name": display,
                        "exchange": item.get("exchange", ""),
                        "score": round(float(score), 4),
                    }
                )
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best = candidates[0] if candidates else {}
            return {
                "security_name": name,
                "yahoo_query": query,
                "yahoo_ticker": best.get("symbol", ""),
                "yahoo_match_name": best.get("name", ""),
                "yahoo_match_score": best.get("score", 0.0),
                "yahoo_candidates_json": json.dumps(candidates, ensure_ascii=True),
                "yahoo_error": "",
            }
        except Exception as exc:  # network audit path
            error = str(exc)
            time.sleep((attempt + 1) * 0.5)
    return {
        "security_name": name,
        "yahoo_query": query,
        "yahoo_ticker": "",
        "yahoo_match_name": "",
        "yahoo_match_score": 0.0,
        "yahoo_candidates_json": "[]",
        "yahoo_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    events = pd.read_csv(args.events)
    names = sorted(events["security_name"].dropna().astype(str).unique())

    existing = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame()
    done = set(existing.get("security_name", pd.Series(dtype=str)).astype(str))
    pending = [name for name in names if name not in done]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(query_one, name): name for name in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if number % 25 == 0:
                print(f"queried={number}/{len(pending)}", flush=True)

    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    result = result.drop_duplicates("security_name", keep="last").sort_values("security_name")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"names={len(result)} suggestions={result['yahoo_ticker'].fillna('').ne('').sum()} "
        f"errors={result['yahoo_error'].fillna('').ne('').sum()}"
    )


if __name__ == "__main__":
    main()
