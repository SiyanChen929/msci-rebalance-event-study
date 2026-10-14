"""Parse public MSCI review notices into a point-in-time US event table.

The public notices identify constituent additions/deletions but do not provide
tickers or final index weights.  Ticker resolution is therefore kept as a
separate, audited step and every match records its provenance and confidence.
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


DATE_RE = re.compile(r"(?:close of|effective)\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", re.I)
ANNOUNCEMENT_RE = re.compile(r"(?:Geneva|London|New York),\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", re.I)
SECTION_RE = re.compile(r"^MSCI\s+(.+?)\s+INDEX(?:ES)?\s*$")
PAGE_RE = re.compile(r"^Page\s+\d+")


@dataclass(frozen=True)
class ParsedReview:
    announcement_date: pd.Timestamp
    effective_close_date: pd.Timestamp
    segment: str
    source_file: str
    source_url: str
    additions: tuple[str, ...]
    deletions: tuple[str, ...]


def pdf_to_layout_text(path: str | Path) -> str:
    """Extract layout-preserving text using Poppler's ``pdftotext``."""

    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_us_review_pdf(path: str | Path, source_url: str = "") -> ParsedReview:
    """Parse the MSCI USA section from a Global Standard/Small Cap notice."""

    path = Path(path)
    text = pdf_to_layout_text(path)
    ann_match = ANNOUNCEMENT_RE.search(text)
    eff_match = DATE_RE.search(text)
    if not ann_match or not eff_match:
        raise ValueError(f"Could not parse review dates from {path}")

    upper_head = text[:1500].upper()
    if "GLOBAL SMALL CAP" in upper_head:
        segment = "small_cap"
    elif "GLOBAL STANDARD" in upper_head:
        segment = "standard"
    else:
        raise ValueError(f"Could not identify index segment in {path}")

    lines = text.splitlines()
    usa_summary = re.search(r"^\s*USA\s+(\d+)\s+(\d+)\s*$", text, flags=re.M)
    expected_additions = int(usa_summary.group(1)) if usa_summary else 0
    expected_deletions = int(usa_summary.group(2)) if usa_summary else 0
    starts = [i for i, line in enumerate(lines) if line.strip() == "MSCI USA INDEX"]
    # A notice omits the country section when there are no US changes.  This is
    # a real zero-event review, not a parser failure (for example February 2020).
    if not starts:
        return ParsedReview(
            announcement_date=pd.Timestamp(ann_match.group(1)).normalize(),
            effective_close_date=pd.Timestamp(eff_match.group(1)).normalize(),
            segment=segment,
            source_file=path.name,
            source_url=source_url,
            additions=(),
            deletions=(),
        )
    if len(starts) != 1:
        raise ValueError(f"Expected at most one MSCI USA section in {path}; found {len(starts)}")
    start = starts[0]

    header = None
    for i in range(start + 1, min(start + 8, len(lines))):
        if "Additions" in lines[i] and "Deletions" in lines[i]:
            header = i
            break
    if header is None:
        raise ValueError(f"Could not locate Additions/Deletions header in {path}")

    base_indent = len(lines[header]) - len(lines[header].lstrip())
    additions: list[str] = []
    deletions: list[str] = []
    for raw in lines[header + 1 :]:
        raw = raw.replace("\x0c", "")
        if base_indent and raw.startswith(" " * base_indent):
            raw = raw[base_indent:]
        stripped = raw.strip()
        if not stripped:
            continue
        if SECTION_RE.match(stripped) or stripped in {
            "ASIA PACIFIC",
            "AMERICAS",
            "EUROPE, MIDDLE EAST AND AFRICA",
            "Notice and Disclaimer",
        }:
            break
        if PAGE_RE.match(stripped) or "© MSCI" in raw:
            continue
        leading = len(raw) - len(raw.lstrip())
        if leading >= 2:
            left, right = "", raw.strip()
        else:
            gaps = list(re.finditer(r" {2,}", raw.rstrip()))
            if gaps:
                # The widest alignment gap is the two-column separator.
                gap = max(gaps, key=lambda match: match.end() - match.start())
                left, right = raw[: gap.start()].strip(), raw[gap.end() :].strip()
            else:
                left, right = raw.strip(), ""
        if left and left.upper() != "NONE":
            additions.append(_clean_security_name(left))
        if right and right.upper() != "NONE":
            deletions.append(_clean_security_name(right))

    if (len(additions), len(deletions)) != (expected_additions, expected_deletions):
        raise ValueError(
            f"USA row-count mismatch in {path}: parsed "
            f"{len(additions)}/{len(deletions)}, expected "
            f"{expected_additions}/{expected_deletions}"
        )

    return ParsedReview(
        announcement_date=pd.Timestamp(ann_match.group(1)).normalize(),
        effective_close_date=pd.Timestamp(eff_match.group(1)).normalize(),
        segment=segment,
        source_file=path.name,
        source_url=source_url,
        additions=tuple(additions),
        deletions=tuple(deletions),
    )


def _clean_security_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x0c", " ")).strip()


def review_to_frame(review: ParsedReview) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for action, names in (("add", review.additions), ("delete", review.deletions)):
        for name in names:
            rows.append(
                {
                    "announcement_date": review.announcement_date,
                    "effective_close_date": review.effective_close_date,
                    "segment": review.segment,
                    "action": action,
                    "security_name": name,
                    "source_file": review.source_file,
                    "source_url": review.source_url,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "announcement_date",
            "effective_close_date",
            "segment",
            "action",
            "security_name",
            "source_file",
            "source_url",
        ],
    )


def build_standard_event_frame(reviews: Iterable[ParsedReview]) -> pd.DataFrame:
    """Return Standard changes and classify Standard/Small-Cap migrations."""

    all_events = pd.concat([review_to_frame(r) for r in reviews], ignore_index=True)
    standard = all_events.loc[all_events["segment"].eq("standard")].copy()
    small = all_events.loc[all_events["segment"].eq("small_cap")].copy()
    small["name_key"] = small["security_name"].map(normalize_company_name)
    standard["name_key"] = standard["security_name"].map(normalize_company_name)

    migration: list[str] = []
    migration_match: list[str] = []
    for row in standard.itertuples(index=False):
        opposite = "delete" if row.action == "add" else "add"
        candidates = small.loc[
            small["effective_close_date"].eq(row.effective_close_date)
            & small["action"].eq(opposite)
        ]
        matched_name, score = best_name_match(row.security_name, candidates["security_name"])
        if score >= 0.92:
            migration.append("up_from_small" if row.action == "add" else "down_to_small")
            migration_match.append(matched_name)
        else:
            migration.append("new_to_standard" if row.action == "add" else "exit_standard")
            migration_match.append("")
    standard["change_type"] = migration
    standard["small_cap_match"] = migration_match
    return standard.sort_values(["effective_close_date", "action", "security_name"]).reset_index(drop=True)


CORPORATE_STOPWORDS = {
    "A",
    "B",
    "CLASS",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "GROUP",
    "GRP",
    "HOLDING",
    "HOLDINGS",
    "HLDG",
    "HLDGS",
    "INC",
    "INCORPORATED",
    "INTL",
    "LTD",
    "PLC",
    "THE",
}


def normalize_company_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.upper().replace("&", " AND ")
    tokens = re.findall(r"[A-Z0-9]+", value)
    tokens = [token for token in tokens if token not in CORPORATE_STOPWORDS]
    return " ".join(tokens)


def name_similarity(left: str, right: str) -> float:
    a = normalize_company_name(left)
    b = normalize_company_name(right)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    token = len(ta & tb) / max(len(ta | tb), 1)
    containment = min(len(a), len(b)) / max(len(a), len(b)) if (a in b or b in a) else 0.0
    return max(seq, token, containment)


def best_name_match(name: str, candidates: Iterable[str]) -> tuple[str, float]:
    best_name = ""
    best_score = 0.0
    for candidate in candidates:
        score = name_similarity(name, str(candidate))
        if score > best_score:
            best_name, best_score = str(candidate), score
    return best_name, best_score


def load_sec_ticker_map(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    frame = pd.DataFrame(payload.values())
    frame = frame.rename(columns={"title": "sec_title"})
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame[["ticker", "sec_title", "cik_str"]]


def resolve_tickers(
    events: pd.DataFrame,
    sec_map: pd.DataFrame,
    overrides: dict[str, str] | None = None,
    minimum_score: float = 0.70,
) -> pd.DataFrame:
    """Fuzzy-resolve tickers with explicit overrides and an audit trail."""

    overrides = {normalize_company_name(k): v.upper() for k, v in (overrides or {}).items()}
    titles = sec_map["sec_title"].astype(str).tolist()
    # The SEC file can contain multiple listed instruments for one registrant
    # (common stock, preferreds, warrants).  The shortest plain ticker is the
    # safest first-pass common-stock candidate; all mappings remain auditable.
    title_lookup: dict[str, str] = {}
    for title, group in sec_map.groupby("sec_title", sort=False):
        tickers = group["ticker"].astype(str).tolist()
        title_lookup[str(title)] = sorted(tickers, key=lambda x: ("-" in x, len(x), x))[0]
    normalized_titles = [normalize_company_name(title) for title in titles]
    token_index: dict[str, set[int]] = {}
    for idx, normalized in enumerate(normalized_titles):
        for token in normalized.split():
            if len(token) >= 3:
                token_index.setdefault(token, set()).add(idx)
    rows: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        name = str(event["security_name"])
        key = normalize_company_name(name)
        if key in overrides:
            ticker = overrides[key]
            matched_title = "manual override"
            score = 1.0
            method = "override"
        else:
            candidate_indices: set[int] = set()
            for token in key.split():
                candidate_indices.update(token_index.get(token, ()))
            # Exact tokens provide a fast and interpretable shortlist.  Fall
            # back to the full SEC file only for heavily abbreviated names.
            candidate_titles = [titles[i] for i in candidate_indices] if candidate_indices else titles
            matched_title, score = best_name_match(name, candidate_titles)
            ticker = title_lookup.get(matched_title, "") if score >= minimum_score else ""
            method = "sec_fuzzy" if ticker else "unresolved"
        event.update(
            {
                "ticker": ticker,
                "ticker_match_method": method,
                "ticker_match_title": matched_title,
                "ticker_match_score": round(float(score), 4),
            }
        )
        rows.append(event)
    return pd.DataFrame(rows)
