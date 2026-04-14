#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_11_extract_urban_dictionary_slang.py

Purpose:
- Collect modern slang / contemporary idiom candidates from public Urban Dictionary term pages
- Save them in the same normalized source schema you already use in IdiomX
- Keep this as a separate source dataset first, then merge later if useful

Output schema:
- idiom
- meaning_en
- example
- source
- source_type
- pos
- tags
- idiom_confidence
- source_url

Recommended usage:
    python collect_11_extract_urban_dictionary_slang.py

Optional:
    python collect_11_extract_urban_dictionary_slang.py --max-terms 300 --delay 1.5
    python collect_11_extract_urban_dictionary_slang.py --seed-file seeds_modern_slang.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"
DATA_LOG_DIR = DATA_DIR / "logs"

DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
DATA_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_CSV = DATA_PROCESS_DIR / "idioms_source_urban_dictionary.csv"
DEFAULT_OUTPUT_JSON = DATA_PROCESS_DIR / "idioms_source_urban_dictionary_stats.json"
DEFAULT_SEED_FILE = DATA_PROCESS_DIR / "urban_dictionary_seed_terms.txt"


# ============================================================
# Config
# ============================================================

BASE_URL = "https://www.urbandictionary.com/define.php?term={term}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_DELAY = 1.5
DEFAULT_TIMEOUT = 25
DEFAULT_MAX_TERMS = 200
DEFAULT_MAX_RELATED_PER_PAGE = 20

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")
TOKEN_RE = re.compile(r"\S+")

# light safety / noise filtering
NSFW_HINTS = {
    "porn", "rape", "murder", "suicide", "kill yourself", "nazi",
    "slur", "racist", "cocaine", "meth", "heroin", "weapon", "gun"
}

STOP_LINES = {
    "share definition",
    "listen to pronunciation",
    "flag",
    "more random definitions",
    "sign in to vote",
    "copy link",
}


DEFAULT_SEEDS = [
    "spill the tea",
    "throw shade",
    "ghost someone",
    "catch feelings",
    "main character energy",
    "hit different",
    "low key",
    "high key",
    "no cap",
    "living rent free",
    "hard launch",
    "soft launch",
    "red flag",
    "green flag",
    "out of pocket",
    "under the radar",
    "down bad",
    "touch grass",
    "quiet quitting",
    "doomscrolling",
    "plot twist",
    "clap back",
    "drag someone",
    "read someone",
    "it’s giving",
    "in your flop era",
    "keep it real",
    "ride or die",
    "spill the beans",
    "throw in the towel",
]


# ============================================================
# Helpers
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return MULTISPACE_RE.sub(" ", text)


def normalize_term(term: str) -> str:
    term = normalize_text(term).lower()
    term = term.replace("’", "'")
    return term


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(normalize_text(text)))


def looks_like_phrase(term: str) -> bool:
    term = normalize_term(term)

    if not term:
        return False

    if not LETTER_RE.search(term):
        return False

    if BAD_SYMBOL_RE.search(term):
        return False

    n = token_count(term)
    if n < 2 or n > 8:
        return False

    return True


def contains_nsfw_hint(text: str) -> bool:
    text = normalize_text(text).lower()
    return any(h in text for h in NSFW_HINTS)


def load_seed_terms(seed_file: Path | None) -> list[str]:
    if seed_file is None or not seed_file.exists():
        return sorted({normalize_term(x) for x in DEFAULT_SEEDS if looks_like_phrase(x)})

    terms = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            term = normalize_term(line)
            if looks_like_phrase(term):
                terms.append(term)

    return sorted(set(terms))


def fetch_html(term: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    url = BASE_URL.format(term=quote_plus(term))
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_related_terms(soup: BeautifulSoup, current_term: str, max_related: int) -> list[str]:
    current_term = normalize_term(current_term)
    related = []

    for a in soup.select('a[href*="define.php?term="]'):
        href = a.get("href", "")
        if not href:
            continue

        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "term" not in qs:
            continue

        candidate = normalize_term(unquote(qs["term"][0]))
        if candidate == current_term:
            continue
        if not looks_like_phrase(candidate):
            continue

        related.append(candidate)

    deduped = []
    seen = set()
    for x in related:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return deduped[:max_related]


def extract_entries_by_css(soup: BeautifulSoup, requested_term: str, source_url: str) -> list[dict]:
    """
    First attempt:
    try common UD-style HTML classes if available.
    """
    rows = []

    term_nodes = soup.select(".word, a.word, h1, h2")
    meaning_nodes = soup.select(".meaning, div.meaning")
    example_nodes = soup.select(".example, div.example")

    if not term_nodes or not meaning_nodes:
        return rows

    n = min(len(term_nodes), len(meaning_nodes))
    for i in range(n):
        term = normalize_term(term_nodes[i].get_text(" ", strip=True))
        meaning = normalize_text(meaning_nodes[i].get_text(" ", strip=True))
        example = ""
        if i < len(example_nodes):
            example = normalize_text(example_nodes[i].get_text(" ", strip=True))

        if term != normalize_term(requested_term):
            continue
        if not looks_like_phrase(term):
            continue
        if not meaning:
            continue
        if contains_nsfw_hint(meaning) or contains_nsfw_hint(example):
            continue

        rows.append({
            "idiom": term,
            "meaning_en": meaning,
            "example": example,
            "source": "urban_dictionary",
            "source_type": "crowdsourced_dictionary",
            "pos": "",
            "tags": "slang,colloquial,informal,modern",
            "idiom_confidence": "low",
            "source_url": source_url,
        })

    return rows


def extract_entries_by_visible_text(soup: BeautifulSoup, requested_term: str, source_url: str) -> list[dict]:
    """
    Fallback parser:
    parse visible text lines because page structure may change.
    """
    requested_term = normalize_term(requested_term)

    text = soup.get_text("\n", strip=True)
    lines = [normalize_text(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    rows = []
    i = 0
    while i < len(lines):
        line = normalize_term(lines[i])

        if line == requested_term:
            j = i + 1

            while j < len(lines) and normalize_term(lines[j]) in STOP_LINES:
                j += 1

            meaning = ""
            example = ""

            collected = []
            while j < len(lines):
                cur = normalize_text(lines[j])
                cur_norm = normalize_term(cur)

                if cur_norm.startswith("by "):
                    break
                if cur_norm in STOP_LINES:
                    j += 1
                    continue
                if cur_norm == requested_term and collected:
                    break
                if cur.startswith("Get the ") or cur.startswith("© "):
                    break

                collected.append(cur)
                j += 1

            if collected:
                meaning = collected[0]
            if len(collected) > 1:
                example = collected[1]

            if looks_like_phrase(requested_term) and meaning:
                if not contains_nsfw_hint(meaning) and not contains_nsfw_hint(example):
                    rows.append({
                        "idiom": requested_term,
                        "meaning_en": meaning,
                        "example": example,
                        "source": "urban_dictionary",
                        "source_type": "crowdsourced_dictionary",
                        "pos": "",
                        "tags": "slang,colloquial,informal,modern",
                        "idiom_confidence": "low",
                        "source_url": source_url,
                    })

            i = j
            continue

        i += 1

    return rows


def parse_term_page(html: str, requested_term: str) -> tuple[list[dict], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    source_url = BASE_URL.format(term=quote_plus(requested_term))

    rows = extract_entries_by_css(soup, requested_term, source_url)
    if not rows:
        rows = extract_entries_by_visible_text(soup, requested_term, source_url)

    related_terms = extract_related_terms(
        soup=soup,
        current_term=requested_term,
        max_related=DEFAULT_MAX_RELATED_PER_PAGE,
    )

    return rows, related_terms


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["dedup_key"] = (
        df["idiom"].str.lower().str.strip()
        + " || " +
        df["meaning_en"].str.lower().str.strip()
    )

    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"]).reset_index(drop=True)
    return df


def build_stats(df: pd.DataFrame, processed_terms: int, discovered_terms: int, failed_terms: list[str]) -> dict:
    if df.empty:
        return {
            "processed_terms": processed_terms,
            "discovered_terms": discovered_terms,
            "rows": 0,
            "unique_idioms": 0,
            "failed_terms": failed_terms,
        }

    return {
        "processed_terms": processed_terms,
        "discovered_terms": discovered_terms,
        "rows": int(len(df)),
        "unique_idioms": int(df["idiom"].nunique()),
        "rows_with_example": int((df["example"].fillna("").astype(str).str.strip() != "").sum()),
        "source_distribution": df["source"].value_counts().to_dict(),
        "confidence_distribution": df["idiom_confidence"].value_counts().to_dict(),
        "failed_terms": failed_terms,
    }


# ============================================================
# Main
# ============================================================

def collect_urban_dictionary_idioms(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    seed_file: Path | None = None,
    max_terms: int = DEFAULT_MAX_TERMS,
    delay: float = DEFAULT_DELAY,
) -> tuple[pd.DataFrame, dict]:
    output_csv = Path(output_csv)
    output_json = Path(output_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    seed_terms = load_seed_terms(seed_file)
    queue = deque(seed_terms)
    seen_terms = set()
    all_rows = []
    failed_terms = []

    pbar = tqdm(total=max_terms, desc="Scraping modern idiom candidates")

    while queue and len(seen_terms) < max_terms:
        term = normalize_term(queue.popleft())

        if term in seen_terms:
            continue
        if not looks_like_phrase(term):
            continue

        seen_terms.add(term)
        pbar.update(1)

        try:
            html = fetch_html(term)
            rows, related_terms = parse_term_page(html, term)

            if rows:
                all_rows.extend(rows)

            for rel in related_terms:
                if len(seen_terms) + len(queue) >= max_terms:
                    break
                if rel not in seen_terms and rel not in queue and looks_like_phrase(rel):
                    queue.append(rel)

            time.sleep(delay)

        except Exception as e:
            failed_terms.append(f"{term} :: {type(e).__name__}: {e}")
            time.sleep(delay)

    pbar.close()

    df = pd.DataFrame(all_rows)
    df = deduplicate_rows(df)

    # Keep only entries with at least a non-empty meaning
    if not df.empty:
        df = df[df["meaning_en"].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)

    stats = build_stats(
        df=df,
        processed_terms=len(seen_terms),
        discovered_terms=len(queue) + len(seen_terms),
        failed_terms=failed_terms,
    )

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nSaved source dataset:", output_csv)
    print("Saved stats:", output_json)
    print("Rows:", len(df))
    print("Unique idioms:", df["idiom"].nunique() if not df.empty else 0)

    if not df.empty:
        print("\nPreview:")
        print(df.head(10))

    if failed_terms:
        print("\nFailed terms:", len(failed_terms))
        print("Example failure:", failed_terms[0])

    return df, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect modern slang / contemporary idiom candidates from Urban Dictionary term pages."
    )
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--seed-file", type=str, default=str(DEFAULT_SEED_FILE))
    parser.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    return parser.parse_args()


def main():
    args = parse_args()

    seed_file = Path(args.seed_file) if args.seed_file else None
    if seed_file is not None and not seed_file.exists():
        print(f"Seed file not found, using built-in seeds instead: {seed_file}")
        seed_file = None

    collect_urban_dictionary_idioms(
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        seed_file=seed_file,
        max_terms=args.max_terms,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()