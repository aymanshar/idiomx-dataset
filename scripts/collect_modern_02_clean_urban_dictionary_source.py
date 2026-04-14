#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_12_clean_urban_dictionary_source.py

Purpose:
- Clean the raw Urban Dictionary slang / modern idiom source dataset
- Remove noisy, weak, malformed, duplicated, or unsafe entries
- Keep only one best candidate row per idiom
- Preserve the IdiomX normalized source schema

Input:
- data/processed/idioms_source_urban_dictionary.csv

Output:
- data/processed/idioms_source_urban_dictionary_cleaned.csv
- data/processed/idioms_source_urban_dictionary_cleaned_stats.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"

DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESS_DIR / "idioms_source_urban_dictionary.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_urban_dictionary_cleaned.csv"
OUTPUT_STATS = DATA_PROCESS_DIR / "idioms_source_urban_dictionary_cleaned_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")

BAD_MEANING_PATTERNS = [
    r"^a person who\b",
    r"^someone who\b",
    r"^a girl who\b",
    r"^a guy who\b",
    r"^when\b",
    r"^the act of\b",
    r"^an act of\b",
    r"^used to describe\b",
    r"^a word used\b",
    r"^something you say\b",
    r"^something said\b",
]

STRONG_SLANG_HINTS = {
    "slang",
    "colloquial",
    "informal",
    "modern",
    "internet",
    "social media",
    "viral",
    "meme",
}

WEAK_OR_NOISY_TERMS = {
    "bro",
    "sis",
    "vibe",
    "mood",
    "energy",
    "tea",
    "shade",
    "ghost",
    "cap",
}

NSFW_HINTS = {
    "porn", "rape", "suicide", "kill yourself", "nazi",
    "racist", "cocaine", "meth", "heroin"
}


# ============================================================
# Helpers
# ============================================================

def norm(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    return MULTISPACE_RE.sub(" ", x)


def normalize_idiom_text(text: str) -> str:
    text = norm(text).lower()
    text = text.replace("’", "'")
    return text


def normalize_meaning_text(text: str) -> str:
    return normalize_idiom_text(text)


def token_count(text: str) -> int:
    return len(norm(text).split())


def has_letters(text: str) -> bool:
    return bool(LETTER_RE.search(norm(text)))


def bad_symbolic(text: str) -> bool:
    return bool(BAD_SYMBOL_RE.search(norm(text)))


def contains_nsfw(text: str) -> bool:
    text = norm(text).lower()
    return any(x in text for x in NSFW_HINTS)


def looks_like_good_idiom(idiom: str) -> bool:
    idiom = normalize_idiom_text(idiom)

    if not idiom:
        return False

    if not has_letters(idiom):
        return False

    if bad_symbolic(idiom):
        return False

    n = token_count(idiom)
    if n < 2 or n > 8:
        return False

    return True


def bad_meaning(meaning: str) -> bool:
    meaning = normalize_meaning_text(meaning)

    if not meaning:
        return True

    for pat in BAD_MEANING_PATTERNS:
        if re.search(pat, meaning):
            return True

    return False


def too_short_example(example: str) -> bool:
    return token_count(example) < 3


def build_quality_score(row: pd.Series) -> int:
    score = 0

    idiom = normalize_idiom_text(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    tags = norm(row["tags"]).lower()

    # better rows have stronger explanation and example
    if meaning:
        score += 3
    if example and not too_short_example(example):
        score += 2

    # phrase length preference
    n = token_count(idiom)
    if 2 <= n <= 5:
        score += 2
    elif 6 <= n <= 8:
        score += 1

    # slang / informal indicators
    if any(h in tags for h in STRONG_SLANG_HINTS):
        score += 2

    # prefer entries where idiom appears in example
    if idiom and example and idiom in example.lower():
        score += 1

    # penalize weak generic terms
    if idiom in WEAK_OR_NOISY_TERMS:
        score -= 3

    # penalize very short meanings
    if token_count(meaning) < 4:
        score -= 1

    return score


def compute_stats(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    return {
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "removed_rows": int(len(df_before) - len(df_after)),
        "unique_idioms_before": int(df_before["idiom"].nunique()) if len(df_before) else 0,
        "unique_idioms_after": int(df_after["idiom"].nunique()) if len(df_after) else 0,
        "rows_with_example_after": int((df_after["example"].fillna("").astype(str).str.strip() != "").sum()) if len(df_after) else 0,
        "source_distribution_after": df_after["source"].value_counts().to_dict() if len(df_after) else {},
        "confidence_distribution_after": df_after["idiom_confidence"].value_counts().to_dict() if len(df_after) else {},
    }


# ============================================================
# Main cleaning
# ============================================================

def clean_urban_dictionary_source(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    output_stats: Path = OUTPUT_STATS,
) -> tuple[pd.DataFrame, dict]:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_stats = Path(output_stats)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    df_before = df.copy()

    # ensure expected schema
    expected_cols = [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # normalize key text
    df["idiom"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_en"] = df["meaning_en"].apply(norm)
    df["example"] = df["example"].apply(norm)

    # base filters
    df = df[df["idiom"].apply(looks_like_good_idiom)]
    df = df[~df["meaning_en"].apply(bad_meaning)]
    df = df[~df["idiom"].apply(contains_nsfw)]
    df = df[~df["meaning_en"].apply(contains_nsfw)]
    df = df[~df["example"].apply(contains_nsfw)]

    # remove exact duplicate idiom + meaning
    df["dedup_key"] = (
        df["idiom"].apply(normalize_idiom_text)
        + " || " +
        df["meaning_en"].apply(normalize_meaning_text)
    )
    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    # keep best row per idiom
    df["quality_score"] = df.apply(build_quality_score, axis=1)
    df = df.sort_values(
        by=["idiom", "quality_score", "meaning_en", "example"],
        ascending=[True, False, True, True]
    )
    df = df.drop_duplicates(subset=["idiom"], keep="first")

    # preserve schema order
    df = df[expected_cols].sort_values("idiom").reset_index(drop=True)

    stats = compute_stats(df_before=df_before, df_after=df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved cleaned dataset:", output_file)
    print("Saved stats:", output_stats)
    print("Rows:", len(df))
    print("Unique idioms:", df["idiom"].nunique() if len(df) else 0)

    if len(df):
        print("\nPreview:")
        print(df.head(10))

    return df, stats


def main():
    clean_urban_dictionary_source()


if __name__ == "__main__":
    main()