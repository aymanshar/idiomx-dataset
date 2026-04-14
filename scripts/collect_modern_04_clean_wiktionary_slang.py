#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_14_clean_wiktionary_slang.py

Purpose:
- Clean the raw Wiktionary slang / modern idiom source dataset
- Remove noisy, weak, malformed, duplicated, or low-value entries
- Keep only one best candidate row per idiom
- Preserve the IdiomX normalized source schema

Input:
- data/processed/idioms_source_wiktionary_slang.csv

Output:
- data/processed/idioms_source_wiktionary_slang_cleaned.csv
- data/processed/idioms_source_wiktionary_slang_cleaned_stats.json
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

INPUT_FILE = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang_cleaned.csv"
OUTPUT_STATS = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang_cleaned_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")

BAD_MEANING_PATTERNS = [
    r"^a person who\b",
    r"^someone who\b",
    r"^a woman who\b",
    r"^a man who\b",
    r"^a guy who\b",
    r"^a girl who\b",
    r"^a boy who\b",
    r"^a word used\b",
    r"^a term used\b",
    r"^something you say\b",
    r"^something said\b",
    r"^used to describe\b",
    r"^used to refer to\b",
    r"^used when\b",
    r"^when\b",
    r"^the act of\b",
    r"^an act of\b",
    r"^the state of\b",
    r"^the process of\b",
    r"^alternative spelling of\b",
    r"^alternative form of\b",
    r"^misspelling of\b",
    r"^nonstandard spelling of\b",
    r"^abbreviation of\b",
    r"^initialism of\b",
    r"^acronym of\b",
    r"^clipping of\b",
    r"^ellipsis of\b",
    r"^plural of\b",
    r"^past tense of\b",
    r"^past participle of\b",
    r"^present participle of\b",
    r"^comparative of\b",
    r"^superlative of\b",
    r"^used other than figuratively or idiomatically\b",
    r"^used other than idiomatically\b",
    r"^used other than figuratively\b",
]

STRONG_TAG_HINTS = {
    "slang",
    "informal",
    "colloquial",
    "internet",
    "humorous",
    "sarcastic",
    "figurative",
    "figuratively",
    "idiomatic",
    "idiom",
    "meme",
}

WEAK_OR_NOISY_TERMS = {
    "all good",
    "for real",
    "my bad",
    "no way",
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "at all",
    "right now",
}

VERY_GENERIC_CONTENT_WORDS = {
    "thing", "stuff", "person", "people", "someone", "somebody",
    "something", "anything", "everything", "state", "process", "act"
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


def tags_to_set(tags: str) -> set[str]:
    items = []
    for x in norm(tags).lower().split(","):
        x = x.strip()
        if x:
            items.append(x)
    return set(items)


def weak_generic_phrase(idiom: str, meaning: str, tags: str) -> bool:
    """
    Remove very generic conversational fragments unless there is
    strong idiomatic/slang evidence.
    """
    idiom_norm = normalize_idiom_text(idiom)
    meaning_norm = normalize_meaning_text(meaning)
    tag_set = tags_to_set(tags)

    if idiom_norm in WEAK_OR_NOISY_TERMS:
        if not (tag_set & STRONG_TAG_HINTS):
            return True
        if not any(x in meaning_norm for x in ["figurative", "idiomatic", "slang", "informal", "colloquial"]):
            return True

    return False


def too_literal_or_definition_like(idiom: str, meaning: str, tags: str, pos: str) -> bool:
    """
    Conservative literal/noise filtering:
    reject rows with weak lexical-definition style meanings
    and no strong slang/idiomatic evidence.
    """
    idiom_norm = normalize_idiom_text(idiom)
    meaning_norm = normalize_meaning_text(meaning)
    tag_set = tags_to_set(tags)
    pos_norm = norm(pos).lower()

    strong_signal = False

    if tag_set & STRONG_TAG_HINTS:
        strong_signal = True

    if any(x in meaning_norm for x in ["figurative", "idiomatic", "slang", "informal", "colloquial", "sarcastic", "humorous"]):
        strong_signal = True

    if pos_norm in {"phrase", "idiom", "proverb"}:
        strong_signal = True

    # If the meaning is too generic and there is no strong evidence, drop it
    if not strong_signal:
        if any(x in meaning_norm for x in VERY_GENERIC_CONTENT_WORDS):
            return True

        # Very short generic meanings are often weak/noisy
        if token_count(meaning_norm) <= 3:
            return True

        # Plain compositional-looking phrase with no signal
        if token_count(idiom_norm) <= 3 and token_count(meaning_norm) <= 6:
            return True

    return False


def build_quality_score(row: pd.Series) -> int:
    score = 0

    idiom = normalize_idiom_text(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    tags = norm(row["tags"]).lower()
    pos = norm(row["pos"]).lower()

    # Meaning and example quality
    if meaning:
        score += 3
    if example and not too_short_example(example):
        score += 2

    # Phrase length preference
    n = token_count(idiom)
    if 2 <= n <= 5:
        score += 2
    elif 6 <= n <= 8:
        score += 1

    # Better POS
    if pos in {"phrase", "idiom", "proverb"}:
        score += 3
    elif pos in {"verb", "adjective", "adverb", "noun", "interjection"}:
        score += 1

    # Strong slang / idiom evidence
    if any(x in tags for x in ["slang", "informal", "colloquial", "internet"]):
        score += 3

    if any(x in tags for x in ["idiomatic", "figurative", "idiom", "sarcastic", "humorous", "meme"]):
        score += 2

    if any(x in meaning.lower() for x in ["figurative", "idiomatic", "slang", "informal", "colloquial"]):
        score += 2

    # Prefer cleaner non-fragment phrases slightly
    if not idiom.startswith(("i ", "you ", "we ", "they ", "he ", "she ")):
        score += 1

    return score


# ============================================================
# Main cleaning
# ============================================================

def clean_wiktionary_slang_source(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    output_stats: Path = OUTPUT_STATS,
):
    """
    Clean the extracted Wiktionary slang source dataset.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_stats = Path(output_stats)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    rows_before = len(df)

    expected_cols = [
        "idiom",
        "meaning_en",
        "example",
        "source",
        "source_type",
        "pos",
        "tags",
        "idiom_confidence",
        "source_url",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Normalize main fields
    df["idiom"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_en"] = df["meaning_en"].apply(norm)
    df["example"] = df["example"].apply(norm)
    df["pos"] = df["pos"].apply(norm)
    df["tags"] = df["tags"].apply(norm)
    df["source"] = df["source"].apply(norm)
    df["source_type"] = df["source_type"].apply(norm)
    df["idiom_confidence"] = df["idiom_confidence"].apply(norm)
    df["source_url"] = df["source_url"].apply(norm)

    # Row-level filtering
    keep_mask = []
    for _, row in df.iterrows():
        idiom = row["idiom"]
        meaning = row["meaning_en"]
        example = row["example"]
        tags = row["tags"]
        pos = row["pos"]

        keep = True

        if not looks_like_good_idiom(idiom):
            keep = False
        elif contains_nsfw(idiom) or contains_nsfw(meaning) or contains_nsfw(example):
            keep = False
        elif bad_meaning(meaning):
            keep = False
        elif weak_generic_phrase(idiom, meaning, tags):
            keep = False
        elif too_literal_or_definition_like(idiom, meaning, tags, pos):
            keep = False

        keep_mask.append(keep)

    df = df[keep_mask].copy()
    rows_after_filtering = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["idiom", "meaning_en", "example"]).copy()
    rows_after_exact_dedup = len(df)

    # Score rows and keep the strongest row per idiom
    df["quality_score"] = df.apply(build_quality_score, axis=1)
    df = df.sort_values(
        by=["idiom", "quality_score", "meaning_en", "example"],
        ascending=[True, False, True, True]
    ).copy()

    df = df.drop_duplicates(subset=["idiom"], keep="first").copy()
    rows_final = len(df)

    # Final sort and column order
    df = df.sort_values(by=["idiom", "meaning_en"]).copy()

    out_cols = [
        "idiom",
        "meaning_en",
        "example",
        "source",
        "source_type",
        "pos",
        "tags",
        "idiom_confidence",
        "source_url",
    ]
    df = df[out_cols].reset_index(drop=True)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    stats = {
        "rows_before": int(rows_before),
        "rows_after_filtering": int(rows_after_filtering),
        "rows_after_exact_dedup": int(rows_after_exact_dedup),
        "rows_final": int(rows_final),
        "unique_idioms_final": int(df["idiom"].nunique()),
        "rows_with_example_final": int((df["example"].str.strip() != "").sum()),
        "avg_idiom_token_length": round(df["idiom"].apply(token_count).mean(), 4) if len(df) else 0.0,
        "source_distribution": df["source"].value_counts().to_dict(),
        "confidence_distribution": df["idiom_confidence"].value_counts().to_dict(),
        "pos_distribution_top20": df["pos"].value_counts().head(20).to_dict(),
    }

    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved cleaned dataset:", output_file)
    print("Saved stats:", output_stats)
    print("Rows before:", rows_before)
    print("Rows final:", rows_final)
    print("Rows with example:", stats["rows_with_example_final"])

    return df


def main():
    clean_wiktionary_slang_source(INPUT_FILE, OUTPUT_FILE, OUTPUT_STATS)


if __name__ == "__main__":
    main()