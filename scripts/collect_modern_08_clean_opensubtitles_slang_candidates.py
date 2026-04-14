#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_18_clean_opensubtitles_slang_candidates.py

Purpose:
- Clean the raw OpenSubtitles candidate source dataset
- Remove weak, overly generic, literal, malformed, or low-value repeated phrases
- Keep one best candidate row per idiom
- Preserve the IdiomX normalized source schema

Input:
- data/processed/idioms_source_opensubtitles_candidates.csv

Output:
- data/processed/idioms_source_opensubtitles_candidates_cleaned.csv
- data/processed/idioms_source_opensubtitles_candidates_cleaned_stats.json
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

INPUT_FILE = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates_cleaned.csv"
OUTPUT_STATS = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates_cleaned_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

PRONOUNS = {"i", "you", "we", "he", "she", "they", "it"}
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "from", "with", "by", "as", "is", "am", "are", "was", "were", "be",
    "been", "being", "it", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "so", "just", "very", "too", "not", "no", "yes", "do", "did",
    "does", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "all", "any", "some", "one", "two", "three",
    "then", "than", "here", "there", "now", "well", "also", "only", "even", "still",
}

CONTENT_CUE_WORDS = {
    "get", "give", "take", "keep", "cut", "lose", "make", "play", "come", "go",
    "back", "over", "off", "out", "down", "up", "through", "around", "away",
    "mind", "break", "slack", "point", "deal", "catch", "mean", "mess", "hang",
    "blow", "fall", "pull", "push", "drag", "drop", "sell", "buy", "call", "turn",
    "hold", "bring", "throw", "pick", "run", "work", "stick", "hit", "cool", "save",
    "snap", "shut", "spill", "burn", "move", "read", "ride", "flip", "rip",
}

ENDING_PARTICLES = {"up", "down", "off", "out", "over", "away", "around", "back", "through"}

WEAK_EXACT = {
    "how are you",
    "where are you",
    "what are you",
    "what do you",
    "do you know",
    "do you want",
    "can you help",
    "can you see",
    "i need you",
    "i want you",
    "i love you",
    "i miss you",
    "let me go",
    "come with me",
    "look at me",
    "talk to me",
    "wait for me",
    "go with him",
    "what is that",
    "what is this",
    "who is that",
    "who is this",
    "there you are",
    "here you are",
    "here we are",
    "at the end",
    "in the end",
    "at this point",
    "for the first",
    "for the last",
    "one of them",
    "one of us",
}

WEAK_STARTS = (
    "i want ",
    "i need ",
    "i have ",
    "i had ",
    "i was ",
    "i am ",
    "you want ",
    "you need ",
    "you have ",
    "you had ",
    "you are ",
    "you were ",
    "we want ",
    "we need ",
    "we have ",
    "we had ",
    "he was ",
    "she was ",
    "they were ",
    "it was ",
    "there is ",
    "there are ",
    "this is ",
    "that is ",
    "what is ",
    "who is ",
    "why is ",
    "how is ",
    "do you ",
    "can you ",
    "will you ",
    "would you ",
    "did you ",
    "are you ",
    "have you ",
)

LITERAL_PREFIXES = (
    "go to ",
    "come to ",
    "walk to ",
    "drive to ",
    "open the ",
    "close the ",
    "pick up ",
    "put it ",
    "take the ",
    "sit on ",
    "look at ",
    "talk to ",
    "wait for ",
    "work on ",
    "call the ",
    "bring the ",
    "leave the ",
    "hold the ",
)

WEAK_CONTENT_WORDS = {
    "thing", "stuff", "person", "people", "someone", "somebody", "something",
    "anything", "everything", "nothing", "time", "place", "way", "kind", "sort",
    "part", "side", "moment", "minute", "day", "night", "man", "woman",
}

STRONG_IDIOM_PATTERNS = [
    r"\bget\b.*\bover\b",
    r"\bgive\b.*\bup\b",
    r"\bmake\b.*\bup\b",
    r"\bcut\b.*\bslack\b",
    r"\bblow\b.*\boff\b",
    r"\bhang\b.*\bout\b",
    r"\bback\b.*\boff\b",
    r"\bshut\b.*\bup\b",
    r"\bkeep\b.*\bup\b",
    r"\blose\b.*\bmind\b",
    r"\bcatch\b.*\bup\b",
    r"\bfall\b.*\bapart\b",
    r"\bpull\b.*\boff\b",
    r"\bwork\b.*\bout\b",
    r"\bcome\b.*\bthrough\b",
    r"\bturn\b.*\bdown\b",
    r"\bturn\b.*\bup\b",
    r"\bstick\b.*\bout\b",
    r"\bcall\b.*\boff\b",
]

WEAK_PATTERN_RE = re.compile(
    r"^(?:"
    r"(?:there|this|that|what|who|why|how)\s+\w+\s+\w+\b|"
    r"(?:do|did|can|will|would|are|have)\s+you\s+\w+\b"
    r")",
    re.IGNORECASE,
)


# ============================================================
# Helpers
# ============================================================

def norm(x) -> str:
    if pd.isna(x):
        return ""
    return MULTISPACE_RE.sub(" ", str(x).strip())


def normalize_idiom_text(text: str) -> str:
    text = norm(text).lower()
    text = text.replace("’", "'")
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_idiom_text(text))


def token_count(text: str) -> int:
    return len(tokenize(text))


def has_letters(text: str) -> bool:
    return bool(LETTER_RE.search(norm(text)))


def bad_symbolic(text: str) -> bool:
    return bool(BAD_SYMBOL_RE.search(norm(text)))


def content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def contains_strong_idiom_pattern(text: str) -> bool:
    text = normalize_idiom_text(text)
    return any(re.search(pat, text) for pat in STRONG_IDIOM_PATTERNS)


def example_quality_score(example: str, idiom: str) -> int:
    score = 0
    ex = norm(example)
    idiom = normalize_idiom_text(idiom)

    if ex:
        score += 2

    n = len(tokenize(ex))
    if 5 <= n <= 18:
        score += 3
    elif 4 <= n <= 24:
        score += 2
    else:
        score += 1

    if idiom and idiom in ex.lower():
        score += 2

    if ex.isupper():
        score -= 2

    bad_char_count = sum(1 for c in ex if c in "<>[]{}=_*\\|")
    score -= bad_char_count

    return score


def looks_like_good_idiom(idiom: str) -> bool:
    idiom = normalize_idiom_text(idiom)

    if not idiom:
        return False

    if not has_letters(idiom):
        return False

    if bad_symbolic(idiom):
        return False

    n = token_count(idiom)
    if n < 3 or n > 5:
        return False

    toks = idiom.split()

    if toks[0] in PRONOUNS:
        return False

    if idiom in WEAK_EXACT:
        return False

    if idiom.startswith(WEAK_STARTS):
        return False

    if idiom.startswith(LITERAL_PREFIXES):
        return False

    if WEAK_PATTERN_RE.match(idiom):
        return False

    non_stop = content_words(toks)
    if len(non_stop) < 2:
        return False

    weak_count = sum(1 for t in non_stop if t in WEAK_CONTENT_WORDS)
    if weak_count >= 2:
        return False

    if toks[-1] in STOPWORDS and toks[-1] not in ENDING_PARTICLES:
        return False

    return True


def too_literal_or_generic(idiom: str, example: str) -> bool:
    idiom = normalize_idiom_text(idiom)
    ex = normalize_idiom_text(example)
    toks = idiom.split()
    non_stop = content_words(toks)

    # Strong patterns should survive unless very malformed
    if contains_strong_idiom_pattern(idiom):
        return False

    if idiom in WEAK_EXACT:
        return True

    if idiom.startswith(WEAK_STARTS):
        return True

    if idiom.startswith(LITERAL_PREFIXES):
        return True

    if WEAK_PATTERN_RE.match(idiom):
        return True

    # Weak lexical structure
    if len(non_stop) < 2:
        return True

    # No idiom-like cue at all
    if not any(t in CONTENT_CUE_WORDS for t in toks) and toks[-1] not in ENDING_PARTICLES:
        if len(non_stop) <= 2:
            return True

    # Generic examples with little context
    if example_quality_score(ex, idiom) <= 1:
        return True

    return False


def build_quality_score(row: pd.Series) -> int:
    score = 0

    idiom = normalize_idiom_text(row["idiom"])
    example = norm(row["example"])
    tags = norm(row["tags"]).lower()

    toks = idiom.split()
    non_stop = content_words(toks)

    # Strong idiom-like shape
    if contains_strong_idiom_pattern(idiom):
        score += 5

    if any(t in CONTENT_CUE_WORDS for t in toks):
        score += 3

    if toks and toks[-1] in ENDING_PARTICLES:
        score += 2

    # Prefer richer phrases
    if len(set(non_stop)) >= 3:
        score += 2
    elif len(set(non_stop)) == 2:
        score += 1

    # Better example
    score += example_quality_score(example, idiom)

    # Slight preference for 3-4 token phrases
    n = len(toks)
    if n in {3, 4}:
        score += 2
    elif n == 5:
        score += 1

    # Prefer candidate/repeated metadata
    if "repeated_ngram" in tags:
        score += 1

    return score


# ============================================================
# Main cleaning
# ============================================================

def clean_opensubtitles_candidates(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    output_stats: Path = OUTPUT_STATS,
):
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

    # Normalize
    df["idiom"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_en"] = df["meaning_en"].apply(norm)
    df["example"] = df["example"].apply(norm)
    df["source"] = df["source"].apply(norm)
    df["source_type"] = df["source_type"].apply(norm)
    df["pos"] = df["pos"].apply(norm)
    df["tags"] = df["tags"].apply(norm)
    df["idiom_confidence"] = df["idiom_confidence"].apply(norm)
    df["source_url"] = df["source_url"].apply(norm)

    keep_mask = []
    for _, row in df.iterrows():
        idiom = row["idiom"]
        example = row["example"]

        keep = True

        if not looks_like_good_idiom(idiom):
            keep = False
        elif too_literal_or_generic(idiom, example):
            keep = False

        keep_mask.append(keep)

    df = df[keep_mask].copy()
    rows_after_filtering = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["idiom", "example"]).copy()
    rows_after_exact_dedup = len(df)

    # Score and keep best row per idiom
    df["quality_score"] = df.apply(build_quality_score, axis=1)
    df = df.sort_values(
        by=["idiom", "quality_score", "example"],
        ascending=[True, False, True]
    ).copy()

    df = df.drop_duplicates(subset=["idiom"], keep="first").copy()
    rows_final = len(df)

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

    df = df[out_cols].sort_values(by=["idiom"]).reset_index(drop=True)

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
        "top_examples_preview": df.head(20)[["idiom", "example"]].to_dict(orient="records"),
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
    clean_opensubtitles_candidates(INPUT_FILE, OUTPUT_FILE, OUTPUT_STATS)


if __name__ == "__main__":
    main()