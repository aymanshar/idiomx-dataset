#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_20_filter_global_modern_idioms.py

Purpose:
- Apply a balanced global filtering stage to the merged modern idioms/slang dataset
- Produce:
    1. a broad usable dataset
    2. a stronger high-quality subset
- Preserve the shared IdiomX source schema

Input:
- data/processed/idioms_merged_modern_stage2_opensubtitles.csv

Outputs:
- data/processed/idioms_modern_dataset_broad.csv
- data/processed/idioms_modern_dataset_high_quality.csv
- data/processed/idioms_modern_filter_stats.json
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

INPUT_FILE = DATA_PROCESS_DIR / "idioms_merged_modern_stage2_opensubtitles.csv"
OUTPUT_BROAD = DATA_PROCESS_DIR / "idioms_modern_dataset_broad.csv"
OUTPUT_HIGH = DATA_PROCESS_DIR / "idioms_modern_dataset_high_quality.csv"
OUTPUT_STATS = DATA_PROCESS_DIR / "idioms_modern_filter_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

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

ENDING_PARTICLES = {"up", "down", "off", "out", "over", "away", "around", "back", "through"}

CONTENT_CUE_WORDS = {
    "get", "give", "take", "keep", "cut", "lose", "make", "play", "come", "go",
    "back", "over", "off", "out", "down", "up", "through", "around", "away",
    "mind", "break", "slack", "point", "deal", "catch", "mean", "mess", "hang",
    "blow", "fall", "pull", "push", "drag", "drop", "sell", "buy", "call", "turn",
    "hold", "bring", "throw", "pick", "run", "work", "stick", "hit", "cool", "save",
    "snap", "shut", "spill", "burn", "move", "read", "ride", "flip", "rip",
}

PRONOUN_STARTERS = {"i", "you", "we", "he", "she", "they", "it"}

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
    "a lot of",
    "one of the",
    "in the middle",
    "at the same",
    "right now",
    "at all",
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

BAD_MEANING_PATTERNS = [
    r"^a person who\b",
    r"^someone who\b",
    r"^a woman who\b",
    r"^a man who\b",
    r"^a guy who\b",
    r"^a girl who\b",
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
    "dialogue",
    "conversational",
}

STRONG_IDIOM_PATTERNS = [
    r"\bget\b.*\bover\b",
    r"\bgive\b.*\bup\b",
    r"\bmake\b.*\bup\b",
    r"\bcut\b.*\bslack\b",
    r"\bblow\b.*\boff\b",
    r"\bhang\b.*\bout\b",
    r"\bback\b.*\boff\b",
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

STANDARD_COLUMNS = [
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


# ============================================================
# Helpers
# ============================================================

def norm(x) -> str:
    if pd.isna(x):
        return ""
    return MULTISPACE_RE.sub(" ", str(x).strip())


def normalize_text(text: str) -> str:
    text = norm(text).lower()
    text = text.replace("’", "'")
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def token_count(text: str) -> int:
    return len(tokenize(text))


def has_letters(text: str) -> bool:
    return bool(LETTER_RE.search(norm(text)))


def bad_symbolic(text: str) -> bool:
    return bool(BAD_SYMBOL_RE.search(norm(text)))


def content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def tags_to_set(tags: str) -> set[str]:
    items = []
    for x in norm(tags).lower().split(","):
        x = x.strip()
        if x:
            items.append(x)
    return set(items)


def confidence_score(conf: str) -> int:
    conf = norm(conf).lower()
    mapping = {"high": 3, "medium": 2, "low": 1}
    return mapping.get(conf, 0)


def source_score(src: str) -> int:
    src = norm(src).lower()
    mapping = {
        "wiktionary_slang_kaikki": 4,
        "kaikki_wiktionary": 4,
        "urban_dictionary": 2,
        "opensubtitles_dialogue": 1,
    }
    return mapping.get(src, 0)


def contains_strong_idiom_pattern(text: str) -> bool:
    text = normalize_text(text)
    return any(re.search(pat, text) for pat in STRONG_IDIOM_PATTERNS)


def bad_meaning(meaning: str) -> bool:
    meaning = normalize_text(meaning)
    if not meaning:
        return False
    for pat in BAD_MEANING_PATTERNS:
        if re.search(pat, meaning):
            return True
    return False


def idiom_shape_score(idiom: str) -> int:
    idiom = normalize_text(idiom)
    toks = idiom.split()
    non_stop = content_words(toks)

    score = 0

    if len(toks) in {3, 4}:
        score += 2
    elif len(toks) == 5:
        score += 1

    if toks and toks[-1] in ENDING_PARTICLES:
        score += 2

    if any(t in CONTENT_CUE_WORDS for t in toks):
        score += 2

    if len(set(non_stop)) >= 3:
        score += 1

    if contains_strong_idiom_pattern(idiom):
        score += 3

    return score


def example_quality_score(example: str, idiom: str = "") -> int:
    score = 0
    ex = norm(example)
    idiom = normalize_text(idiom)

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


def looks_like_good_phrase(idiom: str) -> bool:
    idiom = normalize_text(idiom)

    if not idiom:
        return False

    if not has_letters(idiom):
        return False

    if bad_symbolic(idiom):
        return False

    toks = idiom.split()
    if len(toks) < 3 or len(toks) > 5:
        return False

    if idiom in WEAK_EXACT:
        return False

    if idiom.startswith(WEAK_STARTS):
        return False

    if idiom.startswith(LITERAL_PREFIXES):
        return False

    if toks[0] in PRONOUN_STARTERS:
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


def broad_keep(row: pd.Series) -> bool:
    idiom = normalize_text(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    tags = norm(row["tags"]).lower()
    pos = norm(row["pos"]).lower()
    source = norm(row["source"]).lower()

    if not looks_like_good_phrase(idiom):
        return False

    if meaning and bad_meaning(meaning):
        return False

    # At least one useful payload field
    if not meaning and not example:
        return False

    # Strong dictionary rows survive if structurally good
    if source in {"wiktionary_slang_kaikki", "urban_dictionary"}:
        return True

    # OpenSubtitles rows need stronger structure or example support
    if source == "opensubtitles_dialogue":
        if contains_strong_idiom_pattern(idiom):
            return True
        if idiom_shape_score(idiom) >= 4 and example_quality_score(example, idiom) >= 4:
            return True
        return False

    # fallback for other future sources
    if idiom_shape_score(idiom) >= 3:
        return True

    if any(x in tags for x in ["slang", "figurative", "idiomatic", "colloquial", "informal"]):
        return True

    if pos in {"phrase", "idiom", "proverb"}:
        return True

    return False


def high_quality_keep(row: pd.Series) -> bool:
    idiom = normalize_text(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    tags = norm(row["tags"]).lower()
    pos = norm(row["pos"]).lower()
    source = norm(row["source"]).lower()
    conf = norm(row["idiom_confidence"]).lower()

    shape = idiom_shape_score(idiom)
    ex_score = example_quality_score(example, idiom)
    has_strong_pattern = contains_strong_idiom_pattern(idiom)
    has_strong_tag = any(x in tags for x in ["slang", "figurative", "idiomatic", "colloquial", "informal"])
    has_meaning = bool(meaning)

    # Strong lexical rows
    if has_meaning:
        if bad_meaning(meaning):
            return False

        if source in {"wiktionary_slang_kaikki", "urban_dictionary"}:
            if shape >= 3 or has_strong_tag or pos in {"phrase", "idiom", "proverb"}:
                return True

        if conf in {"high", "medium"} and (shape >= 4 or has_strong_tag):
            return True

    # Strong subtitle/dialogue rows can survive even without meaning
    if source == "opensubtitles_dialogue":
        if has_strong_pattern and ex_score >= 4:
            return True
        if shape >= 6 and ex_score >= 5:
            return True

    return False


def build_best_row_score(row: pd.Series) -> int:
    idiom = normalize_text(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    source = norm(row["source"])
    tags = norm(row["tags"]).lower()
    pos = norm(row["pos"]).lower()
    conf = norm(row["idiom_confidence"])

    score = 0

    if meaning:
        score += 5
    if example:
        score += 4

    score += source_score(source) * 3
    score += confidence_score(conf) * 2
    score += idiom_shape_score(idiom)
    score += example_quality_score(example, idiom)

    if any(x in tags for x in ["slang", "figurative", "idiomatic", "colloquial", "informal"]):
        score += 2

    if pos in {"phrase", "idiom", "proverb"}:
        score += 2

    if contains_strong_idiom_pattern(idiom):
        score += 3

    return score


def finalize_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    df["idiom_key"] = df["idiom"].apply(normalize_text)
    df["meaning_key"] = df["meaning_en"].apply(lambda x: normalize_text(x) if norm(x) else "")
    df["best_score"] = df.apply(build_best_row_score, axis=1)

    df = df.sort_values(
        by=["idiom_key", "meaning_key", "best_score"],
        ascending=[True, True, False]
    ).copy()

    # Keep unique idiom+meaning first
    df = df.drop_duplicates(subset=["idiom_key", "meaning_key"], keep="first").copy()

    # Also keep one best row per idiom for cleaner broad/high subsets
    best_per_idiom = (
        df.sort_values(by=["idiom_key", "best_score"], ascending=[True, False])
          .drop_duplicates(subset=["idiom_key"], keep="first")
          .copy()
    )

    # Merge best_per_idiom with rows having explicit meanings
    with_meaning = df[df["meaning_en"].str.strip() != ""].copy()
    merged = pd.concat([best_per_idiom, with_meaning], ignore_index=True)

    merged["idiom_key"] = merged["idiom"].apply(normalize_text)
    merged["meaning_key"] = merged["meaning_en"].apply(lambda x: normalize_text(x) if norm(x) else "")
    merged["best_score"] = merged.apply(build_best_row_score, axis=1)

    merged = merged.sort_values(
        by=["idiom_key", "meaning_key", "best_score"],
        ascending=[True, True, False]
    ).drop_duplicates(subset=["idiom_key", "meaning_key"], keep="first")

    merged = merged[STANDARD_COLUMNS].sort_values(by=["idiom", "meaning_en", "source"]).reset_index(drop=True)
    return merged


# ============================================================
# Main
# ============================================================

def filter_global_modern_idioms(
    input_file: Path = INPUT_FILE,
    output_broad: Path = OUTPUT_BROAD,
    output_high: Path = OUTPUT_HIGH,
    output_stats: Path = OUTPUT_STATS,
):
    input_file = Path(input_file)
    output_broad = Path(output_broad)
    output_high = Path(output_high)
    output_stats = Path(output_stats)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    rows_before = len(df)

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Normalize key text fields
    df["idiom"] = df["idiom"].apply(normalize_text)
    df["meaning_en"] = df["meaning_en"].apply(norm)
    df["example"] = df["example"].apply(norm)
    df["source"] = df["source"].apply(norm)
    df["source_type"] = df["source_type"].apply(norm)
    df["pos"] = df["pos"].apply(norm)
    df["tags"] = df["tags"].apply(norm)
    df["idiom_confidence"] = df["idiom_confidence"].apply(norm)
    df["source_url"] = df["source_url"].apply(norm)

    broad_df = df[df.apply(broad_keep, axis=1)].copy()
    broad_rows_after_filter = len(broad_df)
    broad_df = finalize_subset(broad_df)
    broad_rows_final = len(broad_df)

    high_df = broad_df[broad_df.apply(high_quality_keep, axis=1)].copy()
    high_rows_after_filter = len(high_df)
    high_df = finalize_subset(high_df)
    high_rows_final = len(high_df)

    output_broad.parent.mkdir(parents=True, exist_ok=True)
    output_high.parent.mkdir(parents=True, exist_ok=True)
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    broad_df.to_csv(output_broad, index=False, encoding="utf-8-sig")
    high_df.to_csv(output_high, index=False, encoding="utf-8-sig")

    stats = {
        "rows_before": int(rows_before),

        "broad_rows_after_filter": int(broad_rows_after_filter),
        "broad_rows_final": int(broad_rows_final),
        "broad_unique_idioms": int(broad_df["idiom"].nunique()) if len(broad_df) else 0,
        "broad_rows_with_meaning": int((broad_df["meaning_en"].str.strip() != "").sum()) if len(broad_df) else 0,
        "broad_rows_with_example": int((broad_df["example"].str.strip() != "").sum()) if len(broad_df) else 0,
        "broad_source_distribution": broad_df["source"].value_counts().to_dict() if len(broad_df) else {},

        "high_rows_after_filter": int(high_rows_after_filter),
        "high_rows_final": int(high_rows_final),
        "high_unique_idioms": int(high_df["idiom"].nunique()) if len(high_df) else 0,
        "high_rows_with_meaning": int((high_df["meaning_en"].str.strip() != "").sum()) if len(high_df) else 0,
        "high_rows_with_example": int((high_df["example"].str.strip() != "").sum()) if len(high_df) else 0,
        "high_source_distribution": high_df["source"].value_counts().to_dict() if len(high_df) else {},

        "broad_preview": broad_df.head(20)[["idiom", "meaning_en", "source"]].to_dict(orient="records") if len(broad_df) else [],
        "high_preview": high_df.head(20)[["idiom", "meaning_en", "source"]].to_dict(orient="records") if len(high_df) else [],
    }

    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved broad dataset:", output_broad)
    print("Saved high-quality dataset:", output_high)
    print("Saved stats:", output_stats)
    print("Rows before:", rows_before)
    print("Broad rows final:", broad_rows_final)
    print("High-quality rows final:", high_rows_final)

    return broad_df, high_df


def main():
    filter_global_modern_idioms(INPUT_FILE, OUTPUT_BROAD, OUTPUT_HIGH, OUTPUT_STATS)


if __name__ == "__main__":
    main()