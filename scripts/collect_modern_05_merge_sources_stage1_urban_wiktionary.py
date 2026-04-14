#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_15_merge_sources_stage1_urban_wiktionary.py

Purpose:
- Merge the cleaned Urban Dictionary slang source dataset
  with the cleaned Wiktionary slang source dataset
- Align both sources to the shared IdiomX source schema
- Remove duplicate idiom-meaning rows while keeping the best candidate
- Save the first merged modern idioms/slang dataset stage

Inputs:
- data/processed/idioms_source_urban_dictionary_cleaned.csv
- data/processed/idioms_source_wiktionary_slang_cleaned.csv

Output:
- data/processed/idioms_merged_modern_stage1_urban_wiktionary.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import re


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URBAN_FILE = DATA_PROCESS_DIR / "idioms_source_urban_dictionary_cleaned.csv"
DEFAULT_WIKTIONARY_FILE = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang_cleaned.csv"
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_merged_modern_stage1_urban_wiktionary.csv"


# ============================================================
# Shared schema
# ============================================================

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

MULTISPACE_RE = re.compile(r"\s+")


# ============================================================
# Helpers
# ============================================================

def norm(x) -> str:
    """
    Safely normalize a single scalar value into a stripped string.
    """
    if pd.isna(x):
        return ""
    return MULTISPACE_RE.sub(" ", str(x).strip())


def normalize_idiom_text(text: str) -> str:
    """
    Light idiom normalization for matching and deduplication.
    """
    text = norm(text).lower()
    text = text.replace("’", "'")
    return text


def normalize_meaning_text(text: str) -> str:
    """
    Light meaning normalization for matching and deduplication.
    """
    return norm(text).lower()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected columns exist and normalize their values.
    """
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["idiom"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_en"] = df["meaning_en"].apply(norm)
    df["example"] = df["example"].apply(norm)
    df["source"] = df["source"].apply(norm)
    df["source_type"] = df["source_type"].apply(norm)
    df["pos"] = df["pos"].apply(norm)
    df["tags"] = df["tags"].apply(norm)
    df["idiom_confidence"] = df["idiom_confidence"].apply(norm)
    df["source_url"] = df["source_url"].apply(norm)

    return df


def confidence_score(conf: str) -> int:
    """
    Rank confidence labels numerically for tie-breaking.
    """
    conf = norm(conf).lower()
    mapping = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return mapping.get(conf, 0)


def source_priority(source: str) -> int:
    """
    Prefer more structured dictionary-like sources over noisier ones.
    Higher is better.
    """
    source = norm(source).lower()
    mapping = {
        "wiktionary_slang_kaikki": 3,
        "kaikki_wiktionary": 3,
        "urban_dictionary": 1,
    }
    return mapping.get(source, 0)


def build_merge_score(row: pd.Series) -> int:
    """
    Score rows to keep the best candidate among duplicates.
    """
    score = 0

    example = norm(row["example"])
    source = norm(row["source"])
    conf = norm(row["idiom_confidence"])
    pos = norm(row["pos"]).lower()
    tags = norm(row["tags"]).lower()

    if example:
        score += 4

    score += confidence_score(conf) * 3
    score += source_priority(source) * 2

    if pos in {"phrase", "idiom", "proverb"}:
        score += 2

    if any(x in tags for x in ["slang", "informal", "colloquial", "figurative", "idiomatic"]):
        score += 1

    return score


# ============================================================
# Main merge
# ============================================================

def merge_stage1_urban_wiktionary(
    urban_file: Path = DEFAULT_URBAN_FILE,
    wiktionary_file: Path = DEFAULT_WIKTIONARY_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
):
    """
    Merge the cleaned Urban Dictionary slang source dataset
    with the cleaned Wiktionary slang source dataset.
    """
    urban_file = Path(urban_file)
    wiktionary_file = Path(wiktionary_file)
    output_file = Path(output_file)

    if not urban_file.exists():
        raise FileNotFoundError(f"Urban cleaned dataset not found: {urban_file}")

    if not wiktionary_file.exists():
        raise FileNotFoundError(f"Wiktionary cleaned dataset not found: {wiktionary_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read both cleaned source datasets
    df_urban = pd.read_csv(urban_file, encoding="utf-8-sig")
    df_wiktionary = pd.read_csv(wiktionary_file, encoding="utf-8-sig")

    rows_urban = len(df_urban)
    rows_wiktionary = len(df_wiktionary)

    # Normalize and align schema
    df_urban = normalize_dataframe(df_urban)
    df_wiktionary = normalize_dataframe(df_wiktionary)

    # Concatenate
    df = pd.concat([df_urban, df_wiktionary], ignore_index=True)

    rows_before_dedup = len(df)

    # Build lightweight normalized keys
    df["idiom_key"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_key"] = df["meaning_en"].apply(normalize_meaning_text)

    # Score rows so the best duplicate candidate is kept
    df["merge_score"] = df.apply(build_merge_score, axis=1)

    df = df.sort_values(
        by=["idiom_key", "meaning_key", "merge_score"],
        ascending=[True, True, False]
    ).copy()

    # Deduplicate at idiom + meaning level
    df = df.drop_duplicates(subset=["idiom_key", "meaning_key"], keep="first").copy()

    rows_after_dedup = len(df)

    # Drop helper columns
    df = df[STANDARD_COLUMNS].copy()

    # Final sort
    df = df.sort_values(by=["idiom", "meaning_en"]).reset_index(drop=True)

    # Save merged dataset
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved merged dataset:", output_file)
    print("Urban rows:", rows_urban)
    print("Wiktionary rows:", rows_wiktionary)
    print("Rows before dedup:", rows_before_dedup)
    print("Rows after dedup:", rows_after_dedup)
    print("Unique idioms:", df["idiom"].nunique())
    print("Rows with example:", int((df["example"].str.strip() != "").sum()))

    return df


def main():
    merge_stage1_urban_wiktionary(
        DEFAULT_URBAN_FILE,
        DEFAULT_WIKTIONARY_FILE,
        DEFAULT_OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()