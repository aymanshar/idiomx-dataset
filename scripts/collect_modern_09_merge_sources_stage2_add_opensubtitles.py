#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_19_merge_sources_stage2_add_opensubtitles.py

Purpose:
- Merge the current modern idioms/slang stage1 dataset
  with the cleaned OpenSubtitles candidate source dataset
- Align all rows to the shared IdiomX source schema
- Remove duplicate idiom-meaning rows while keeping the best candidate
- Save the merged stage2 modern dataset

Inputs:
- data/processed/idioms_merged_modern_stage1_urban_wiktionary.csv
- data/processed/idioms_source_opensubtitles_candidates_cleaned.csv

Output:
- data/processed/idioms_merged_modern_stage2_opensubtitles.csv
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

DEFAULT_STAGE1_FILE = DATA_PROCESS_DIR / "idioms_merged_modern_stage1_urban_wiktionary.csv"
DEFAULT_OPENSUB_FILE = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates_cleaned.csv"
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_merged_modern_stage2_opensubtitles.csv"


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
ENDING_PARTICLES = {"up", "down", "off", "out", "over", "away", "around", "back", "through"}
CONTENT_CUE_WORDS = {
    "get", "give", "take", "keep", "cut", "lose", "make", "play", "come", "go",
    "back", "over", "off", "out", "down", "up", "through", "around", "away",
    "mind", "break", "slack", "point", "deal", "catch", "mean", "mess", "hang",
    "blow", "fall", "pull", "push", "drag", "drop", "sell", "buy", "call", "turn",
    "hold", "bring", "throw", "pick", "run", "work", "stick", "hit", "cool", "save",
    "snap", "shut", "spill", "burn", "move", "read", "ride", "flip", "rip",
}


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


def tokenize(text: str) -> list[str]:
    return normalize_idiom_text(text).split()


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
    Prefer more structured lexical sources over noisier dialogue-mined sources.
    Higher is better.
    """
    source = norm(source).lower()
    mapping = {
        "wiktionary_slang_kaikki": 4,
        "kaikki_wiktionary": 4,
        "urban_dictionary": 2,
        "opensubtitles_dialogue": 1,
    }
    return mapping.get(source, 0)


def idiom_shape_score(idiom: str) -> int:
    """
    Light preference for more idiom-like phrase shapes.
    """
    idiom = normalize_idiom_text(idiom)
    toks = tokenize(idiom)

    score = 0

    if len(toks) in {3, 4}:
        score += 2
    elif len(toks) == 5:
        score += 1

    if toks and toks[-1] in ENDING_PARTICLES:
        score += 2

    if any(t in CONTENT_CUE_WORDS for t in toks):
        score += 2

    return score


def build_merge_score(row: pd.Series) -> int:
    """
    Score rows to keep the best candidate among duplicates.
    """
    score = 0

    idiom = norm(row["idiom"])
    meaning = norm(row["meaning_en"])
    example = norm(row["example"])
    source = norm(row["source"])
    conf = norm(row["idiom_confidence"])
    pos = norm(row["pos"]).lower()
    tags = norm(row["tags"]).lower()

    # Prefer rows with examples
    if example:
        score += 4

    # Prefer rows with explicit meaning
    if meaning:
        score += 5
    else:
        score += 1

    # Confidence and source trust
    score += confidence_score(conf) * 3
    score += source_priority(source) * 3

    # POS preference
    if pos in {"phrase", "idiom", "proverb"}:
        score += 2

    # Tag evidence
    if any(x in tags for x in ["slang", "informal", "colloquial", "figurative", "idiomatic"]):
        score += 2

    # Better idiom-like shape
    score += idiom_shape_score(idiom)

    return score


# ============================================================
# Main merge
# ============================================================

def merge_stage2_add_opensubtitles(
    stage1_file: Path = DEFAULT_STAGE1_FILE,
    opensub_file: Path = DEFAULT_OPENSUB_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
):
    """
    Merge the stage1 modern dataset with cleaned OpenSubtitles candidates.
    """
    stage1_file = Path(stage1_file)
    opensub_file = Path(opensub_file)
    output_file = Path(output_file)

    if not stage1_file.exists():
        raise FileNotFoundError(f"Stage1 dataset not found: {stage1_file}")

    if not opensub_file.exists():
        raise FileNotFoundError(f"OpenSubtitles cleaned dataset not found: {opensub_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read both datasets
    df_stage1 = pd.read_csv(stage1_file, encoding="utf-8-sig")
    df_opensub = pd.read_csv(opensub_file, encoding="utf-8-sig")

    rows_stage1 = len(df_stage1)
    rows_opensub = len(df_opensub)

    # Normalize and align schema
    df_stage1 = normalize_dataframe(df_stage1)
    df_opensub = normalize_dataframe(df_opensub)

    # Concatenate
    df = pd.concat([df_stage1, df_opensub], ignore_index=True)
    rows_before_dedup = len(df)

    # Build matching keys
    df["idiom_key"] = df["idiom"].apply(normalize_idiom_text)
    df["meaning_key"] = df["meaning_en"].apply(normalize_meaning_text)

    # For rows without meaning, also prepare idiom-only duplicate handling
    df["has_meaning"] = (df["meaning_en"].str.strip() != "").astype(int)

    # Score rows
    df["merge_score"] = df.apply(build_merge_score, axis=1)

    # Sort so strongest row wins
    df = df.sort_values(
        by=["idiom_key", "meaning_key", "has_meaning", "merge_score"],
        ascending=[True, True, False, False]
    ).copy()

    # Dedup at idiom + meaning level first
    df = df.drop_duplicates(subset=["idiom_key", "meaning_key"], keep="first").copy()

    # For idioms that have both empty-meaning subtitle rows and richer meaning rows,
    # keep subtitle rows too for now, because they provide real usage examples.
    rows_after_dedup = len(df)

    # Drop helper columns
    df = df[STANDARD_COLUMNS].copy()

    # Final sort
    df = df.sort_values(by=["idiom", "meaning_en", "source"]).reset_index(drop=True)

    # Save
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved merged dataset:", output_file)
    print("Stage1 rows:", rows_stage1)
    print("OpenSubtitles rows:", rows_opensub)
    print("Rows before dedup:", rows_before_dedup)
    print("Rows after dedup:", rows_after_dedup)
    print("Unique idioms:", df["idiom"].nunique())
    print("Rows with example:", int((df["example"].str.strip() != "").sum()))
    print("Rows with meaning:", int((df["meaning_en"].str.strip() != "").sum()))

    return df


def main():
    merge_stage2_add_opensubtitles(
        DEFAULT_STAGE1_FILE,
        DEFAULT_OPENSUB_FILE,
        DEFAULT_OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()