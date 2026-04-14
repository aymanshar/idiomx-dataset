#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_23_dedup_modern_by_idiom_only_pre_llm.py

Purpose:
- Keep ONLY one best row per idiom_canonical
- Prepare clean idiom list for LLM enrichment
- Do NOT over-clean meanings (LLM will regenerate them)

Input:
- idiomx_modern_pre_enrichment_deduped_against_main.csv

Outputs:
- idiomx_modern_pre_llm_unique_idioms.csv
- idiomx_modern_pre_llm_unique_idioms.parquet
- stats json
"""

from pathlib import Path
import pandas as pd
import json
import re

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"

INPUT_FILE = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment_deduped_against_main.csv"

OUTPUT_CSV = DATA_PROCESS_DIR / "idiomx_modern_pre_llm_unique_idioms.csv"
OUTPUT_PARQUET = DATA_PROCESS_DIR / "idiomx_modern_pre_llm_unique_idioms.parquet"
OUTPUT_STATS = DATA_PROCESS_DIR / "idiomx_modern_pre_llm_unique_idioms_stats.json"


# ============================================================
# Helpers
# ============================================================

def normalize(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def source_score(source: str) -> int:
    s = normalize(source)
    if "wiktionary" in s:
        return 4
    if "urban" in s:
        return 3
    if "opensubtitles" in s:
        return 1
    return 0


def confidence_score(conf: str) -> int:
    c = normalize(conf)
    if c == "high":
        return 3
    if c == "medium":
        return 2
    if c == "low":
        return 1
    return 0


def build_score(row):
    score = 0

    meaning = normalize(row.get("idiom_canonical_meaning"))
    example = normalize(row.get("example"))
    source = row.get("source", "")
    conf = row.get("idiom_confidence", "")
    tags = normalize(row.get("tags"))
    pos = normalize(row.get("pos"))

    if meaning:
        score += 5

    if example:
        score += 1

    score += source_score(source) * 3
    score += confidence_score(conf) * 2

    if any(x in tags for x in ["idiomatic", "figurative", "slang", "colloquial"]):
        score += 2

    if pos in {"idiom", "phrase", "proverb"}:
        score += 1

    return score


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    rows_before = len(df)

    # normalize idiom key
    df["idiom_key"] = df["idiom_canonical"].astype(str).str.lower().str.strip()

    # build score
    df["row_score"] = df.apply(build_score, axis=1)

    # sort by best row per idiom
    df = df.sort_values(
        by=["idiom_key", "row_score"],
        ascending=[True, False]
    )

    # keep only best row per idiom
    df = df.drop_duplicates(subset=["idiom_key"], keep="first")

    rows_final = len(df)

    # cleanup
    df = df.drop(columns=["idiom_key", "row_score"], errors="ignore")

    # save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    stats = {
        "rows_before": int(rows_before),
        "rows_final": int(rows_final),
        "unique_idioms": int(df["idiom_canonical"].nunique()),
        "source_distribution": df["source"].value_counts().to_dict()
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Saved:", OUTPUT_CSV)
    print("Rows before:", rows_before)
    print("Rows final:", rows_final)


if __name__ == "__main__":
    main()