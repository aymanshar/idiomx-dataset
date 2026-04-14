#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_22_compare_modern_with_main_idiomx.py

Purpose:
- Compare the modern pre-enrichment dataset against the main IdiomX pre-enrichment dataset
- Remove modern rows that are duplicates of the main IdiomX dataset
- Keep:
    1) completely new idioms
    2) new meanings for existing idioms
- Remove:
    1) exact duplicates (idiom + meaning + example)
    2) duplicate idiom + meaning rows
    3) near-duplicate meanings for the same idiom
    4) weak rows for existing idioms when the modern row adds no real semantic novelty

Inputs:
- modern dataset:
    data/processed/idiomx_modern_pre_enrichment.csv
- main IdiomX dataset:
    data/processed/idiomx_pre_enrichment.csv
  or a custom path you provide

Outputs:
- data/processed/idiomx_modern_pre_enrichment_deduped_against_main.csv
- data/processed/idiomx_modern_pre_enrichment_deduped_against_main.parquet
- data/processed/idiomx_modern_rows_removed_as_duplicates.csv
- data/processed/idiomx_modern_vs_main_audit.csv
- data/processed/idiomx_modern_vs_main_stats.json

Recommended usage:
    python collect_22_compare_modern_with_main_idiomx.py

Custom main file example:
    python collect_22_compare_modern_with_main_idiomx.py --main-file "C:/Users/ayman/Documents/IdiomX/github_idiomX/idiomx-dataset/data/processed/idiomx_pre_enrichment.csv"
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"

DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODERN_FILE = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment.csv"
DEFAULT_MAIN_FILE = DATA_PROCESS_DIR / "idiomx_pre_enrichment.csv"

DEFAULT_OUTPUT_KEEP_CSV = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment_deduped_against_main.csv"
DEFAULT_OUTPUT_KEEP_PARQUET = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment_deduped_against_main.parquet"
DEFAULT_OUTPUT_REMOVED_CSV = DATA_PROCESS_DIR / "idiomx_modern_rows_removed_as_duplicates.csv"
DEFAULT_OUTPUT_AUDIT_CSV = DATA_PROCESS_DIR / "idiomx_modern_vs_main_audit.csv"
DEFAULT_OUTPUT_STATS_JSON = DATA_PROCESS_DIR / "idiomx_modern_vs_main_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")

MEANING_SIMILARITY_RATIO_THRESHOLD = 0.92
MEANING_TOKEN_JACCARD_THRESHOLD = 0.85


# ============================================================
# Helpers
# ============================================================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(x) -> str:
    text = safe_str(x).lower()
    text = text.replace("’", "'")
    text = MULTISPACE_RE.sub(" ", text)
    return text


def normalize_example(x) -> str:
    text = normalize_text(x)
    # light punctuation normalization
    text = re.sub(r"[\"“”`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)

    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def is_near_duplicate_meaning(a: str, b: str) -> bool:
    """
    Conservative near-duplicate meaning check.
    Only used within the same idiom.
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    ratio = sequence_ratio(a_norm, b_norm)
    jac = jaccard_similarity(a_norm, b_norm)

    return (
        ratio >= MEANING_SIMILARITY_RATIO_THRESHOLD
        or jac >= MEANING_TOKEN_JACCARD_THRESHOLD
    )


def resolve_column(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None


def standardize_pre_enrichment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map likely pre-enrichment columns to a standard comparison view
    without deleting any original columns.
    """
    df = df.copy()

    col_idiom = resolve_column(df, ["idiom_canonical", "idiom"], required=True)
    col_meaning = resolve_column(df, ["idiom_canonical_meaning", "meaning_en"], required=False)
    col_example = resolve_column(df, ["example"], required=False)
    col_id = resolve_column(df, ["idiom_id"], required=False)

    df["_idiom_cmp"] = df[col_idiom].map(normalize_text)
    df["_meaning_cmp"] = df[col_meaning].map(normalize_text) if col_meaning else ""
    df["_example_cmp"] = df[col_example].map(normalize_example) if col_example else ""
    df["_row_id_cmp"] = df[col_id].map(safe_str) if col_id else ""

    return df


def build_main_lookup(df_main: pd.DataFrame) -> tuple[set[str], set[str], dict[str, list[str]], set[str]]:
    """
    Build fast lookup structures from main IdiomX dataset.
    """
    idiom_set = set()
    idiom_meaning_set = set()
    idiom_to_meanings: dict[str, list[str]] = {}
    full_key_set = set()

    for _, row in df_main.iterrows():
        idiom = row["_idiom_cmp"]
        meaning = row["_meaning_cmp"]
        example = row["_example_cmp"]

        if not idiom:
            continue

        idiom_set.add(idiom)
        idiom_meaning_set.add(f"{idiom} || {meaning}")
        full_key_set.add(f"{idiom} || {meaning} || {example}")

        idiom_to_meanings.setdefault(idiom, [])
        if meaning and meaning not in idiom_to_meanings[idiom]:
            idiom_to_meanings[idiom].append(meaning)

    return idiom_set, idiom_meaning_set, idiom_to_meanings, full_key_set


# ============================================================
# Main logic
# ============================================================

def compare_modern_with_main_idiomx(
    modern_file: Path = DEFAULT_MODERN_FILE,
    main_file: Path = DEFAULT_MAIN_FILE,
    output_keep_csv: Path = DEFAULT_OUTPUT_KEEP_CSV,
    output_keep_parquet: Path = DEFAULT_OUTPUT_KEEP_PARQUET,
    output_removed_csv: Path = DEFAULT_OUTPUT_REMOVED_CSV,
    output_audit_csv: Path = DEFAULT_OUTPUT_AUDIT_CSV,
    output_stats_json: Path = DEFAULT_OUTPUT_STATS_JSON,
):
    modern_file = Path(modern_file)
    main_file = Path(main_file)
    output_keep_csv = Path(output_keep_csv)
    output_keep_parquet = Path(output_keep_parquet)
    output_removed_csv = Path(output_removed_csv)
    output_audit_csv = Path(output_audit_csv)
    output_stats_json = Path(output_stats_json)

    if not modern_file.exists():
        raise FileNotFoundError(f"Modern dataset not found: {modern_file}")

    if not main_file.exists():
        raise FileNotFoundError(f"Main IdiomX dataset not found: {main_file}")

    df_modern = pd.read_csv(modern_file, encoding="utf-8-sig", low_memory=False)
    df_main = pd.read_csv(main_file, encoding="utf-8-sig", low_memory=False)

    df_modern = standardize_pre_enrichment_columns(df_modern)
    df_main = standardize_pre_enrichment_columns(df_main)

    idiom_set, idiom_meaning_set, idiom_to_meanings, full_key_set = build_main_lookup(df_main)

    audit_rows = []
    keep_indices = []
    remove_indices = []

    for idx, row in df_modern.iterrows():
        idiom = row["_idiom_cmp"]
        meaning = row["_meaning_cmp"]
        example = row["_example_cmp"]

        idiom_meaning_key = f"{idiom} || {meaning}"
        full_key = f"{idiom} || {meaning} || {example}"

        status = ""
        matched_main_meaning = ""

        # Missing idiom should always be removed
        if not idiom:
            status = "remove_missing_idiom"

        # Exact duplicate: idiom + meaning + example
        elif full_key in full_key_set:
            status = "remove_exact_duplicate_of_main"

        # Duplicate idiom + meaning even if example differs
        elif idiom_meaning_key in idiom_meaning_set:
            status = "remove_duplicate_idiom_meaning_of_main"

        # Idiom exists in main but this modern row has no meaning
        elif idiom in idiom_set and not meaning:
            status = "remove_existing_idiom_without_new_meaning"

        # Idiom exists in main, compare meaning against known meanings
        elif idiom in idiom_set and meaning:
            main_meanings = idiom_to_meanings.get(idiom, [])

            near_dup_found = False
            for mm in main_meanings:
                if is_near_duplicate_meaning(meaning, mm):
                    near_dup_found = True
                    matched_main_meaning = mm
                    break

            if near_dup_found:
                status = "remove_near_duplicate_meaning_of_main"
            else:
                status = "keep_new_meaning_for_existing_idiom"

        # Completely new idiom
        else:
            status = "keep_new_idiom"

        audit_rows.append({
            "modern_row_index": int(idx),
            "idiom_id": safe_str(row.get("idiom_id", "")),
            "idiom_canonical": safe_str(row.get("idiom_canonical", row.get("idiom", ""))),
            "idiom_canonical_meaning": safe_str(row.get("idiom_canonical_meaning", row.get("meaning_en", ""))),
            "example": safe_str(row.get("example", "")),
            "source": safe_str(row.get("source", "")),
            "status": status,
            "matched_main_meaning": matched_main_meaning,
        })

        if status.startswith("keep_"):
            keep_indices.append(idx)
        else:
            remove_indices.append(idx)

    df_keep = df_modern.loc[keep_indices].copy()
    df_removed = df_modern.loc[remove_indices].copy()
    df_audit = pd.DataFrame(audit_rows)

    # Remove helper columns before saving
    helper_cols = ["_idiom_cmp", "_meaning_cmp", "_example_cmp", "_row_id_cmp"]
    df_keep = df_keep.drop(columns=[c for c in helper_cols if c in df_keep.columns], errors="ignore")
    df_removed = df_removed.drop(columns=[c for c in helper_cols if c in df_removed.columns], errors="ignore")

    output_keep_csv.parent.mkdir(parents=True, exist_ok=True)
    output_keep_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_removed_csv.parent.mkdir(parents=True, exist_ok=True)
    output_audit_csv.parent.mkdir(parents=True, exist_ok=True)
    output_stats_json.parent.mkdir(parents=True, exist_ok=True)

    df_keep.to_csv(output_keep_csv, index=False, encoding="utf-8-sig")
    df_keep.to_parquet(output_keep_parquet, index=False)
    df_removed.to_csv(output_removed_csv, index=False, encoding="utf-8-sig")
    df_audit.to_csv(output_audit_csv, index=False, encoding="utf-8-sig")

    stats = {
        "modern_input_rows": int(len(df_modern)),
        "main_input_rows": int(len(df_main)),
        "rows_kept": int(len(df_keep)),
        "rows_removed": int(len(df_removed)),
        "status_distribution": {
            str(k): int(v)
            for k, v in df_audit["status"].value_counts().sort_index().items()
        },
    }

    with open(output_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved keep dataset:", output_keep_csv)
    print("Saved keep parquet:", output_keep_parquet)
    print("Saved removed rows:", output_removed_csv)
    print("Saved audit file:", output_audit_csv)
    print("Saved stats:", output_stats_json)
    print("Modern input rows:", len(df_modern))
    print("Rows kept:", len(df_keep))
    print("Rows removed:", len(df_removed))

    return df_keep, df_removed, df_audit, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare modern IdiomX branch against main IdiomX and remove duplicates."
    )
    parser.add_argument("--modern-file", type=str, default=str(DEFAULT_MODERN_FILE))
    parser.add_argument("--main-file", type=str, default=str(DEFAULT_MAIN_FILE))
    parser.add_argument("--output-keep-csv", type=str, default=str(DEFAULT_OUTPUT_KEEP_CSV))
    parser.add_argument("--output-keep-parquet", type=str, default=str(DEFAULT_OUTPUT_KEEP_PARQUET))
    parser.add_argument("--output-removed-csv", type=str, default=str(DEFAULT_OUTPUT_REMOVED_CSV))
    parser.add_argument("--output-audit-csv", type=str, default=str(DEFAULT_OUTPUT_AUDIT_CSV))
    parser.add_argument("--output-stats-json", type=str, default=str(DEFAULT_OUTPUT_STATS_JSON))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    compare_modern_with_main_idiomx(
        modern_file=Path(args.modern_file),
        main_file=Path(args.main_file),
        output_keep_csv=Path(args.output_keep_csv),
        output_keep_parquet=Path(args.output_keep_parquet),
        output_removed_csv=Path(args.output_removed_csv),
        output_audit_csv=Path(args.output_audit_csv),
        output_stats_json=Path(args.output_stats_json),
    )