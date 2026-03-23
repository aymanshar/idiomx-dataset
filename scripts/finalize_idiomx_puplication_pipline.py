"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
Year: 2026

Description:
Finalize the IdiomX dataset publication pipeline.

This script takes the final high-precision idiom dataset and produces:
- a reproducibility pre-enrichment dataset
- a finalized enriched dataset copy (if available)
- the main public dataset
- a human-only subset
- a sample dataset
- dataset statistics JSON
- source distribution CSV

It standardizes column names, creates missing required columns, fixes values
such as idiom_confidence, and writes publication-ready CSV/Parquet files.

Notes:
- Safe to run offline
- No API calls
- If an enriched final dataset already exists, it will be used
- Otherwise, the script will still build publication files from the pre-enrichment base
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from typing import Iterable, Optional

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"
DATA_ENRICHED_DIR = DATA_DIR / "enriched"
DATA_FINAL_DIR = DATA_DIR / "final"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

for p in [DATA_PROCESS_DIR, DATA_ENRICHED_DIR, DATA_FINAL_DIR, DATA_SAMPLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Input files
# ============================================================

DEFAULT_HIGH_PRECISION_INPUT = DATA_PROCESS_DIR / "idioms_dataset_stage5_high_precision.csv"

# Optional enriched files, checked in order
ENRICHED_CANDIDATES = [
    DATA_ENRICHED_DIR / "idiomx_enriched_full_final.csv",
    DATA_ENRICHED_DIR / "idiomx_enriched_full_validated.csv",
    DATA_ENRICHED_DIR / "idiomx_enriched_full.csv",
]

# ============================================================
# Output files
# ============================================================

PRE_ENRICHMENT_PARQUET = DATA_PROCESS_DIR / "idiomx_pre_enrichment.parquet"
PRE_ENRICHMENT_SAMPLE_PARQUET = DATA_SAMPLE_DIR / "idiomx_pre_enrichment_sample.parquet"

FINAL_DATASET_CSV = DATA_FINAL_DIR / "idiomx_dataset_v1.csv"
FINAL_DATASET_PARQUET = DATA_FINAL_DIR / "idiomx_dataset_v1.parquet"

FINAL_ENRICHED_FULL_CSV = DATA_FINAL_DIR / "idiomx_enriched_full.csv"
FINAL_ENRICHED_FULL_PARQUET = DATA_FINAL_DIR / "idiomx_enriched_full.parquet"

PUBLIC_CORE_CSV = DATA_FINAL_DIR / "idiomx_core.csv"
PUBLIC_CORE_PARQUET = DATA_FINAL_DIR / "idiomx_core.parquet"

HUMAN_ONLY_CSV = DATA_FINAL_DIR / "idiomx_human_examples_only.csv"
HUMAN_ONLY_PARQUET = DATA_FINAL_DIR / "idiomx_human_examples_only.parquet"

FINAL_SAMPLE_CSV = DATA_SAMPLE_DIR / "idiomx_sample.csv"
FINAL_SAMPLE_PARQUET = DATA_SAMPLE_DIR / "idiomx_sample.parquet"

STATS_JSON = DATA_FINAL_DIR / "dataset_statistics.json"
SOURCE_DIST_CSV = DATA_FINAL_DIR / "source_distribution.csv"


# ============================================================
# Publication schema
# ============================================================

PUBLIC_COLUMNS = [
    "idiom_id",
    "idiom_canonical",
    "idiom_surface",
    "idiom_in_example",
    "idiom_in_example_arabic",
    "idiom_in_example_meaning_en",
    "idiom_in_example_meaning_arabic",
    "idiom_canonical_meaning",
    "idiom_canonical_meaning_arabic",
    "ambiguity_flag",
    "idiom_compositionality_level",
    "idiom_register",
    "idiom_domain",
    "learner_difficulty",
    "is_example_idiom",
    "example_usage_label",
    "source",
    "source_type",
    "pos",
    "tags",
    "idiom_confidence",
    "source_url",
    "validation_status",
]

PRE_ENRICHMENT_COLUMNS = [
    "idiom_id",
    "idiom_canonical",
    "idiom_canonical_meaning",
    "idiom_canonical_meaning_arabic",
    "ambiguity_flag",
    "idiom_compositionality_level",
    "idiom_register",
    "idiom_domain",
    "learner_difficulty",
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

def safe_str(x) -> str:
    """Convert null-like values to empty string and strip whitespace."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(x) -> str:
    """Normalize text by trimming and collapsing repeated spaces."""
    return re.sub(r"\s+", " ", safe_str(x))


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], fill_value: str = "") -> pd.DataFrame:
    """Ensure the dataframe contains all listed columns."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Normalize selected dataframe columns to clean strings."""
    df = ensure_columns(df, columns)
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna("").astype(str).map(normalize_text)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with UTF-8-SIG encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def make_idiom_id(idiom_canonical: str, meaning_en: str) -> str:
    """Create a stable idiom ID from canonical idiom + English meaning."""
    key = f"{normalize_text(idiom_canonical).lower()} || {normalize_text(meaning_en).lower()}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"idiomx_{digest}"


def fix_idiom_confidence(source: str, current_value: str) -> str:
    """
    Standardize idiom_confidence values.

    Rules:
    - keep valid existing values if present
    - otherwise infer from source
    """
    current_value = normalize_text(current_value).lower()
    source = normalize_text(source).lower()

    if current_value in {"high", "medium", "low"}:
        return current_value

    if source in {"kaikki_wiktionary", "phrasefinder", "kaggle_english_idioms", "lidioms"}:
        return "high"

    if source == "wordnet":
        return "medium"

    return "medium"


def choose_enriched_input() -> Optional[Path]:
    """Return the first existing enriched dataset candidate, if any."""
    for path in ENRICHED_CANDIDATES:
        if path.exists():
            return path
    return None


def deduplicate_on_public_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate using the most stable public row identity.
    """
    df = df.copy()

    key_parts = (
        df["idiom_canonical"].fillna("").astype(str).str.lower().str.strip()
        + " || " +
        df["idiom_surface"].fillna("").astype(str).str.lower().str.strip()
        + " || " +
        df["idiom_in_example"].fillna("").astype(str).str.lower().str.strip()
        + " || " +
        df["idiom_in_example_meaning_en"].fillna("").astype(str).str.lower().str.strip()
    )

    df["_dedup_key"] = key_parts
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"]).reset_index(drop=True)
    return df


def token_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(normalize_text(text).split())


def build_dataset_statistics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compute dataset statistics and source distribution.
    """
    source_distribution = df["source"].value_counts().sort_index()

    stats = {
        "rows_total": int(len(df)),
        "unique_idioms": int(df["idiom_canonical"].nunique()),
        "unique_meanings_en": int(df["idiom_canonical_meaning"].nunique()),
        "rows_with_example_en": int((df["idiom_in_example"].str.len() > 0).sum()),
        "rows_with_example_ar": int((df["idiom_in_example_arabic"].str.len() > 0).sum()),
        "rows_with_meaning_en": int((df["idiom_canonical_meaning"].str.len() > 0).sum()),
        "rows_with_meaning_ar": int((df["idiom_canonical_meaning_arabic"].str.len() > 0).sum()),
        "rows_idiomatic_examples": int((df["example_usage_label"] == "idiomatic").sum()),
        "rows_literal_examples": int((df["example_usage_label"] == "literal").sum()),
        "rows_human_examples_only": int(((df["source"] != "generated") & (df["idiom_in_example"].str.len() > 0)).sum()),
        "avg_idiom_tokens": round(df["idiom_canonical"].apply(token_count).mean(), 2) if len(df) else 0.0,
        "min_idiom_tokens": int(df["idiom_canonical"].apply(token_count).min()) if len(df) else 0,
        "max_idiom_tokens": int(df["idiom_canonical"].apply(token_count).max()) if len(df) else 0,
        "source_distribution": {k: int(v) for k, v in source_distribution.items()},
    }

    source_df = (
        source_distribution.rename_axis("source")
        .reset_index(name="count")
        .sort_values(["count", "source"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return stats, source_df


# ============================================================
# Step 1: Load final high-precision base dataset
# ============================================================

def load_base_dataset(input_csv: Path = DEFAULT_HIGH_PRECISION_INPUT) -> pd.DataFrame:
    """
    Load the final high-precision base idiom dataset.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Base input dataset not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)

    # Standard source-level columns expected from the base pipeline
    base_cols = [
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
    df = normalize_columns(df, base_cols)

    # Fix confidence values
    df["idiom_confidence"] = [
        fix_idiom_confidence(src, conf)
        for src, conf in zip(df["source"], df["idiom_confidence"])
    ]

    return df


# ============================================================
# Step 2: Create reproducibility pre-enrichment dataset
# ============================================================

def build_pre_enrichment_dataset(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Map base dataset columns into the ideal pre-enrichment schema.
    """
    out = pd.DataFrame({
        "idiom_id": [
            make_idiom_id(idiom, meaning)
            for idiom, meaning in zip(df_base["idiom"], df_base["meaning_en"])
        ],
        "idiom_canonical": df_base["idiom"].map(normalize_text),
        "idiom_canonical_meaning": df_base["meaning_en"].map(normalize_text),
        "idiom_canonical_meaning_arabic": "",
        "ambiguity_flag": "",
        "idiom_compositionality_level": "",
        "idiom_register": "",
        "idiom_domain": "",
        "learner_difficulty": "",
        "example": df_base["example"].map(normalize_text),
        "source": df_base["source"].map(normalize_text),
        "source_type": df_base["source_type"].map(normalize_text),
        "pos": df_base["pos"].map(normalize_text),
        "tags": df_base["tags"].map(normalize_text),
        "idiom_confidence": df_base["idiom_confidence"].map(normalize_text),
        "source_url": df_base["source_url"].map(normalize_text),
    })

    out = ensure_columns(out, PRE_ENRICHMENT_COLUMNS)
    out = normalize_columns(out, PRE_ENRICHMENT_COLUMNS)
    out = out[PRE_ENRICHMENT_COLUMNS].copy()

    # Deduplicate by idiom_id
    out = out.drop_duplicates(subset=["idiom_id"]).reset_index(drop=True)
    return out


# ============================================================
# Step 3: Load or synthesize enriched/public dataset
# ============================================================

def load_or_build_public_dataset(df_base: pd.DataFrame, df_pre: pd.DataFrame) -> tuple[pd.DataFrame, Optional[Path]]:
    """
    Load enriched final dataset if available; otherwise build a fallback
    publication dataset from the pre-enrichment/base dataset.
    """
    enriched_input = choose_enriched_input()

    if enriched_input is None:
        # Fallback publication dataset from base/pre-enrichment
        out = pd.DataFrame({
            "idiom_id": df_pre["idiom_id"],
            "idiom_canonical": df_pre["idiom_canonical"],
            "idiom_surface": df_pre["idiom_canonical"],
            "idiom_in_example": df_pre["example"],
            "idiom_in_example_arabic": "",
            "idiom_in_example_meaning_en": df_pre["idiom_canonical_meaning"],
            "idiom_in_example_meaning_arabic": "",
            "idiom_canonical_meaning": df_pre["idiom_canonical_meaning"],
            "idiom_canonical_meaning_arabic": df_pre["idiom_canonical_meaning_arabic"],
            "ambiguity_flag": df_pre["ambiguity_flag"],
            "idiom_compositionality_level": df_pre["idiom_compositionality_level"],
            "idiom_register": df_pre["idiom_register"],
            "idiom_domain": df_pre["idiom_domain"],
            "learner_difficulty": df_pre["learner_difficulty"],
            "is_example_idiom": "",
            "example_usage_label": "",
            "source": df_pre["source"],
            "source_type": df_pre["source_type"],
            "pos": df_pre["pos"],
            "tags": df_pre["tags"],
            "idiom_confidence": df_pre["idiom_confidence"],
            "source_url": df_pre["source_url"],
            "validation_status": "",
        })

        out = ensure_columns(out, PUBLIC_COLUMNS)
        out = normalize_columns(out, PUBLIC_COLUMNS)
        out = out[PUBLIC_COLUMNS].copy()
        out["idiom_confidence"] = [
            fix_idiom_confidence(src, conf)
            for src, conf in zip(out["source"], out["idiom_confidence"])
        ]
        out = deduplicate_on_public_key(out)
        return out, None

    # Real enriched dataset path found
    df_enriched = pd.read_csv(enriched_input, encoding="utf-8-sig", low_memory=False)

    # Map possible current names to the publication schema
    rename_map = {
        "idiom": "idiom_canonical",
        "meaning_en": "idiom_canonical_meaning",
        "example": "idiom_in_example",
    }
    df_enriched = df_enriched.rename(columns={k: v for k, v in rename_map.items() if k in df_enriched.columns})

    # Create missing required fields
    df_enriched = ensure_columns(df_enriched, PUBLIC_COLUMNS)

    # Fill idiom_id if missing
    if (df_enriched["idiom_id"].fillna("").astype(str).str.strip() == "").all():
        df_enriched["idiom_id"] = [
            make_idiom_id(idiom, meaning)
            for idiom, meaning in zip(
                df_enriched["idiom_canonical"],
                df_enriched["idiom_canonical_meaning"],
            )
        ]

    # Map fallback values for missing publication fields
    mask_surface_empty = df_enriched["idiom_surface"].fillna("").astype(str).str.strip() == ""
    df_enriched.loc[mask_surface_empty, "idiom_surface"] = df_enriched.loc[mask_surface_empty, "idiom_canonical"]

    mask_example_meaning_en_empty = df_enriched["idiom_in_example_meaning_en"].fillna("").astype(str).str.strip() == ""
    df_enriched.loc[mask_example_meaning_en_empty, "idiom_in_example_meaning_en"] = df_enriched.loc[
        mask_example_meaning_en_empty, "idiom_canonical_meaning"
    ]

    # Normalize values and reorder
    df_enriched = normalize_columns(df_enriched, PUBLIC_COLUMNS)
    df_enriched = df_enriched[PUBLIC_COLUMNS].copy()

    # Fix idiom_confidence
    df_enriched["idiom_confidence"] = [
        fix_idiom_confidence(src, conf)
        for src, conf in zip(df_enriched["source"], df_enriched["idiom_confidence"])
    ]

    # Deduplicate
    df_enriched = deduplicate_on_public_key(df_enriched)

    return df_enriched, enriched_input


# ============================================================
# Step 4: Build publication subsets
# ============================================================

def build_human_only_subset(df_public: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Build a human-only subset.

    Preferred rule:
    - examples that come from non-generated sources and have example text

    Fallback:
    - if empty, derive from the original base dataset examples
    """
    df_human = df_public[
        (df_public["source"].str.lower() != "generated") &
        (df_public["idiom_in_example"].str.len() > 0)
    ].copy()

    if len(df_human) > 0:
        df_human = df_human.reset_index(drop=True)
        return df_human

    # Fallback from base
    df_base_examples = df_base[df_base["example"].str.len() > 0].copy()

    out = pd.DataFrame({
        "idiom_id": [
            make_idiom_id(idiom, meaning)
            for idiom, meaning in zip(df_base_examples["idiom"], df_base_examples["meaning_en"])
        ],
        "idiom_canonical": df_base_examples["idiom"].map(normalize_text),
        "idiom_surface": df_base_examples["idiom"].map(normalize_text),
        "idiom_in_example": df_base_examples["example"].map(normalize_text),
        "idiom_in_example_arabic": "",
        "idiom_in_example_meaning_en": df_base_examples["meaning_en"].map(normalize_text),
        "idiom_in_example_meaning_arabic": "",
        "idiom_canonical_meaning": df_base_examples["meaning_en"].map(normalize_text),
        "idiom_canonical_meaning_arabic": "",
        "ambiguity_flag": "",
        "idiom_compositionality_level": "",
        "idiom_register": "",
        "idiom_domain": "",
        "learner_difficulty": "",
        "is_example_idiom": "",
        "example_usage_label": "",
        "source": df_base_examples["source"].map(normalize_text),
        "source_type": df_base_examples["source_type"].map(normalize_text),
        "pos": df_base_examples["pos"].map(normalize_text),
        "tags": df_base_examples["tags"].map(normalize_text),
        "idiom_confidence": df_base_examples["idiom_confidence"].map(normalize_text),
        "source_url": df_base_examples["source_url"].map(normalize_text),
        "validation_status": "",
    })

    out = ensure_columns(out, PUBLIC_COLUMNS)
    out = normalize_columns(out, PUBLIC_COLUMNS)
    out = out[PUBLIC_COLUMNS].copy()
    out["idiom_confidence"] = [
        fix_idiom_confidence(src, conf)
        for src, conf in zip(out["source"], out["idiom_confidence"])
    ]
    out = deduplicate_on_public_key(out)
    return out


def build_sample_dataset(df_public: pd.DataFrame, n_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Create a compact reproducible sample dataset.
    """
    if len(df_public) == 0:
        return df_public.copy()

    n_rows = min(n_rows, len(df_public))
    return df_public.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


# ============================================================
# Main pipeline
# ============================================================

def finalize_pipeline():
    """
    Run the full finalization pipeline.
    """
    print("Loading base high-precision dataset...")
    df_base = load_base_dataset()

    print("Building reproducibility pre-enrichment dataset...")
    df_pre = build_pre_enrichment_dataset(df_base)

    write_parquet(df_pre, PRE_ENRICHMENT_PARQUET)
    print(f"Saved: {PRE_ENRICHMENT_PARQUET}")

    df_pre_sample = build_sample_dataset(df_pre.rename(columns={"idiom_canonical": "idiom_canonical"}), n_rows=50)
    write_parquet(df_pre_sample, PRE_ENRICHMENT_SAMPLE_PARQUET)
    print(f"Saved: {PRE_ENRICHMENT_SAMPLE_PARQUET}")

    print("Loading enriched dataset if available, otherwise building fallback public dataset...")
    df_public, enriched_input = load_or_build_public_dataset(df_base, df_pre)

    # Save publication-named main dataset
    write_csv(df_public, FINAL_DATASET_CSV)
    write_parquet(df_public, FINAL_DATASET_PARQUET)
    print(f"Saved: {FINAL_DATASET_CSV}")
    print(f"Saved: {FINAL_DATASET_PARQUET}")

    # Save enriched full copy in final/
    write_csv(df_public, FINAL_ENRICHED_FULL_CSV)
    write_parquet(df_public, FINAL_ENRICHED_FULL_PARQUET)
    print(f"Saved: {FINAL_ENRICHED_FULL_CSV}")
    print(f"Saved: {FINAL_ENRICHED_FULL_PARQUET}")

    # Save main public dataset alias
    write_csv(df_public, PUBLIC_CORE_CSV)
    write_parquet(df_public, PUBLIC_CORE_PARQUET)
    print(f"Saved: {PUBLIC_CORE_CSV}")
    print(f"Saved: {PUBLIC_CORE_PARQUET}")

    print("Building human-only subset...")
    df_human = build_human_only_subset(df_public, df_base)
    write_csv(df_human, HUMAN_ONLY_CSV)
    write_parquet(df_human, HUMAN_ONLY_PARQUET)
    print(f"Saved: {HUMAN_ONLY_CSV}")
    print(f"Saved: {HUMAN_ONLY_PARQUET}")

    print("Building publication sample dataset...")
    df_sample = build_sample_dataset(df_public, n_rows=200)
    write_csv(df_sample, FINAL_SAMPLE_CSV)
    write_parquet(df_sample, FINAL_SAMPLE_PARQUET)
    print(f"Saved: {FINAL_SAMPLE_CSV}")
    print(f"Saved: {FINAL_SAMPLE_PARQUET}")

    print("Computing final dataset statistics...")
    stats, source_df = build_dataset_statistics(df_public)

    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    write_csv(source_df, SOURCE_DIST_CSV)

    print(f"Saved: {STATS_JSON}")
    print(f"Saved: {SOURCE_DIST_CSV}")

    print("\nFinal Dataset Summary")
    print("---------------------")
    print("Rows total:", stats["rows_total"])
    print("Unique idioms:", stats["unique_idioms"])
    print("Rows with English examples:", stats["rows_with_example_en"])
    print("Rows with Arabic examples:", stats["rows_with_example_ar"])
    print("Rows with English meanings:", stats["rows_with_meaning_en"])
    print("Rows with Arabic meanings:", stats["rows_with_meaning_ar"])
    print("Idiomatic examples:", stats["rows_idiomatic_examples"])
    print("Literal examples:", stats["rows_literal_examples"])
    print("Human-only example rows:", stats["rows_human_examples_only"])
    print("Average idiom tokens:", stats["avg_idiom_tokens"])
    print("\nSource distribution:")
    print(source_df)

    if enriched_input is None:
        print("\nNote: No enriched dataset was found under data/enriched/.")
        print("A fallback public dataset was built from the pre-enrichment/base dataset.")
    else:
        print(f"\nEnriched source used: {enriched_input}")

    return {
        "base_df": df_base,
        "pre_enrichment_df": df_pre,
        "public_df": df_public,
        "human_only_df": df_human,
        "sample_df": df_sample,
        "stats": stats,
        "source_distribution_df": source_df,
    }


def main():
    """
    Command-line entry point.
    """
    finalize_pipeline()


if __name__ == "__main__":
    main()