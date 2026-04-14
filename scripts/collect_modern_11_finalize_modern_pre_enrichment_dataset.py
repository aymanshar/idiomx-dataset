#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_21_finalize_modern_pre_enrichment_dataset.py

Purpose:
- Finalize the modern idioms/slang dataset before LLM enrichment
- Combine broad and high-quality subsets
- Preserve multiple meanings when they are genuinely different
- Prefer the strongest row when duplicate idiom+meaning pairs exist
- Export reproducible pre-enrichment files and summary statistics

Inputs:
- data/processed/idioms_modern_dataset_broad.csv
- data/processed/idioms_modern_dataset_high_quality.csv

Outputs:
- data/processed/idiomx_modern_pre_enrichment.csv
- data/processed/idiomx_modern_pre_enrichment.parquet
- data/processed/idiomx_modern_pre_enrichment_statistics.json
- data/processed/idiomx_modern_pre_enrichment_source_distribution.csv
- data/sample/idiomx_modern_pre_enrichment_sample.csv
"""

from pathlib import Path
import hashlib
import argparse
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Defaults
# ============================================================

DEFAULT_BROAD_FILE = DATA_PROCESS_DIR / "idioms_modern_dataset_broad.csv"
DEFAULT_HIGH_FILE = DATA_PROCESS_DIR / "idioms_modern_dataset_high_quality.csv"

DEFAULT_STATS_JSON = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment_statistics.json"
DEFAULT_STATS_CSV = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment_source_distribution.csv"

DEFAULT_PRE_ENRICHMENT_CSV = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment.csv"
DEFAULT_PRE_ENRICHMENT_PARQUET = DATA_PROCESS_DIR / "idiomx_modern_pre_enrichment.parquet"

DEFAULT_SAMPLE_CSV = DATA_SAMPLE_DIR / "idiomx_modern_pre_enrichment_sample.csv"


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
    return " ".join(safe_str(x).split())


def make_stable_id(idiom_canonical: str, meaning: str) -> str:
    """Create a stable deterministic ID from idiom + meaning."""
    key = f"{normalize_text(idiom_canonical).lower()} || {normalize_text(meaning).lower()}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"idiomx_modern_{digest}"


def infer_license(src: str) -> str:
    """Infer source-level license metadata."""
    src = normalize_text(src).lower()

    if src == "wiktionary_slang_kaikki":
        return "wiktionary_cc_by_sa_4_0"
    elif src == "urban_dictionary":
        return "unknown"
    elif src == "opensubtitles_dialogue":
        return "opensubtitles"
    return "unknown"


def standardize_source_name(src: str) -> str:
    """Normalize source names into consistent canonical values."""
    src = normalize_text(src)
    mapping = {
        "wiktionary_slang_kaikki": "wiktionary_slang_kaikki",
        "urban_dictionary": "urban_dictionary",
        "opensubtitles_dialogue": "opensubtitles_dialogue",
    }
    return mapping.get(src, src)


def normalize_for_key(x: str) -> str:
    """Light normalization for deduplication keys."""
    return normalize_text(x).lower()


def build_dataset_statistics(
    df: pd.DataFrame,
    output_json: Path,
    output_csv: Path,
):
    """
    Build summary statistics and source distribution
    in the same spirit as the main pipeline finalizer.
    """
    output_json = Path(output_json)
    output_csv = Path(output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()

    for col in [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]:
        if col not in work_df.columns:
            work_df[col] = ""
        work_df[col] = work_df[col].fillna("").astype(str).str.strip()

    source_distribution = work_df["source"].value_counts().sort_index()
    source_type_distribution = work_df["source_type"].value_counts().sort_index()
    confidence_distribution = work_df["idiom_confidence"].value_counts().sort_index()

    stats = {
        "rows_total": int(len(work_df)),
        "unique_idioms": int(work_df["idiom"].nunique()),
        "unique_meanings": int(work_df["meaning_en"].nunique()),
        "rows_with_meaning": int((work_df["meaning_en"].str.len() > 0).sum()),
        "rows_with_example": int((work_df["example"].str.len() > 0).sum()),
        "rows_without_meaning": int((work_df["meaning_en"].str.len() == 0).sum()),
        "rows_without_example": int((work_df["example"].str.len() == 0).sum()),
        "avg_idiom_token_length": round(
            work_df["idiom"].fillna("").astype(str).str.split().str.len().mean(), 4
        ) if len(work_df) else 0.0,
        "avg_meaning_token_length": round(
            work_df["meaning_en"].fillna("").astype(str).str.split().str.len().mean(), 4
        ) if len(work_df) else 0.0,
        "avg_example_token_length": round(
            work_df["example"].fillna("").astype(str).str.split().str.len().mean(), 4
        ) if len(work_df) else 0.0,
        "source_distribution": {str(k): int(v) for k, v in source_distribution.items()},
        "source_type_distribution": {str(k): int(v) for k, v in source_type_distribution.items()},
        "confidence_distribution": {str(k): int(v) for k, v in confidence_distribution.items()},
    }

    df_source_stats = pd.DataFrame({
        "source": source_distribution.index.astype(str),
        "rows": source_distribution.values.astype(int),
    })

    import json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    df_source_stats.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return stats, df_source_stats


# ============================================================
# Main finalization
# ============================================================

def finalize_modern_pre_enrichment_dataset(
    broad_file: Path = DEFAULT_BROAD_FILE,
    high_file: Path = DEFAULT_HIGH_FILE,
    stats_json: Path = DEFAULT_STATS_JSON,
    stats_csv: Path = DEFAULT_STATS_CSV,
    output_csv: Path = DEFAULT_PRE_ENRICHMENT_CSV,
    output_parquet: Path = DEFAULT_PRE_ENRICHMENT_PARQUET,
    sample_csv: Path = DEFAULT_SAMPLE_CSV,
    sample_n: int = 500,
):
    """
    Finalize the modern idioms/slang pre-enrichment dataset.

    Strategy:
    - use high-quality as backbone
    - add broad rows that are not duplicate idiom+meaning pairs
    - preserve multiple meanings when genuinely distinct
    """
    broad_file = Path(broad_file)
    high_file = Path(high_file)
    stats_json = Path(stats_json)
    stats_csv = Path(stats_csv)
    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)
    sample_csv = Path(sample_csv)

    if not broad_file.exists():
        raise FileNotFoundError(f"Broad dataset not found: {broad_file}")

    if not high_file.exists():
        raise FileNotFoundError(f"High-quality dataset not found: {high_file}")

    # --------------------------------------------------------
    # Step 1: Load both datasets
    # --------------------------------------------------------
    df_broad = pd.read_csv(broad_file, encoding="utf-8-sig", low_memory=False)
    df_high = pd.read_csv(high_file, encoding="utf-8-sig", low_memory=False)

    # --------------------------------------------------------
    # Step 2: Ensure shared columns exist
    # --------------------------------------------------------
    expected_cols = [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]

    for df in [df_broad, df_high]:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str).str.strip()

    # --------------------------------------------------------
    # Step 3: Use high-quality as backbone
    # Add broad rows only when idiom+meaning pair is new
    # --------------------------------------------------------
    df_high["dedup_key"] = (
        df_high["idiom"].map(normalize_for_key) + " || " +
        df_high["meaning_en"].map(normalize_for_key)
    )

    df_broad["dedup_key"] = (
        df_broad["idiom"].map(normalize_for_key) + " || " +
        df_broad["meaning_en"].map(normalize_for_key)
    )

    existing_keys = set(df_high["dedup_key"].tolist())
    df_broad_extra = df_broad[~df_broad["dedup_key"].isin(existing_keys)].copy()

    # Keep high-quality first, then extra broad rows
    df_final = pd.concat([df_high, df_broad_extra], ignore_index=True).copy()

    # --------------------------------------------------------
    # Step 4: Drop exact duplicates again for safety
    # --------------------------------------------------------
    df_final = df_final.drop_duplicates(subset=["dedup_key"]).copy()
    df_final = df_final.drop(columns=["dedup_key"], errors="ignore")

    # --------------------------------------------------------
    # Step 5: Build statistics from final combined dataset
    # --------------------------------------------------------
    stats, df_source_stats = build_dataset_statistics(
        df=df_final,
        output_json=stats_json,
        output_csv=stats_csv
    )

    # --------------------------------------------------------
    # Step 6: Rename only minimum required columns
    # Keep everything else unchanged
    # --------------------------------------------------------
    rename_map = {
        "idiom": "idiom_canonical",
        "meaning_en": "idiom_canonical_meaning",
    }
    df_clean = df_final.rename(columns=rename_map).copy()

    # --------------------------------------------------------
    # Step 7: Add required pre-enrichment columns if missing
    # --------------------------------------------------------
    if "idiom_surface" not in df_clean.columns:
        df_clean["idiom_surface"] = df_clean["idiom_canonical"]

    if "idiom_id" not in df_clean.columns:
        df_clean.insert(
            0,
            "idiom_id",
            [
                make_stable_id(idiom, meaning)
                for idiom, meaning in zip(
                    df_clean["idiom_canonical"],
                    df_clean["idiom_canonical_meaning"]
                )
            ]
        )

    if "record_origin" not in df_clean.columns:
        df_clean["record_origin"] = "source_only"

    if "license_source" not in df_clean.columns:
        df_clean["license_source"] = df_clean["source"].apply(infer_license)

    if "example_language" not in df_clean.columns:
        df_clean["example_language"] = "en"

    if "meaning_language" not in df_clean.columns:
        df_clean["meaning_language"] = "en"

    # --------------------------------------------------------
    # Step 8: Standardize text/object columns
    # --------------------------------------------------------
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].map(normalize_text)
        df_clean[col] = df_clean[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if "source" in df_clean.columns:
        df_clean["source"] = df_clean["source"].map(standardize_source_name)

    if "idiom_confidence" in df_clean.columns:
        df_clean["idiom_confidence"] = df_clean["idiom_confidence"].astype("string").str.lower().str.strip()

    # --------------------------------------------------------
    # Step 9: Reorder important columns first, keep the rest unchanged
    # --------------------------------------------------------
    preferred_order = [
        "idiom_id",
        "idiom_canonical",
        "idiom_surface",
        "example",
        "idiom_canonical_meaning",
        "source",
        "source_type",
        "pos",
        "tags",
        "idiom_confidence",
        "source_url",
        "record_origin",
        "license_source",
        "example_language",
        "meaning_language",
    ]

    existing_first = [c for c in preferred_order if c in df_clean.columns]
    remaining_cols = [c for c in df_clean.columns if c not in existing_first]
    df_clean = df_clean[existing_first + remaining_cols].copy()

    # --------------------------------------------------------
    # Step 10: Save pre-enrichment files
    # --------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_csv.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df_clean.to_parquet(output_parquet, index=False)

    # --------------------------------------------------------
    # Step 11: Create sample dataset for GitHub / quick inspection
    # --------------------------------------------------------
    sample_n = min(sample_n, len(df_clean))
    df_sample = df_clean.sample(sample_n, random_state=42).copy()
    df_sample.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    sample_parquet = sample_csv.with_suffix(".parquet")
    df_sample.to_parquet(sample_parquet, index=False)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("Saved pre-enrichment files:")
    print(output_csv)
    print(output_parquet)
    print(stats_json)
    print(stats_csv)
    print(sample_csv)

    print("\nModern pre-enrichment dataset shape:", df_clean.shape)
    print("Sample dataset shape:", df_sample.shape)

    return stats, df_source_stats, df_clean, df_sample


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Finalize modern IdiomX pre-enrichment dataset.")
    parser.add_argument("--broad-file", type=str, default=str(DEFAULT_BROAD_FILE))
    parser.add_argument("--high-file", type=str, default=str(DEFAULT_HIGH_FILE))
    parser.add_argument("--stats-json", type=str, default=str(DEFAULT_STATS_JSON))
    parser.add_argument("--stats-csv", type=str, default=str(DEFAULT_STATS_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_PRE_ENRICHMENT_CSV))
    parser.add_argument("--output-parquet", type=str, default=str(DEFAULT_PRE_ENRICHMENT_PARQUET))
    parser.add_argument("--sample-csv", type=str, default=str(DEFAULT_SAMPLE_CSV))
    parser.add_argument("--sample-n", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    finalize_modern_pre_enrichment_dataset(
        broad_file=Path(args.broad_file),
        high_file=Path(args.high_file),
        stats_json=Path(args.stats_json),
        stats_csv=Path(args.stats_csv),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        sample_csv=Path(args.sample_csv),
        sample_n=args.sample_n,
    )