"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
Year: 2026

Description:
Finalize the pre-enrichment IdiomX dataset from the high-precision idiom dataset.

This script:
1. computes pre-enrichment dataset statistics
2. standardizes the minimum required schema for LLM enrichment
3. preserves all existing columns
4. creates the reproducibility dataset before LLM enrichment
5. creates a small sample dataset for GitHub / inspection

Outputs:
- data/processed/idiomx_pre_enrichment.csv
- data/processed/idiomx_pre_enrichment.parquet
- data/processed/idiomx_pre_enrichment_statistics.json
- data/processed/idiomx_pre_enrichment_source_distribution.csv
- data/sample/idiomx_pre_enrichment_sample.csv
"""

from pathlib import Path
import hashlib
import argparse
import pandas as pd

from collect_09_dataset_statistics import build_dataset_statistics


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

DEFAULT_INPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_high_precision.csv"

DEFAULT_STATS_JSON = DATA_PROCESS_DIR / "idiomx_pre_enrichment_statistics.json"
DEFAULT_STATS_CSV = DATA_PROCESS_DIR / "idiomx_pre_enrichment_source_distribution.csv"

DEFAULT_PRE_ENRICHMENT_CSV = DATA_PROCESS_DIR / "idiomx_pre_enrichment.csv"
DEFAULT_PRE_ENRICHMENT_PARQUET = DATA_PROCESS_DIR / "idiomx_pre_enrichment.parquet"

DEFAULT_SAMPLE_CSV = DATA_SAMPLE_DIR / "idiomx_pre_enrichment_sample.csv"


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
    return f"idiomx_{digest}"


def infer_license(src: str) -> str:
    """Infer source-level license metadata."""
    src = normalize_text(src).lower()
    if src == "kaikki_wiktionary":
        return "wiktionary_cc_by_sa_4_0"
    elif src == "wordnet":
        return "wordnet"
    return "unknown"


def standardize_source_name(src: str) -> str:
    """Normalize source names into consistent canonical values."""
    src = normalize_text(src)
    mapping = {
        "kaikki": "kaikki_wiktionary",
        "wiktionary": "kaikki_wiktionary",
        "kaikki.org": "kaikki_wiktionary",
        "word_net": "wordnet",
        "WordNet": "wordnet",
    }
    return mapping.get(src, src)

# Remove intermediate and old pipeline files
def cleanup_old_files(base_dir: Path):
    """
    Remove intermediate and old pipeline files safely.
    Keeps only required final inputs.
    """

    patterns_to_delete = [
        # intermediate datasets
        "idioms_dataset_*.csv",

        # old batches/results
        "idiomx_batch*.jsonl",
        "idiomx_results*.jsonl",

        # old enriched outputs
        "idiomx_enriched_full*.csv",
        "idiomx_enriched_full*.parquet",

        # validation outputs
        "idiomx_enriched_full_validated*.csv",
        "idiomx_enriched_full_issues*.csv",
        "idiomx_enriched_full_final*.csv",
    ]

    protected_files = {
        "idioms_dataset_high_precision.csv",
        "idiomx_pre_enrichment.csv",
        "idiomx_pre_enrichment.parquet",
        "idiomx_pre_enrichment_sample.csv",
        "idiomx_pre_enrichment_sample.parquet",
        "idiomx_pre_enrichment_statistics.json",
        "idiomx_pre_enrichment_source_distribution.csv",
    }

    deleted = []

    for pattern in patterns_to_delete:
        for file in base_dir.rglob(pattern):
            if file.name not in protected_files and file.exists():
                try:
                    print("Would delete:", file)
                    file.unlink()
                    deleted.append(str(file))
                except Exception as e:
                    print(f"⚠️ Could not delete {file}: {e}")

    print(f"\n🧹 Cleanup completed. Deleted {len(deleted)} files.")
    return deleted

# ============================================================
# Main finalization
# ============================================================

def finalize_pre_enrichment_dataset(
    input_file: Path = DEFAULT_INPUT_FILE,
    stats_json: Path = DEFAULT_STATS_JSON,
    stats_csv: Path = DEFAULT_STATS_CSV,
    output_csv: Path = DEFAULT_PRE_ENRICHMENT_CSV,
    output_parquet: Path = DEFAULT_PRE_ENRICHMENT_PARQUET,
    sample_csv: Path = DEFAULT_SAMPLE_CSV,
    sample_n: int = 500,
    do_cleanup=True,
):
    """
    Finalize the pre-enrichment IdiomX dataset and sample.

    Returns
    -------
    tuple
        stats, source_distribution_df, pre_enrichment_df, sample_df
    """
    input_file = Path(input_file)
    stats_json = Path(stats_json)
    stats_csv = Path(stats_csv)
    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)
    sample_csv = Path(sample_csv)

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    # --------------------------------------------------------
    # Step 1: Build statistics from the original high-precision file
    # --------------------------------------------------------
    stats, df_source_stats = build_dataset_statistics(
        input_file=input_file,
        output_json=stats_json,
        output_csv=stats_csv
    )

    # --------------------------------------------------------
    # Step 2: Load the high-precision dataset once for finalization
    # --------------------------------------------------------
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)

    # --------------------------------------------------------
    # Step 3: Rename only the minimum required columns
    # Keep everything else unchanged
    # --------------------------------------------------------
    rename_map = {
        "idiom": "idiom_canonical",
        "meaning_en": "idiom_canonical_meaning",
    }
    df_clean = df.rename(columns=rename_map).copy()

    # --------------------------------------------------------
    # Step 4: Add required pre-enrichment columns if missing
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
    # Step 5: Standardize text/object columns
    # --------------------------------------------------------
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].map(normalize_text)
        df_clean[col] = df_clean[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if "source" in df_clean.columns:
        df_clean["source"] = df_clean["source"].map(standardize_source_name)

    if "idiom_confidence" in df_clean.columns:
        df_clean["idiom_confidence"] = df_clean["idiom_confidence"].astype("string").str.lower().str.strip()

    # --------------------------------------------------------
    # Step 6: Reorder important columns first, keep the rest unchanged
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
    # Step 7: Save pre-enrichment files
    # --------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_csv.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df_clean.to_parquet(output_parquet, index=False)

    # --------------------------------------------------------
    # Step 8: Create sample dataset for GitHub / quick inspection
    # --------------------------------------------------------
    sample_n = min(sample_n, len(df_clean))
    df_sample = df_clean.sample(sample_n, random_state=42).copy()
    # Save CSV
    df_sample.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    #  Save Parquet (for LLM pipeline)
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

    print("\nPre-enrichment dataset shape:", df_clean.shape)
    print("Sample dataset shape:", df_sample.shape)

    if do_cleanup:
        cleanup_old_files(BASE_DIR)

    return stats, df_source_stats, df_clean, df_sample


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Finalize IdiomX pre-enrichment dataset.")
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--stats-json", type=str, default=str(DEFAULT_STATS_JSON))
    parser.add_argument("--stats-csv", type=str, default=str(DEFAULT_STATS_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_PRE_ENRICHMENT_CSV))
    parser.add_argument("--output-parquet", type=str, default=str(DEFAULT_PRE_ENRICHMENT_PARQUET))
    parser.add_argument("--sample-csv", type=str, default=str(DEFAULT_SAMPLE_CSV))
    parser.add_argument("--sample-n", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    finalize_pre_enrichment_dataset(
        input_file=Path(args.input_file),
        stats_json=Path(args.stats_json),
        stats_csv=Path(args.stats_csv),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        sample_csv=Path(args.sample_csv),
        sample_n=args.sample_n,
        do_cleanup=True,

    )