"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Merge the Stage 4 idiom dataset with the normalized LIdioms dataset.
This script expands the idiom inventory by adding semantically structured
LIdioms entries to the previously merged dataset. The result is the Stage 5
dataset, which is later refined by global idiom filtering.

Notes:
# LIdioms contributes a small but semantically structured set of idiomatic expressions.
# Unlike WordNet, this source is designed specifically around idiomatic knowledge,
# which improves source diversity in the merged dataset.

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAIN_DATASET = DATA_PROCESS_DIR / "idioms_dataset_stage4.csv"
DEFAULT_LIDIOMS_DATASET = DATA_PROCESS_DIR / "idioms_source_lidioms_normalized.csv"
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage5.csv"

# Shared IdiomX schema used to align heterogeneous lexical resources
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


def merge_main_with_lidioms(
    main_dataset=DEFAULT_MAIN_DATASET,
    lidioms_dataset=DEFAULT_LIDIOMS_DATASET,
    output_file=DEFAULT_OUTPUT_FILE,
    ):
    """
    Merge the Stage 4 idiom dataset with the normalized LIdioms dataset.

    Aligns schemas, normalizes text fields, removes duplicate idiom-meaning pairs,
    and saves the merged Stage 5 dataset for downstream filtering.
    """

    main_dataset = Path(main_dataset)
    lidioms_dataset = Path(lidioms_dataset)
    output_file = Path(output_file)

    if not main_dataset.exists():
        raise FileNotFoundError(f"Main dataset not found: {main_dataset}")

    if not lidioms_dataset.exists():
        raise FileNotFoundError(f"LIdioms dataset not found: {lidioms_dataset}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load the current merged idiom dataset and the normalized LIdioms dataset
    df_main = pd.read_csv(main_dataset, encoding="utf-8-sig")
    df_lidioms = pd.read_csv(lidioms_dataset, encoding="utf-8-sig")

    # Align schema
    # Ensure both datasets follow the same standardized schema
    for col in STANDARD_COLUMNS:
        if col not in df_main.columns:
            df_main[col] = ""
        if col not in df_lidioms.columns:
            df_lidioms[col] = ""

    df_main = df_main[STANDARD_COLUMNS]
    df_lidioms = df_lidioms[STANDARD_COLUMNS]

    # Merge the main idiom inventory with semantically curated LIdioms entries
    df_merged = pd.concat([df_main, df_lidioms], ignore_index=True)

    # Normalize text fields to improve consistency and deduplication
    for col in STANDARD_COLUMNS:
        df_merged[col] = df_merged[col].fillna("").astype(str).str.strip()

    # Deduplicate by idiom + meaning
    # Build a stable deduplication key from idiom text and English meaning
    df_merged["dedup_key"] = (
        df_merged["idiom"].str.lower().str.strip()
        + " || " +
        df_merged["meaning_en"].str.lower().str.strip()
    )

    df_merged = (
        df_merged
        .drop_duplicates(subset=["dedup_key"])
        .drop(columns=["dedup_key"])
        .reset_index(drop=True)
    )

    # Save the Stage 5 merged dataset for the global filtering stage
    df_merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df_merged))
    print("Unique idioms:", df_merged["idiom"].nunique())
    print("\nSource distribution:")
    print(df_merged["source"].value_counts())

    return df_merged


def main():
    """
    Run the Stage 4 + LIdioms merge pipeline using the default project paths.
    """
    df = merge_main_with_lidioms()
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()