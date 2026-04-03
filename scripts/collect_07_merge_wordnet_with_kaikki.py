"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Merge the kaikki idiom dataset with the normalized WordNet dataset.
This script expands the idiom inventory by adding WordNet multi-word expressions
to the previously merged idiom dataset. The result is the merged kaikki and wordnet dataset,
which is later refined by global idiom filtering.

Notes:
# WordNet contributes lexical multi-word expressions rather than explicitly labeled idioms.
# These entries are merged as medium-confidence idiom candidates and refined later
# by the global idiom filtering stage.

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

# Current main dataset after previous merging steps
DEFAULT_KAIKKI_FILE = DATA_PROCESS_DIR / "idioms_source_kaikki.csv"

# Newly extracted WordNet normalized dataset
DEFAULT_WORDNET_FILE = DATA_PROCESS_DIR / "idioms_source_wordnet.csv"

# Output merged dataset
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_merged_kaikki_wordnet.csv"

# Any missing columns will be added automatically to keep

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


def merge_kaikki_with_wordnet(
    kaikki_file=DEFAULT_KAIKKI_FILE,
    wordnet_file=DEFAULT_WORDNET_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
):
    """
    Merge the kaikki idiom dataset with the normalized WordNet dataset.

    Aligns schemas, normalizes text fields, removes duplicate idiom-meaning pairs,
    and saves the merged kaikki_wordnet dataset for downstream filtering.

    Parameters
    ----------
    kaikki_file : path-like
        Path to the current merged idiom dataset.
    wordnet_file : path-like
        Path to the normalized WordNet source dataset.
    output_file : path-like
        Path where the merged output CSV will be saved.

    Returns
    -------
    pandas.DataFrame
        The merged dataframe after schema alignment and deduplication.
    """

    # Convert all incoming paths to Path objects
    kaikki_file = Path(kaikki_file)
    wordnet_file = Path(wordnet_file)
    output_file = Path(output_file)

    # Validate input files before processing
    if not kaikki_file.exists():
        raise FileNotFoundError(f"kaikki dataset not found: {kaikki_file}")

    if not wordnet_file.exists():
        raise FileNotFoundError(f"WordNet dataset not found: {wordnet_file}")

    # Ensure the output folder exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read both datasets
    # Load the current merged idiom dataset and the WordNet source dataset
    df_kaikki= pd.read_csv(kaikki_file, encoding="utf-8-sig")
    df_wordnet = pd.read_csv(wordnet_file, encoding="utf-8-sig")

    # Add missing standard columns if needed
    # Ensure both datasets share the same schema before concatenation
    for col in STANDARD_COLUMNS:
        if col not in df_kaikki.columns:
            df_kaikki[col] = ""
        if col not in df_wordnet.columns:
            df_wordnet[col] = ""

    # Keep only the standard columns and preserve column order
    df_kaikki = df_kaikki[STANDARD_COLUMNS]
    df_wordnet = df_wordnet[STANDARD_COLUMNS]

    # Merge curated idioms with WordNet-derived multi-word expressions
    df_merged = pd.concat([df_kaikki, df_wordnet], ignore_index=True)

    # Normalize text fields to improve matching and deduplication consistency
    for col in STANDARD_COLUMNS:
        df_merged[col] = df_merged[col].fillna("").astype(str).str.strip()

    # Build a stable deduplication key from idiom text and English meaning
    df_merged["dedup_key"] = (
        df_merged["idiom"].str.lower().str.strip()
        + " || " +
        df_merged["meaning_en"].str.lower().str.strip()
    )

    # Remove exact duplicate idiom-meaning pairs
    df_merged = (
        df_merged
        .drop_duplicates(subset=["dedup_key"])
        .drop(columns=["dedup_key"])
        .reset_index(drop=True)
    )

    # Save the kaikki and wordnet merged dataset for the next filtering stage
    df_merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    # Print a quick summary
    print("Saved:", output_file)
    print("Rows:", len(df_merged))
    print("Unique idioms:", df_merged["idiom"].nunique())
    print("\nSource distribution:")
    print(df_merged["source"].value_counts())

    return df_merged


def main():
    """
    Run the kaikki + WordNet merge pipeline using the default project paths.
    """
    df_merged = merge_kaikki_with_wordnet()
    print("\nPreview:")
    print(df_merged.head())


if __name__ == "__main__":
    main()