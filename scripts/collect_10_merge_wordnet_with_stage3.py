from pathlib import Path
import pandas as pd

# ============================================================
# Default project paths
# These defaults allow the script to run directly from CMD,
# while still allowing custom paths from a notebook.
# ============================================================

BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Current main dataset after previous merging steps
DEFAULT_STAGE3_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage3.csv"

# Newly extracted WordNet normalized dataset
DEFAULT_WORDNET_FILE = DATA_PROCESS_DIR / "idioms_source_wordnet_normalized.csv"

# Output merged dataset
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage4.csv"

# ============================================================
# Standard schema
# Any missing columns will be added automatically to keep
# the merged dataset consistent across all sources.
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


def merge_stage3_with_wordnet(
    stage3_file=DEFAULT_STAGE3_FILE,
    wordnet_file=DEFAULT_WORDNET_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
):
    """
    Merge the current master idiom dataset (stage3) with the
    normalized WordNet multi-word expressions dataset.

    Parameters
    ----------
    stage3_file : path-like
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
    stage3_file = Path(stage3_file)
    wordnet_file = Path(wordnet_file)
    output_file = Path(output_file)

    # Validate input files before processing
    if not stage3_file.exists():
        raise FileNotFoundError(f"Stage3 dataset not found: {stage3_file}")

    if not wordnet_file.exists():
        raise FileNotFoundError(f"WordNet dataset not found: {wordnet_file}")

    # Ensure the output folder exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read both datasets
    df_stage3 = pd.read_csv(stage3_file, encoding="utf-8-sig")
    df_wordnet = pd.read_csv(wordnet_file, encoding="utf-8-sig")

    # Add missing standard columns if needed
    for col in STANDARD_COLUMNS:
        if col not in df_stage3.columns:
            df_stage3[col] = ""
        if col not in df_wordnet.columns:
            df_wordnet[col] = ""

    # Keep only the standard columns and preserve column order
    df_stage3 = df_stage3[STANDARD_COLUMNS]
    df_wordnet = df_wordnet[STANDARD_COLUMNS]

    # Concatenate the datasets
    df_merged = pd.concat([df_stage3, df_wordnet], ignore_index=True)

    # Normalize text values to reduce duplicate mismatches
    for col in STANDARD_COLUMNS:
        df_merged[col] = df_merged[col].fillna("").astype(str).str.strip()

    # Create a deduplication key using idiom + meaning
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

    # Save the merged dataset
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
    Command-line entry point using the default project paths.
    """
    df_merged = merge_stage3_with_wordnet()
    print("\nPreview:")
    print(df_merged.head())


if __name__ == "__main__":
    main()