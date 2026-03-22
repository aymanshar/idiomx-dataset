from pathlib import Path
import pandas as pd
import json
import re

"""
Compute summary statistics for the final IdiomX dataset.

This script analyzes dataset size, lexical diversity, token-length patterns,
coverage of meanings/examples, and source distributions, then exports
the results as JSON and CSV files for reporting and reproducibility.
"""

# NOTE:
# This stage provides the descriptive statistics reported in the paper,
# including dataset size, lexical diversity, coverage, and source composition.

# paths

BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage5_high_precision.csv"
OUTPUT_JSON = DATA_PROCESS_DIR / "idioms_dataset_stage5_statistics.json"
OUTPUT_CSV = DATA_PROCESS_DIR / "idioms_dataset_stage5_source_distribution.csv"


def norm(x):
    """
    Normalize a value by converting null-like values to an empty string
    and trimming surrounding whitespace.
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def token_count(text):
    """
    Count the number of whitespace-separated tokens in a text field.
    """
    return len(norm(text).split())


def build_dataset_statistics(
    input_file=INPUT_FILE,
    output_json=OUTPUT_JSON,
    output_csv=OUTPUT_CSV,
    ):
    """
    Compute summary statistics for the final idiom dataset.

    Generates overall counts, lexical coverage metrics, token-length statistics,
    and source distribution summaries, then saves them to JSON and CSV files.
    """

    input_file = Path(input_file)
    output_json = Path(output_json)
    output_csv = Path(output_csv)

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load the final high-precision idiom dataset
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Ensure expected schema fields exist and normalize text values
    for col in [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Compute source-level metadata distributions for dataset analysis
    source_distribution = df["source"].value_counts().sort_index()
    source_type_distribution = df["source_type"].value_counts().sort_index()
    confidence_distribution = df["idiom_confidence"].value_counts().sort_index()

    # Build core dataset statistics used in the notebook, paper, and README
    stats = {
        "rows_total": int(len(df)),
        "unique_idioms": int(df["idiom"].nunique()),
        "unique_meanings": int(df["meaning_en"].nunique()),
        "rows_with_meaning": int((df["meaning_en"].str.len() > 0).sum()),
        "rows_with_example": int((df["example"].str.len() > 0).sum()),
        "rows_without_meaning": int((df["meaning_en"].str.len() == 0).sum()),
        "rows_without_example": int((df["example"].str.len() == 0).sum()),
        "avg_idiom_tokens": round(df["idiom"].apply(token_count).mean(), 2) if len(df) else 0.0,
        "min_idiom_tokens": int(df["idiom"].apply(token_count).min()) if len(df) else 0,
        "max_idiom_tokens": int(df["idiom"].apply(token_count).max()) if len(df) else 0,
        "source_distribution": {k: int(v) for k, v in source_distribution.items()},
        "source_type_distribution": {k: int(v) for k, v in source_type_distribution.items()},
        "idiom_confidence_distribution": {k: int(v) for k, v in confidence_distribution.items()},
    }

    # Convert source distribution into a tabular format for easy inspection
    source_distribution_df = (
        source_distribution.rename_axis("source")
        .reset_index(name="count")
        .sort_values(["count", "source"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Save machine-readable statistics summary
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save source distribution table for reporting and visualization
    source_distribution_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Saved JSON stats:", output_json)
    print("Saved source distribution CSV:", output_csv)
    print("\nSummary:")
    print("Rows total:", stats["rows_total"])
    print("Unique idioms:", stats["unique_idioms"])
    print("Rows with meaning:", stats["rows_with_meaning"])
    print("Rows with example:", stats["rows_with_example"])
    print("Average idiom tokens:", stats["avg_idiom_tokens"])
    print("\nSource distribution:")
    print(source_distribution)

    return stats, source_distribution_df


def main():
    """
    Run the dataset statistics pipeline using the default dataset paths.
    """
    stats, source_df = build_dataset_statistics()
    print("\nTop sources preview:")
    print(source_df.head())


if __name__ == "__main__":
    main()