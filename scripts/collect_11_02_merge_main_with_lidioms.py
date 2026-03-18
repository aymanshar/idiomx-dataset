from pathlib import Path
import pandas as pd

# ============================================================
# Default project paths
# ============================================================

BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAIN_DATASET = DATA_PROCESS_DIR / "idioms_dataset_stage4.csv"
DEFAULT_LIDIOMS_DATASET = DATA_PROCESS_DIR / "idioms_source_lidioms_normalized.csv"
DEFAULT_OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage5.csv"

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
    Merge the current main idiom dataset with the normalized LIdioms dataset.
    """

    main_dataset = Path(main_dataset)
    lidioms_dataset = Path(lidioms_dataset)
    output_file = Path(output_file)

    if not main_dataset.exists():
        raise FileNotFoundError(f"Main dataset not found: {main_dataset}")

    if not lidioms_dataset.exists():
        raise FileNotFoundError(f"LIdioms dataset not found: {lidioms_dataset}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_main = pd.read_csv(main_dataset, encoding="utf-8-sig")
    df_lidioms = pd.read_csv(lidioms_dataset, encoding="utf-8-sig")

    # Align schema
    for col in STANDARD_COLUMNS:
        if col not in df_main.columns:
            df_main[col] = ""
        if col not in df_lidioms.columns:
            df_lidioms[col] = ""

    df_main = df_main[STANDARD_COLUMNS]
    df_lidioms = df_lidioms[STANDARD_COLUMNS]

    # Merge
    df_merged = pd.concat([df_main, df_lidioms], ignore_index=True)

    # Normalize text
    for col in STANDARD_COLUMNS:
        df_merged[col] = df_merged[col].fillna("").astype(str).str.strip()

    # Deduplicate by idiom + meaning
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

    df_merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df_merged))
    print("Unique idioms:", df_merged["idiom"].nunique())
    print("\nSource distribution:")
    print(df_merged["source"].value_counts())

    return df_merged


def main():
    df = merge_main_with_lidioms()
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()