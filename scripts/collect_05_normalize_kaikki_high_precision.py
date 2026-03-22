"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Normalize the high-precision Kaikki idiom dataset.
This script converts the high-precision Wiktionary idiom subset into the
standardized source schema used across all IdiomX lexical resources.

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""
import pandas as pd
from pathlib import Path
import sys
import re

# Project directories
BASE_DIR = Path("..")

DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"

# Make sure directories exist
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Raw dataset file
INPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_high_precision.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_kaikki_normalized.csv"


def normalize_high_precision_idioms(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Normalize the high-precision Kaikki idiom dataset into the unified source schema.

    Maps the Kaikki-specific columns into the standard IdiomX source format
    used for later merging with other idiom resources.
    """
    # Load the high-precision Kaikki idiom dataset from the previous pipeline stage
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Map Kaikki columns into the common IdiomX source schema
    out = pd.DataFrame({
        "idiom": df["idiom"].fillna("").astype(str).str.strip(),
        "meaning_en": df["meaning"].fillna("").astype(str).str.strip(),
        "example": df["example"].fillna("").astype(str).str.strip(),
        "source": "kaikki_wiktionary",
        "source_type": "dictionary",
        "pos": df["pos"].fillna("").astype(str).str.strip(),
        "tags": df["tags"].fillna("").astype(str).str.strip(),
        "idiom_confidence": "high",
        "source_url": ""
    })
    # Remove duplicate idiom-meaning pairs to keep the source dataset compact
    out = out.drop_duplicates(subset=["idiom", "meaning_en"]).reset_index(drop=True)
    # Save the normalized Kaikki idiom source dataset for later merging
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(out))
    return out

def main():
    """
    Run the Kaikki normalization pipeline using the default input and output paths.
    """
    normalize_high_precision_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()