#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "generated" / "synthetic_master_merged_v1.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "synthetic_pre_enrichment_for_llm_v1.csv"


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if "idiom_id" not in df.columns or "idiom_canonical" not in df.columns:
        raise ValueError("Input file must contain at least: idiom_id, idiom_canonical")

    # ---------------------------------
    # keep original example as example_raw
    # ---------------------------------
    if "example" in df.columns:
        if "example_raw" not in df.columns:
            df = df.rename(columns={"example": "example_raw"})
        else:
            # if both exist, keep them, but normalize
            df["example_raw"] = df["example_raw"].fillna(df["example"])
            df = df.drop(columns=["example"])
    else:
        df["example_raw"] = ""

    # ---------------------------------
    # create fresh empty example column
    # ---------------------------------
    df["example"] = ""

    # ---------------------------------
    # fill required fields safely
    # ---------------------------------
    if "idiom_surface" not in df.columns:
        df["idiom_surface"] = df["idiom_canonical"]

    df["idiom_canonical"] = df["idiom_canonical"].apply(normalize_text).str.lower()
    df["idiom_surface"] = df["idiom_surface"].fillna(df["idiom_canonical"]).apply(normalize_text)
    df["example_raw"] = df["example_raw"].fillna("").apply(normalize_text)
    df["example"] = df["example"].fillna("")

    if "idiom_canonical_meaning" not in df.columns:
        df["idiom_canonical_meaning"] = ""

    if "source" not in df.columns:
        df["source"] = "llm_generated_inventory"

    if "source_type" not in df.columns:
        df["source_type"] = "synthetic_generation"

    if "pos" not in df.columns:
        df["pos"] = "phrase"

    if "tags" not in df.columns:
        df["tags"] = ""

    if "idiom_confidence" not in df.columns:
        df["idiom_confidence"] = 0.7

    if "source_url" not in df.columns:
        df["source_url"] = "synthetic"

    if "record_origin" not in df.columns:
        df["record_origin"] = "synthetic_generation_pipeline"

    if "license_source" not in df.columns:
        df["license_source"] = "synthetic_llm"

    if "example_language" not in df.columns:
        df["example_language"] = "en"

    if "meaning_language" not in df.columns:
        df["meaning_language"] = "en"

    # ---------------------------------
    # required main enrichment input columns
    # ---------------------------------
    core_cols = [
        "idiom_id",
        "idiom_canonical",
        "idiom_surface",
        "example",
        "example_raw",
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

    # keep extra metadata after core columns
    extra_cols = [c for c in df.columns if c not in core_cols]

    df = df[core_cols + extra_cols].copy()

    # final dedup safety
    df = df.drop_duplicates(subset=["idiom_canonical"]).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved clean pre-enrichment file: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print("\nColumns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:02d}. {col}")


if __name__ == "__main__":
    main()