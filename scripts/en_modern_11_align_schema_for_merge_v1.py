#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_final_ready_v1.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_merge_aligned_v1.csv"


def align_schema_for_merge(input_csv: Path, output_csv: Path):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    # --------------------------------------------------------
    # Drop temporary helper columns
    # --------------------------------------------------------
    drop_cols = ["_has_example", "_example_len", "_has_surface"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --------------------------------------------------------
    # Add missing core IdiomX columns
    # --------------------------------------------------------
    core_main_cols = [
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
        "idiom_canonical_meaning_arabic",
        "is_idiom",
        "ambiguity_flag",
        "idiom_compositionality_level",
        "idiom_register",
        "idiom_domain",
        "learner_difficulty",
        "idiom_in_example",
        "idiom_in_example_arabic",
        "idiom_in_example_meaning_en",
        "idiom_in_example_meaning_arabic",
        "is_example_idiom",
        "example_usage_label",
        "is_generated_example",
        "enrichment_model",
        "enrichment_version",
        "validation_status",
        "context_type",
        "source_style",
        "hard_negative_idioms",
        "meaning_paraphrases_en",
        "meaning_paraphrases_ar",
        "idiom_level_explanation_en",
        "idiom_level_explanation_ar",
        "explanation_en",
        "explanation_ar",
        "minimal_pair_id",
        "paraphrase_group_id",
        "is_adversarial_example",
        "adversarial_type",
        "expected_label",
        "row_type",
    ]

    for col in core_main_cols:
        if col not in df.columns:
            if col == "is_generated_example":
                df[col] = True
            elif col == "validation_status":
                df[col] = "valid"
            elif col == "example_language":
                df[col] = "en"
            elif col == "meaning_language":
                df[col] = "en"
            else:
                df[col] = np.nan

    # --------------------------------------------------------
    # Keep modern extension columns too
    # --------------------------------------------------------
    modern_extra_cols = [
        "idiom_validity_label",
        "idiom_canonical_meaning_french",
        "slang_strength",
        "regionality",
        "offensive_flag",
        "idiom_level_explanation_fr",
        "meaning_paraphrases_fr",
        "idiom_in_example_french",
        "idiom_in_example_meaning_french",
        "explanation_fr",
    ]

    for col in modern_extra_cols:
        if col not in df.columns:
            df[col] = np.nan

    # optional provenance column
    if "source_dataset" not in df.columns:
        df["source_dataset"] = "modern_extension"

    # --------------------------------------------------------
    # Final order:
    # main schema first, then modern extension cols
    # --------------------------------------------------------
    ordered_cols = core_main_cols + [c for c in modern_extra_cols if c in df.columns] + ["source_dataset"]

    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved schema-aligned modern dataset to: {output_csv}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Align modern IdiomX dataset schema before merge.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    align_schema_for_merge(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
    )