#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# =========================
# FULL MODE (synth)
# =========================
DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_synth_enriched_full_v1.csv"
DEFAULT_FULL_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_synth_enriched_valid_only_v1.csv"

# =========================
# SAMPLE MODE (synth)
# =========================
DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_synth_enriched_sample_v1.csv"
DEFAULT_SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_synth_enriched_valid_only_sample_v1.csv"


def get_mode_paths(use_sample: bool = False):
    if use_sample:
        return DEFAULT_SAMPLE_INPUT_CSV, DEFAULT_SAMPLE_OUTPUT_CSV
    return DEFAULT_FULL_INPUT_CSV, DEFAULT_FULL_OUTPUT_CSV


def filter_valid_idioms(input_csv: Path, output_csv: Path):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    # =========================
    # CORE FILTER
    # =========================
    df = df[
        (df["is_idiom"] == True) &
        (df["idiom_validity_label"] == "valid_idiom")
    ].copy()

    # =========================
    # CLEAN (recommended)
    # =========================
    df.fillna("", inplace=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Saved valid-only dataset:", output_csv)
    print("Rows:", len(df))

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Filter synth enriched dataset to valid idioms only.")

    parser.add_argument("--sample", action="store_true", help="Use sample mode paths")
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    default_input, default_output = get_mode_paths(use_sample=args.sample)

    input_csv = Path(args.input_csv) if args.input_csv else default_input
    output_csv = Path(args.output_csv) if args.output_csv else default_output

    filter_valid_idioms(
        input_csv=input_csv,
        output_csv=output_csv,
    )