#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_full_v1.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_valid_only_v1.csv"


def filter_valid_idioms(input_csv: Path, output_csv: Path):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    df = df[
        (df["is_idiom"] == True) &
        (df["idiom_validity_label"] == "valid_idiom")
    ].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Saved valid-only dataset:", output_csv)
    print("Rows:", len(df))

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Filter modern enriched dataset to valid idioms only.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_valid_idioms(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
    )