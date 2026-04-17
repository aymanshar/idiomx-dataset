#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

# =========================
# FULL MODE (synth)
# =========================
DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_synth_enriched_valid_only_v1.csv"
DEFAULT_FULL_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_synth_enriched_final_ready_v1.csv"

# =========================
# SAMPLE MODE (synth)
# =========================
DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_synth_enriched_valid_only_sample_v1.csv"
DEFAULT_SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_synth_enriched_final_ready_sample_v1.csv"


def get_mode_paths(use_sample=False):
    if use_sample:
        return DEFAULT_SAMPLE_INPUT_CSV, DEFAULT_SAMPLE_OUTPUT_CSV
    return DEFAULT_FULL_INPUT_CSV, DEFAULT_FULL_OUTPUT_CSV


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def bool_from_any(x):
    s = safe_str(x).lower()
    if s in {"true", "1"}:
        return True
    if s in {"false", "0"}:
        return False
    return x


def finalize_for_merge(input_csv: Path, output_csv: Path):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    rows_before = len(df)

    # ---------------------------
    # 1) safety filter
    # ---------------------------
    if "is_idiom" in df.columns:
        df = df[df["is_idiom"] == True].copy()

    if "idiom_validity_label" in df.columns:
        df = df[df["idiom_validity_label"] == "valid_idiom"].copy()

    # ---------------------------
    # 2) normalize
    # ---------------------------
    cols = [
        "idiom_id","idiom_canonical","row_type","context_type",
        "example_usage_label","expected_label","source_style",
        "idiom_surface","idiom_in_example","is_example_idiom",
        "adversarial_type"
    ]

    for col in cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(safe_str)

    df.fillna("", inplace=True)

    if "is_example_idiom" in df.columns:
        df["is_example_idiom"] = df["is_example_idiom"].apply(bool_from_any)

    # ---------------------------
    # 3) adversarial flag
    # ---------------------------
    df["is_adversarial_example"] = df["row_type"].str.lower().eq("adversarial_example")

    # ---------------------------
    # 4) split adversarial
    # ---------------------------
    adv_mask = df["is_adversarial_example"]

    if adv_mask.any():
        df_adv = df[adv_mask].copy()
        df_non_adv = df[~adv_mask].copy()

        df_adv = df_adv.sort_values(["idiom_id", "idiom_in_example"]).copy()
        df_adv["_rank"] = df_adv.groupby("idiom_id").cumcount() + 1
        df_adv = df_adv[df_adv["_rank"] <= 2]

        df_adv["row_type"] = df_adv["_rank"].map({
            1: "adversarial_example_1",
            2: "adversarial_example_2"
        })

        df_adv.drop(columns=["_rank"], inplace=True)

        df = pd.concat([df_non_adv, df_adv], ignore_index=True)
    else:
        df["row_type"] = df["row_type"].replace({"adversarial_example": "adversarial_example_1"})

    # ---------------------------
    # 5) expected_label fix
    # ---------------------------
    main_mask = df["row_type"] == "main_example"
    df.loc[main_mask, "expected_label"] = df.loc[main_mask, "example_usage_label"]

    adv_mask = df["row_type"].str.startswith("adversarial_example")
    adv_missing = adv_mask & (df["expected_label"] == "")
    df.loc[adv_missing, "expected_label"] = df.loc[adv_missing, "example_usage_label"]

    # ---------------------------
    # 6) minimal pairs
    # ---------------------------
    df["minimal_pair_id"] = ""

    counter = 0
    for (idiom, ctx), g in df[df["row_type"] == "main_example"].groupby(["idiom_id","context_type"]):
        counter += 1
        pair_id = f"{idiom}_pair_{counter}"
        df.loc[g.index, "minimal_pair_id"] = pair_id

    # ---------------------------
    # 7) paraphrase group
    # ---------------------------
    df["paraphrase_group_id"] = df["idiom_id"]

    # ---------------------------
    # 8) ordering
    # ---------------------------
    order = {"main_example":1,"adversarial_example_1":2,"adversarial_example_2":3}
    df["_order"] = df["row_type"].map(order).fillna(99)

    df = df.sort_values(["idiom_id","_order","context_type","idiom_in_example"])
    df.drop(columns=["_order"], inplace=True)

    # ---------------------------
    # 9) save
    # ---------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Saved FINAL dataset:", output_csv)
    print("Rows before:", rows_before)
    print("Rows after:", len(df))
    print("\nRow types:\n", df["row_type"].value_counts())

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    default_in, default_out = get_mode_paths(args.sample)

    input_csv = Path(args.input_csv) if args.input_csv else default_in
    output_csv = Path(args.output_csv) if args.output_csv else default_out

    finalize_for_merge(input_csv, output_csv)