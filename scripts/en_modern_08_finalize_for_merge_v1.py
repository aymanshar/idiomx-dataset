#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_valid_only_v1.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_final_ready_v1.csv"


def safe_str(x) -> str:
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

    # --------------------------------------------------------
    # 1) Keep only valid idioms again, just to be safe
    # --------------------------------------------------------
    if "is_idiom" in df.columns:
        df = df[df["is_idiom"] == True].copy()

    if "idiom_validity_label" in df.columns:
        df = df[df["idiom_validity_label"] == "valid_idiom"].copy()

    # --------------------------------------------------------
    # 2) Normalize helper columns
    # --------------------------------------------------------
    for col in [
        "idiom_id",
        "idiom_canonical",
        "row_type",
        "context_type",
        "example_usage_label",
        "expected_label",
        "source_style",
        "idiom_surface",
        "idiom_in_example",
        "is_example_idiom",
        "adversarial_type",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(safe_str)

    if "is_example_idiom" in df.columns:
        df["is_example_idiom"] = df["is_example_idiom"].apply(bool_from_any)

    # --------------------------------------------------------
    # 3) Mark adversarial rows cleanly
    # --------------------------------------------------------
    df["is_adversarial_example"] = df["row_type"].astype(str).str.lower().eq("adversarial_example")

    # --------------------------------------------------------
    # 4) Split adversarial rows into adversarial_example_1 / _2
    # --------------------------------------------------------
    adv_mask = df["is_adversarial_example"] == True

    if adv_mask.any():
        df_adv = df[adv_mask].copy()
        df_non_adv = df[~adv_mask].copy()

        # deterministic ordering inside each idiom
        sort_cols = ["idiom_id"]
        for extra in ["adversarial_type", "idiom_in_example"]:
            if extra in df_adv.columns:
                sort_cols.append(extra)

        df_adv = df_adv.sort_values(by=sort_cols).copy()
        df_adv["_adv_rank"] = df_adv.groupby("idiom_id").cumcount() + 1

        # keep only first 2 if somehow more exist
        df_adv = df_adv[df_adv["_adv_rank"] <= 2].copy()
        df_adv["row_type"] = df_adv["_adv_rank"].map({
            1: "adversarial_example_1",
            2: "adversarial_example_2"
        })
        df_adv.drop(columns=["_adv_rank"], inplace=True, errors="ignore")

        df = pd.concat([df_non_adv, df_adv], ignore_index=True)
    else:
        df["row_type"] = df["row_type"].replace({"adversarial_example": "adversarial_example_1"})

    # --------------------------------------------------------
    # 5) Rebuild expected_label logic
    # --------------------------------------------------------
    # Main rows: expected_label should match example_usage_label
    main_mask = df["row_type"] == "main_example"
    df.loc[main_mask, "expected_label"] = df.loc[main_mask, "example_usage_label"]

    # Adversarial rows: keep existing expected_label if present
    # otherwise fallback to example_usage_label
    adv_mask = df["row_type"].astype(str).str.startswith("adversarial_example")
    adv_missing = adv_mask & (df["expected_label"].apply(safe_str) == "")
    df.loc[adv_missing, "expected_label"] = df.loc[adv_missing, "example_usage_label"]

    # --------------------------------------------------------
    # 6) Rebuild minimal_pair_id correctly for main examples
    # --------------------------------------------------------
    df["minimal_pair_id"] = np.nan

    if main_mask.any():
        df_main = df[main_mask].copy()

        # deterministic order
        sort_cols = ["idiom_id", "context_type", "example_usage_label", "idiom_in_example"]
        existing_sort_cols = [c for c in sort_cols if c in df_main.columns]
        df_main = df_main.sort_values(by=existing_sort_cols).copy()

        pair_counter = 0
        pair_ids = {}

        for (idiom_id, context_type), g in df_main.groupby(["idiom_id", "context_type"], sort=False):
            g_idio = g[g["example_usage_label"] == "idiomatic"]
            g_lit = g[g["example_usage_label"] == "literal"]

            if len(g_idio) >= 1 and len(g_lit) >= 1:
                # take first idiomatic + first literal as one pair
                pair_counter += 1
                pair_id = f"{idiom_id}_pair_{pair_counter}"

                idx_idio = g_idio.index[0]
                idx_lit = g_lit.index[0]

                pair_ids[idx_idio] = pair_id
                pair_ids[idx_lit] = pair_id

                # if extra rows exist in same context, pair them to same group id
                for idx in g.index:
                    if idx not in pair_ids:
                        pair_ids[idx] = pair_id
            else:
                # fallback: still assign a context-based pair id
                pair_counter += 1
                pair_id = f"{idiom_id}_pair_{pair_counter}"
                for idx in g.index:
                    pair_ids[idx] = pair_id

        df.loc[list(pair_ids.keys()), "minimal_pair_id"] = [pair_ids[i] for i in pair_ids.keys()]

    # --------------------------------------------------------
    # 7) Rebuild paraphrase_group_id by idiom
    # --------------------------------------------------------
    df["paraphrase_group_id"] = df["idiom_id"].astype(str)

    # --------------------------------------------------------
    # 8) Normalize row ordering
    # --------------------------------------------------------
    row_type_order = {
        "main_example": 1,
        "adversarial_example_1": 2,
        "adversarial_example_2": 3,
    }
    df["_row_type_order"] = df["row_type"].map(row_type_order).fillna(99)

    sort_cols = ["idiom_id", "_row_type_order", "context_type", "example_usage_label", "idiom_in_example"]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(by=existing_sort_cols).copy()
    df.drop(columns=["_row_type_order"], inplace=True, errors="ignore")

    # --------------------------------------------------------
    # 9) Save
    # --------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved final-ready modern dataset to: {output_csv}")
    print(f"Rows before: {rows_before}")
    print(f"Rows after: {len(df)}")
    print("\nRow type distribution:")
    print(df["row_type"].value_counts(dropna=False))

    if "is_adversarial_example" in df.columns:
        print("\nAdversarial flag distribution:")
        print(df["is_adversarial_example"].value_counts(dropna=False))

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize modern enriched dataset for merge with IdiomX.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize_for_merge(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
    )