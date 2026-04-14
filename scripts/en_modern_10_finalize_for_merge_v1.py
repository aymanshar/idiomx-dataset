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


def normalize_text(x) -> str:
    return safe_str(x).lower().strip()


def bool_from_any(x):
    s = safe_str(x).lower()
    if s in {"true", "1"}:
        return True
    if s in {"false", "0"}:
        return False
    return x


def pick_best_row(group: pd.DataFrame) -> pd.Series:
    """
    Pick one best row from a small candidate group.
    Preference:
    - non-empty example
    - longer example
    - non-empty surface
    """
    g = group.copy()

    if "idiom_in_example" not in g.columns:
        g["idiom_in_example"] = ""
    if "idiom_surface" not in g.columns:
        g["idiom_surface"] = ""

    g["_has_example"] = g["idiom_in_example"].apply(lambda x: 1 if safe_str(x) else 0)
    g["_example_len"] = g["idiom_in_example"].apply(lambda x: len(safe_str(x)))
    g["_has_surface"] = g["idiom_surface"].apply(lambda x: 1 if safe_str(x) else 0)

    g = g.sort_values(
        by=["_has_example", "_example_len", "_has_surface"],
        ascending=[False, False, False],
    )

    row = g.iloc[0].copy()
    return row


def finalize_for_merge(input_csv: Path, output_csv: Path):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    rows_before = len(df)

    print(f"Loaded rows: {rows_before}")

    # --------------------------------------------------------
    # 1) Keep only valid idioms again, just to be safe
    # --------------------------------------------------------
    if "is_idiom" in df.columns:
        df = df[df["is_idiom"] == True].copy()

    if "idiom_validity_label" in df.columns:
        df = df[df["idiom_validity_label"] == "valid_idiom"].copy()

    print(f"Rows after valid idiom filter: {len(df)}")

    # --------------------------------------------------------
    # 2) Ensure important columns exist
    # --------------------------------------------------------
    required_cols = [
        "idiom_id",
        "idiom_canonical",
        "row_type",
        "context_type",
        "example_usage_label",
        "expected_label",
        "source_style",
        "idiom_surface",
        "idiom_in_example",
        "idiom_in_example_arabic",
        "idiom_in_example_meaning_en",
        "idiom_in_example_meaning_arabic",
        "explanation_en",
        "explanation_ar",
        "idiom_level_explanation_en",
        "idiom_level_explanation_ar",
        "is_example_idiom",
        "adversarial_type",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # normalize basic text
    for col in required_cols:
        if col == "is_example_idiom":
            continue
        df[col] = df[col].apply(safe_str)

    if "is_example_idiom" in df.columns:
        df["is_example_idiom"] = df["is_example_idiom"].apply(bool_from_any)

    # normalize text fields used by validator
    for col in [
        "idiom_surface",
        "idiom_in_example",
        "idiom_in_example_arabic",
        "idiom_in_example_meaning_en",
        "idiom_in_example_meaning_arabic",
        "explanation_en",
        "explanation_ar",
        "idiom_level_explanation_en",
        "idiom_level_explanation_ar",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)

    # --------------------------------------------------------
    # 3) Identify row families
    # --------------------------------------------------------
    df["row_type"] = df["row_type"].apply(safe_str)

    # main rows
    main_mask = df["row_type"] == "main_example"

    # treat anything adversarial-like as adversarial
    adv_mask = df["row_type"].str.contains("adversarial", case=False, na=False)

    # rows that are neither main nor adversarial are not expected in final merge-ready file
    other_mask = ~(main_mask | adv_mask)
    if other_mask.any():
        print(f"Dropping non-main/non-adversarial rows: {int(other_mask.sum())}")
        df = df[~other_mask].copy()

    # recompute masks after drop
    main_mask = df["row_type"] == "main_example"
    adv_mask = df["row_type"].str.contains("adversarial", case=False, na=False)

    # --------------------------------------------------------
    # 4) Rebuild main examples:
    # Keep at most one idiomatic + one literal per (idiom_id, context_type)
    # --------------------------------------------------------
    df_main = df[main_mask].copy()
    main_kept_rows = []

    if not df_main.empty:
        for (idiom_id, context_type), g in df_main.groupby(["idiom_id", "context_type"], sort=False):
            g_idio = g[g["example_usage_label"] == "idiomatic"].copy()
            g_lit = g[g["example_usage_label"] == "literal"].copy()

            # Keep one best idiomatic if available
            if len(g_idio) >= 1:
                main_kept_rows.append(pick_best_row(g_idio))

            # Keep one best literal if available
            if len(g_lit) >= 1:
                main_kept_rows.append(pick_best_row(g_lit))

        if main_kept_rows:
            df_main = pd.DataFrame(main_kept_rows).copy()
        else:
            df_main = df_main.iloc[0:0].copy()

    # --------------------------------------------------------
    # 5) Rebuild adversarial rows:
    # Keep at most two per idiom and rename to _1 / _2
    # --------------------------------------------------------
    df_adv = df[adv_mask].copy()
    adv_kept_rows = []

    if not df_adv.empty:
        sort_cols = ["idiom_id"]
        for extra in ["adversarial_type", "idiom_in_example"]:
            if extra in df_adv.columns:
                sort_cols.append(extra)

        df_adv = df_adv.sort_values(by=sort_cols).copy()

        for idiom_id, g in df_adv.groupby("idiom_id", sort=False):
            # keep best two adversarial rows max
            g = g.copy()
            g["_has_example"] = g["idiom_in_example"].apply(lambda x: 1 if safe_str(x) else 0)
            g["_example_len"] = g["idiom_in_example"].apply(lambda x: len(safe_str(x)))
            g = g.sort_values(
                by=["_has_example", "_example_len", "adversarial_type"],
                ascending=[False, False, True]
            )
            g = g.head(2).copy()

            # rename row types to expected final shape
            g = g.reset_index(drop=True)
            for i in range(len(g)):
                row = g.iloc[i].copy()
                row["row_type"] = f"adversarial_example_{i+1}"
                adv_kept_rows.append(row)

        if adv_kept_rows:
            df_adv = pd.DataFrame(adv_kept_rows).copy()
        else:
            df_adv = df_adv.iloc[0:0].copy()

        for col in ["_has_example", "_example_len"]:
            if col in df_adv.columns:
                df_adv.drop(columns=[col], inplace=True, errors="ignore")

    # --------------------------------------------------------
    # 6) Combine back
    # --------------------------------------------------------
    df = pd.concat([df_main, df_adv], ignore_index=True)

    # --------------------------------------------------------
    # 7) Rebuild flags and labels
    # --------------------------------------------------------
    df["is_adversarial_example"] = df["row_type"].astype(str).str.startswith("adversarial_example")

    # Main rows: expected_label must equal example_usage_label
    main_mask = df["row_type"] == "main_example"
    df.loc[main_mask, "expected_label"] = df.loc[main_mask, "example_usage_label"]

    # Adversarial rows: expected_label must be valid
    adv_mask = df["row_type"].astype(str).str.startswith("adversarial_example")

    valid_adv_labels = {"idiomatic", "literal", "borderline"}
    df.loc[adv_mask, "expected_label"] = df.loc[adv_mask, "expected_label"].apply(safe_str)
    df.loc[adv_mask & (~df["expected_label"].isin(valid_adv_labels)), "expected_label"] = "borderline"

    # adversarial example_usage_label must also be valid for validator
    df.loc[adv_mask, "example_usage_label"] = df.loc[adv_mask, "example_usage_label"].apply(safe_str)
    df.loc[adv_mask & (~df["example_usage_label"].isin(valid_adv_labels)), "example_usage_label"] = "borderline"

    # is_example_idiom cleanup for main rows
    if "is_example_idiom" in df.columns:
        df.loc[(main_mask) & (df["example_usage_label"] == "idiomatic"), "is_example_idiom"] = True
        df.loc[(main_mask) & (df["example_usage_label"] == "literal"), "is_example_idiom"] = False

    # --------------------------------------------------------
    # 8) Rebuild minimal_pair_id correctly for main examples
    # One pair per (idiom_id, context_type) if both labels exist
    # --------------------------------------------------------
    df["minimal_pair_id"] = np.nan

    if main_mask.any():
        pair_assignments = {}

        pair_counter = 0
        main_df = df[main_mask].copy()

        for (idiom_id, context_type), g in main_df.groupby(["idiom_id", "context_type"], sort=False):
            g_idio = g[g["example_usage_label"] == "idiomatic"]
            g_lit = g[g["example_usage_label"] == "literal"]

            if len(g_idio) >= 1 and len(g_lit) >= 1:
                pair_counter += 1
                pair_id = f"{idiom_id}_pair_{pair_counter}"

                # assign pair to both rows in this context
                for idx in g.index:
                    pair_assignments[idx] = pair_id

        if pair_assignments:
            df.loc[list(pair_assignments.keys()), "minimal_pair_id"] = [
                pair_assignments[idx] for idx in pair_assignments.keys()
            ]

    # --------------------------------------------------------
    # 9) Rebuild paraphrase group id
    # --------------------------------------------------------
    df["paraphrase_group_id"] = df["idiom_id"].astype(str)

    # --------------------------------------------------------
    # 10) Light normalization to reduce surface mismatch noise
    # --------------------------------------------------------
    for col in ["idiom_surface", "idiom_in_example"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)

    # --------------------------------------------------------
    # 11) Sort cleanly
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
    # 12) Save
    # --------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved final-ready modern dataset to: {output_csv}")
    print(f"Rows before: {rows_before}")
    print(f"Rows after: {len(df)}")

    print("\nRow type distribution:")
    print(df["row_type"].value_counts(dropna=False))

    print("\nAdversarial flag distribution:")
    print(df["is_adversarial_example"].value_counts(dropna=False))

    if "example_usage_label" in df.columns:
        print("\nExample usage label distribution:")
        print(df["example_usage_label"].value_counts(dropna=False))

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