#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_GENERATED = BASE_DIR / "data" / "generated"
ACCEPTED_ROUNDS_DIR = DATA_GENERATED / "accepted_rounds"
DATA_FINAL = BASE_DIR / "data" / "final" / "publication"

MAIN_DATASET_PATH = DATA_FINAL / "idiomx_extended_full.parquet"

OUTPUT_MERGED = DATA_GENERATED / "synthetic_master_merged_v1.csv"
OUTPUT_BLACKLIST = DATA_GENERATED / "synthetic_master_blacklist_v1.txt"


def normalize(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def main():
    print("=== STEP 1: Load accepted synthetic rounds ===")

    files = sorted(ACCEPTED_ROUNDS_DIR.glob("synthetic_pre_enrichment_*.csv"))

    if not files:
        raise FileNotFoundError("No accepted round files found.")

    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)

            if "idiom_canonical" not in df.columns:
                print(f"Skipping (no idiom_canonical): {f.name}")
                continue

            df["idiom_canonical"] = df["idiom_canonical"].apply(normalize)
            df = df[df["idiom_canonical"] != ""].copy()

            df["source_file"] = f.name

            dfs.append(df)

            print(f"Loaded: {f.name} | rows={len(df)}")

        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    if not dfs:
        raise ValueError("No valid data loaded from accepted rounds.")

    df_all = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal rows before dedup: {len(df_all)}")

    df_all = df_all.drop_duplicates(subset=["idiom_canonical"]).copy()

    print(f"After dedup (synthetic): {len(df_all)}")

    df_all.to_csv(OUTPUT_MERGED, index=False, encoding="utf-8-sig")

    print(f"Saved merged synthetic: {OUTPUT_MERGED}")

    print("\n=== STEP 2: Load main IdiomX dataset ===")

    df_main = pd.read_parquet(MAIN_DATASET_PATH)

    if "idiom_canonical" not in df_main.columns:
        raise ValueError("Main dataset missing idiom_canonical.")

    df_main["idiom_canonical"] = df_main["idiom_canonical"].apply(normalize)

    main_set = set(df_main["idiom_canonical"])
    synthetic_set = set(df_all["idiom_canonical"])

    print(f"Main idioms: {len(main_set)}")
    print(f"Synthetic idioms: {len(synthetic_set)}")

    print("\n=== STEP 3: Build blacklist ===")

    blacklist = sorted(main_set.union(synthetic_set))

    print(f"Total blacklist size: {len(blacklist)}")

    with open(OUTPUT_BLACKLIST, "w", encoding="utf-8") as f:
        for idiom in blacklist:
            f.write(idiom + "\n")

    print(f"Saved blacklist: {OUTPUT_BLACKLIST}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()