import pandas as pd
from pathlib import Path
import sys
import re

# =========================
# CONFIG
# =========================

# Project directories
BASE_DIR = Path("..")

DATA_DIR = BASE_DIR / "data"
DATA_PROCESS_DIR = DATA_DIR / "processed"

# Make sure directories exist
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Raw dataset file
INPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_cleaned.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_high_precision.csv"


def high_precision_idioms(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    def norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    tags_lower = df["tags"].str.lower()
    pos_lower = df["pos"].str.lower()
    meaning_lower = df["meaning"].str.lower()

    strong_tag_signal = tags_lower.str.contains(
        "idiomatic|figurative|figuratively|proverb",
        regex=True,
        na=False
    )

    strong_pos_signal = pos_lower.isin(["phrase", "idiom", "proverb"])

    strong_meaning_signal = meaning_lower.str.contains(
        "idiomatic|figurative|figuratively|proverb",
        regex=True,
        na=False
    )

    idiom_hint_signal = df["idiom_hint"].fillna(0).astype(int) == 1

    df_hp = df[
        strong_tag_signal |
        strong_pos_signal |
        strong_meaning_signal |
        idiom_hint_signal
    ].copy()

    # Score rows for better ranking
    df_hp["score"] = 0
    df_hp.loc[strong_tag_signal, "score"] += 3
    df_hp.loc[strong_pos_signal, "score"] += 3
    df_hp.loc[strong_meaning_signal, "score"] += 2
    df_hp.loc[idiom_hint_signal, "score"] += 2
    df_hp.loc[df_hp["example"].str.len() > 0, "score"] += 1

    # Keep one best row per idiom
    df_hp = df_hp.sort_values(["idiom", "score"], ascending=[True, False])
    df_hp = df_hp.drop_duplicates(subset=["idiom"], keep="first")

    df_hp = df_hp.drop(columns=["score"]).sort_values("idiom").reset_index(drop=True)

    df_hp.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(df_hp))

    return df_hp

def main():
    high_precision_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()