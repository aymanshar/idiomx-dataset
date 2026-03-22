import pandas as pd
from pathlib import Path
import sys
import re

"""
High-precision idiom selection stage.

This script refines the cleaned Wiktionary idiom dataset by selecting only
entries with strong idiomatic evidence. It applies signal-based filtering
and ranking to produce a high-confidence idiom subset for downstream tasks.
"""

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
    """
    Build a high-precision subset of idioms from the cleaned dataset.

    Selects entries with strong idiomatic signals using tags, POS, meaning,
    and heuristic hints, then ranks and keeps the best candidate per idiom.
    """
    def norm(x):
        """
        Normalize a value by converting nulls to empty string and trimming whitespace.
        """
        if pd.isna(x):
            return ""
        return str(x).strip()

    # Load cleaned idiom dataset from previous pipeline stage
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Normalize key text fields for consistent filtering and scoring
    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    tags_lower = df["tags"].str.lower()
    pos_lower = df["pos"].str.lower()
    meaning_lower = df["meaning"].str.lower()

    # Define strong idiomatic signals from tags, POS, meaning, and heuristic hints
    strong_tag_signal = tags_lower.str.contains(
        "idiomatic|figurative|figuratively|proverb",
        regex=True,
        na=False
    )

    strong_pos_signal = pos_lower.isin(["phrase", "idiom", "proverb"])
    # Define strong idiomatic signals from tags, POS, meaning, and heuristic hints
    strong_meaning_signal = meaning_lower.str.contains(
        "idiomatic|figurative|figuratively|proverb",
        regex=True,
        na=False
    )

    idiom_hint_signal = df["idiom_hint"].fillna(0).astype(int) == 1
    # Keep only rows with at least one strong idiomatic signal
    df_hp = df[
        strong_tag_signal |
        strong_pos_signal |
        strong_meaning_signal |
        idiom_hint_signal
    ].copy()

    # Score rows for better ranking
    # Assign a weighted score to rank stronger idiom candidates higher
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

    df_hp.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)

    print("Rows:", len(df_hp))

    return df_hp

def main():
    """
    Run the high-precision idiom extraction pipeline using default paths.
    """
    high_precision_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()