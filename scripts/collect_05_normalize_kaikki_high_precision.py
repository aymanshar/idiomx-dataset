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
INPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_high_precision.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_kaikki_normalized.csv"


def normalize_high_precision_idioms(input_file=INPUT_FILE, output_file=OUTPUT_FILE):

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    out = pd.DataFrame({
        "idiom": df["idiom"].fillna("").astype(str).str.strip(),
        "meaning_en": df["meaning"].fillna("").astype(str).str.strip(),
        "example": df["example"].fillna("").astype(str).str.strip(),
        "source": "kaikki_wiktionary",
        "source_type": "dictionary",
        "pos": df["pos"].fillna("").astype(str).str.strip(),
        "tags": df["tags"].fillna("").astype(str).str.strip(),
        "idiom_confidence": "high",
    })

    out = out.drop_duplicates(subset=["idiom", "meaning_en"]).reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(out))
    return out

def main():
    normalize_high_precision_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()