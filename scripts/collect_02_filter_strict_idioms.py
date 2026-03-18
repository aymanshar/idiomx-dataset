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
INPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_strict.csv"

def filter_strict_idioms(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    IDIOM_TAG_HINTS = {
        "idiomatic",
        "idiom",
        "figurative",
        "figuratively",
        "proverb",
        "sarcastic",
        "slang",
        "informal",
        "colloquial",
    }

    BAD_POS = {
        "name",
        "num",
        "symbol",
        "punct",
        "character",
        "det",
        "intj",
    }

    BAD_MEANING_PATTERNS = [
        r"^initialism of\b",
        r"^abbreviation of\b",
        r"^acronym of\b",
        r"^clipping of\b",
        r"^ellipsis of\b",
        r"^alternative form of\b",
        r"^alternative spelling of\b",
        r"^misspelling of\b",
        r"^plural of\b",
        r"^past tense of\b",
        r"^past participle of\b",
        r"^present participle of\b",
        r"^comparative of\b",
        r"^superlative of\b",
        r"^initialism\b",
        r"^abbreviation\b",
    ]

    def normalize_text(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    def looks_like_good_phrase(idiom):
        idiom = normalize_text(idiom)

        if not idiom:
            return False

        # Must contain space
        if " " not in idiom:
            return False

        # 2 to 7 words is a good idiom range
        n = len(idiom.split())
        if n < 2 or n > 7:
            return False

        # Avoid all-caps abbreviations
        if idiom.isupper():
            return False

        # Avoid entries with too many non-letter chars
        if re.search(r"[<>[\]{}_=+*/\\]", idiom):
            return False

        return True

    def has_idiom_signal(row):
        tags = normalize_text(row["tags"]).lower()
        meaning = normalize_text(row["meaning"]).lower()
        pos = normalize_text(row["pos"]).lower()
        idiom_hint = int(row["idiom_hint"]) if str(row["idiom_hint"]).strip() else 0

        if pos in {"phrase", "idiom", "proverb"}:
            return True

        if idiom_hint == 1:
            return True

        if any(h in tags for h in IDIOM_TAG_HINTS):
            return True

        if any(h in meaning for h in ["idiomatic", "figurative", "figuratively", "proverb"]):
            return True

        return False

    def bad_meaning(meaning):
        meaning = normalize_text(meaning).lower()
        for pat in BAD_MEANING_PATTERNS:
            if re.search(pat, meaning):
                return True
        return False

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Normalize
    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Basic shape filters
    df = df[df["idiom"].apply(looks_like_good_phrase)]

    # Remove bad POS
    df = df[~df["pos"].str.lower().isin(BAD_POS)]

    # Remove obvious non-idiomatic lexical/meta definitions
    df = df[~df["meaning"].apply(bad_meaning)]

    # Keep only rows with strong idiom signal
    df = df[df.apply(has_idiom_signal, axis=1)]

    # Deduplicate
    df["dedup_key"] = (
        df["idiom"].str.lower().str.strip() + " || " +
        df["meaning"].str.lower().str.strip()
    )
    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    # Sort
    df = df.sort_values(by=["idiom", "meaning"]).reset_index(drop=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))

    return df

def main():
    filter_strict_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()