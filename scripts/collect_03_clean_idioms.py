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
INPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_strict.csv"
OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki_cleaned.csv"

def clean_idioms(input_file=INPUT_FILE, output_file=OUTPUT_FILE):

    BAD_MEANING_PATTERNS = [
        r"^used other than figuratively or idiomatically\b",
        r"^used other than idiomatically\b",
        r"^used other than figuratively\b",
        r"^obsolete form of\b",
        r"^archaic form of\b",
        r"^alternative form of\b",
        r"^alternative spelling of\b",
        r"^misspelling of\b",
        r"^nonstandard spelling of\b",
        r"^clipping of\b",
        r"^ellipsis of\b",
        r"^abbreviation of\b",
        r"^initialism of\b",
        r"^synonym of\b",
    ]

    BAD_TAG_HINTS = {
        "alt-of",
        "form-of",
        "obsolete",
        "misspelling",
    }

    def norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    def is_good_idiom_text(text):
        text = norm(text)
        if not text:
            return False

        words = text.split()
        if len(words) < 2 or len(words) > 7:
            return False

        # Too many symbols → likely noisy lexical entry
        if re.search(r"[<>[\]{}_=+*/\\]", text):
            return False

        # Reject entries with no letters
        if not re.search(r"[A-Za-z]", text):
            return False

        return True

    def bad_meaning(meaning):
        meaning = norm(meaning).lower()
        for pat in BAD_MEANING_PATTERNS:
            if re.search(pat, meaning):
                return True
        return False

    def bad_tags(tags):
        tags = norm(tags).lower()
        return any(t in tags for t in BAD_TAG_HINTS)

    def too_orthographic(idiom):
        idiom = norm(idiom)

        # keep contractions like 'nuff said, 'fraid so, etc.
        # but reject extremely symbolic items
        non_alnum = sum(1 for c in idiom if not c.isalnum() and not c.isspace() and c not in {"'", "-", "$"})
        return non_alnum > 2

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["idiom"].apply(is_good_idiom_text)]
    df = df[~df["meaning"].apply(bad_meaning)]
    df = df[~df["tags"].apply(bad_tags)]
    df = df[~df["idiom"].apply(too_orthographic)]

    # Prefer rows with better idiom evidence
    df["score"] = 0
    df.loc[df["idiom_hint"] == 1, "score"] += 3
    df.loc[df["pos"].str.lower().isin(["phrase", "idiom", "proverb"]), "score"] += 3
    df.loc[df["tags"].str.lower().str.contains("idiomatic|figurative|figuratively|slang|colloquial|informal|proverb", regex=True), "score"] += 2
    df.loc[df["example"].str.len() > 0, "score"] += 1

    # Deduplicate by idiom+meaning, keep best row
    df["dedup_key"] = df["idiom"].str.lower().str.strip() + " || " + df["meaning"].str.lower().str.strip()
    df = df.sort_values(["dedup_key", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["dedup_key"], keep="first")

    # Optional: deduplicate by idiom only, keeping the best sense first
    # Comment this block if you want multiple senses per idiom
    df = df.sort_values(["idiom", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["idiom"], keep="first")

    df = df.drop(columns=["score", "dedup_key"]).sort_values("idiom").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(df))

    return df

def main():
    clean_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()