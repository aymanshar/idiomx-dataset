import pandas as pd
from pathlib import Path
import sys
import re

"""
Strict filtering stage for Wiktionary idiom candidates.

This script takes the broad Kaikki extraction output and applies
higher-precision filtering rules to retain more reliable idiomatic expressions.

Main operations:
1. structural phrase filtering
2. part-of-speech filtering
3. lexical meaning filtering
4. idiom signal detection
5. deduplication

Output:
- a stricter idiom subset used in later cleaning, normalization, and merging steps
"""

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
    """
    Apply strict filtering to the extracted Wiktionary idiom candidates.

    This step removes noisy lexical entries, keeps stronger idiomatic phrases,
    and outputs a higher-precision subset for later cleaning and merging.
    """
    # Lexical hints that strongly suggest figurative or idiomatic usage
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
    # POS categories that are usually not useful for idiom extraction
    BAD_POS = {
        "name",
        "num",
        "symbol",
        "punct",
        "character",
        "det",
        "intj",
    }
    # Patterns indicating lexical/meta definitions rather than true idiom meanings
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

    # Internal helper functions (used only within this filtering step)
    def normalize_text(x):
        """
        Normalize a single value by converting null-like values to empty strings
        and trimming surrounding whitespace.
        """
        if pd.isna(x):
            return ""
        return str(x).strip()

    def looks_like_good_phrase(idiom):
        """
        Check whether an idiom candidate has the basic structural form of
        a plausible multiword idiomatic expression.
        """
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
        """
        Detect whether a row contains strong evidence of idiomatic usage.

        Uses part-of-speech, idiom hints, tags, and meaning text.
        """
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
        """
        Identify meanings that indicate lexical metadata or non-idiomatic forms,
        such as inflections, abbreviations, or spelling variants.
        """
        meaning = normalize_text(meaning).lower()
        for pat in BAD_MEANING_PATTERNS:
            if re.search(pat, meaning):
                return True
        return False

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load the previously extracted Kaikki idiom candidate dataset
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Normalize text fields to ensure consistent filtering and matching
    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Basic shape filters
    # Keep only phrase-like multiword expressions within a reasonable length range
    df = df[df["idiom"].apply(looks_like_good_phrase)]

    # Remove bad POS
    # Remove entries whose part-of-speech category is unlikely to represent idioms
    df = df[~df["pos"].str.lower().isin(BAD_POS)]

    # Remove obvious non-idiomatic lexical/meta definitions
    df = df[~df["meaning"].apply(bad_meaning)]


    # Keep only rows with strong idiom signal / idiomatic evidence from tags, POS, hints, or meanings
    df = df[df.apply(has_idiom_signal, axis=1)]

    # Remove duplicate idiom-meaning pairs to keep the dataset compact and clean
    df["dedup_key"] = (
        df["idiom"].str.lower().str.strip() + " || " +
        df["meaning"].str.lower().str.strip()
    )
    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    # Sort results for stable output and easier downstream inspection
    df = df.sort_values(by=["idiom", "meaning"]).reset_index(drop=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))

    return df

def main():
    """
    Run the strict filtering pipeline using the default input and output paths.
    """
    filter_strict_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()