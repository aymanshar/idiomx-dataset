"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Clean the strictly filtered Wiktionary idiom dataset.
This script refines the strict Kaikki idiom subset by removing lexical noise,
filtering low-quality meanings and tags, ranking rows by idiom evidence,
and deduplicating the final cleaned idiom list.

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

import pandas as pd
from pathlib import Path
import sys
import re

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
    """
    Clean the strictly filtered Wiktionary idiom dataset.

    Removes noisy lexical entries, orthographic artifacts, and low-quality senses,
    then prioritizes stronger idiom evidence and keeps the best candidate rows.
    """
    # Meaning patterns that usually indicate lexical metadata rather than true idiom senses
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
    # Tags that often correspond to spelling/form variants rather than canonical idioms
    BAD_TAG_HINTS = {
        "alt-of",
        "form-of",
        "obsolete",
        "misspelling",
    }

    def norm(x):
        """
        Normalize a single value by converting null-like values to an empty string
        and trimming surrounding whitespace.
        """
        if pd.isna(x):
            return ""
        return str(x).strip()

    def is_good_idiom_text(text):
        """
        Check whether an idiom candidate has a valid phrase-like structure.

        Keeps multiword expressions with letters and removes clearly noisy entries.
        """
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
        """
        Identify meanings that indicate lexical metadata, spelling variants,
        or non-idiomatic usages.
        """
        meaning = norm(meaning).lower()
        for pat in BAD_MEANING_PATTERNS:
            if re.search(pat, meaning):
                return True
        return False

    def bad_tags(tags):
        """
        Detect tag patterns that suggest non-canonical or low-quality lexical entries.
        """
        tags = norm(tags).lower()
        return any(t in tags for t in BAD_TAG_HINTS)

    def too_orthographic(idiom):
        """
        Detect overly symbolic or orthographic entries that are unlikely to be useful idioms.
        """
        idiom = norm(idiom)

        # keep contractions like 'nuff said, 'fraid so, etc.
        # but reject extremely symbolic items
        non_alnum = sum(1 for c in idiom if not c.isalnum() and not c.isspace() and c not in {"'", "-", "$"})
        return non_alnum > 2

    # Load the strictly filtered Kaikki idiom dataset
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Normalize text fields to ensure consistent filtering and scoring
    for col in ["idiom", "meaning", "example", "pos", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Keep only phrase-like idiom candidates with valid surface forms
    df = df[df["idiom"].apply(is_good_idiom_text)]
    # Remove entries whose meanings indicate lexical metadata or non-idiomatic usage
    df = df[~df["meaning"].apply(bad_meaning)]
    # Remove entries with tags suggesting spelling variants or obsolete forms
    df = df[~df["tags"].apply(bad_tags)]
    # Remove overly symbolic or orthographic entries that are unlikely to be useful idioms
    df = df[~df["idiom"].apply(too_orthographic)]

    # Prefer rows with better idiom evidence
    # Assign a simple evidence score to prefer stronger idiom candidates
    df["score"] = 0
    df.loc[df["idiom_hint"] == 1, "score"] += 3
    df.loc[df["pos"].str.lower().isin(["phrase", "idiom", "proverb"]), "score"] += 3
    df.loc[df["tags"].str.lower().str.contains("idiomatic|figurative|figuratively|slang|colloquial|informal|proverb", regex=True), "score"] += 2
    df.loc[df["example"].str.len() > 0, "score"] += 1

    # Deduplicate by idiom + meaning and keep the strongest row
    df["dedup_key"] = df["idiom"].str.lower().str.strip() + " || " + df["meaning"].str.lower().str.strip()
    df = df.sort_values(["dedup_key", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["dedup_key"], keep="first")

    # For the current dataset design, keep one strongest sense per idiom.
    # Disable this block if a multi-sense idiom inventory is preferred.
    df = df.sort_values(["idiom", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["idiom"], keep="first")

    df = df.drop(columns=["score", "dedup_key"]).sort_values("idiom").reset_index(drop=True)

    # Save the cleaned idiom dataset for downstream normalization and merging
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))

    return df

def main():
    """
    Run the idiom cleaning pipeline using the default input and output paths.
    """
    clean_idioms(INPUT_FILE,OUTPUT_FILE )
    #df = filter_strict_idioms()
    #print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()