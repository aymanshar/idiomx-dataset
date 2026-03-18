from pathlib import Path
import pandas as pd
import re

# ============================================================
# Default project paths
# These defaults make the script easy to run from the command
# line while still allowing custom paths from a notebook.
# ============================================================

BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESS_DIR / "idioms_dataset_stage5.csv"
OUTPUT_BROAD = DATA_PROCESS_DIR / "idioms_dataset_stage5_broad.csv"
OUTPUT_HIGH = DATA_PROCESS_DIR / "idioms_dataset_stage5_high_precision.csv"

# ============================================================
# Heuristic hints used to keep likely idiomatic expressions
# and remove overly literal / academic / technical phrases.
# ============================================================

IDIOM_TAG_HINTS = {
    "idiomatic",
    "idiom",
    "figurative",
    "figuratively",
    "proverb",
    "slang",
    "colloquial",
    "informal",
    "humorous",
    "sarcastic",
    "archaic",
}

BAD_MEANING_PATTERNS = [
    r"^a .*? (school|college|university|institution|city|country|person|place)\b",
    r"^the branch of\b",
    r"^the study of\b",
    r"^a field of\b",
    r"^an academic\b",
    r"^a scientific\b",
    r"^a kind of\b",
    r"^a type of\b",
    r"^a genus of\b",
    r"^a family of\b",
    r"^a species of\b",
    r"^a unit of\b",
    r"^a measure of\b",
    r"^a department of\b",
    r"^a system of\b",
    r"^a person who\b",
    r"^a woman who\b",
    r"^a man who\b",
    r"^someone who\b",
]

# Optional literal-domain hints if you want to expand later
BAD_LITERAL_WORDS = {
    "school", "science", "engineering", "mathematics", "government", "ministry",
    "department", "language", "linguistics", "computer", "network", "hospital",
    "bank", "office", "police station", "data science"
}


# ============================================================
# Helper functions
# ============================================================

def norm(x):
    """Safely normalize a single scalar value into a stripped string."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_dataframe(df):
    """
    Normalize the expected text columns:
    - fill nulls
    - cast to string
    - strip surrounding whitespace
    """
    expected_cols = [
        "idiom", "meaning_en", "example", "source", "source_type",
        "pos", "tags", "idiom_confidence", "source_url"
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def good_length(idiom):
    """Keep only multi-word expressions with a reasonable token length."""
    idiom = norm(idiom)
    n = len(idiom.split())
    return 2 <= n <= 7


def has_letters(idiom):
    """Reject entries that do not contain alphabetic characters."""
    return bool(re.search(r"[A-Za-z]", norm(idiom)))


def bad_symbolic(idiom):
    """Reject symbolic or malformed expressions."""
    idiom = norm(idiom)
    return bool(re.search(r"[<>[\]{}_=+*/\\]", idiom))


def bad_meaning(meaning):
    """Reject overly literal / academic / taxonomic meaning patterns."""
    meaning = norm(meaning).lower()
    for pat in BAD_MEANING_PATTERNS:
        if re.search(pat, meaning):
            return True
    return False


def normalize_idiom_text(text):
    """
    Light normalization for deduplication:
    - lowercase
    - collapse internal whitespace
    """
    text = norm(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_meaning_text(text):
    """
    Light normalization for meaning text before deduplication.
    """
    text = norm(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strong_idiom_signal(row):
    """
    High-precision idiom signal:
    keep records with strong evidence that they are idioms.
    """
    tags = norm(row["tags"]).lower()
    pos = norm(row["pos"]).lower()
    meaning = norm(row["meaning_en"]).lower()
    source = norm(row["source"]).lower()
    conf = norm(row["idiom_confidence"]).lower()

    if source in {"phrasefinder", "kaggle_english_idioms"}:
        return True

    if "idiomatic" in tags or "figurative" in tags or "proverb" in tags:
        return True

    if pos in {"phrase", "idiom", "proverb"}:
        return True

    if any(h in meaning for h in ["idiomatic", "figurative", "proverb"]):
        return True

    if conf == "high" and source == "kaikki_wiktionary":
        return True

    return False


def broad_signal(row):
    """
    Broader idiom signal:
    includes strong idioms plus some lexicalized multi-word expressions
    from WordNet when they look more phrase-like.
    """
    idiom = norm(row["idiom"]).lower()
    meaning = norm(row["meaning_en"]).lower()
    tags = norm(row["tags"]).lower()
    source = norm(row["source"]).lower()

    if strong_idiom_signal(row):
        return True

    if source == "wordnet":
        if any(x in idiom for x in [
            " of ", " in ", " on ", " at ", " out ", " up ", " down ",
            " over ", " under ", " for ", " with ", " by "
        ]):
            return True

        if any(x in meaning for x in [
            "figurative", "idiomatic", "informal", "slang", "colloquial"
        ]):
            return True

    if any(h in tags for h in IDIOM_TAG_HINTS):
        return True

    return False


def deduplicate_idiom_meaning(df):
    """
    Remove exact duplicates based on normalized idiom + normalized meaning.
    """
    df = df.copy()

    df["dedup_key"] = (
        df["idiom"].apply(normalize_idiom_text)
        + " || " +
        df["meaning_en"].apply(normalize_meaning_text)
    )

    df = df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])
    df = df.reset_index(drop=True)

    return df


def build_high_precision_dataset(df):
    """
    Build the high-precision dataset and score candidates so that, when
    duplicates exist, the strongest candidate is kept.
    """
    df_high = df[df.apply(strong_idiom_signal, axis=1)].copy()

    df_high["score"] = 0
    df_high.loc[df_high["source"].isin(["phrasefinder", "kaggle_english_idioms"]), "score"] += 4
    df_high.loc[df_high["source"].eq("kaikki_wiktionary"), "score"] += 3
    df_high.loc[df_high["source"].eq("wordnet"), "score"] += 1
    df_high.loc[
        df_high["tags"].str.lower().str.contains("idiomatic|figurative|proverb", regex=True, na=False),
        "score"
    ] += 3
    df_high.loc[df_high["pos"].str.lower().isin(["phrase", "idiom", "proverb"]), "score"] += 2
    df_high.loc[df_high["example"].str.len() > 0, "score"] += 1

    df_high["dedup_key"] = (
        df_high["idiom"].apply(normalize_idiom_text)
        + " || " +
        df_high["meaning_en"].apply(normalize_meaning_text)
    )

    df_high = df_high.sort_values(["dedup_key", "score"], ascending=[True, False])
    df_high = df_high.drop_duplicates(subset=["dedup_key"], keep="first")
    df_high = df_high.drop(columns=["dedup_key", "score"]).reset_index(drop=True)

    return df_high


def build_statistics(df_input, df_broad, df_high):
    """
    Build a compact statistics dictionary for notebook inspection or logging.
    """
    stats = {
        "rows_input": len(df_input),
        "rows_broad": len(df_broad),
        "rows_high_precision": len(df_high),
        "unique_idioms_input": df_input["idiom"].nunique(),
        "unique_idioms_broad": df_broad["idiom"].nunique(),
        "unique_idioms_high_precision": df_high["idiom"].nunique(),
        "broad_source_counts": df_broad["source"].value_counts().to_dict(),
        "high_source_counts": df_high["source"].value_counts().to_dict(),
    }
    return stats


# ============================================================
# Main pipeline function
# ============================================================

def filter_global_idioms(
    input_file=INPUT_FILE,
    output_broad=OUTPUT_BROAD,
    output_high=OUTPUT_HIGH,
):
    """
    Run the global filtering pipeline on the merged idiom dataset:
    - cleaning
    - normalization
    - broad filtering
    - high-precision filtering
    - deduplication
    - statistics

    Parameters
    ----------
    input_file : path-like
        Input merged dataset path.
    output_broad : path-like
        Output path for the broad filtered dataset.
    output_high : path-like
        Output path for the high-precision dataset.

    Returns
    -------
    tuple
        (df_broad, df_high, stats)
    """

    input_file = Path(input_file)
    output_broad = Path(output_broad)
    output_high = Path(output_high)

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    output_broad.parent.mkdir(parents=True, exist_ok=True)
    output_high.parent.mkdir(parents=True, exist_ok=True)

    # Read and normalize input
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    df = normalize_dataframe(df)

    # Basic structural cleaning
    df = df[df["idiom"].apply(good_length)]
    df = df[df["idiom"].apply(has_letters)]
    df = df[~df["idiom"].apply(bad_symbolic)]
    df = df[~df["meaning_en"].apply(bad_meaning)]
    df = df.reset_index(drop=True)

    # Broad dataset
    df_broad = df[df.apply(broad_signal, axis=1)].copy()
    df_broad = deduplicate_idiom_meaning(df_broad)
    df_broad = df_broad.sort_values(["idiom", "meaning_en"]).reset_index(drop=True)

    # High-precision dataset
    df_high = build_high_precision_dataset(df)
    df_high = df_high.sort_values(["idiom", "meaning_en"]).reset_index(drop=True)

    # Save outputs
    df_broad.to_csv(output_broad, index=False, encoding="utf-8-sig")
    df_high.to_csv(output_high, index=False, encoding="utf-8-sig")

    # Build stats
    stats = build_statistics(df, df_broad, df_high)

    # Print summary for command-line usage
    print("Saved broad:", output_broad)
    print("Saved high :", output_high)
    print("Rows after basic cleaning:", stats["rows_input"])
    print("Rows broad:", stats["rows_broad"])
    print("Rows high precision:", stats["rows_high_precision"])
    print("Unique idioms broad:", stats["unique_idioms_broad"])
    print("Unique idioms high precision:", stats["unique_idioms_high_precision"])
    print("\nBroad per source:")
    print(df_broad["source"].value_counts())
    print("\nHigh per source:")
    print(df_high["source"].value_counts())

    return df_broad, df_high, stats


def main():
    """
    Command-line entry point using default project paths.
    """
    df_broad, df_high, stats = filter_global_idioms()
    print("\nBroad preview:")
    print(df_broad.head())
    print("\nHigh-precision preview:")
    print(df_high.head())


if __name__ == "__main__":
    main()