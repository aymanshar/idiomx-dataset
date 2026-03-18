from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

# ============================================================
# Default project paths
# This allows the script to run directly from CMD while also
# letting the notebook pass custom output paths if needed.
# ============================================================

BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_wordnet_normalized.csv"


def ensure_wordnet_downloaded():
    """
    Ensure the NLTK WordNet corpus is available locally.
    If not found, download it once.
    """
    try:
        wn.synsets("dog")
    except LookupError:
        print("WordNet corpus not found. Downloading...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def extract_wordnet_multiword_expressions(output_file=OUTPUT_FILE):
    """
    Extract multi-word expressions from WordNet and save them
    in the normalized project schema.

    Parameters
    ----------
    output_file : path-like
        Path to the output CSV file.

    Returns
    -------
    pandas.DataFrame
        Extracted WordNet multi-word expressions.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Make sure the WordNet corpus is available
    ensure_wordnet_downloaded()

    rows = []

    # Iterate through all WordNet synsets
    for synset in wn.all_synsets():
        definition = synset.definition()
        pos = synset.pos()

        # Collect lemma names and keep only multi-word expressions
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").strip()

            # Keep only expressions made of more than one word
            if " " not in name:
                continue

            rows.append({
                "idiom": name,
                "meaning_en": definition,
                "example": "",
                "source": "wordnet",
                "source_type": "lexical_database",
                "pos": pos,
                "tags": "",
                "idiom_confidence": "medium",
                "source_url": "",
            })

    df = pd.DataFrame(rows)

    # Remove exact duplicate idiom-meaning pairs
    df = df.drop_duplicates(subset=["idiom", "meaning_en"]).reset_index(drop=True)

    # Save normalized output
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))
    print("Unique idioms:", df["idiom"].nunique())

    return df


def main():
    """
    Command-line entry point using the default output path.
    """
    df = extract_wordnet_multiword_expressions()
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()