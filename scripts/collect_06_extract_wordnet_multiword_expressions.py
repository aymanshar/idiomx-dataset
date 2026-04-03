"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

# WordNet does not explicitly label idioms.
# This extraction captures multi-word expressions, which may include  collocations, compounds, and idiomatic phrases.
# Therefore, this source is labeled as medium-confidence.

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn


BASE_DIR = Path("..")
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_PROCESS_DIR / "idioms_source_wordnet.csv"


def ensure_wordnet_downloaded():
    """
    Ensure the NLTK WordNet corpus is available locally.

    If not present, it downloads the required resources automatically.
    """
    try:
        wn.synsets("dog")
    except LookupError:
        print("WordNet corpus not found. Downloading...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def extract_wordnet_multiword_expressions(output_file=OUTPUT_FILE):
    """
    Extract multi-word expressions from WordNet as candidate idioms.

    This function iterates over all WordNet synsets, extracts lemma names,
    filters multi-word expressions, and maps them into the standardized
    IdiomX dataset schema.

    These entries are considered medium-confidence idioms and serve as
    a lexical expansion resource rather than strictly validated idioms.

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

    # Iterate through all WordNet synsets to extract lexical expressions
    for synset in wn.all_synsets():
        definition = synset.definition()
        pos = synset.pos()

        # Extract lemma names and convert WordNet format (underscores → spaces)
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").strip()

            # Normalize to lowercase for consistency across sources
            name = name.lower()

            # Keep only multi-word expressions (potential idioms or phrases)
            if " " not in name:
                continue

            # Map extracted expression into unified IdiomX schema
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

    # Remove duplicate idiom-meaning pairs to reduce redundancy
    df = df.drop_duplicates(subset=["idiom", "meaning_en"]).reset_index(drop=True)

    # Save normalized output
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Saved:", output_file)
    print("Rows:", len(df))
    print("Unique idioms:", df["idiom"].nunique())

    return df


def main():
    """
    Run WordNet extraction pipeline using default output path.
    """
    df = extract_wordnet_multiword_expressions()
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()