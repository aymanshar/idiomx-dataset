"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

import json
import csv
import re
from pathlib import Path
from tqdm import tqdm
import sys


# Project directories
BASE_DIR = Path("..")

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESS_DIR = DATA_DIR / "processed"

# python file path
sys.path.append(str(BASE_DIR / "scripts"))

# Make sure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

# Raw dataset file
KAIKKI_FILE = DATA_RAW_DIR / "kaikki.org-dictionary-English-words.jsonl"

# External dataset sources
KAIKKI_DATASET_URL = "https://kaikki.org/dictionary/English/words/kaikki.org-dictionary-English-words.jsonl"

# Set True if you want stricter filtering
STRICT_MODE = False

# Common POS values that are often useful for idioms / phrases
GOOD_POS = {
    "phrase",
    "proverb",
    "idiom",
    "verb",
    "adjective",
    "adverb",
    "noun",
    "interjection",
}

# Terms in glosses/tags that strongly suggest idiomatic usage
IDIOM_HINTS = {
    "idiomatic",
    "idiom",
    "figuratively",
    "figurative",
    "metaphoric",
    "metaphorical",
    "proverb",
}

# Exclude obvious noisy/meta entries
BAD_PREFIXES = (
    "Template:",
    "Module:",
    "Appendix:",
    "Wiktionary:",
    "Category:",
    "Help:",
    "Special:",
    "Citations:",
    "Reconstruction:",
    "Thesaurus:",
    "Index:",
    "Rhymes:",
    "Concordance:",
)

BAD_EXACT = {
    "",
    "-",
    "—",
}

WORD_RE = re.compile(r"[A-Za-z]")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize text by trimming whitespace and collapsing multiple spaces.
    """
    if text is None:
        return ""
    text = str(text).strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def looks_like_english_phrase(word: str) -> bool:
    """
    Check if a word is a valid English multiword expression candidate.
    Filters out noise, non-English patterns, and non-phrase entries.
    """
    word = normalize_text(word)
    if not word or word in BAD_EXACT:
        return False

    if word.startswith(BAD_PREFIXES):
        return False

    # We mainly want multi-word expressions
    if " " not in word and "-" not in word:
        return False

    # Avoid extremely long titles
    token_count = len(word.split())
    if token_count < 2 or token_count > 8:
        return False

    # Must contain Latin letters
    if not WORD_RE.search(word):
        return False

    return True


def contains_idiom_hint(sense: dict) -> bool:
    """
    Detect whether a sense contains signals of idiomatic or figurative usage.
    Uses tags and gloss text to identify idiom-related patterns.
    """
    tags = [str(x).lower() for x in sense.get("tags", []) if x]
    raw_glosses = sense.get("raw_glosses", []) or []
    glosses = sense.get("glosses", []) or []

    bag = " ".join(
        tags
        + [str(x).lower() for x in raw_glosses]
        + [str(x).lower() for x in glosses]
    )

    return any(hint in bag for hint in IDIOM_HINTS)


def pick_example(sense: dict) -> str:
    """
    Extract the first valid example sentence from a sense entry.
    """
    examples = sense.get("examples", []) or []
    for ex in examples:
        if isinstance(ex, dict):
            txt = normalize_text(ex.get("text", ""))
            if txt:
                return txt
        elif isinstance(ex, str):
            txt = normalize_text(ex)
            if txt:
                return txt
    return ""


def pick_meaning(sense: dict) -> str:
    """
    Extract the primary meaning from a sense using glosses or raw glosses.
    """
    glosses = sense.get("glosses", []) or []
    for g in glosses:
        g = normalize_text(g)
        if g:
            return g

    raw_glosses = sense.get("raw_glosses", []) or []
    for g in raw_glosses:
        g = normalize_text(g)
        if g:
            return g

    return ""


def sense_is_useful(sense: dict) -> bool:
    """
    Filter out non-lexical or noisy senses such as inflections or spelling variants.
    """
    meaning = pick_meaning(sense)
    if not meaning:
        return False

    lower_meaning = meaning.lower()

    # Exclude some very meta/non-lexical senses
    noisy_patterns = [
        "inflection of",
        "plural of",
        "alternative form of",
        "alternative spelling of",
        "misspelling of",
        "comparative of",
        "superlative of",
        "past tense of",
        "past participle of",
        "present participle of",
    ]
    if any(p in lower_meaning for p in noisy_patterns):
        return False

    return True


def entry_to_rows(entry: dict, strict_mode: bool = False):
    """
    Convert a single Kaikki dictionary entry into structured idiom candidate rows.
    Applies filtering, extracts meaning/example, and labels idiom hints.
    """
    word = normalize_text(entry.get("word", ""))
    lang = normalize_text(entry.get("lang", ""))
    pos = normalize_text(entry.get("pos", "")).lower()

    if lang != "English":
        return []

    if not looks_like_english_phrase(word):
        return []

    if pos and pos not in GOOD_POS and STRICT_MODE:
        return []

    senses = entry.get("senses", []) or []
    if not senses:
        return []

    rows = []

    for idx, sense in enumerate(senses):
        if not isinstance(sense, dict):
            continue

        if not sense_is_useful(sense):
            continue

        meaning = pick_meaning(sense)
        example = pick_example(sense)
        tags = ", ".join([str(x) for x in sense.get("tags", []) if x])

        hint = contains_idiom_hint(sense)

        # Filtering logic:
        # STRICT_MODE = only take clearly idiomatic/phrase-like cases
        # otherwise take broader multiword entries and keep idiom_hint as a feature
        if STRICT_MODE:
            if pos not in {"phrase", "idiom", "proverb"} and not hint:
                continue

        rows.append({
            "idiom": word,
            "meaning": meaning,
            "example": example,
            "pos": pos,
            "tags": tags,
            "idiom_hint": int(hint),
            "source": "kaikki_wiktionary",
            "sense_index": idx,
        })

    return rows


def extract_kaikki_idioms(
            input_file,
            output_csv,
            output_jsonl,
            strict_mode: bool = False,
    ):
    """
        Extract idiom-like multiword expressions from Kaikki English Wiktionary JSONL.

        Processes entries, filters valid multiword phrases, removes noise,
        and outputs structured idiom candidates to CSV and JSONL formats.

        Supports optional strict filtering for higher precision extraction.

        Parameters
        ----------
        input_file : str or Path
            Path to Kaikki JSONL file.
        output_csv : str or Path
            Output CSV path.
        output_jsonl : str or Path
            Output JSONL path.
        strict_mode : bool
            If True, keep only stronger idiom/phrase/proverb candidates.

        Returns
        -------
        dict
            Summary information about extraction.
        """
    input_path = Path(input_file)
    output_csv = Path(output_csv)
    output_jsonl = Path(output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    total_lines = 0
    kept_rows = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8-sig") as fout_csv, \
         open(output_jsonl, "w", encoding="utf-8") as fout_jsonl:

        writer = csv.DictWriter(
            fout_csv,
            fieldnames=[
                "idiom",
                "meaning",
                "example",
                "pos",
                "tags",
                "idiom_hint",
                "source",
                "sense_index",
            ]
        )
        writer.writeheader()

        for line in tqdm(fin, desc="Reading Kaikki JSONL"):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except Exception:
                continue

            # Convert entry into candidate idiom rows using filtering logic
            #rows = entry_to_rows(entry) 
            rows = entry_to_rows(entry, strict_mode=strict_mode)

            for row in rows:
                # deduplicate on idiom + meaning
                key = (row["idiom"].lower(), row["meaning"].lower())
                if key in seen:
                    continue
                seen.add(key)

                writer.writerow(row)
                fout_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_rows += 1

    summary = {
        "input_file": str(input_path),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "strict_mode": strict_mode,
        "total_lines_scanned": total_lines,
        "rows_kept": kept_rows,
    }

    print("\nExtraction finished.")
    print(summary)

    return summary

def main():
    """
    Entry point for running the Kaikki idiom extraction pipeline.
    Defines input/output paths and executes the extraction process.
    """
    # CONFIG input and output dataset
    
    input_file = KAIKKI_FILE
    output_csv = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki.csv"
    output_jsonl = DATA_PROCESS_DIR / "idioms_wiktionary_kaikki.jsonl"

    extract_kaikki_idioms(
        input_file=input_file,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        strict_mode=False,
    )


if __name__ == "__main__":
    main()