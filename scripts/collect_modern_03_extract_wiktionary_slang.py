#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_13_extract_wiktionary_slang.py

Purpose:
- Extract modern slang / informal / colloquial multiword expression candidates
  from the Kaikki Wiktionary English dump
- Keep only phrase-like entries with strong slang / informal / figurative signals
- Save them in the same normalized source schema used across the IdiomX pipeline

Output schema:
- idiom
- meaning_en
- example
- source
- source_type
- pos
- tags
- idiom_confidence
- source_url

Recommended usage:
    python collect_13_extract_wiktionary_slang.py

Optional:
    python collect_13_extract_wiktionary_slang.py --input-file ../data/raw/kaikki.org-dictionary-English-words.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import quote

from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESS_DIR = DATA_DIR / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_FILE = DATA_RAW_DIR / "kaikki.org-dictionary-English-words.jsonl"
DEFAULT_OUTPUT_CSV = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang.csv"
DEFAULT_OUTPUT_JSON = DATA_PROCESS_DIR / "idioms_source_wiktionary_slang_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")
TOKEN_RE = re.compile(r"\S+")

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

SLANG_TAG_HINTS = {
    "slang",
    "informal",
    "colloquial",
    "internet",
    "humorous",
    "sarcastic",
    "meme",
    "vernacular",
    "nonstandard",
}

IDIOM_HINTS = {
    "idiomatic",
    "idiom",
    "figurative",
    "figuratively",
    "metaphorical",
    "metaphoric",
    "proverb",
}

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

BAD_MEANING_PATTERNS = [
    r"^alternative spelling of\b",
    r"^alternative form of\b",
    r"^misspelling of\b",
    r"^nonstandard spelling of\b",
    r"^abbreviation of\b",
    r"^initialism of\b",
    r"^acronym of\b",
    r"^clipping of\b",
    r"^ellipsis of\b",
    r"^plural of\b",
    r"^past tense of\b",
    r"^past participle of\b",
    r"^present participle of\b",
    r"^comparative of\b",
    r"^superlative of\b",
    r"^used other than figuratively or idiomatically\b",
    r"^used other than idiomatically\b",
    r"^used other than figuratively\b",
]

BAD_TAG_HINTS = {
    "alt-of",
    "form-of",
    "misspelling",
    "obsolete",
}


# ============================================================
# Helpers
# ============================================================

def normalize_text(text) -> str:
    """
    Normalize a single text value by trimming whitespace
    and collapsing repeated spaces.
    """
    if text is None:
        return ""
    text = str(text).strip()
    return MULTISPACE_RE.sub(" ", text)


def normalize_term(term: str) -> str:
    """
    Normalize idiom surface text for consistent downstream use.
    """
    term = normalize_text(term)
    term = term.replace("’", "'")
    return term


def token_count(text: str) -> int:
    """
    Count whitespace-separated tokens in a text string.
    """
    return len(TOKEN_RE.findall(normalize_text(text)))


def looks_like_phrase(term: str) -> bool:
    """
    Keep phrase-like multiword expression candidates only.
    """
    term = normalize_term(term)

    if not term or term in BAD_EXACT:
        return False

    if term.startswith(BAD_PREFIXES):
        return False

    if not LETTER_RE.search(term):
        return False

    if BAD_SYMBOL_RE.search(term):
        return False

    n = token_count(term)
    if n < 2 or n > 8:
        return False

    return True


def build_source_url(word: str) -> str:
    """
    Build a Wiktionary URL from the entry word.
    """
    word = normalize_term(word)
    if not word:
        return ""
    return f"https://en.wiktionary.org/wiki/{quote(word)}"


def pick_meaning(sense: dict) -> str:
    """
    Extract the primary meaning from a Kaikki sense.
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


def pick_example(sense: dict) -> str:
    """
    Extract the first valid example sentence from a Kaikki sense.
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


def collect_tags(sense: dict) -> list[str]:
    """
    Collect normalized sense-level tags.
    """
    tags = []
    for x in sense.get("tags", []) or []:
        txt = normalize_text(x).lower()
        if txt:
            tags.append(txt)
    return tags


def collect_topics(sense: dict) -> list[str]:
    """
    Collect normalized sense-level topics if present.
    """
    topics = []
    for x in sense.get("topics", []) or []:
        txt = normalize_text(x).lower()
        if txt:
            topics.append(txt)
    return topics


def contains_slang_signal(sense: dict, pos: str) -> bool:
    """
    Detect strong slang / informal / figurative signals.
    """
    tags = collect_tags(sense)
    topics = collect_topics(sense)
    meaning = pick_meaning(sense).lower()
    raw_glosses = " ".join(normalize_text(x).lower() for x in (sense.get("raw_glosses", []) or []))
    bag = " ".join(tags + topics + [meaning, raw_glosses, pos.lower()])

    if pos.lower() in {"phrase", "idiom"} and any(h in bag for h in SLANG_TAG_HINTS | IDIOM_HINTS):
        return True

    if any(h in bag for h in SLANG_TAG_HINTS):
        return True

    if any(h in bag for h in IDIOM_HINTS):
        return True

    return False


def bad_meaning(meaning: str) -> bool:
    """
    Reject lexical metadata and clearly non-meaning definitions.
    """
    meaning = normalize_text(meaning).lower()

    if not meaning:
        return True

    for pat in BAD_MEANING_PATTERNS:
        if re.search(pat, meaning):
            return True

    return False


def bad_tags(tags: list[str]) -> bool:
    """
    Reject entries with low-quality lexical variant tags.
    """
    return any(t in BAD_TAG_HINTS for t in tags)


def make_tags_text(tags: list[str], topics: list[str]) -> str:
    """
    Build the output tags field as a compact comma-separated string.
    """
    merged = []
    seen = set()

    for item in list(tags) + list(topics):
        item = normalize_text(item).lower()
        if item and item not in seen:
            seen.add(item)
            merged.append(item)

    return ",".join(merged)


def build_quality_score(row: dict) -> int:
    """
    Score rows so stronger candidates are preferred during deduplication.
    """
    score = 0

    pos = normalize_text(row.get("pos", "")).lower()
    tags = normalize_text(row.get("tags", "")).lower()
    meaning = normalize_text(row.get("meaning_en", "")).lower()
    example = normalize_text(row.get("example", ""))

    if pos in {"phrase", "idiom", "proverb"}:
        score += 3

    if any(x in tags for x in ["slang", "informal", "colloquial", "internet"]):
        score += 3

    if any(x in tags for x in ["idiomatic", "figurative", "proverb"]):
        score += 2

    if any(x in meaning for x in ["figurative", "idiomatic", "informal", "slang", "colloquial"]):
        score += 2

    if example:
        score += 1

    n = token_count(row.get("idiom", ""))
    if 2 <= n <= 5:
        score += 1

    return score


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    """
    Deduplicate by idiom + meaning and keep the strongest row.
    Then keep one best row per idiom.
    """
    if not rows:
        return rows

    for row in rows:
        row["score"] = build_quality_score(row)

    rows = sorted(
        rows,
        key=lambda r: (
            normalize_text(r["idiom"]).lower(),
            normalize_text(r["meaning_en"]).lower(),
            -r["score"],
        )
    )

    dedup = {}
    for row in rows:
        key = (
            normalize_text(row["idiom"]).lower()
            + " || " +
            normalize_text(row["meaning_en"]).lower()
        )
        if key not in dedup:
            dedup[key] = row

    one_per_idiom = {}
    for row in dedup.values():
        idiom_key = normalize_text(row["idiom"]).lower()
        if idiom_key not in one_per_idiom:
            one_per_idiom[idiom_key] = row
        else:
            if row["score"] > one_per_idiom[idiom_key]["score"]:
                one_per_idiom[idiom_key] = row

    final_rows = []
    for row in one_per_idiom.values():
        row = row.copy()
        row.pop("score", None)
        final_rows.append(row)

    final_rows = sorted(final_rows, key=lambda r: normalize_text(r["idiom"]).lower())
    return final_rows


def build_stats(rows: list[dict], total_lines: int, parsed_lines: int, extracted_before_dedup: int) -> dict:
    """
    Build compact extraction statistics for inspection.
    """
    rows_with_example = sum(1 for r in rows if normalize_text(r.get("example", "")) != "")
    idioms = {normalize_text(r["idiom"]).lower() for r in rows}

    source_distribution = {}
    confidence_distribution = {}

    for r in rows:
        src = normalize_text(r.get("source", ""))
        conf = normalize_text(r.get("idiom_confidence", ""))

        source_distribution[src] = source_distribution.get(src, 0) + 1
        confidence_distribution[conf] = confidence_distribution.get(conf, 0) + 1

    return {
        "total_lines_read": total_lines,
        "parsed_lines": parsed_lines,
        "extracted_before_dedup": extracted_before_dedup,
        "rows_after_dedup": len(rows),
        "unique_idioms": len(idioms),
        "rows_with_example": rows_with_example,
        "source_distribution": source_distribution,
        "confidence_distribution": confidence_distribution,
    }


# ============================================================
# Main extraction
# ============================================================

def extract_wiktionary_slang(
    input_file: Path = DEFAULT_INPUT_FILE,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> tuple[list[dict], dict]:
    """
    Extract slang/informal/figurative multiword expressions
    from the Kaikki Wiktionary English dump.
    """
    input_file = Path(input_file)
    output_csv = Path(output_csv)
    output_json = Path(output_json)

    if not input_file.exists():
        raise FileNotFoundError(f"Kaikki input file not found: {input_file}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_lines = 0
    parsed_lines = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting Wiktionary slang candidates"):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                parsed_lines += 1
            except Exception:
                continue

            word = normalize_term(obj.get("word", ""))
            pos = normalize_text(obj.get("pos", ""))

            if not looks_like_phrase(word):
                continue

            if pos and pos.lower() not in GOOD_POS:
                continue

            senses = obj.get("senses", []) or []
            if not senses:
                continue

            for sense in senses:
                if not isinstance(sense, dict):
                    continue

                meaning = pick_meaning(sense)
                example = pick_example(sense)
                tags = collect_tags(sense)
                topics = collect_topics(sense)

                if bad_meaning(meaning):
                    continue

                if bad_tags(tags):
                    continue

                if not contains_slang_signal(sense, pos):
                    continue

                rows.append({
                    "idiom": word.lower(),
                    "meaning_en": meaning,
                    "example": example,
                    "source": "wiktionary_slang_kaikki",
                    "source_type": "dictionary",
                    "pos": pos,
                    "tags": make_tags_text(tags, topics),
                    "idiom_confidence": "medium",
                    "source_url": obj.get("wiktionary_url", "") or build_source_url(word),
                })

    extracted_before_dedup = len(rows)
    rows = deduplicate_rows(rows)
    stats = build_stats(rows, total_lines, parsed_lines, extracted_before_dedup)

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idiom",
                "meaning_en",
                "example",
                "source",
                "source_type",
                "pos",
                "tags",
                "idiom_confidence",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nSaved source dataset:", output_csv)
    print("Saved stats:", output_json)
    print("Rows after dedup:", stats["rows_after_dedup"])
    print("Unique idioms:", stats["unique_idioms"])
    print("Rows with example:", stats["rows_with_example"])

    return rows, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract slang / informal / figurative multiword expressions from Kaikki Wiktionary."
    )
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()

    extract_wiktionary_slang(
        input_file=Path(args.input_file),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()