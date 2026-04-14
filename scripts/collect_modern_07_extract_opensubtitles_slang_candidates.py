#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_17_extract_opensubtitles_slang_candidates.py

Purpose:
- Mine conversational multi-word candidate expressions from prepared OpenSubtitles raw text
- Keep repeated phrase-like candidates that may be slang, informal expressions,
  or semi-idiomatic spoken phrases
- Save them in the standard IdiomX source schema for later cleaning

Input:
- data/raw/opensubtitles/opensubtitles_en_raw.txt

Outputs:
- data/processed/idioms_source_opensubtitles_candidates.csv
- data/processed/idioms_source_opensubtitles_candidates_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESS_DIR = DATA_DIR / "processed"

OPEN_SUB_DIR = DATA_RAW_DIR / "opensubtitles"

DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
OPEN_SUB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_FILE = OPEN_SUB_DIR / "opensubtitles_en_raw.txt"
DEFAULT_OUTPUT_CSV = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates.csv"
DEFAULT_OUTPUT_JSON = DATA_PROCESS_DIR / "idioms_source_opensubtitles_candidates_stats.json"


# ============================================================
# Config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
BAD_SYMBOL_RE = re.compile(r"[<>[\]{}_=+*/\\|]")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

DEFAULT_MIN_NGRAM = 3
DEFAULT_MAX_NGRAM = 5
DEFAULT_MIN_COUNT = 5
DEFAULT_MAX_ROWS = 0  # 0 = no limit

PRONOUN_STARTERS = {"i", "you", "we", "he", "she", "they", "it"}

BLOCKLIST_EXACT = {
    "thank you",
    "thanks a lot",
    "excuse me",
    "good morning",
    "good night",
    "see you later",
    "see you soon",
    "yes sir",
    "no sir",
    "oh my god",
    "come on",
    "let's go",
    "shut up",
    "i don't know",
    "i do not know",
    "you know what",
    "what are you doing",
    "what do you mean",
    "where are you going",
    "how are you",
    "i love you",
    "i miss you",
    "leave me alone",
    "get out of here",
    "what's going on",
    "right now",
    "at all",
    "a lot of",
    "one of the",
    "in the middle",
    "at the same",
    "i have to",
    "we have to",
    "you have to",
    "do you want",
    "do you know",
    "what do you think",
    "as soon as",
    "at the end",
    "at the time",
    "in the end",
    "on the other",
    "for the first",
    "for the last",
    "there you go",
    "here we go",
    "all right now",
    "what is this",
    "who is that",
    "how do you",
    "why do you",
}

BLOCKLIST_STARTS = (
    "i want ",
    "i need ",
    "i have ",
    "i had ",
    "i was ",
    "i am ",
    "you want ",
    "you need ",
    "you have ",
    "you had ",
    "you were ",
    "you are ",
    "we want ",
    "we need ",
    "we have ",
    "we had ",
    "he was ",
    "she was ",
    "they were ",
    "it was ",
    "there is ",
    "there are ",
    "this is ",
    "that is ",
    "what is ",
    "who is ",
    "why is ",
    "how is ",
    "do you ",
    "can you ",
    "will you ",
    "would you ",
    "did you ",
    "are you ",
    "have you ",
)

LITERAL_PREFIXES = (
    "go to ",
    "come to ",
    "walk to ",
    "drive to ",
    "open the ",
    "close the ",
    "pick up ",
    "put it ",
    "take the ",
    "sit on ",
    "look at ",
    "talk to ",
    "come with ",
    "wait for ",
    "work on ",
    "call the ",
    "bring the ",
    "leave the ",
    "hold the ",
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "from", "with", "by", "as", "is", "am", "are", "was", "were", "be",
    "been", "being", "it", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "so", "just", "very", "too", "not", "no", "yes", "do", "did",
    "does", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "all", "any", "some", "one", "two", "three",
    "then", "than", "here", "there", "now", "well", "also", "only", "even", "still",
}

CONTENT_CUE_WORDS = {
    "get", "give", "take", "keep", "cut", "lose", "make", "play", "come", "go",
    "back", "over", "off", "out", "down", "up", "through", "around", "away",
    "mind", "break", "slack", "point", "deal", "catch", "mean", "mess", "hang",
    "blow", "fall", "pull", "push", "drag", "drop", "sell", "buy", "call", "turn",
    "hold", "bring", "throw", "pick", "run", "work", "stick", "hit", "cool", "save",
    "snap", "shut", "spill", "burn", "move", "read", "ride", "backfire", "flip",
}

ENDING_PARTICLES = {"up", "down", "off", "out", "over", "away", "around", "back", "through"}

WEAK_CONTENT_WORDS = {
    "thing", "stuff", "person", "people", "someone", "somebody", "something",
    "anything", "everything", "nothing", "time", "place", "way", "kind", "sort",
}

# Very common weak trigram/bigger scaffolds
WEAK_PATTERN_RE = re.compile(
    r"^(?:"
    r"(?:i|you|we|he|she|they|it)\s+\w+\s+(?:to|the|a|an|my|your|his|her|our|their)\b|"
    r"(?:there|this|that|what|who|why|how)\s+\w+\s+\w+\b|"
    r"(?:do|did|can|will|would|are|have)\s+you\s+\w+\b"
    r")",
    re.IGNORECASE,
)


# ============================================================
# Helpers
# ============================================================

def norm(x) -> str:
    if x is None:
        return ""
    return MULTISPACE_RE.sub(" ", str(x).strip())


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into simple word tokens with contractions retained.
    """
    return TOKEN_RE.findall(text.lower())


def looks_like_good_line(text: str) -> bool:
    """
    Keep reasonably natural dialogue lines.
    """
    text = norm(text)

    if not text:
        return False

    if len(text) < 8:
        return False

    if len(text) > 180:
        return False

    if not LETTER_RE.search(text):
        return False

    bad_char_count = sum(1 for c in text if c in "<>[]{}=_*\\|")
    if bad_char_count > 2:
        return False

    token_n = len(tokenize(text))
    if token_n < 3 or token_n > 25:
        return False

    return True


def candidate_tokens_to_text(tokens: list[str]) -> str:
    return " ".join(tokens).strip().lower()


def content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def looks_like_candidate_phrase(phrase: str, n: int) -> bool:
    """
    Structural filter for candidate n-grams.
    """
    phrase = norm(phrase).lower()

    if not phrase:
        return False

    if phrase in BLOCKLIST_EXACT:
        return False

    if phrase.startswith(BLOCKLIST_STARTS):
        return False

    if BAD_SYMBOL_RE.search(phrase):
        return False

    if not LETTER_RE.search(phrase):
        return False

    toks = phrase.split()
    if len(toks) < 3 or len(toks) > 5:
        return False

    # reject pronoun-heavy generic starters
    if toks[0] in PRONOUN_STARTERS:
        return False

    non_stop = content_words(toks)

    # require stronger semantic content
    if len(non_stop) < 2:
        return False

    if phrase.startswith(LITERAL_PREFIXES):
        return False

    # Avoid phrases ending with weak determiners / stopwords
    if toks[-1] in STOPWORDS and toks[-1] not in ENDING_PARTICLES:
        return False

    # Too many weak content words
    weak_count = sum(1 for t in non_stop if t in WEAK_CONTENT_WORDS)
    if weak_count >= 2:
        return False

    # reject common sentence scaffolds
    if WEAK_PATTERN_RE.match(phrase):
        return False

    return True


def weak_generic_candidate(phrase: str, n: int) -> bool:
    """
    Reject obvious generic / compositional fragments.
    """
    phrase = norm(phrase).lower()
    toks = phrase.split()
    non_stop = content_words(toks)

    if phrase in BLOCKLIST_EXACT:
        return True

    if phrase.startswith(BLOCKLIST_STARTS):
        return True

    if phrase.startswith(LITERAL_PREFIXES):
        return True

    if len(non_stop) < 2:
        return True

    # Weak 3-grams are especially noisy
    if n == 3:
        if len(non_stop) == 2 and all(t in WEAK_CONTENT_WORDS for t in non_stop):
            return True

    return False


def spoken_pattern_signal(phrase: str) -> bool:
    """
    Soft heuristic to preserve more conversational / idiom-like phrase shapes.
    """
    toks = phrase.split()
    if not toks:
        return False

    if any(t in CONTENT_CUE_WORDS for t in toks):
        return True

    if len(toks) >= 2:
        if toks[0] in CONTENT_CUE_WORDS:
            return True
        if toks[-1] in ENDING_PARTICLES:
            return True

    # common idiom-like structure: verb + object/content + particle
    if len(toks) == 3:
        if toks[0] in CONTENT_CUE_WORDS and (toks[2] in ENDING_PARTICLES or toks[1] in CONTENT_CUE_WORDS):
            return True

    return False


def lexical_diversity_score(phrase: str) -> int:
    """
    Prefer phrases with stronger lexical content.
    """
    toks = phrase.split()
    non_stop = content_words(toks)

    score = 0
    score += len(set(non_stop))

    if any(t in CONTENT_CUE_WORDS for t in toks):
        score += 2

    if toks[-1] in ENDING_PARTICLES:
        score += 2

    if len(non_stop) >= 3:
        score += 1

    return score


def example_quality_score(example: str, phrase: str) -> int:
    """
    Prefer clean, natural examples for each candidate.
    """
    score = 0
    ex = norm(example)
    phrase = norm(phrase).lower()

    if ex:
        score += 2

    token_n = len(tokenize(ex))
    if 5 <= token_n <= 16:
        score += 3
    elif 4 <= token_n <= 20:
        score += 2
    else:
        score += 1

    if phrase in ex.lower():
        score += 2

    if ex.isupper():
        score -= 2

    bad_char_count = sum(1 for c in ex if c in "<>[]{}=_*\\|")
    score -= bad_char_count

    return score


def select_best_example(examples: list[str], phrase: str) -> str:
    """
    Keep one best example line per phrase candidate.
    """
    if not examples:
        return ""

    best_ex = ""
    best_score = -10**9

    for ex in examples:
        s = example_quality_score(ex, phrase)
        if s > best_score:
            best_score = s
            best_ex = ex

    return best_ex


# ============================================================
# Main extraction
# ============================================================

def extract_opensubtitles_candidates(
    input_file: Path = DEFAULT_INPUT_FILE,
    min_ngram: int = DEFAULT_MIN_NGRAM,
    max_ngram: int = DEFAULT_MAX_NGRAM,
    min_count: int = DEFAULT_MIN_COUNT,
    max_rows: int = DEFAULT_MAX_ROWS,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
):
    """
    Mine repeated multi-word spoken phrase candidates from prepared subtitle/dialogue text.
    """
    input_file = Path(input_file)
    output_csv = Path(output_csv)
    output_json = Path(output_json)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if min_ngram < 3:
        raise ValueError("min_ngram must be at least 3 for this stricter extractor")

    if max_ngram < min_ngram:
        raise ValueError("max_ngram must be >= min_ngram")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cleaned_lines = []
    total_lines_read = 0

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines_read += 1
            line = norm(line)
            if looks_like_good_line(line):
                cleaned_lines.append(line)

            if max_rows > 0 and total_lines_read >= max_rows:
                break

    lines_kept = len(cleaned_lines)

    phrase_counter = Counter()
    phrase_examples = defaultdict(list)
    raw_ngram_total = 0
    structural_pass_total = 0

    for line in tqdm(cleaned_lines, desc="Mining OpenSubtitles candidates"):
        toks = tokenize(line)
        if len(toks) < min_ngram:
            continue

        seen_in_line = set()

        for n in range(min_ngram, max_ngram + 1):
            if len(toks) < n:
                continue

            for i in range(len(toks) - n + 1):
                raw_ngram_total += 1

                cand_tokens = toks[i:i+n]
                phrase = candidate_tokens_to_text(cand_tokens)

                if not looks_like_candidate_phrase(phrase, n=n):
                    continue

                if weak_generic_candidate(phrase, n=n):
                    continue

                # stricter rule for 3-grams
                if n == 3 and not spoken_pattern_signal(phrase):
                    continue

                structural_pass_total += 1

                # Count once per line to reduce local repetition inflation
                if phrase not in seen_in_line:
                    phrase_counter[phrase] += 1
                    seen_in_line.add(phrase)

                if len(phrase_examples[phrase]) < 20:
                    phrase_examples[phrase].append(line)

    kept_candidates = {
        phrase: count
        for phrase, count in phrase_counter.items()
        if count >= min_count
    }

    rows = []
    for phrase, count in kept_candidates.items():
        example = select_best_example(phrase_examples.get(phrase, []), phrase)

        rows.append({
            "idiom": phrase,
            "meaning_en": "",
            "example": example,
            "source": "opensubtitles_dialogue",
            "source_type": "subtitle_dialogue",
            "pos": "",
            "tags": "dialogue,conversational,candidate,weak_candidate,repeated_ngram",
            "idiom_confidence": "low",
            "source_url": "",
            "frequency_count": int(count),
            "lexical_score": lexical_diversity_score(phrase),
        })

    df = pd.DataFrame(rows)

    if len(df):
        df["example_nonempty"] = (df["example"].fillna("").astype(str).str.strip() != "").astype(int)
        df["token_length"] = df["idiom"].fillna("").astype(str).str.split().str.len()

        df = df.sort_values(
            by=["frequency_count", "lexical_score", "example_nonempty", "token_length", "idiom"],
            ascending=[False, False, False, True, True]
        ).copy()

        df = df.drop_duplicates(subset=["idiom"], keep="first").copy()

        out_cols = [
            "idiom",
            "meaning_en",
            "example",
            "source",
            "source_type",
            "pos",
            "tags",
            "idiom_confidence",
            "source_url",
        ]
        df = df[out_cols].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=[
            "idiom",
            "meaning_en",
            "example",
            "source",
            "source_type",
            "pos",
            "tags",
            "idiom_confidence",
            "source_url",
        ])

    top_candidates_preview = phrase_counter.most_common(30)
    rows_with_example = int((df["example"].fillna("").astype(str).str.strip() != "").sum()) if len(df) else 0
    avg_token_len = round(df["idiom"].fillna("").astype(str).str.split().str.len().mean(), 4) if len(df) else 0.0

    stats = {
        "input_file": str(input_file),
        "total_lines_read": int(total_lines_read),
        "lines_kept_after_cleaning": int(lines_kept),
        "raw_ngram_total": int(raw_ngram_total),
        "candidates_after_structural_filtering": int(structural_pass_total),
        "candidates_after_frequency_filtering": int(len(kept_candidates)),
        "rows_final": int(len(df)),
        "unique_idioms_final": int(df["idiom"].nunique()) if len(df) else 0,
        "rows_with_example_final": int(rows_with_example),
        "avg_idiom_token_length": avg_token_len,
        "min_ngram": int(min_ngram),
        "max_ngram": int(max_ngram),
        "min_count": int(min_count),
        "top_repeated_candidates_preview": [
            {"phrase": phrase, "count": int(count)}
            for phrase, count in top_candidates_preview
        ],
    }

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved source dataset:", output_csv)
    print("Saved stats:", output_json)
    print("Total lines read:", total_lines_read)
    print("Lines kept after cleaning:", lines_kept)
    print("Rows final:", len(df))
    print("Rows with example:", rows_with_example)

    return df, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine OpenSubtitles phrase candidates for modern idiom/slang pipeline."
    )
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--min-ngram", type=int, default=DEFAULT_MIN_NGRAM)
    parser.add_argument("--max-ngram", type=int, default=DEFAULT_MAX_NGRAM)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="0 means no limit")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()

    extract_opensubtitles_candidates(
        input_file=Path(args.input_file),
        min_ngram=args.min_ngram,
        max_ngram=args.max_ngram,
        min_count=args.min_count,
        max_rows=args.max_rows,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()