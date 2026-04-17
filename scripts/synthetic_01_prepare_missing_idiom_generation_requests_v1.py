#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import argparse
import re
import pandas as pd
import random

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FULL_INPUT_FILE = BASE_DIR / "data" / "generated" / "synthetic_master_blacklist_v1.txt"
DEFAULT_EXCLUSION_OUTPUT_CSV = BASE_DIR / "data" / "generated" / "synthetic_existing_idiom_inventory.csv"
DEFAULT_BATCH_OUTPUT_JSONL = BASE_DIR / "data" / "batches" / "synthetic_idiom_generation_batch_v1.jsonl"

MODEL_NAME = "gpt-4.1-mini"

GENERATION_CATEGORIES = [
    "everyday_conversation",
    "relationships_emotions",
    "conflict_attitude",
    "workplace_business",
    "money_hustle",
    "sports_competition",
]

DEFAULT_CANDIDATES_PER_CATEGORY = 1000


def normalize_idiom(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def load_blacklist(path: Path) -> list[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {path}")

    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            items = [normalize_idiom(line) for line in f if line.strip()]
        return sorted(set([x for x in items if x]))

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)

        if "idiom_canonical" in df.columns:
            col = "idiom_canonical"
        elif "idiom_canonical_normalized" in df.columns:
            col = "idiom_canonical_normalized"
        else:
            col = df.columns[0]

        items = (
            df[col]
            .dropna()
            .astype(str)
            .map(normalize_idiom)
            .tolist()
        )
        return sorted(set([x for x in items if x]))

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)

        if "idiom_canonical" not in df.columns:
            raise ValueError("Input parquet must contain column: idiom_canonical")

        items = (
            df["idiom_canonical"]
            .dropna()
            .astype(str)
            .map(normalize_idiom)
            .tolist()
        )
        return sorted(set([x for x in items if x]))

    raise ValueError(f"Unsupported blacklist format: {path}")

def build_prompt(category: str, exclusion_sample: list[str], n_candidates: int) -> str:
    exclusion_text = "\n".join(f"- {x}" for x in exclusion_sample[:300])

    return f"""
You are an expert linguist, lexicographer, and dataset designer.

Generate NEW English idiomatic expressions and modern slang phrases that are:

- currently plausible in real conversational English
- natural and human-like
- short multiword expressions
- figurative, implied, or socially meaningful
- not already in the exclusion inventory

CATEGORY: {category}

STRICT OUTPUT LIMIT
- Generate EXACTLY {n_candidates} candidates only.
- Keep the JSON compact and complete.
- Do not include any explanation outside the JSON.

STRICT QUALITY RULES
Each candidate must:
- be 2–5 words preferred
- sound like something a real person might actually say
- carry a figurative or socially understood meaning
- be reusable across contexts
- not sound like a product, feature, workflow label, or literal task description

DO NOT GENERATE
- single words
- literal phrases
- generic action phrases
- platform feature names
- corporate jargon labels
- sentence fragments
- hashtags
- obvious variants of known idioms
- phrases whose meaning is fully transparent from the words

EXCLUSION INVENTORY SAMPLE
{exclusion_text}

RETURN ONLY VALID JSON:
{{
  "generation_category": "{category}",
  "candidates": [
    {{
      "idiom_candidate": "...",
      "proposed_meaning_en": "...",
      "idiom_domain": "...",
      "idiom_register": "...",
      "slang_strength": "...",
      "regionality": "...",
      "notes": "..."
    }}
  ]
}}

Allowed values:
- idiom_domain: general, business, sports, legal, politics, regional, internet
- idiom_register: formal, neutral, informal, slang, archaic
- slang_strength: none, weak, medium, strong
- regionality: general_english, american, british, online_global, uncertain

FINAL SELF-CHECK
Before returning the JSON, remove any candidate that is:
- too literal
- too generic
- too transparent
- too similar to another candidate
- not clearly idiomatic or slang-like

Return JSON only.
""".strip()


def get_response_schema(n_candidates: int):
    return {
        "type": "json_schema",
        "name": "synthetic_missing_idiom_generation_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "generation_category": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "minItems": n_candidates,
                    "maxItems": n_candidates,
                    "items": {
                        "type": "object",
                        "properties": {
                            "idiom_candidate": {"type": "string"},
                            "proposed_meaning_en": {"type": "string"},
                            "idiom_domain": {
                                "type": "string",
                                "enum": [
                                    "general", "business", "sports", "military", "religious",
                                    "biblical", "regional", "legal", "politics", "internet"
                                ],
                            },
                            "idiom_register": {
                                "type": "string",
                                "enum": ["formal", "neutral", "informal", "slang", "archaic"],
                            },
                            "slang_strength": {
                                "type": "string",
                                "enum": ["none", "weak", "medium", "strong"],
                            },
                            "regionality": {
                                "type": "string",
                                "enum": ["general_english", "american", "british", "online_global", "uncertain"],
                            },
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "idiom_candidate",
                            "proposed_meaning_en",
                            "idiom_domain",
                            "idiom_register",
                            "slang_strength",
                            "regionality",
                            "notes",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["generation_category", "candidates"],
            "additionalProperties": False,
        },
    }


def prepare_requests(
    input_file: Path,
    exclusion_output_csv: Path,
    output_jsonl: Path,
    candidates_per_category: int,
):
    inventory = load_blacklist(input_file)
    print(f"Unique existing idioms in blacklist: {len(inventory)}")

    exclusion_output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idiom_canonical_normalized": inventory}).to_csv(
        exclusion_output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved exclusion inventory: {exclusion_output_csv}")
    print(f"Unique existing idioms in blacklist: {len(inventory)}")

    schema = get_response_schema(candidates_per_category)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, category in enumerate(GENERATION_CATEGORIES, start=1):
            if len(inventory) > 0:
                rng = random.Random(42 + i)
                exclusion_sample = rng.sample(inventory, k=min(500, len(inventory)))
            else:
                exclusion_sample = []

            prompt = build_prompt(
                category=category,
                exclusion_sample=exclusion_sample,
                n_candidates=candidates_per_category,
            )

            request = {
                "custom_id": f"synthetic_missing_idioms_{category}_v1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL_NAME,
                    "input": prompt,
                    "text": {
                        "format": schema
                    }
                }
            }

            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"Saved batch requests: {output_jsonl}")
    print(f"Total category requests: {len(GENERATION_CATEGORIES)}")
    print(f"Expected raw candidates: {len(GENERATION_CATEGORIES) * candidates_per_category}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare synthetic missing idiom generation requests.")
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_FULL_INPUT_FILE))
    parser.add_argument("--exclusion-output-csv", type=str, default=str(DEFAULT_EXCLUSION_OUTPUT_CSV))
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_BATCH_OUTPUT_JSONL))
    parser.add_argument("--candidates-per-category", type=int, default=DEFAULT_CANDIDATES_PER_CATEGORY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prepare_requests(
        input_file=Path(args.input_file),
        exclusion_output_csv=Path(args.exclusion_output_csv),
        output_jsonl=Path(args.output_jsonl),
        candidates_per_category=args.candidates_per_category,
    )