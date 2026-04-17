#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import pandas as pd
import re
import hashlib


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "generated" / "synthetic_idiom_generation_results_v1.jsonl"
BLACKLIST_FILE = BASE_DIR / "data" / "generated" / "synthetic_existing_idiom_inventory.csv"
OUTPUT_CLEAN = BASE_DIR / "data" / "generated" / "synthetic_pre_enrichment_dataset_v1.csv"


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def normalize_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_idiom(text):
    if not isinstance(text, str):
        return False

    text = text.strip()
    token_count = len(text.split())

    # keep multiword expressions only
    if token_count < 2 or token_count > 6:
        return False

    # reject very generic literal phrase patterns
    bad_patterns = [
        "go to",
        "make a",
        "have a",
        "do a",
        "get a",
        "take a",
        "go for",
        "come to",
    ]
    lowered = text.lower()
    for p in bad_patterns:
        if lowered.startswith(p):
            return False

    return True


def generate_idiom_id(text: str) -> str:
    return "syn_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------
# Load blacklist
# ---------------------------------------------------

print("Loading blacklist...")
blacklist_df = pd.read_csv(BLACKLIST_FILE)

if "idiom_canonical" in blacklist_df.columns:
    blacklist_col = "idiom_canonical"
elif "idiom_canonical_normalized" in blacklist_df.columns:
    blacklist_col = "idiom_canonical_normalized"
else:
    blacklist_col = blacklist_df.columns[0]

blacklist = set(
    blacklist_df[blacklist_col]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

print(f"Blacklist size: {len(blacklist)}")


# ---------------------------------------------------
# Parse batch output
# ---------------------------------------------------

records = []

print("Parsing batch output...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        try:
            content = obj["response"]["body"]["output"][0]["content"][0]["text"]
            data = json.loads(content)

            generation_category = data.get("generation_category", "")

            for item in data.get("candidates", []):
                idiom_candidate = item.get("idiom_candidate", "")
                proposed_meaning_en = item.get("proposed_meaning_en", "")

                records.append({
                    "generation_category": generation_category,
                    "idiom_candidate": idiom_candidate,
                    "proposed_meaning_en": proposed_meaning_en,
                    "idiom_domain": item.get("idiom_domain", ""),
                    "idiom_register": item.get("idiom_register", ""),
                    "slang_strength": item.get("slang_strength", ""),
                    "regionality": item.get("regionality", ""),
                    "notes": item.get("notes", ""),
                })

        except Exception:
            continue

df = pd.DataFrame(records)

print(f"Raw generated: {len(df)}")

if df.empty:
    raise ValueError("No parsed candidates found in batch output.")


# ---------------------------------------------------
# Normalize and clean
# ---------------------------------------------------

df["idiom_candidate"] = df["idiom_candidate"].apply(normalize_text)
df["proposed_meaning_en"] = df["proposed_meaning_en"].astype(str).str.strip()

df = df.dropna(subset=["idiom_candidate"])
df = df[df["idiom_candidate"].astype(str).str.strip() != ""]

print(f"After null/empty removal: {len(df)}")

df = df[df["idiom_candidate"].apply(is_valid_idiom)]

print(f"After validity filter: {len(df)}")

df = df[~df["idiom_candidate"].isin(blacklist)]

print(f"After blacklist removal: {len(df)}")

df = df.drop_duplicates(subset=["idiom_candidate"]).copy()

print(f"After deduplication: {len(df)}")


# ---------------------------------------------------
# Convert to EXACT pre-enrichment schema
# ---------------------------------------------------

# tags: use generation category + optional light metadata
df["tags"] = df["generation_category"].fillna("").astype(str).str.strip()

# final schema only
df_final = pd.DataFrame({
    "idiom_id": df["idiom_candidate"].apply(generate_idiom_id),
    "idiom_canonical": df["idiom_candidate"],
    "idiom_surface": df["idiom_candidate"],
    "example": "",  # intentionally empty before enrichment
    "idiom_canonical_meaning": df["proposed_meaning_en"],
    "source": "llm_generated_inventory",
    "source_type": "synthetic_generation",
    "pos": "phrase",
    "tags": df["tags"],
    "idiom_confidence": 0.7,
    "source_url": "synthetic",
    "record_origin": "synthetic_generation_pipeline",
    "license_source": "synthetic_llm",
    "example_language": "en",
    "meaning_language": "en",
})

# exact order enforcement
required_columns = [
    "idiom_id",
    "idiom_canonical",
    "idiom_surface",
    "example",
    "idiom_canonical_meaning",
    "source",
    "source_type",
    "pos",
    "tags",
    "idiom_confidence",
    "source_url",
    "record_origin",
    "license_source",
    "example_language",
    "meaning_language",
]

df_final = df_final[required_columns].copy()


# ---------------------------------------------------
# Save
# ---------------------------------------------------

from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "generated" / "accepted_rounds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

output_file = OUTPUT_DIR / f"synthetic_pre_enrichment_{timestamp}.csv"

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\nSaved accepted round file: {output_file}")
print(f"Final accepted rows: {len(df)}")