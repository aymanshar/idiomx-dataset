import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODERN_VALID_PATH = BASE_DIR / "data/enriched/idiomx_modern_enriched_valid_only_v1.csv"
MODERN_SOURCE_PATH = BASE_DIR / "data/processed/idiomx_modern_pre_llm_unique_idioms.csv"

OUTPUT_PATH = BASE_DIR / "data/enriched/idiomx_modern_enriched_aligned_v1.csv"

ENRICHMENT_MODEL = "gpt-4.1-mini"
ENRICHMENT_VERSION = "modern_v1"

# =========================================================
# LOAD DATA
# =========================================================

print("Loading datasets...")

df = pd.read_csv(MODERN_VALID_PATH)
df_source = pd.read_csv(MODERN_SOURCE_PATH)

print(f"Modern rows: {len(df)}")
print(f"Source rows: {len(df_source)}")

# =========================================================
# MERGE SOURCE METADATA
# =========================================================

print("Merging source metadata...")

merge_cols = [
    "idiom_id",
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
    "example"
]

existing_cols = [c for c in merge_cols if c in df_source.columns]

df = df.merge(
    df_source[existing_cols],
    on="idiom_id",
    how="left"
)

# =========================================================
# ADD / COMPUTE MISSING COLUMNS
# =========================================================

print("Adding computed columns...")

# Generated example flag
df["is_generated_example"] = True

# Model tracking
df["enrichment_model"] = ENRICHMENT_MODEL
df["enrichment_version"] = ENRICHMENT_VERSION

# Validation status (to be updated later if needed)
df["validation_status"] = ""

# Adversarial detection
df["is_adversarial_example"] = df["row_type"].str.contains("adversarial", na=False)

# Minimal pair id (group main examples per idiom)
df["minimal_pair_id"] = (
    df["idiom_id"].astype(str) + "_" +
    df.groupby(["idiom_id", "row_type"]).cumcount().astype(str)
)

# Paraphrase group (group by idiom)
df["paraphrase_group_id"] = df["idiom_id"]

# =========================================================
# ENSURE ALL FINAL COLUMNS EXIST
# =========================================================

print("Ensuring schema compatibility...")

final_columns = [
    "idiom_id", "idiom_canonical", "idiom_surface", "example",
    "idiom_canonical_meaning", "source", "source_type", "pos", "tags",
    "idiom_confidence", "source_url", "record_origin", "license_source",
    "example_language", "meaning_language",
    "idiom_canonical_meaning_arabic",
    "is_idiom", "ambiguity_flag", "idiom_compositionality_level",
    "idiom_register", "idiom_domain", "learner_difficulty",
    "idiom_in_example", "idiom_in_example_arabic",
    "idiom_in_example_meaning_en", "idiom_in_example_meaning_arabic",
    "is_example_idiom", "example_usage_label",
    "is_generated_example", "enrichment_model", "enrichment_version",
    "validation_status", "context_type", "source_style",
    "hard_negative_idioms", "meaning_paraphrases_en", "meaning_paraphrases_ar",
    "idiom_level_explanation_en", "idiom_level_explanation_ar",
    "explanation_en", "explanation_ar",
    "minimal_pair_id", "paraphrase_group_id",
    "is_adversarial_example", "adversarial_type",
    "expected_label", "row_type"
]

# Add missing columns
for col in final_columns:
    if col not in df.columns:
        df[col] = np.nan

# =========================================================
# REORDER COLUMNS
# =========================================================

df = df[final_columns + [c for c in df.columns if c not in final_columns]]

# =========================================================
# SAVE
# =========================================================

print("Saving aligned dataset...")

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(f"Final rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")