"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Merge LLM batch results into the final enriched IdiomX dataset.

This script aligns raw batch outputs with the original pre-enrichment dataset,
repairs malformed JSON if needed, expands generated examples into structured rows,
and exports the enriched dataset for downstream analysis and modeling.

Notes:
# This stage transforms schema-constrained LLM outputs into the final row-level
# enrichment dataset, enabling direct use in bilingual idiom modeling,
# retrieval tasks, and downstream analysis.

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""
from pathlib import Path
import json
import pandas as pd
from json_repair import repair_json as repair_llm_json

# paths
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_FILE = BASE_DIR / "data" / "processed" / "idiomx_pre_enrichment.parquet"
DEFAULT_RESULTS_JSONL = BASE_DIR / "data" / "results" / "idiomx_results.jsonl"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full.csv"
DEFAULT_OUTPUT_PARQUET = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full.parquet"

# ==============================
# Sample paths
# ==============================
SAMPLE_RESULTS_JSONL = BASE_DIR / "data" / "sample" / "idiomx_results_sample.jsonl"
SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample.csv"
SAMPLE_OUTPUT_PARQUET = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample.parquet"

# Final standardized schema for the enriched IdiomX dataset
FINAL_COLUMNS = [
    "idiom_id", "idiom_canonical", "idiom_surface", "example",
    "idiom_canonical_meaning", "source", "source_type", "pos", "tags",
    "idiom_confidence_score", "record_origin", "license_source",
    "example_language", "meaning_language", "idiom_canonical_meaning_arabic",
    "is_idiom", "ambiguity_flag", "idiom_compositionality_level",
    "idiom_register", "idiom_domain", "learner_difficulty",
    "idiom_in_example", "idiom_in_example_arabic",
    "idiom_in_example_meaning_en", "idiom_in_example_meaning_arabic",
    "is_example_idiom", "example_usage_label", "is_generated_example",
    "enrichment_model", "enrichment_version", "validation_status"
]


def extract_output_text(body: dict):
    """
    Extract the generated text content from the batch response body.

    Returns the first output_text message if available, otherwise None.
    """
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    return content_item.get("text")
    return None


def merge_results(
    raw_file: Path = DEFAULT_RAW_FILE,
    results_jsonl: Path = DEFAULT_RESULTS_JSONL,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_parquet: Path = DEFAULT_OUTPUT_PARQUET,
    ):
    """
    Merge LLM batch outputs with the raw pre-enrichment dataset.

    Parses batch results, repairs malformed JSON when possible, expands generated
    examples into row-level records, and saves the final enriched dataset in CSV
    and Parquet formats.
    """
    print("Raw file:", raw_file)
    print("Results file:", results_jsonl)

    # Load the original pre-enrichment dataset used to create the batch requests
    raw_df = pd.read_parquet(raw_file)
    rows = []

    # Iterate through batch results and align each response with its source row
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Recover the original row index from the batch custom_id
            custom_id = record.get("custom_id", "")
            # Assumes custom_id ends with the original row index (e.g., idiomx_v2_123)
            idx = int(custom_id.split("_")[-1])

            raw_row = raw_df.iloc[idx]
            body = record["response"]["body"]

            # Extract the structured text payload returned by the LLM
            output_text = extract_output_text(body)
            if not output_text:
                print(f"Skipping {custom_id}: no output_text")
                continue

            try:
                obj = json.loads(output_text)

            except Exception:
                try:
                    # Parse the LLM JSON output; attempt repair if the response is malformed
                    repaired = repair_llm_json(output_text)
                    obj = json.loads(repaired)
                    # Attempt lightweight recovery for malformed but recoverable LLM JSON responses
                    print(f"Repaired JSON for {custom_id}")
                except Exception as e:
                    print(f"Skipping {custom_id}: invalid JSON - {e}")
                    continue

            # Expand example-level outputs into one row per generated example
            examples = obj.get("examples", [])
            if len(examples) != 8:
                print(f"Warning: {custom_id} returned {len(examples)} examples")

            for ex in examples:
                # Combine source metadata, idiom-level enrichment, and example-level enrichment
                row = {
                    "idiom_id": raw_row.get("idiom_id", ""),
                    "idiom_canonical": obj.get("idiom_canonical", raw_row.get("idiom_canonical", "")),
                    "example": raw_row.get("example", ""),
                    "source": raw_row.get("source", ""),
                    "source_type": raw_row.get("source_type", ""),
                    "pos": raw_row.get("pos", ""),
                    "tags": raw_row.get("tags", ""),
                    "idiom_confidence_score": raw_row.get("idiom_confidence_score", ""),
                    "record_origin": "llm_enriched",
                    "license_source": raw_row.get("license_source", ""),
                    "example_language": raw_row.get("example_language", "en"),
                    "meaning_language": raw_row.get("meaning_language", "en"),

                    "idiom_canonical_meaning": obj.get("idiom_canonical_meaning", ""),
                    "idiom_canonical_meaning_arabic": obj.get("idiom_canonical_meaning_arabic", ""),

                    "is_idiom": obj.get("is_idiom", None),
                    "ambiguity_flag": obj.get("ambiguity_flag", ""),
                    "idiom_compositionality_level": obj.get("idiom_compositionality_level", ""),
                    "idiom_register": obj.get("idiom_register", ""),
                    "idiom_domain": obj.get("idiom_domain", ""),
                    "learner_difficulty": obj.get("learner_difficulty", ""),

                    "idiom_surface": ex.get("idiom_surface", ""),
                    "idiom_in_example": ex.get("idiom_in_example", ""),
                    "idiom_in_example_arabic": ex.get("idiom_in_example_arabic", ""),
                    "idiom_in_example_meaning_en": ex.get("idiom_in_example_meaning_en", ""),
                    "idiom_in_example_meaning_arabic": ex.get("idiom_in_example_meaning_arabic", ""),
                    "is_example_idiom": ex.get("is_example_idiom", None),
                    "example_usage_label": ex.get("example_usage_label", ""),
                    "is_generated_example": 1,
                    "enrichment_model": body.get("model", "gpt-4.1-mini"),
                    "enrichment_version": "v2",
                    "validation_status": "pending"
                }

                rows.append(row)

    out_df = pd.DataFrame(rows)

    # Reorder columns to match the final IdiomX enriched schema
    out_df = out_df.reindex(columns=FINAL_COLUMNS)

    # Save enriched dataset in both CSV and Parquet formats
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    out_df.to_parquet(output_parquet, index=False)

    print("Saved CSV:", output_csv)
    print("Saved Parquet:", output_parquet)
    print("Total rows:", len(out_df))

    if not out_df.empty:
        print(out_df["example_usage_label"].value_counts(dropna=False))

    return out_df


if __name__ == "__main__":
    merge_results()