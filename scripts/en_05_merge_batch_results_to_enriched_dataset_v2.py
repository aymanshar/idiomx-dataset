"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Merge LLM batch results into the final enriched IdiomX dataset (v2).

This script aligns raw batch outputs with the original pre-enrichment dataset,
repairs malformed JSON if needed, expands generated examples into structured rows,
and exports the enriched dataset for downstream analysis and modeling.

Supports:
- notebook execution with explicit paths
- command-line execution
- sample mode and full mode

Notes:
This stage transforms schema-constrained LLM outputs into the final row-level
enrichment dataset, enabling direct use in idiom detection, retrieval,
generation, bilingual analysis, and robustness evaluation.
"""

from pathlib import Path
import json
import argparse
import hashlib
from typing import Optional

import pandas as pd
from json_repair import repair_json as repair_llm_json


BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_RAW_FILE = BASE_DIR / "data" / "processed" / "idiomx_pre_enrichment.parquet"
DEFAULT_FULL_RESULTS_JSONL = BASE_DIR / "data" / "results" / "idiomx_results_v2.jsonl"
DEFAULT_FULL_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_v2.csv"
DEFAULT_FULL_OUTPUT_PARQUET = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_v2.parquet"

# Sample-mode defaults
DEFAULT_SAMPLE_RAW_FILE = BASE_DIR / "data" / "sample" / "idiomx_pre_enrichment_sample_v2.parquet"
DEFAULT_SAMPLE_RESULTS_JSONL = BASE_DIR / "data" / "sample" / "idiomx_results_sample_v2.jsonl"
DEFAULT_SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_sample_v2.csv"
DEFAULT_SAMPLE_OUTPUT_PARQUET = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_sample_v2.parquet"


FINAL_COLUMNS = [
    "idiom_id",
    "idiom_canonical",
    "idiom_surface",
    "example",  # original source example from pre-enrichment
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
    "idiom_canonical_meaning_arabic",
    "is_idiom",
    "ambiguity_flag",
    "idiom_compositionality_level",
    "idiom_register",
    "idiom_domain",
    "learner_difficulty",
    "idiom_in_example",
    "idiom_in_example_arabic",
    "idiom_in_example_meaning_en",
    "idiom_in_example_meaning_arabic",
    "is_example_idiom",
    "example_usage_label",
    "is_generated_example",
    "enrichment_model",
    "enrichment_version",
    "validation_status",
    # v2 additions
    "context_type",
    "source_style",
    "hard_negative_idioms",
    "meaning_paraphrases_en",
    "meaning_paraphrases_ar",
    "idiom_level_explanation_en",
    "idiom_level_explanation_ar",
    "explanation_en",
    "explanation_ar",
    "minimal_pair_id",
    "paraphrase_group_id",
    "is_adversarial_example",
    "adversarial_type",
    "expected_label",
    "row_type",
]


def get_mode_paths(use_sample: bool = False) -> tuple[Path, Path, Path, Path]:
    """
    Return raw input, results JSONL, output CSV, and output Parquet paths
    based on execution mode.
    """
    if use_sample:
        return (
            DEFAULT_SAMPLE_RAW_FILE,
            DEFAULT_SAMPLE_RESULTS_JSONL,
            DEFAULT_SAMPLE_OUTPUT_CSV,
            DEFAULT_SAMPLE_OUTPUT_PARQUET,
        )
    return (
        DEFAULT_FULL_RAW_FILE,
        DEFAULT_FULL_RESULTS_JSONL,
        DEFAULT_FULL_OUTPUT_CSV,
        DEFAULT_FULL_OUTPUT_PARQUET,
    )


def extract_output_text(body: dict) -> Optional[str]:
    """
    Extract the generated text content from the batch response body.
    """
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    return content_item.get("text")
    return None


def normalize_json_list(value) -> str:
    """
    Store list-like values as JSON strings for CSV/Parquet compatibility.
    """
    if value is None:
        return "[]"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps([value], ensure_ascii=False)


def make_pair_id(idiom_id: str, context_type: str) -> str:
    raw = f"{idiom_id}||{context_type}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"pair_{digest}"


def make_paraphrase_group_id(idiom_id: str) -> str:
    digest = hashlib.md5(f"{idiom_id}||paraphrase".encode("utf-8")).hexdigest()[:12]
    return f"paraphrase_{digest}"


def safe_expected_label_to_bool(expected_label: str):
    """
    Convert adversarial expected label to boolean if possible.
    """
    val = str(expected_label).strip().lower()
    if val == "idiomatic":
        return True
    if val == "literal":
        return False
    return pd.NA


def build_raw_lookup(raw_df: pd.DataFrame) -> tuple[dict, bool]:
    """
    Build a lookup by idiom_id when available.
    Returns (mapping, uses_idiom_id).
    """
    if "idiom_id" in raw_df.columns and raw_df["idiom_id"].notna().any():
        mapping = {
            str(row["idiom_id"]): row
            for _, row in raw_df.iterrows()
            if pd.notna(row.get("idiom_id"))
        }
        return mapping, True
    return {}, False


def get_raw_row(custom_id: str, raw_df: pd.DataFrame, raw_lookup: dict, uses_idiom_id: bool):
    """
    Resolve the source row from custom_id.
    Priority:
    1) direct idiom_id lookup
    2) fallback to suffix index parse
    """
    if uses_idiom_id and custom_id in raw_lookup:
        return raw_lookup[custom_id]

    try:
        idx = int(str(custom_id).split("_")[-1])
        return raw_df.iloc[idx]
    except Exception:
        return None


def merge_results(
    raw_file: Path = None,
    results_jsonl: Path = None,
    output_csv: Path = None,
    output_parquet: Path = None,
    use_sample: bool = False,
):
    """
    Merge LLM batch outputs with the raw pre-enrichment dataset.
    """
    default_raw_file, default_results_jsonl, default_output_csv, default_output_parquet = get_mode_paths(
        use_sample=use_sample
    )

    raw_file = Path(raw_file) if raw_file is not None else default_raw_file
    results_jsonl = Path(results_jsonl) if results_jsonl is not None else default_results_jsonl
    output_csv = Path(output_csv) if output_csv is not None else default_output_csv
    output_parquet = Path(output_parquet) if output_parquet is not None else default_output_parquet

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    if not results_jsonl.exists():
        raise FileNotFoundError(f"Results JSONL not found: {results_jsonl}")

    print("Raw file:", raw_file)
    print("Results file:", results_jsonl)

    raw_df = pd.read_parquet(raw_file)
    raw_lookup, uses_idiom_id = build_raw_lookup(raw_df)

    rows = []

    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            custom_id = str(record.get("custom_id", "")).strip()

            raw_row = get_raw_row(custom_id, raw_df, raw_lookup, uses_idiom_id)
            if raw_row is None:
                print(f"Skipping line {line_num}: could not resolve raw row for custom_id={custom_id}")
                continue

            response = record.get("response", {})
            body = response.get("body", {})

            output_text = extract_output_text(body)
            if not output_text:
                print(f"Skipping {custom_id}: no output_text")
                continue

            try:
                obj = json.loads(output_text)
            except Exception:
                try:
                    repaired = repair_llm_json(output_text)
                    obj = json.loads(repaired)
                    print(f"Repaired JSON for {custom_id}")
                except Exception as e:
                    print(f"Skipping {custom_id}: invalid JSON - {e}")
                    continue

            idiom_id = str(raw_row.get("idiom_id", custom_id))
            paraphrase_group_id = make_paraphrase_group_id(idiom_id)

            idiom_level = {
                "idiom_id": idiom_id,
                "idiom_canonical": obj.get("idiom_canonical", raw_row.get("idiom_canonical", "")),
                "example": raw_row.get("example", ""),
                "source": raw_row.get("source", ""),
                "source_type": raw_row.get("source_type", ""),
                "pos": raw_row.get("pos", ""),
                "tags": raw_row.get("tags", ""),
                "idiom_confidence": raw_row.get("idiom_confidence", raw_row.get("idiom_confidence_score", "")),
                "source_url": raw_row.get("source_url", ""),
                "record_origin": "llm_enriched_v2",
                "license_source": raw_row.get("license_source", ""),
                "example_language": raw_row.get("example_language", "en"),
                "meaning_language": raw_row.get("meaning_language", "en"),
                "idiom_canonical_meaning": obj.get("idiom_canonical_meaning", raw_row.get("idiom_canonical_meaning", "")),
                "idiom_canonical_meaning_arabic": obj.get("idiom_canonical_meaning_arabic", ""),
                "is_idiom": obj.get("is_idiom", None),
                "ambiguity_flag": obj.get("ambiguity_flag", ""),
                "idiom_compositionality_level": obj.get("idiom_compositionality_level", ""),
                "idiom_register": obj.get("idiom_register", ""),
                "idiom_domain": obj.get("idiom_domain", ""),
                "learner_difficulty": obj.get("learner_difficulty", ""),
                "hard_negative_idioms": normalize_json_list(obj.get("hard_negative_idioms", [])),
                "meaning_paraphrases_en": normalize_json_list(obj.get("meaning_paraphrases_en", [])),
                "meaning_paraphrases_ar": normalize_json_list(obj.get("meaning_paraphrases_ar", [])),
                "idiom_level_explanation_en": obj.get("explanation_en", ""),
                "idiom_level_explanation_ar": obj.get("explanation_ar", ""),
                "paraphrase_group_id": paraphrase_group_id,
                "is_generated_example": 1,
                "enrichment_model": body.get("model", "gpt-4.1-mini"),
                "enrichment_version": "v2",
                "validation_status": "pending",
            }

            examples = obj.get("examples", [])
            if len(examples) != 12:
                print(f"Warning: {custom_id} returned {len(examples)} main examples (expected 12)")

            for ex in examples:
                context_type = ex.get("context_type", "")
                minimal_pair_id = make_pair_id(idiom_id, context_type)

                row = {
                    **idiom_level,
                    "idiom_surface": ex.get("idiom_surface", ""),
                    "idiom_in_example": ex.get("idiom_in_example", ""),
                    "idiom_in_example_arabic": ex.get("idiom_in_example_arabic", ""),
                    "idiom_in_example_meaning_en": ex.get("idiom_in_example_meaning_en", ""),
                    "idiom_in_example_meaning_arabic": ex.get("idiom_in_example_meaning_arabic", ""),
                    "is_example_idiom": ex.get("is_example_idiom", None),
                    "example_usage_label": ex.get("example_usage_label", ""),
                    "context_type": context_type,
                    "source_style": ex.get("source_style", ""),
                    "explanation_en": ex.get("explanation_en", ""),
                    "explanation_ar": ex.get("explanation_ar", ""),
                    "minimal_pair_id": minimal_pair_id,
                    "is_adversarial_example": 0,
                    "adversarial_type": "",
                    "expected_label": ex.get("example_usage_label", ""),
                    "row_type": "main_example",
                }
                rows.append(row)

            adversarial_examples = obj.get("adversarial_examples", [])
            if len(adversarial_examples) != 2:
                print(f"Warning: {custom_id} returned {len(adversarial_examples)} adversarial examples (expected 2)")

            for adv_idx, adv in enumerate(adversarial_examples, start=1):
                row = {
                    **idiom_level,
                    "idiom_surface": pd.NA,
                    "idiom_in_example": adv.get("idiom_in_example", ""),
                    "idiom_in_example_arabic": adv.get("idiom_in_example_arabic", ""),
                    "idiom_in_example_meaning_en": adv.get("idiom_in_example_meaning_en", ""),
                    "idiom_in_example_meaning_arabic": adv.get("idiom_in_example_meaning_arabic", ""),
                    "is_example_idiom": safe_expected_label_to_bool(adv.get("expected_label", "")),
                    "example_usage_label": adv.get("expected_label", ""),
                    "context_type": "adversarial",
                    "source_style": adv.get("source_style", "synthetic_adversarial"),
                    "explanation_en": adv.get("explanation_en", ""),
                    "explanation_ar": adv.get("explanation_ar", ""),
                    "minimal_pair_id": pd.NA,
                    "is_adversarial_example": 1,
                    "adversarial_type": adv.get("adversarial_type", ""),
                    "expected_label": adv.get("expected_label", ""),
                    "row_type": f"adversarial_example_{adv_idx}",
                }
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=FINAL_COLUMNS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    out_df.to_parquet(output_parquet, index=False)

    print("Saved CSV:", output_csv)
    print("Saved Parquet:", output_parquet)
    print("Total rows:", len(out_df))

    if not out_df.empty:
        print("\nexample_usage_label distribution:")
        print(out_df["example_usage_label"].value_counts(dropna=False))
        print("\nrow_type distribution:")
        print(out_df["row_type"].value_counts(dropna=False))

    return out_df


def parse_args():
    """
    Parse command-line arguments for merging batch results.
    """
    parser = argparse.ArgumentParser(description="Merge IdiomX batch results into enriched dataset v2.")
    parser.add_argument("--sample", action="store_true", help="Use sample-mode default paths.")
    parser.add_argument("--raw-file", type=str, default=None, help="Path to raw pre-enrichment parquet.")
    parser.add_argument("--results-jsonl", type=str, default=None, help="Path to downloaded results JSONL.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to output CSV.")
    parser.add_argument("--output-parquet", type=str, default=None, help="Path to output Parquet.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    merge_results(
        raw_file=Path(args.raw_file) if args.raw_file else None,
        results_jsonl=Path(args.results_jsonl) if args.results_jsonl else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        output_parquet=Path(args.output_parquet) if args.output_parquet else None,
        use_sample=args.sample,
    )