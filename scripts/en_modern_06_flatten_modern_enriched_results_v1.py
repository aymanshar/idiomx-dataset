#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_JSON = BASE_DIR / "data" / "results" / "idiomx_modern_enriched_full_v1.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_modern_enriched_full_v1.csv"


def flatten_results(input_json: Path, output_csv: Path):
    input_json = Path(input_json)
    output_csv = Path(output_csv)

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        idiom_id = item.get("_custom_id", "")
        idiom_canonical = item.get("idiom_canonical", "")
        is_idiom = item.get("is_idiom", False)
        validity = item.get("idiom_validity_label", "")

        base = {
            "idiom_id": idiom_id,
            "is_idiom": is_idiom,
            "idiom_validity_label": validity,
            "idiom_canonical": idiom_canonical,
            "idiom_canonical_meaning": item.get("idiom_canonical_meaning", ""),
            "idiom_canonical_meaning_arabic": item.get("idiom_canonical_meaning_arabic", ""),
            "idiom_canonical_meaning_french": item.get("idiom_canonical_meaning_french", ""),
            "ambiguity_flag": item.get("ambiguity_flag", ""),
            "idiom_compositionality_level": item.get("idiom_compositionality_level", ""),
            "idiom_register": item.get("idiom_register", ""),
            "idiom_domain": item.get("idiom_domain", ""),
            "learner_difficulty": item.get("learner_difficulty", ""),
            "slang_strength": item.get("slang_strength", ""),
            "regionality": item.get("regionality", ""),
            "offensive_flag": item.get("offensive_flag", ""),
            "idiom_level_explanation_en": item.get("explanation_en", ""),
            "idiom_level_explanation_ar": item.get("explanation_ar", ""),
            "idiom_level_explanation_fr": item.get("explanation_fr", ""),
            "hard_negative_idioms": json.dumps(item.get("hard_negative_idioms", []), ensure_ascii=False),
            "meaning_paraphrases_en": json.dumps(item.get("meaning_paraphrases_en", []), ensure_ascii=False),
            "meaning_paraphrases_ar": json.dumps(item.get("meaning_paraphrases_ar", []), ensure_ascii=False),
            "meaning_paraphrases_fr": json.dumps(item.get("meaning_paraphrases_fr", []), ensure_ascii=False),
        }

        examples = item.get("examples", [])
        adv_examples = item.get("adversarial_examples", [])

        if not is_idiom:
            rows.append({
                **base,
                "row_type": "idiom_only",
                "context_type": "",
                "source_style": "",
                "idiom_surface": "",
                "idiom_in_example": "",
                "idiom_in_example_arabic": "",
                "idiom_in_example_french": "",
                "idiom_in_example_meaning_en": "",
                "idiom_in_example_meaning_arabic": "",
                "idiom_in_example_meaning_french": "",
                "explanation_en": "",
                "explanation_ar": "",
                "explanation_fr": "",
                "is_example_idiom": "",
                "example_usage_label": "",
                "expected_label": "",
                "adversarial_type": "",
            })
            continue

        for ex in examples:
            rows.append({
                **base,
                "row_type": "main_example",
                "context_type": ex.get("context_type", ""),
                "source_style": ex.get("source_style", ""),
                "idiom_surface": ex.get("idiom_surface", ""),
                "idiom_in_example": ex.get("idiom_in_example", ""),
                "idiom_in_example_arabic": ex.get("idiom_in_example_arabic", ""),
                "idiom_in_example_french": ex.get("idiom_in_example_french", ""),
                "idiom_in_example_meaning_en": ex.get("idiom_in_example_meaning_en", ""),
                "idiom_in_example_meaning_arabic": ex.get("idiom_in_example_meaning_arabic", ""),
                "idiom_in_example_meaning_french": ex.get("idiom_in_example_meaning_french", ""),
                "explanation_en": ex.get("explanation_en", ""),
                "explanation_ar": ex.get("explanation_ar", ""),
                "explanation_fr": ex.get("explanation_fr", ""),
                "is_example_idiom": ex.get("is_example_idiom", ""),
                "example_usage_label": ex.get("example_usage_label", ""),
                "expected_label": "",
                "adversarial_type": "",
            })

        for ex in adv_examples:
            rows.append({
                **base,
                "row_type": "adversarial_example",
                "context_type": "adversarial",
                "source_style": ex.get("source_style", ""),
                "idiom_surface": idiom_canonical,
                "idiom_in_example": ex.get("idiom_in_example", ""),
                "idiom_in_example_arabic": ex.get("idiom_in_example_arabic", ""),
                "idiom_in_example_french": ex.get("idiom_in_example_french", ""),
                "idiom_in_example_meaning_en": ex.get("idiom_in_example_meaning_en", ""),
                "idiom_in_example_meaning_arabic": ex.get("idiom_in_example_meaning_arabic", ""),
                "idiom_in_example_meaning_french": ex.get("idiom_in_example_meaning_french", ""),
                "explanation_en": ex.get("explanation_en", ""),
                "explanation_ar": ex.get("explanation_ar", ""),
                "explanation_fr": ex.get("explanation_fr", ""),
                "is_example_idiom": "",
                "example_usage_label": "",
                "expected_label": ex.get("expected_label", ""),
                "adversarial_type": ex.get("adversarial_type", ""),
            })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Saved flattened CSV:", output_csv)
    print("Rows:", len(df))

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Flatten parsed modern enrichment results.")
    parser.add_argument("--input-json", type=str, default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flatten_results(
        input_json=Path(args.input_json),
        output_csv=Path(args.output_csv),
    )