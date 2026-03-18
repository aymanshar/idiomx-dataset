from pathlib import Path
import json
import pandas as pd
from llm_enrichment.config.api_config import client
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2_validated.csv"
OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2_final.csv"

SCHEMA = {
    "type": "json_schema",
    "name": "idiomx_v2_row_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_valid": {"type": "boolean"},
            "correction_needed": {"type": "boolean"},
            "corrected_entry": {
                "type": "object",
                "properties": {
                    "idiom_surface": {"type": "string"},
                    "idiom_in_example": {"type": "string"},
                    "idiom_in_example_arabic": {"type": "string"},
                    "idiom_in_example_meaning_en": {"type": "string"},
                    "idiom_in_example_meaning_arabic": {"type": "string"},
                    "is_example_idiom": {"type": "boolean"},
                    "example_usage_label": {
                        "type": "string",
                        "enum": ["idiomatic", "literal"]
                    }
                },
                "required": [
                    "idiom_surface",
                    "idiom_in_example",
                    "idiom_in_example_arabic",
                    "idiom_in_example_meaning_en",
                    "idiom_in_example_meaning_arabic",
                    "is_example_idiom",
                    "example_usage_label"
                ],
                "additionalProperties": False
            }
        },
        "required": ["is_valid", "correction_needed", "corrected_entry"],
        "additionalProperties": False
    }
}

def verify_suspicious_rows(input_csv: Path = INPUT_CSV,
                           output_csv: Path = OUTPUT_CSV):
    df = pd.read_csv(input_csv, low_memory=False)
    review_df = df[df["validation_status"] == "needs_review"].copy()
    from tqdm import tqdm
    for idx, row in tqdm(review_df.iterrows(), total=len(review_df)):
        prompt = f"""
You are verifying a row in the IdiomX v2 dataset.

Check whether the entry is valid.

Focus on:
1. Is the example natural and grammatically correct?
2. Does idiom_surface match the wording used in idiom_in_example?
3. Is example_usage_label correct?
4. Is is_example_idiom consistent with the example?
5. Is the English meaning correct for that example usage?
6. Is the Arabic translation natural and correct?

Return JSON only.

idiom_canonical: {row.get('idiom_canonical', '')}
idiom_canonical_meaning: {row.get('idiom_canonical_meaning', '')}
ambiguity_flag: {row.get('ambiguity_flag', '')}
idiom_compositionality_level: {row.get('idiom_compositionality_level', '')}

idiom_surface: {row.get('idiom_surface', '')}
idiom_in_example: {row.get('idiom_in_example', '')}
idiom_in_example_arabic: {row.get('idiom_in_example_arabic', '')}
idiom_in_example_meaning_en: {row.get('idiom_in_example_meaning_en', '')}
idiom_in_example_meaning_arabic: {row.get('idiom_in_example_meaning_arabic', '')}
is_example_idiom: {row.get('is_example_idiom', '')}
example_usage_label: {row.get('example_usage_label', '')}
"""

        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                text={"format": SCHEMA}
            )

            obj = json.loads(response.output_text)

            if obj["is_valid"]:
                df.loc[idx, "validation_status"] = "verified"
            else:
                df.loc[idx, "validation_status"] = "invalid"

            if obj["correction_needed"]:
                corrected = obj["corrected_entry"]
                for k, v in corrected.items():
                    df.loc[idx, k] = v
                df.loc[idx, "validation_status"] = "corrected"

        except Exception as e:
            df.loc[idx, "validation_status"] = f"verification_error: {e}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved final v2 dataset to: {output_csv}")
    print(df["validation_status"].value_counts(dropna=False))

if __name__ == "__main__":
    verify_suspicious_rows()