from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "raw" / "idiomx_dataset_v1.csv"
OUTPUT_JSONL = BASE_DIR / "data" / "batches" / "idiomx_batch_v2.jsonl"

MODEL_NAME = "gpt-4.1-mini"

def build_prompt(idiom: str, meaning_en: str, example: str) -> str:
    return f"""
You are an expert linguist specializing in English idioms, literal language usage, bilingual semantics, and dataset construction for NLP research.

Your task is to enrich the following idiom entry into a high-quality structured dataset.

Input:
- idiom: "{idiom}"
- existing_meaning_en: "{meaning_en}"
- existing_example: "{example}"

Instructions:

A. Idiom-level analysis
1. Determine whether this phrase is truly an idiom.
2. Normalize the canonical idiom form.
3. Provide the best canonical English idiomatic meaning.
4. Translate the canonical idiomatic meaning into natural professional Arabic.
5. Assign an ambiguity flag:
   - strongly_idiomatic
   - ambiguous
   - semi_literal_possible
6. Assign compositionality level:
   - opaque
   - semi_opaque
   - transparent
7. Assign idiom register:
   - formal
   - neutral
   - informal
   - slang
   - archaic
8. Assign idiom domain:
   - general
   - business
   - sports
   - military
   - religious
   - biblical
   - regional
   - legal
   - politics
   - internet
9. Assign learner difficulty:
   - easy
   - medium
   - hard

B. Example generation
Generate exactly:
- 4 idiomatic example sentences
- 4 literal example sentences

For EACH generated example:
1. Extract the exact idiom surface used in the sentence.
2. Provide the English sentence.
3. Provide a natural Arabic translation of the sentence.
4. Provide the meaning of the sentence usage in English:
   - if idiomatic: the intended figurative meaning in that sentence
   - if literal: the literal meaning in plain English
5. Provide the Arabic translation of that sentence meaning.
6. Mark whether the example is idiomatic or literal.

Important rules:
- The 4 idiomatic examples must be clearly figurative.
- The 4 literal examples must be clearly literal uses of the same phrase or close surface form.
- Keep examples natural, diverse, and grammatically correct.
- Arabic must be natural and professional.
- Return JSON only.
"""

def get_response_schema():
    return {
        "type": "json_schema",
        "name": "idiomx_v2_enrichment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_idiom": {"type": "boolean"},
                "idiom_canonical": {"type": "string"},
                "idiom_canonical_meaning": {"type": "string"},
                "idiom_canonical_meaning_arabic": {"type": "string"},
                "ambiguity_flag": {
                    "type": "string",
                    "enum": ["strongly_idiomatic", "ambiguous", "semi_literal_possible"]
                },
                "idiom_compositionality_level": {
                    "type": "string",
                    "enum": ["opaque", "semi_opaque", "transparent"]
                },
                "idiom_register": {
                    "type": "string",
                    "enum": ["formal", "neutral", "informal", "slang", "archaic"]
                },
                "idiom_domain": {
                    "type": "string",
                    "enum": ["general", "business", "sports", "military", "religious", "biblical", "regional", "legal", "politics", "internet"]
                },
                "learner_difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"]
                },
                "examples": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 8,
                    "items": {
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
                }
            },
            "required": [
                "is_idiom",
                "idiom_canonical",
                "idiom_canonical_meaning",
                "idiom_canonical_meaning_arabic",
                "ambiguity_flag",
                "idiom_compositionality_level",
                "idiom_register",
                "idiom_domain",
                "learner_difficulty",
                "examples"
            ],
            "additionalProperties": False
        }
    }

def prepare_batch_requests(input_csv: Path = INPUT_CSV, output_jsonl: Path = OUTPUT_JSONL):
    df = pd.read_csv(input_csv)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    schema = get_response_schema()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            idiom = str(row.get("idiom", "") if pd.notna(row.get("idiom", "")) else "")
            meaning_en = str(row.get("meaning_en", "") if pd.notna(row.get("meaning_en", "")) else "")
            example = str(row.get("example", "") if pd.notna(row.get("example", "")) else "")

            prompt = build_prompt(idiom, meaning_en, example)

            request = {
                "custom_id": f"idiomx_v2_{idx}",
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

    print(f"Saved batch file to: {output_jsonl}")
    print(f"Total requests: {len(df)}")

if __name__ == "__main__":
    prepare_batch_requests()