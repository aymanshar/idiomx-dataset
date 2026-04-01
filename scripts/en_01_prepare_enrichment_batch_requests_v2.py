"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Prepare advanced batch requests for IdiomX enrichment v2.

This script is safe to run offline.
It does NOT call the API.
It reads the pre-enrichment dataset and writes JSONL batch requests
for a richer enrichment schema with:
- 6 idiomatic examples
- 6 literal examples
- fixed context-style coverage
- hard negative idioms
- paraphrase augmentation
- adversarial examples
- richer bilingual explanations

Supports:
- notebook execution with explicit paths
- command-line execution
- sample mode and full mode

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

from pathlib import Path
import json
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# Default full-data paths
DEFAULT_FULL_INPUT_FILE = BASE_DIR / "data" / "processed" / "idiomx_pre_enrichment.parquet"
DEFAULT_FULL_OUTPUT_JSONL = BASE_DIR / "data" / "batches" / "idiomx_batch_v2.jsonl"

# Default sample-data paths
DEFAULT_SAMPLE_DIR = BASE_DIR / "data" / "sample"
DEFAULT_SAMPLE_INPUT_FILE = DEFAULT_SAMPLE_DIR / "idiomx_pre_enrichment_sample_v2.parquet"
DEFAULT_SAMPLE_OUTPUT_JSONL = DEFAULT_SAMPLE_DIR / "idiomx_batch_sample_v2.jsonl"

MODEL_NAME = "gpt-4.1-mini"
MAX_REQUESTS_PER_BATCH = 2000

CONTEXT_TYPES = [
    "dialogue",
    "narrative",
    "formal",
    "social_media",
    "question",
    "sarcastic",
]

SOURCE_STYLES = [
    "synthetic_dialogue",
    "synthetic_narrative",
    "synthetic_formal",
    "synthetic_social_media",
    "synthetic_question",
    "synthetic_sarcastic",
    "synthetic_adversarial",
]


def build_prompt(idiom: str, meaning_en: str, example: str) -> str:
    """
    Build the advanced enrichment prompt for one idiom entry.
    """
    context_list = ", ".join(CONTEXT_TYPES)

    return f"""
You are an expert linguist, bilingual lexicographer, benchmark dataset designer, and figurative-language specialist for English idioms and Arabic semantic alignment.

Your task is to enrich ONE idiom entry into a high-quality structured research dataset.

INPUT
- idiom: "{idiom}"
- existing_meaning_en: "{meaning_en}"
- existing_example: "{example}"

OBJECTIVES
1. Determine whether the phrase is truly idiomatic.
2. Normalize the best canonical idiom form.
3. Provide the best canonical English idiomatic meaning.
4. Provide a natural professional Arabic translation of the canonical meaning.
5. Provide semantically confusable hard negative idioms.
6. Provide semantic paraphrases of the idiomatic meaning.
7. Provide a small adversarial set of borderline or tricky examples.
8. Generate a controlled balanced set of idiomatic and literal examples across discourse styles.

IDIOM-LEVEL OUTPUT
Return:
- is_idiom
- idiom_canonical
- idiom_canonical_meaning
- idiom_canonical_meaning_arabic
- ambiguity_flag:
  strongly_idiomatic / ambiguous / semi_literal_possible
- idiom_compositionality_level:
  opaque / semi_opaque / transparent
- idiom_register:
  formal / neutral / informal / slang / archaic
- idiom_domain:
  general / business / sports / military / religious / biblical / regional / legal / politics / internet
- learner_difficulty:
  easy / medium / hard
- explanation_en:
  brief explanation of why this expression is idiomatic
- explanation_ar:
  Arabic explanation of why this expression is idiomatic
- hard_negative_idioms:
  exactly 3 semantically confusable idioms, distinct from the target idiom
- meaning_paraphrases_en:
  exactly 3 short English paraphrases of the idiomatic meaning
- meaning_paraphrases_ar:
  exactly 3 Arabic paraphrases of the idiomatic meaning

MAIN EXAMPLES
Generate EXACTLY 12 examples total:
- 6 idiomatic examples
- 6 literal examples

Use EXACTLY these context types, once each for idiomatic and once each for literal:
{context_list}

For each context_type, create:
- 1 idiomatic example
- 1 literal example

PAIRING RULE
For each context_type, the idiomatic and literal examples should feel like a matched pair in topic/style, but not in meaning.

FOR EACH MAIN EXAMPLE RETURN
- context_type
- source_style
- idiom_surface
- idiom_in_example
- idiom_in_example_arabic
- idiom_in_example_meaning_en
- idiom_in_example_meaning_arabic
- explanation_en
- explanation_ar
- is_example_idiom
- example_usage_label

ADVERSARIAL EXAMPLES
Generate EXACTLY 2 adversarial examples that are intentionally tricky or borderline.
These may include:
- negation
- interruption
- partial literal overlap
- context that could confuse a model

These adversarial examples should still be natural and realistic.

FOR EACH ADVERSARIAL EXAMPLE RETURN
- source_style
- idiom_in_example
- idiom_in_example_arabic
- idiom_in_example_meaning_en
- idiom_in_example_meaning_arabic
- explanation_en
- explanation_ar
- expected_label
- adversarial_type

STRICT RULES
- Each main example must be natural, grammatically correct, and meaningfully different.
- Do NOT reuse the same sentence template across context types.
- Avoid repetitive openings such as "He was..." and "She was..." unless truly necessary.
- Make the idiomatic examples clearly figurative.
- Make the literal examples clearly literal and non-idiomatic.
- The literal example should use the same phrase or a very close surface form naturally.
- Arabic must be natural Modern Standard Arabic.
- The meaning field must describe the actual meaning in context, not just repeat the idiom.
- Keep sentences reasonably concise, usually 8 to 20 words.
- Avoid profanity unless strongly required by the idiom itself.
- Return valid JSON only.
""".strip()


def get_response_schema():
    """
    Strict JSON schema for advanced enrichment v2.
    """
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
                    "enum": ["strongly_idiomatic", "ambiguous", "semi_literal_possible"],
                },
                "idiom_compositionality_level": {
                    "type": "string",
                    "enum": ["opaque", "semi_opaque", "transparent"],
                },
                "idiom_register": {
                    "type": "string",
                    "enum": ["formal", "neutral", "informal", "slang", "archaic"],
                },
                "idiom_domain": {
                    "type": "string",
                    "enum": [
                        "general", "business", "sports", "military", "religious",
                        "biblical", "regional", "legal", "politics", "internet"
                    ],
                },
                "learner_difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                },
                "explanation_en": {"type": "string"},
                "explanation_ar": {"type": "string"},
                "hard_negative_idioms": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "meaning_paraphrases_en": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "meaning_paraphrases_ar": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "examples": {
                    "type": "array",
                    "minItems": 12,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "context_type": {
                                "type": "string",
                                "enum": CONTEXT_TYPES,
                            },
                            "source_style": {
                                "type": "string",
                                "enum": SOURCE_STYLES[:6],
                            },
                            "idiom_surface": {"type": "string"},
                            "idiom_in_example": {"type": "string"},
                            "idiom_in_example_arabic": {"type": "string"},
                            "idiom_in_example_meaning_en": {"type": "string"},
                            "idiom_in_example_meaning_arabic": {"type": "string"},
                            "explanation_en": {"type": "string"},
                            "explanation_ar": {"type": "string"},
                            "is_example_idiom": {"type": "boolean"},
                            "example_usage_label": {
                                "type": "string",
                                "enum": ["idiomatic", "literal"],
                            },
                        },
                        "required": [
                            "context_type",
                            "source_style",
                            "idiom_surface",
                            "idiom_in_example",
                            "idiom_in_example_arabic",
                            "idiom_in_example_meaning_en",
                            "idiom_in_example_meaning_arabic",
                            "explanation_en",
                            "explanation_ar",
                            "is_example_idiom",
                            "example_usage_label",
                        ],
                        "additionalProperties": False,
                    },
                },
                "adversarial_examples": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_style": {
                                "type": "string",
                                "enum": ["synthetic_adversarial"],
                            },
                            "idiom_in_example": {"type": "string"},
                            "idiom_in_example_arabic": {"type": "string"},
                            "idiom_in_example_meaning_en": {"type": "string"},
                            "idiom_in_example_meaning_arabic": {"type": "string"},
                            "explanation_en": {"type": "string"},
                            "explanation_ar": {"type": "string"},
                            "expected_label": {
                                "type": "string",
                                "enum": ["idiomatic", "literal", "borderline"],
                            },
                            "adversarial_type": {
                                "type": "string",
                                "enum": [
                                    "negation",
                                    "partial_overlap",
                                    "contrastive_context",
                                    "literal_trap",
                                    "figurative_trap",
                                    "borderline_ambiguity",
                                ],
                            },
                        },
                        "required": [
                            "source_style",
                            "idiom_in_example",
                            "idiom_in_example_arabic",
                            "idiom_in_example_meaning_en",
                            "idiom_in_example_meaning_arabic",
                            "explanation_en",
                            "explanation_ar",
                            "expected_label",
                            "adversarial_type",
                        ],
                        "additionalProperties": False,
                    },
                },
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
                "explanation_en",
                "explanation_ar",
                "hard_negative_idioms",
                "meaning_paraphrases_en",
                "meaning_paraphrases_ar",
                "examples",
                "adversarial_examples",
            ],
            "additionalProperties": False,
        },
    }


def create_sample_dataset(
    full_input_file: Path = DEFAULT_FULL_INPUT_FILE,
    sample_output_file: Path = DEFAULT_SAMPLE_INPUT_FILE,
    n_rows: int = 10,
    method: str = "head",
    random_state: int = 42,
):
    """
    Create a small sample dataset from the full pre-enrichment dataset.
    """
    full_input_file = Path(full_input_file)
    sample_output_file = Path(sample_output_file)

    if not full_input_file.exists():
        raise FileNotFoundError(f"Full input file not found: {full_input_file}")

    df = pd.read_parquet(full_input_file)

    if df.empty:
        raise ValueError("Full dataset is empty; cannot create sample.")

    if method == "random":
        n_rows = min(n_rows, len(df))
        df_sample = df.sample(n=n_rows, random_state=random_state).copy()
    else:
        df_sample = df.head(n_rows).copy()

    sample_output_file.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_parquet(sample_output_file, index=False)

    print(f"Sample dataset created: {sample_output_file}")
    print(f"Sample shape: {df_sample.shape}")
    return sample_output_file


def get_mode_paths(use_sample: bool = False):
    """
    Return input and output paths based on execution mode.
    """
    if use_sample:
        return DEFAULT_SAMPLE_INPUT_FILE, DEFAULT_SAMPLE_OUTPUT_JSONL
    return DEFAULT_FULL_INPUT_FILE, DEFAULT_FULL_OUTPUT_JSONL

def prepare_batch_requests(
    input_file: Path = None,
    output_jsonl: Path = None,
    use_sample: bool = False,
):
    """
    Convert the input dataset into JSONL batch requests for the LLM API.
    """
    if input_file is None or output_jsonl is None:
        default_input, default_output = get_mode_paths(use_sample=use_sample)
        input_file = input_file or default_input
        output_jsonl = output_jsonl or default_output

    input_file = Path(input_file)
    output_jsonl = Path(output_jsonl)

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    df = pd.read_parquet(input_file)

    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_file}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if output_jsonl.exists():
        print(f"[WARNING] Output batch file already exists and will be overwritten: {output_jsonl}")

    schema = get_response_schema()
    requests = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            idiom = str(row.get("idiom_canonical", "") if pd.notna(row.get("idiom_canonical", "")) else "")
            meaning_en = str(
                row.get("idiom_canonical_meaning", "")
                if pd.notna(row.get("idiom_canonical_meaning", ""))
                else ""
            )
            example = str(row.get("example", "") if pd.notna(row.get("example", "")) else "")

            row_id = row.get("idiom_id", None)
            custom_id = str(row_id) if pd.notna(row_id) else f"idiomx_v2_{idx}"

            prompt = build_prompt(idiom, meaning_en, example)

            request = {
                "custom_id": custom_id,
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
    print(f"Input dataset: {input_file}")
    print(f"Total requests: {len(df)}")
    return output_jsonl


def parse_args():
    """
    Parse command-line arguments for preparing enrichment batch requests.
    """
    parser = argparse.ArgumentParser(description="Prepare IdiomX enrichment batch requests v2.")
    parser.add_argument("--sample", action="store_true", help="Use sample dataset mode.")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size if sample mode is used.")
    parser.add_argument("--sample-method", type=str, default="random", choices=["head", "random"])
    parser.add_argument("--input-file", type=str, default=None, help="Custom input parquet file.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Custom output JSONL path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_file = Path(args.input_file) if args.input_file else None
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None

    if args.sample:
        print("\n[INFO] Running in SAMPLE mode")
        create_sample_dataset(
            full_input_file=DEFAULT_FULL_INPUT_FILE,
            sample_output_file=DEFAULT_SAMPLE_INPUT_FILE,
            n_rows=args.sample_size,
            method=args.sample_method,
            random_state=42,
        )

    prepare_batch_requests(
        input_file=input_file,
        output_jsonl=output_jsonl,
        use_sample=args.sample,
    )