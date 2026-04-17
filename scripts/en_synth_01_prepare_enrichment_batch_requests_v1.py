#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
en_synth_01_prepare_enrichment_batch_requests_v1.py

Prepare enrichment batch requests for the synthetic idiom branch.
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# Full-data defaults
DEFAULT_FULL_INPUT_FILE = BASE_DIR / "data" / "processed" / "synthetic_pre_enrichment_for_llm_v1.csv"
DEFAULT_FULL_OUTPUT_JSONL = BASE_DIR / "data" / "batches" / "idiomx_synth_batch_v1.jsonl"

# Sample-data defaults
DEFAULT_SAMPLE_DIR = BASE_DIR / "data" / "sample"
DEFAULT_SAMPLE_INPUT_FILE = DEFAULT_SAMPLE_DIR / "synthetic_pre_enrichment_for_llm_sample_v1.parquet"
DEFAULT_SAMPLE_OUTPUT_JSONL = DEFAULT_SAMPLE_DIR / "data" / "sample" / "idiomx_synth_batch_sample_v1.jsonl"

MODEL_NAME = "gpt-4.1-mini"

CONTEXT_TYPES = [
    "dialogue",
    "narrative",
    "formal",
    "social_media",
    "question",
    "sarcastic",
]

SOURCE_STYLE_BY_CONTEXT = {
    "dialogue": "synthetic_dialogue",
    "narrative": "synthetic_narrative",
    "formal": "synthetic_formal",
    "social_media": "synthetic_social_media",
    "question": "synthetic_question",
    "sarcastic": "synthetic_sarcastic",
}

SOURCE_STYLES = [
    "synthetic_dialogue",
    "synthetic_narrative",
    "synthetic_formal",
    "synthetic_social_media",
    "synthetic_question",
    "synthetic_sarcastic",
    "synthetic_adversarial",
]


def build_prompt(idiom: str, meaning_en: str, example_raw: str) -> str:
    return f"""
You are an expert linguist, lexicographer, figurative-language researcher, multilingual dataset designer, and specialist in English idioms, slang, figurative expressions, Arabic semantic alignment, and French semantic alignment.

Your task is to enrich ONE candidate idiom/slang expression into a high-quality structured research dataset.

INPUT
- candidate_expression: "{idiom}"
- existing_meaning_en: "{meaning_en}"
- existing_source_example_raw: "{example_raw}"

CORE INTERPRETATION RULE
Treat example_raw only as weak optional background evidence.
It may be noisy, incomplete, misleading, literal, repetitive, synthetic, or low-quality.
Do NOT trust it blindly.
Do NOT copy it.
Do NOT lightly rewrite it.
Do NOT preserve its wording, event, or sentence structure.
Generate all final examples from scratch.

IDIOM VALIDITY DECISION
Be strict.

Mark is_idiom = true only if the expression is:
- a genuine idiom
- a slang expression
- a figurative multiword expression
- a culturally recognized non-literal phrase

Mark is_idiom = false if the expression is mainly:
- numeric
- literal
- compositional
- a generic phrase
- a domain term
- a normal collocation
- a weak sentence fragment
- not a real idiomatic/slang expression

VALIDITY BEHAVIOR RULE (CRITICAL)

If is_idiom = true:
- You MUST return exactly 12 main examples and exactly 2 adversarial examples.
- Do NOT return fewer.
- Do NOT return more.
- Do NOT omit any context type.
- Do NOT omit either idiomatic or literal usage.

If is_idiom = false:
- Return "examples": []
- Return "adversarial_examples": []
- Still provide:
  - idiom_canonical
  - idiom_validity_label
  - idiom_canonical_meaning
  - idiom_canonical_meaning_arabic
  - idiom_canonical_meaning_french
  - explanation_en
  - explanation_ar
  - explanation_fr
  - all metadata fields
- Do NOT invent examples.

OBJECTIVES
1. Judge whether the candidate is a real idiom/slang expression.
2. Normalize the best canonical form.
3. Provide English, Arabic, and French idiom meanings.
4. Provide metadata.
5. Generate a balanced, diverse, multilingual example set.

IDIOM-LEVEL OUTPUT
Return:
- is_idiom
- idiom_validity_label: valid_idiom / borderline_expression / not_idiom
- idiom_canonical
- idiom_canonical_meaning
- idiom_canonical_meaning_arabic
- idiom_canonical_meaning_french
- ambiguity_flag: strongly_idiomatic / ambiguous / semi_literal_possible
- idiom_compositionality_level: opaque / semi_opaque / transparent
- idiom_register: formal / neutral / informal / slang / archaic
- idiom_domain: general / business / sports / military / religious / biblical / regional / legal / politics / internet
- learner_difficulty: easy / medium / hard
- slang_strength: none / weak / medium / strong
- regionality: general_english / american / british / online_global / uncertain
- offensive_flag: none / mild / strong
- explanation_en
- explanation_ar
- explanation_fr
- hard_negative_idioms: exactly 3 items
- meaning_paraphrases_en: exactly 3 items
- meaning_paraphrases_ar: exactly 3 items
- meaning_paraphrases_fr: exactly 3 items

MAIN EXAMPLES
If is_idiom = true, return EXACTLY 12 examples in this EXACT ORDER:

1. dialogue + idiomatic
2. dialogue + literal
3. narrative + idiomatic
4. narrative + literal
5. formal + idiomatic
6. formal + literal
7. social_media + idiomatic
8. social_media + literal
9. question + idiomatic
10. question + literal
11. sarcastic + idiomatic
12. sarcastic + literal

This ordering is mandatory.

FOR EACH MAIN EXAMPLE RETURN
- context_type
- source_style
- idiom_surface
- example
- idiom_in_example_arabic
- idiom_in_example_french
- idiom_in_example_meaning_en
- idiom_in_example_meaning_arabic
- idiom_in_example_meaning_french
- explanation_en
- explanation_ar
- explanation_fr
- is_example_idiom
- example_usage_label

SOURCE STYLE RULE
Source style must match context type exactly:
- dialogue -> synthetic_dialogue
- narrative -> synthetic_narrative
- formal -> synthetic_formal
- social_media -> synthetic_social_media
- question -> synthetic_question
- sarcastic -> synthetic_sarcastic

ADVERSARIAL EXAMPLES
If is_idiom = true, return EXACTLY 2 adversarial examples.
Both must use:
- source_style = synthetic_adversarial
- context_type is NOT used in adversarial_examples array
- expected_label must be one of: idiomatic / literal / borderline
- adversarial_type must be one of:
  negation / partial_overlap / contrastive_context / literal_trap / figurative_trap / borderline_ambiguity

STRICT NATURALNESS RULES
- Every example must sound like something a real person would naturally say or write.
- Avoid textbook-style sentences.
- Avoid dictionary-style demonstration sentences.
- Avoid empty shells where the idiom is simply wrapped with filler words.
- Each example must contain enough context to feel realistic and grounded.

STRICT DIVERSITY RULES
- Each example must represent a different real-life situation.
- Do NOT reuse the same sentence template.
- Do NOT change only names, pronouns, places, adjectives, or one or two words.
- Each example must differ in topic or intent or sentence structure or discourse context.

STRICT SOURCE-EXAMPLE RULES
- Do NOT copy example_raw.
- Do NOT lightly paraphrase example_raw.
- Do NOT reuse its event, sentence frame, or wording.

STRICT IDIOM-SENTENCE RULES
- Sentences should usually be between 8 and 22 words.
- Idiomatic examples must be clearly figurative.
- Literal examples must be clearly literal and natural.

STRICT MEANING RULES
- Meanings must describe actual contextual meaning, not repeat the idiom.
- Explanations must be specific to the example.
- Arabic must be natural Modern Standard Arabic.
- French must be natural contemporary standard French.

FINAL SELF-CHECK (MANDATORY)
Before returning JSON:
- If is_idiom = true:
  - examples must contain exactly 12 items
  - adversarial_examples must contain exactly 2 items
  - all 6 context types must appear exactly twice in examples
  - each context must have one idiomatic and one literal example
- If is_idiom = false:
  - examples must be []
  - adversarial_examples must be []
- no duplicates
- no copied example_raw
- no missing required fields

Return valid JSON only.
""".strip()


def get_response_schema():
    return {
        "type": "json_schema",
        "name": "idiomx_synth_v1_enrichment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_idiom": {"type": "boolean"},
                "idiom_validity_label": {
                    "type": "string",
                    "enum": ["valid_idiom", "borderline_expression", "not_idiom"],
                },
                "idiom_canonical": {"type": "string"},
                "idiom_canonical_meaning": {"type": "string"},
                "idiom_canonical_meaning_arabic": {"type": "string"},
                "idiom_canonical_meaning_french": {"type": "string"},
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
                "slang_strength": {
                    "type": "string",
                    "enum": ["none", "weak", "medium", "strong"],
                },
                "regionality": {
                    "type": "string",
                    "enum": ["general_english", "american", "british", "online_global", "uncertain"],
                },
                "offensive_flag": {
                    "type": "string",
                    "enum": ["none", "mild", "strong"],
                },
                "explanation_en": {"type": "string"},
                "explanation_ar": {"type": "string"},
                "explanation_fr": {"type": "string"},
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
                "meaning_paraphrases_fr": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "examples": {
                    "type": "array",
                    "minItems": 0,
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
                            "example": {"type": "string"},
                            "idiom_in_example_arabic": {"type": "string"},
                            "idiom_in_example_french": {"type": "string"},
                            "idiom_in_example_meaning_en": {"type": "string"},
                            "idiom_in_example_meaning_arabic": {"type": "string"},
                            "idiom_in_example_meaning_french": {"type": "string"},
                            "explanation_en": {"type": "string"},
                            "explanation_ar": {"type": "string"},
                            "explanation_fr": {"type": "string"},
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
                            "example",
                            "idiom_in_example_arabic",
                            "idiom_in_example_french",
                            "idiom_in_example_meaning_en",
                            "idiom_in_example_meaning_arabic",
                            "idiom_in_example_meaning_french",
                            "explanation_en",
                            "explanation_ar",
                            "explanation_fr",
                            "is_example_idiom",
                            "example_usage_label",
                        ],
                        "additionalProperties": False,
                    },
                },
                "adversarial_examples": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_style": {
                                "type": "string",
                                "enum": ["synthetic_adversarial"],
                            },
                            "example": {"type": "string"},
                            "idiom_in_example_arabic": {"type": "string"},
                            "idiom_in_example_french": {"type": "string"},
                            "idiom_in_example_meaning_en": {"type": "string"},
                            "idiom_in_example_meaning_arabic": {"type": "string"},
                            "idiom_in_example_meaning_french": {"type": "string"},
                            "explanation_en": {"type": "string"},
                            "explanation_ar": {"type": "string"},
                            "explanation_fr": {"type": "string"},
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
                            "example",
                            "idiom_in_example_arabic",
                            "idiom_in_example_french",
                            "idiom_in_example_meaning_en",
                            "idiom_in_example_meaning_arabic",
                            "idiom_in_example_meaning_french",
                            "explanation_en",
                            "explanation_ar",
                            "explanation_fr",
                            "expected_label",
                            "adversarial_type",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "is_idiom",
                "idiom_validity_label",
                "idiom_canonical",
                "idiom_canonical_meaning",
                "idiom_canonical_meaning_arabic",
                "idiom_canonical_meaning_french",
                "ambiguity_flag",
                "idiom_compositionality_level",
                "idiom_register",
                "idiom_domain",
                "learner_difficulty",
                "slang_strength",
                "regionality",
                "offensive_flag",
                "explanation_en",
                "explanation_ar",
                "explanation_fr",
                "hard_negative_idioms",
                "meaning_paraphrases_en",
                "meaning_paraphrases_ar",
                "meaning_paraphrases_fr",
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
    full_input_file = Path(full_input_file)
    sample_output_file = Path(sample_output_file)

    if not full_input_file.exists():
        raise FileNotFoundError(f"Full input file not found: {full_input_file}")

    if full_input_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(full_input_file)
    else:
        df = pd.read_csv(full_input_file, low_memory=False)

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
    if use_sample:
        return DEFAULT_SAMPLE_INPUT_FILE, DEFAULT_SAMPLE_OUTPUT_JSONL
    return DEFAULT_FULL_INPUT_FILE, DEFAULT_FULL_OUTPUT_JSONL


def prepare_batch_requests(
    input_file=None,
    output_jsonl=None,
    use_sample=False,
    sample_size=10,
    sample_method="random",
):
    if input_file is None or output_jsonl is None:
        default_input, default_output = get_mode_paths(use_sample=use_sample)
        input_file = input_file or default_input
        output_jsonl = output_jsonl or default_output

    input_file = Path(input_file)
    output_jsonl = Path(output_jsonl)

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    if input_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_file)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    if use_sample:
        if sample_method == "head":
            df = df.head(sample_size).copy()
        elif sample_method == "random":
            df = df.sample(min(sample_size, len(df)), random_state=42).copy()
        else:
            raise ValueError(f"Unsupported sample_method: {sample_method}")

    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_file}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if output_jsonl.exists():
        print(f"[WARNING] Output batch file already exists and will be overwritten: {output_jsonl}")

    schema = get_response_schema()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            idiom = str(row.get("idiom_canonical", "") if pd.notna(row.get("idiom_canonical", "")) else "")
            meaning_en = str(
                row.get("idiom_canonical_meaning", "")
                if pd.notna(row.get("idiom_canonical_meaning", ""))
                else ""
            )
            example_raw = str(
                row.get("example_raw", "")
                if pd.notna(row.get("example_raw", ""))
                else ""
            )

            row_id = row.get("idiom_id", None)
            custom_id = str(row_id) if pd.notna(row_id) else f"idiomx_synth_v1_{idx}"

            prompt = build_prompt(idiom, meaning_en, example_raw)

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

    print(f"Saved synthetic batch file to: {output_jsonl}")
    print(f"Input dataset: {input_file}")
    print(f"Total requests: {len(df)}")
    return output_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare synthetic IdiomX enrichment batch requests v1.")
    parser.add_argument("--sample", action="store_true", help="Use sample dataset mode.")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size if sample mode is used.")
    parser.add_argument("--sample-method", type=str, default="random", choices=["head", "random"])
    parser.add_argument("--input-file", type=str, default=None, help="Custom input parquet/csv file.")
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
        sample_size=args.sample_size,
        sample_method=args.sample_method,
    )