from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# Default full-data paths
DEFAULT_FULL_INPUT_FILE = BASE_DIR / "data" / "processed" / "idiomx_pre_enrichment.parquet"
DEFAULT_FULL_OUTPUT_JSONL = BASE_DIR / "data" / "batches" / "idiomx_batch_v2.jsonl"

# Default sample-data paths
DEFAULT_SAMPLE_DIR = BASE_DIR / "data" / "sample"
DEFAULT_SAMPLE_INPUT_FILE = DEFAULT_SAMPLE_DIR / "idiomx_pre_enrichment_sample.parquet"
DEFAULT_SAMPLE_OUTPUT_JSONL = DEFAULT_SAMPLE_DIR / "idiomx_batch_v2_sample.jsonl"

MODEL_NAME = "gpt-4.1-mini"

"""
Prepare batch requests for IdiomX enrichment.

Safe to run offline.
This script does NOT call the API.
It only reads the pre-enrichment dataset and writes JSONL batch requests.

Supports:
- full dataset mode
- sample dataset mode
- creating a small sample from the full dataset
"""

def build_prompt(idiom: str, meaning_en: str, example: str) -> str:
    """
        Build the LLM instruction prompt for enriching a single idiom entry.

        Includes idiom analysis, bilingual meaning generation, and example generation
        (idiomatic and literal) in a structured format.
    """
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
    """
        Build the LLM instruction prompt for enriching a single idiom entry.

        Includes idiom analysis, bilingual meaning generation, and example generation
        (idiomatic and literal) in a structured format.
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

def create_sample_dataset(
    full_input_file: Path = DEFAULT_FULL_INPUT_FILE,
    sample_output_file: Path = DEFAULT_SAMPLE_INPUT_FILE,
    n_rows: int = 5,
    method: str = "head",
    random_state: int = 42,
):
    """
    Create a small sample dataset from the full pre-enrichment dataset.

    Useful for testing the pipeline quickly before running on the full dataset.
    Supports head or random sampling.

    Parameters
    ----------
    full_input_file : Path
        Path to the full pre-enrichment parquet file.
    sample_output_file : Path
        Path where the sample parquet file will be saved.
    n_rows : int
        Number of rows in the sample.
    method : str
        "head" or "random".
    random_state : int
        Random seed used if method == "random".
    """
    if not full_input_file.exists():
        raise FileNotFoundError(f"Full input file not found: {full_input_file}")

    df = pd.read_parquet(full_input_file)

    if len(df) == 0:
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

    Switches between full dataset mode and sample dataset mode.
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

    Each row becomes one request with a structured prompt and response schema.
    Supports both full dataset and sample dataset modes.
    """
    if input_file is None or output_jsonl is None:
        default_input, default_output = get_mode_paths(use_sample=use_sample)
        input_file = input_file or default_input
        output_jsonl = output_jsonl or default_output

    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    df = pd.read_parquet(input_file)

    if len(df) == 0:
        raise ValueError(f"Input dataset is empty: {input_file}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    schema = get_response_schema()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            idiom = str(row.get("idiom_canonical", "") if pd.notna(row.get("idiom_canonical", "")) else "")
            meaning_en = str(row.get("idiom_canonical_meaning", "") if pd.notna(row.get("idiom_canonical_meaning", "")) else "")
            example = str(row.get("example", "") if pd.notna(row.get("example", "")) else "")

            # Prefer stable idiom_id if available
            row_id = row.get("idiom_id", None)
            if pd.notna(row_id):
                custom_id = str(row_id)
            else:
                custom_id = f"idiomx_v2_{idx}"

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

if __name__ == "__main__":
    # Default behavior: full dataset mode
    prepare_batch_requests()