from pathlib import Path
import json
import argparse
import pandas as pd
from tqdm import tqdm

from config.api_config import client

"""
Verify suspicious IdiomX rows using the API.

LLM-based verification and correction stage for IdiomX dataset.

This script re-evaluates rows flagged during validation using a structured
LLM prompt, optionally correcting inconsistencies in examples, meanings,
labels, and translations.

This step acts as a semi-automatic quality assurance layer.

WARNING:
This script calls the API and may incur cost.
Do not run unless you intentionally want to re-verify flagged rows.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode
"""

# NOTE:
# This stage introduces a human-in-the-loop style refinement using LLMs,
# where only uncertain or problematic samples are re-evaluated,
# significantly improving dataset quality while controlling cost.

BASE_DIR = Path(__file__).resolve().parents[1]

# ==============================
# Full-mode defaults
# ==============================
DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_validated.csv"
DEFAULT_FULL_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_final.csv"

# ==============================
# Sample-mode defaults
# ==============================
DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample_validated.csv"
DEFAULT_SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample_final.csv"


SCHEMA = {
    "type": "json_schema",
    "name": "idiomx_row_verification",
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


def get_mode_paths(use_sample: bool = False):
    """
    Return default input and output paths based on execution mode
    (full dataset or sample dataset).
    """
    if use_sample:
        return DEFAULT_SAMPLE_INPUT_CSV, DEFAULT_SAMPLE_OUTPUT_CSV
    return DEFAULT_FULL_INPUT_CSV, DEFAULT_FULL_OUTPUT_CSV


def verify_suspicious_rows(
    input_csv: Path = None,
    output_csv: Path = None,
    use_sample: bool = False,
    model_name: str = "gpt-4.1-mini",
):
    """
    Re-verify suspicious rows flagged by the validation stage.

    Processes rows flagged during validation, checks semantic and structural
    consistency, and applies corrections when necessary using a structured
    JSON schema.

    This stage improves dataset quality by combining rule-based validation
    with model-assisted refinement.

    Parameters
    ----------
    input_csv : Path
        Input validated CSV.
    output_csv : Path
        Output final CSV after optional correction.
    use_sample : bool
        If True, use sample-mode default paths when explicit paths are not provided.
    model_name : str
        OpenAI model name.

    Returns
    -------
    pd.DataFrame
        Final dataframe after verification/correction.
    """
    default_input_csv, default_output_csv = get_mode_paths(use_sample=use_sample)

    input_csv = Path(input_csv) if input_csv is not None else default_input_csv
    output_csv = Path(output_csv) if output_csv is not None else default_output_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if "validation_status" not in df.columns:
        raise ValueError("Input CSV does not contain 'validation_status' column.")

    # Select rows flagged during validation for re-verification
    review_df = df[df["validation_status"] == "needs_review"].copy()

    print(f"Rows marked for review: {len(review_df)}")

    # If no suspicious rows exist, save dataset unchanged
    if len(review_df) == 0:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"No suspicious rows found. Saved unchanged dataset to: {output_csv}")
        return df

    # Iterate over suspicious rows and validate them using the LLM
    for idx, row in tqdm(review_df.iterrows(), total=len(review_df), desc="Verifying suspicious rows"):
        # Construct structured verification prompt with full idiom context
        prompt = f"""
You are verifying a row in the IdiomX enriched dataset.

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
            # Call LLM with strict JSON schema to enforce structured output
            response = client.responses.create(
                model=model_name,
                input=prompt,
                text={"format": SCHEMA}
            )

            obj = json.loads(response.output_text)

            if obj["is_valid"]:
                df.loc[idx, "validation_status"] = "verified"
            else:
                df.loc[idx, "validation_status"] = "invalid"

            # Update dataset based on LLM validation and optional corrections
            if obj["correction_needed"]:
                corrected = obj["corrected_entry"]
                for k, v in corrected.items():
                    df.loc[idx, k] = v
                df.loc[idx, "validation_status"] = "corrected"

        # Handle API or parsing errors without breaking the pipeline
        except Exception as e:
            df.loc[idx, "validation_status"] = f"verification_error: {e}"

    # Save final verified and corrected dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved final verified dataset to: {output_csv}")
    print("\nVerification summary:")
    print(df["validation_status"].value_counts(dropna=False))

    corrected_ratio = (df["validation_status"] == "corrected").mean()
    print(f"Correction rate: {corrected_ratio:.2%}")

    return df


def parse_args():
    """
    Parse command-line arguments for LLM-based verification of suspicious rows.
    """
    parser = argparse.ArgumentParser(description="Verify suspicious IdiomX rows.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample-mode default paths.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input validated CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output final CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name to use for verification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    verify_suspicious_rows(
        input_csv=Path(args.input_csv) if args.input_csv else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        use_sample=args.sample,
        model_name=args.model,
    )