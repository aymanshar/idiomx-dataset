from pathlib import Path
import re
import argparse
import pandas as pd

"""
Validate the enriched IdiomX dataset.

This script performs rule-based validation to ensure data quality,
consistency, and alignment between idioms, examples, meanings,
and bilingual annotations before model training.

This script is safe to run and does not call the API.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode
"""

# NOTE:
# This stage introduces a rule-based validation layer to ensure that
# LLM-generated outputs meet structural and semantic constraints,
# improving dataset reliability for downstream modeling tasks.

BASE_DIR = Path(__file__).resolve().parents[1]


# Full-mode defaults

DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full.csv"
DEFAULT_FULL_OUTPUT_VALIDATED_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_validated.csv"
DEFAULT_FULL_OUTPUT_ISSUES_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_issues.csv"


# Sample-mode defaults

DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample.csv"
DEFAULT_SAMPLE_OUTPUT_VALIDATED_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample_validated.csv"
DEFAULT_SAMPLE_OUTPUT_ISSUES_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_sample_issues.csv"


def get_mode_paths(use_sample: bool = False):
    """
    Return default input and output file paths based on execution mode
    (full dataset or sample dataset).
    """
    if use_sample:
        return (
            DEFAULT_SAMPLE_INPUT_CSV,
            DEFAULT_SAMPLE_OUTPUT_VALIDATED_CSV,
            DEFAULT_SAMPLE_OUTPUT_ISSUES_CSV,
        )
    return (
        DEFAULT_FULL_INPUT_CSV,
        DEFAULT_FULL_OUTPUT_VALIDATED_CSV,
        DEFAULT_FULL_OUTPUT_ISSUES_CSV,
    )


def tokenize(text: str) -> set:
    """
    Return default input and output file paths based on execution mode
    (full dataset or sample dataset).
    """
    return set(re.findall(r"\w+", str(text).lower()))


def validate_dataset(
    input_csv: Path = None,
    output_validated_csv: Path = None,
    output_issues_csv: Path = None,
    use_sample: bool = False,
):
    """
    Validate the enriched dataset and flag suspicious rows.

    Applies rule-based checks on idiom usage, example alignment,
    bilingual completeness, and label consistency, then flags
    rows that require manual review.

    Outputs:
    - validated dataset with validation_status column
    - issues dataset with detailed problem descriptions
    Parameters
    ----------
    input_csv : Path
        Input enriched CSV file.
    output_validated_csv : Path
        Output CSV with validation_status column.
    output_issues_csv : Path
        Output CSV containing rows/issues flagged for review.
    use_sample : bool
        If True, use sample-mode default paths when explicit paths are not provided.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        validated dataframe, issues dataframe
    """
    default_input_csv, default_output_validated_csv, default_output_issues_csv = get_mode_paths(use_sample=use_sample)

    input_csv = Path(input_csv) if input_csv is not None else default_input_csv
    output_validated_csv = Path(output_validated_csv) if output_validated_csv is not None else default_output_validated_csv
    output_issues_csv = Path(output_issues_csv) if output_issues_csv is not None else default_output_issues_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load enriched dataset generated from LLM augmentation pipeline
    df = pd.read_csv(input_csv, low_memory=False)

    statuses = []
    issues = []

    # Iterate through rows to validate idiom usage, alignment, and completeness
    for i, row in df.iterrows():
        surface = str(row.get("idiom_surface", "")).strip()
        ex = str(row.get("idiom_in_example", "")).strip()
        ex_ar = str(row.get("idiom_in_example_arabic", "")).strip()
        mean_en = str(row.get("idiom_in_example_meaning_en", "")).strip()
        mean_ar = str(row.get("idiom_in_example_meaning_arabic", "")).strip()
        label = str(row.get("example_usage_label", "")).strip()
        is_example_idiom = row.get("is_example_idiom", None)

        problem = None

        # Surface should appear in the example text approximately
        surface_tokens = tokenize(surface)
        ex_tokens = tokenize(ex)
        missing_tokens = surface_tokens - ex_tokens

        # Check that the idiom surface appears in the example (approximate match)
        if label not in {"idiomatic", "literal"}:
            problem = "invalid_example_usage_label"
        elif not surface:
            problem = "missing_surface"
        elif not ex:
            problem = "missing_example_en"
        elif len(missing_tokens) > 1:
            problem = "surface_not_in_example"
        elif not ex_ar:
            problem = "missing_example_arabic"
        elif not mean_en:
            problem = "missing_example_meaning_en"
        elif not mean_ar:
            problem = "missing_example_meaning_arabic"
        elif pd.isna(is_example_idiom):
            problem = "missing_is_example_idiom"
        # Accept both boolean and string representations (True/False, 1/0)
        elif label == "idiomatic" and str(is_example_idiom).lower() not in {"true", "1"}:
            problem = "label_boolean_mismatch_idiomatic"
        elif label == "literal" and str(is_example_idiom).lower() not in {"false", "0"}:
            problem = "label_boolean_mismatch_literal"
        elif len(ex.split()) < 2:
            problem = "example_too_short"

        # Apply validation rules to detect missing fields, mismatches, and inconsistencies
        if problem:
            statuses.append("needs_review")
            # Record problematic rows for manual inspection
            issues.append({
                "row_index": i,
                "problem": problem,
                "idiom_id": row.get("idiom_id", ""),
                "idiom_canonical": row.get("idiom_canonical", ""),
                "idiom_surface": surface,
                "idiom_in_example": ex,
                "example_usage_label": label,
            })
        else:
            statuses.append("valid")

    df["validation_status"] = statuses

    issues_df = pd.DataFrame(issues)

    # Save validated dataset and extracted issues for debugging and analysis
    output_validated_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_validated_csv, index=False, encoding="utf-8-sig")
    issues_df.to_csv(output_issues_csv, index=False, encoding="utf-8-sig")

    print(f"Validated dataset saved to: {output_validated_csv}")
    print(f"Issues saved to: {output_issues_csv}")
    print(df["validation_status"].value_counts(dropna=False))
    valid_ratio = (df["validation_status"] == "valid").mean()
    print(f"\nValidation success rate: {valid_ratio:.2%}")

    return df, issues_df


def parse_args():
    """
    Parse command-line arguments for dataset validation.
    """
    parser = argparse.ArgumentParser(description="Validate IdiomX enriched dataset.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample-mode default paths.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input enriched CSV.",
    )
    parser.add_argument(
        "--output-validated-csv",
        type=str,
        default=None,
        help="Path to output validated CSV.",
    )
    parser.add_argument(
        "--output-issues-csv",
        type=str,
        default=None,
        help="Path to output issues CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    validate_dataset(
        input_csv=Path(args.input_csv) if args.input_csv else None,
        output_validated_csv=Path(args.output_validated_csv) if args.output_validated_csv else None,
        output_issues_csv=Path(args.output_issues_csv) if args.output_issues_csv else None,
        use_sample=args.sample,
    )