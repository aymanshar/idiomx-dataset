"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Validate the enriched IdiomX dataset (v2).

This script performs rule-based validation to ensure data quality,
consistency, and alignment between idioms, examples, meanings,
bilingual annotations, adversarial rows, and new v2 enrichment fields.

This script is safe to run and does not call the API.

Supports:
- notebook execution with explicit paths
- command-line execution
- sample mode and full mode
"""

from pathlib import Path
import re
import json
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_v2.csv"
DEFAULT_FULL_OUTPUT_VALIDATED_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_validated_v2.csv"
DEFAULT_FULL_OUTPUT_ISSUES_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_issues_v2.csv"

# Sample-mode defaults
DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_sample_v2.csv"
DEFAULT_SAMPLE_OUTPUT_VALIDATED_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_validated_sample_v2.csv"
DEFAULT_SAMPLE_OUTPUT_ISSUES_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_issues_sample_v2.csv"

VALID_MAIN_LABELS = {"idiomatic", "literal"}
VALID_ALL_LABELS = {"idiomatic", "literal", "borderline"}
VALID_ROW_TYPES = {"main_example", "adversarial_example_1", "adversarial_example_2"}
VALID_CONTEXT_TYPES = {"dialogue", "narrative", "formal", "social_media", "question", "sarcastic", "adversarial"}
VALID_SOURCE_STYLES = {
    "synthetic_dialogue",
    "synthetic_narrative",
    "synthetic_formal",
    "synthetic_social_media",
    "synthetic_question",
    "synthetic_sarcastic",
    "synthetic_adversarial",
}
VALID_ADVERSARIAL_TYPES = {
    "negation",
    "partial_overlap",
    "contrastive_context",
    "literal_trap",
    "figurative_trap",
    "borderline_ambiguity",
}


def get_mode_paths(use_sample: bool = False):
    """
    Return default input and output file paths based on execution mode.
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
    return set(re.findall(r"\w+", str(text).lower()))


def is_nonempty(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def parse_json_list(value) -> list:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return [obj]
    except Exception:
        return [text]


def bool_is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def bool_is_false(value) -> bool:
    return str(value).strip().lower() in {"false", "0"}


def validate_dataset(
    input_file : Path = None,
    output_validated_csv: Path = None,
    output_issues_csv: Path = None,
    use_sample: bool = False,
):
    """
    Validate the enriched dataset and flag suspicious rows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        validated dataframe, issues dataframe
    """
    default_input_csv, default_output_validated_csv, default_output_issues_csv = get_mode_paths(use_sample=use_sample)

    input_file  = Path(input_file ) if input_file  is not None else default_input_csv
    output_validated_csv = Path(output_validated_csv) if output_validated_csv is not None else default_output_validated_csv
    output_issues_csv = Path(output_issues_csv) if output_issues_csv is not None else default_output_issues_csv

    if not input_file .exists():
        raise FileNotFoundError(f"Input CSV not found: {input_file }")

    if input_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_file)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    statuses = []
    issues = []

    for i, row in df.iterrows():
        idiom_id = row.get("idiom_id", "")
        idiom_canonical = row.get("idiom_canonical", "")
        row_type = str(row.get("row_type", "")).strip()
        is_adv = str(row.get("is_adversarial_example", "")).strip()

        surface = str(row.get("idiom_surface", "")).strip()
        ex = str(row.get("idiom_in_example", "")).strip()
        ex_ar = str(row.get("idiom_in_example_arabic", "")).strip()
        mean_en = str(row.get("idiom_in_example_meaning_en", "")).strip()
        mean_ar = str(row.get("idiom_in_example_meaning_arabic", "")).strip()

        label = str(row.get("example_usage_label", "")).strip()
        expected_label = str(row.get("expected_label", "")).strip()
        is_example_idiom = row.get("is_example_idiom", None)

        context_type = str(row.get("context_type", "")).strip()
        source_style = str(row.get("source_style", "")).strip()
        explanation_en = str(row.get("explanation_en", "")).strip()
        explanation_ar = str(row.get("explanation_ar", "")).strip()
        idiom_level_explanation_en = str(row.get("idiom_level_explanation_en", "")).strip()
        idiom_level_explanation_ar = str(row.get("idiom_level_explanation_ar", "")).strip()
        minimal_pair_id = row.get("minimal_pair_id", pd.NA)
        adversarial_type = str(row.get("adversarial_type", "")).strip()

        hard_negative_idioms = parse_json_list(row.get("hard_negative_idioms", "[]"))
        meaning_paraphrases_en = parse_json_list(row.get("meaning_paraphrases_en", "[]"))
        meaning_paraphrases_ar = parse_json_list(row.get("meaning_paraphrases_ar", "[]"))

        problem = None

        # Core universal checks
        if not is_nonempty(idiom_id):
            problem = "missing_idiom_id"
        elif not is_nonempty(idiom_canonical):
            problem = "missing_idiom_canonical"
        elif row_type not in VALID_ROW_TYPES:
            problem = "invalid_row_type"
        elif context_type not in VALID_CONTEXT_TYPES:
            problem = "invalid_context_type"
        elif source_style not in VALID_SOURCE_STYLES:
            problem = "invalid_source_style"
        elif not is_nonempty(ex):
            problem = "missing_example_en"
        elif not is_nonempty(ex_ar):
            problem = "missing_example_arabic"
        elif not is_nonempty(mean_en):
            problem = "missing_example_meaning_en"
        elif not is_nonempty(mean_ar):
            problem = "missing_example_meaning_arabic"
        elif not is_nonempty(explanation_en):
            problem = "missing_explanation_en"
        elif not is_nonempty(explanation_ar):
            problem = "missing_explanation_ar"
        elif not is_nonempty(idiom_level_explanation_en):
            problem = "missing_idiom_level_explanation_en"
        elif not is_nonempty(idiom_level_explanation_ar):
            problem = "missing_idiom_level_explanation_ar"
        elif len(hard_negative_idioms) < 3:
            problem = "missing_or_short_hard_negative_idioms"
        elif len(meaning_paraphrases_en) < 3:
            problem = "missing_or_short_meaning_paraphrases_en"
        elif len(meaning_paraphrases_ar) < 3:
            problem = "missing_or_short_meaning_paraphrases_ar"
        elif len(ex.split()) < 2:
            problem = "example_too_short"

        # Main examples
        elif row_type == "main_example":
            surface_tokens = tokenize(surface)
            ex_tokens = tokenize(ex)
            missing_tokens = surface_tokens - ex_tokens if surface else set()

            if label not in VALID_MAIN_LABELS:
                problem = "invalid_main_example_usage_label"
            elif context_type == "adversarial":
                problem = "main_example_cannot_have_adversarial_context"
            elif not is_nonempty(surface):
                problem = "missing_surface"
            elif len(missing_tokens) > 1:
                problem = "surface_not_in_example"
            elif pd.isna(is_example_idiom):
                problem = "missing_is_example_idiom"
            elif label == "idiomatic" and not bool_is_true(is_example_idiom):
                problem = "label_boolean_mismatch_idiomatic"
            elif label == "literal" and not bool_is_false(is_example_idiom):
                problem = "label_boolean_mismatch_literal"
            elif pd.isna(minimal_pair_id) or str(minimal_pair_id).strip() == "":
                problem = "missing_minimal_pair_id"
            elif str(is_adv).lower() not in {"0", "false"}:
                problem = "main_example_marked_as_adversarial"

        # Adversarial examples
        elif row_type in {"adversarial_example_1", "adversarial_example_2"}:
            if label not in VALID_ALL_LABELS:
                problem = "invalid_adversarial_example_usage_label"
            elif context_type != "adversarial":
                problem = "adversarial_row_missing_adversarial_context"
            elif source_style != "synthetic_adversarial":
                problem = "invalid_adversarial_source_style"
            elif adversarial_type not in VALID_ADVERSARIAL_TYPES:
                problem = "invalid_adversarial_type"
            elif expected_label not in VALID_ALL_LABELS:
                problem = "invalid_expected_label"
            elif str(is_adv).lower() not in {"1", "true"}:
                problem = "adversarial_row_not_marked_as_adversarial"

        if problem:
            statuses.append("needs_review")
            issues.append({
                "row_index": i,
                "problem": problem,
                "idiom_id": idiom_id,
                "idiom_canonical": idiom_canonical,
                "row_type": row_type,
                "context_type": context_type,
                "example_usage_label": label,
                "expected_label": expected_label,
                "idiom_surface": surface,
                "idiom_in_example": ex,
            })
        else:
            statuses.append("valid")

    df["validation_status"] = statuses
    issues_df = pd.DataFrame(issues)

    # Group-level validation: each idiom_id should have exactly 12 main + 2 adversarial rows
    group_issues = []
    grouped = df.groupby("idiom_id")

    for idiom_id, g in grouped:
        main_count = (g["row_type"] == "main_example").sum()
        adv_count = g["row_type"].astype(str).str.startswith("adversarial_example").sum()

        if main_count != 12:
            group_issues.append({
                "row_index": pd.NA,
                "problem": "unexpected_main_example_count_per_idiom",
                "idiom_id": idiom_id,
                "idiom_canonical": g["idiom_canonical"].iloc[0],
                "row_type": "group_check",
                "context_type": "",
                "example_usage_label": "",
                "expected_label": "",
                "idiom_surface": "",
                "idiom_in_example": f"main_count={main_count}",
            })

        if adv_count != 2:
            group_issues.append({
                "row_index": pd.NA,
                "problem": "unexpected_adversarial_example_count_per_idiom",
                "idiom_id": idiom_id,
                "idiom_canonical": g["idiom_canonical"].iloc[0],
                "row_type": "group_check",
                "context_type": "",
                "example_usage_label": "",
                "expected_label": "",
                "idiom_surface": "",
                "idiom_in_example": f"adv_count={adv_count}",
            })

        main_g = g[g["row_type"] == "main_example"]
        if not main_g.empty:
            idiomatic_count = (main_g["example_usage_label"] == "idiomatic").sum()
            literal_count = (main_g["example_usage_label"] == "literal").sum()

            if idiomatic_count != 6:
                group_issues.append({
                    "row_index": pd.NA,
                    "problem": "unexpected_idiomatic_main_count_per_idiom",
                    "idiom_id": idiom_id,
                    "idiom_canonical": g["idiom_canonical"].iloc[0],
                    "row_type": "group_check",
                    "context_type": "",
                    "example_usage_label": "",
                    "expected_label": "",
                    "idiom_surface": "",
                    "idiom_in_example": f"idiomatic_count={idiomatic_count}",
                })

            if literal_count != 6:
                group_issues.append({
                    "row_index": pd.NA,
                    "problem": "unexpected_literal_main_count_per_idiom",
                    "idiom_id": idiom_id,
                    "idiom_canonical": g["idiom_canonical"].iloc[0],
                    "row_type": "group_check",
                    "context_type": "",
                    "example_usage_label": "",
                    "expected_label": "",
                    "idiom_surface": "",
                    "idiom_in_example": f"literal_count={literal_count}",
                })

            context_counts = main_g["context_type"].value_counts().to_dict()
            for ctx in ["dialogue", "narrative", "formal", "social_media", "question", "sarcastic"]:
                if context_counts.get(ctx, 0) != 2:
                    group_issues.append({
                        "row_index": pd.NA,
                        "problem": "unexpected_context_pair_count_per_idiom",
                        "idiom_id": idiom_id,
                        "idiom_canonical": g["idiom_canonical"].iloc[0],
                        "row_type": "group_check",
                        "context_type": ctx,
                        "example_usage_label": "",
                        "expected_label": "",
                        "idiom_surface": "",
                        "idiom_in_example": f"context_count={context_counts.get(ctx, 0)}",
                    })

    if group_issues:
        issues_df = pd.concat([issues_df, pd.DataFrame(group_issues)], ignore_index=True)

    output_validated_csv.parent.mkdir(parents=True, exist_ok=True)
    output_issues_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_validated_csv, index=False, encoding="utf-8-sig")
    issues_df.to_csv(output_issues_csv, index=False, encoding="utf-8-sig")

    print(f"Validated dataset saved to: {output_validated_csv}")
    print(f"Issues saved to: {output_issues_csv}")
    print(df["validation_status"].value_counts(dropna=False))

    valid_ratio = (df["validation_status"] == "valid").mean()
    print(f"\nValidation success rate: {valid_ratio:.2%}")

    if not issues_df.empty:
        print("\nTop issue counts:")
        print(issues_df["problem"].value_counts().head(15))

    return df, issues_df


def parse_args():
    """
    Parse command-line arguments for dataset validation.
    """
    parser = argparse.ArgumentParser(description="Validate IdiomX enriched dataset v2.")
    parser.add_argument("--sample", action="store_true", help="Use sample-mode default paths.")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input enriched CSV.")
    parser.add_argument("--output-validated-csv", type=str, default=None, help="Path to output validated CSV.")
    parser.add_argument("--output-issues-csv", type=str, default=None, help="Path to output issues CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    validate_dataset(
        input_csv=Path(args.input_file ) if args.input_file  else None,
        output_validated_csv=Path(args.output_validated_csv) if args.output_validated_csv else None,
        output_issues_csv=Path(args.output_issues_csv) if args.output_issues_csv else None,
        use_sample=args.sample,
    )