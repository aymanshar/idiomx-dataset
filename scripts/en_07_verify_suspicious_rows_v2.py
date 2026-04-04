from pathlib import Path
import json
import argparse
import pandas as pd
from tqdm import tqdm
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.api_config import client


# Full-mode defaults
DEFAULT_FULL_INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_validated_v2.csv"
DEFAULT_FULL_OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_full_final_v2.csv"
DEFAULT_FULL_REVIEW_ROWS_CSV = BASE_DIR / "data" / "enriched" / "idiomx_review_rows_v2.csv"
DEFAULT_FULL_CORRECTED_ROWS_CSV = BASE_DIR / "data" / "enriched" / "idiomx_corrected_rows_v2.csv"
DEFAULT_FULL_VERIFICATION_LOG_CSV = BASE_DIR / "data" / "enriched" / "idiomx_verification_log_v2.csv"
DEFAULT_FULL_CHECKPOINT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_verification_checkpoint_v2.csv"
DEFAULT_FULL_CHECKPOINT_JSON = BASE_DIR / "data" / "enriched" / "idiomx_verification_checkpoint_v2.json"

# Sample-mode defaults
DEFAULT_SAMPLE_INPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_validated_sample_v2.csv"
DEFAULT_SAMPLE_OUTPUT_CSV = BASE_DIR / "data" / "sample" / "idiomx_enriched_full_final_sample_v2.csv"
DEFAULT_SAMPLE_REVIEW_ROWS_CSV = BASE_DIR / "data" / "sample" / "idiomx_review_rows_sample_v2.csv"
DEFAULT_SAMPLE_CORRECTED_ROWS_CSV = BASE_DIR / "data" / "sample" / "idiomx_corrected_rows_sample_v2.csv"
DEFAULT_SAMPLE_VERIFICATION_LOG_CSV = BASE_DIR / "data" / "sample" / "idiomx_verification_log_sample_v2.csv"
DEFAULT_SAMPLE_CHECKPOINT_CSV = BASE_DIR / "data" / "sample" / "idiomx_verification_checkpoint_sample_v2.csv"
DEFAULT_SAMPLE_CHECKPOINT_JSON = BASE_DIR / "data" / "sample" / "idiomx_verification_checkpoint_sample_v2.json"


SCHEMA = {
    "type": "json_schema",
    "name": "idiomx_row_verification_v2",
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
                    "explanation_en": {"type": "string"},
                    "explanation_ar": {"type": "string"},
                    "is_example_idiom": {"type": "boolean"},
                    "example_usage_label": {
                        "type": "string",
                        "enum": ["idiomatic", "literal", "borderline"]
                    },
                    "expected_label": {
                        "type": "string",
                        "enum": ["idiomatic", "literal", "borderline"]
                    },
                    "adversarial_type": {"type": "string"}
                },
                "required": [
                    "idiom_surface",
                    "idiom_in_example",
                    "idiom_in_example_arabic",
                    "idiom_in_example_meaning_en",
                    "idiom_in_example_meaning_arabic",
                    "explanation_en",
                    "explanation_ar",
                    "is_example_idiom",
                    "example_usage_label",
                    "expected_label",
                    "adversarial_type"
                ],
                "additionalProperties": False
            }
        },
        "required": ["is_valid", "correction_needed", "corrected_entry"],
        "additionalProperties": False
    }
}


def get_mode_paths(use_sample: bool = False):
    if use_sample:
        return (
            DEFAULT_SAMPLE_INPUT_CSV,
            DEFAULT_SAMPLE_OUTPUT_CSV,
            DEFAULT_SAMPLE_REVIEW_ROWS_CSV,
            DEFAULT_SAMPLE_CORRECTED_ROWS_CSV,
            DEFAULT_SAMPLE_VERIFICATION_LOG_CSV,
            DEFAULT_SAMPLE_CHECKPOINT_CSV,
            DEFAULT_SAMPLE_CHECKPOINT_JSON,
        )
    return (
        DEFAULT_FULL_INPUT_CSV,
        DEFAULT_FULL_OUTPUT_CSV,
        DEFAULT_FULL_REVIEW_ROWS_CSV,
        DEFAULT_FULL_CORRECTED_ROWS_CSV,
        DEFAULT_FULL_VERIFICATION_LOG_CSV,
        DEFAULT_FULL_CHECKPOINT_CSV,
        DEFAULT_FULL_CHECKPOINT_JSON,
    )


def save_progress(
    df: pd.DataFrame,
    review_df_original: pd.DataFrame,
    corrected_indices: list,
    verification_logs: list,
    output_csv: Path,
    review_rows_csv: Path,
    corrected_rows_csv: Path,
    verification_log_csv: Path,
    checkpoint_csv: Path,
    checkpoint_json: Path,
    last_processed_idx: int,
):
    corrected_rows_df = (
        df.loc[sorted(set(corrected_indices))].copy()
        if corrected_indices
        else pd.DataFrame(columns=df.columns)
    )
    verification_log_df = pd.DataFrame(verification_logs)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    review_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    verification_log_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_json.parent.mkdir(parents=True, exist_ok=True)

    # Save main outputs
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    review_df_original.to_csv(review_rows_csv, index=False, encoding="utf-8-sig")
    corrected_rows_df.to_csv(corrected_rows_csv, index=False, encoding="utf-8-sig")
    verification_log_df.to_csv(verification_log_csv, index=False, encoding="utf-8-sig")

    # Save checkpoint snapshot
    df.to_csv(checkpoint_csv, index=False, encoding="utf-8-sig")
    with open(checkpoint_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_processed_idx": int(last_processed_idx),
                "corrected_indices": list(sorted(set(corrected_indices))),
                "verification_log_count": len(verification_logs),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def verify_suspicious_rows(
    input_csv: Path = None,
    output_csv: Path = None,
    review_rows_csv: Path = None,
    corrected_rows_csv: Path = None,
    verification_log_csv: Path = None,
    checkpoint_csv: Path = None,
    checkpoint_json: Path = None,
    use_sample: bool = False,
    model_name: str = "gpt-4.1-mini",
    save_every: int = 25,
    resume: bool = True,
):
    (
        default_input_csv,
        default_output_csv,
        default_review_rows_csv,
        default_corrected_rows_csv,
        default_verification_log_csv,
        default_checkpoint_csv,
        default_checkpoint_json,
    ) = get_mode_paths(use_sample=use_sample)

    input_csv = Path(input_csv) if input_csv is not None else default_input_csv
    output_csv = Path(output_csv) if output_csv is not None else default_output_csv
    review_rows_csv = Path(review_rows_csv) if review_rows_csv is not None else default_review_rows_csv
    corrected_rows_csv = Path(corrected_rows_csv) if corrected_rows_csv is not None else default_corrected_rows_csv
    verification_log_csv = Path(verification_log_csv) if verification_log_csv is not None else default_verification_log_csv
    checkpoint_csv = Path(checkpoint_csv) if checkpoint_csv is not None else default_checkpoint_csv
    checkpoint_json = Path(checkpoint_json) if checkpoint_json is not None else default_checkpoint_json

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Resume from checkpoint CSV if available
    if resume and checkpoint_csv.exists():
        print(f"[INFO] Resuming from checkpoint CSV: {checkpoint_csv}")
        df = pd.read_csv(checkpoint_csv, low_memory=False)
    else:
        df = pd.read_csv(input_csv, low_memory=False)

    if "validation_status" not in df.columns:
        raise ValueError("Input CSV does not contain 'validation_status' column.")

    review_df = df[df["validation_status"] == "needs_review"].copy()
    review_df_original = review_df.copy()
    print(f"Rows marked for review: {len(review_df)}")

    if len(review_df) == 0:
        empty_corrected_df = pd.DataFrame(columns=df.columns)
        empty_log_df = pd.DataFrame(columns=[
            "row_index",
            "idiom_id",
            "row_type",
            "before_validation_status",
            "after_validation_status",
            "changed",
            "notes",
        ])

        save_progress(
            df=df,
            review_df_original=review_df_original,
            corrected_indices=[],
            verification_logs=[],
            output_csv=output_csv,
            review_rows_csv=review_rows_csv,
            corrected_rows_csv=corrected_rows_csv,
            verification_log_csv=verification_log_csv,
            checkpoint_csv=checkpoint_csv,
            checkpoint_json=checkpoint_json,
            last_processed_idx=-1,
        )

        print(f"No suspicious rows found. Saved unchanged dataset to: {output_csv}")
        return df, review_df_original, empty_corrected_df, empty_log_df

    corrected_indices = []
    verification_logs = []
    processed_index_set = set()

    # Resume state
    if resume and checkpoint_json.exists():
        print(f"[INFO] Loading checkpoint state: {checkpoint_json}")
        with open(checkpoint_json, "r", encoding="utf-8") as f:
            state = json.load(f)
        corrected_indices = state.get("corrected_indices", [])
        processed_index_set = set(
            pd.read_csv(verification_log_csv, low_memory=False)["row_index"].tolist()
        ) if verification_log_csv.exists() else set()
        if verification_log_csv.exists():
            verification_logs = pd.read_csv(verification_log_csv, low_memory=False).to_dict("records")

    processed_counter = 0

    for idx, row in tqdm(review_df.iterrows(), total=len(review_df), desc="Verifying suspicious rows"):
        if idx in processed_index_set:
            continue

        row_type = str(row.get("row_type", "")).strip()
        is_adv = str(row.get("is_adversarial_example", "")).strip()

        before_snapshot = {
            "idiom_surface": row.get("idiom_surface", ""),
            "idiom_in_example": row.get("idiom_in_example", ""),
            "idiom_in_example_arabic": row.get("idiom_in_example_arabic", ""),
            "idiom_in_example_meaning_en": row.get("idiom_in_example_meaning_en", ""),
            "idiom_in_example_meaning_arabic": row.get("idiom_in_example_meaning_arabic", ""),
            "explanation_en": row.get("explanation_en", ""),
            "explanation_ar": row.get("explanation_ar", ""),
            "is_example_idiom": row.get("is_example_idiom", ""),
            "example_usage_label": row.get("example_usage_label", ""),
            "expected_label": row.get("expected_label", ""),
            "adversarial_type": row.get("adversarial_type", ""),
        }

        prompt = f"""
You are verifying one row in the IdiomX v2 enriched dataset.

Your task:
1. Check whether the row is valid.
2. If needed, correct the row.
3. Preserve the intended row type.
4. For main examples, ensure the idiom surface appears naturally in the sentence.
5. For adversarial examples, allow difficult or borderline wording if still natural and meaningful.

Return JSON only.

Focus on:
- natural and grammatical English
- natural and correct Arabic
- consistency between idiom_surface and idiom_in_example
- consistency between example_usage_label and is_example_idiom
- correctness of idiom_in_example_meaning_en / arabic
- correctness of explanation_en / explanation_ar
- consistency of expected_label for adversarial rows

ROW METADATA
idiom_id: {row.get('idiom_id', '')}
idiom_canonical: {row.get('idiom_canonical', '')}
idiom_canonical_meaning: {row.get('idiom_canonical_meaning', '')}
idiom_canonical_meaning_arabic: {row.get('idiom_canonical_meaning_arabic', '')}
ambiguity_flag: {row.get('ambiguity_flag', '')}
idiom_compositionality_level: {row.get('idiom_compositionality_level', '')}
idiom_register: {row.get('idiom_register', '')}
idiom_domain: {row.get('idiom_domain', '')}
learner_difficulty: {row.get('learner_difficulty', '')}
row_type: {row_type}
context_type: {row.get('context_type', '')}
source_style: {row.get('source_style', '')}
is_adversarial_example: {is_adv}
adversarial_type: {row.get('adversarial_type', '')}

ROW CONTENT
idiom_surface: {row.get('idiom_surface', '')}
idiom_in_example: {row.get('idiom_in_example', '')}
idiom_in_example_arabic: {row.get('idiom_in_example_arabic', '')}
idiom_in_example_meaning_en: {row.get('idiom_in_example_meaning_en', '')}
idiom_in_example_meaning_arabic: {row.get('idiom_in_example_meaning_arabic', '')}
explanation_en: {row.get('explanation_en', '')}
explanation_ar: {row.get('explanation_ar', '')}
is_example_idiom: {row.get('is_example_idiom', '')}
example_usage_label: {row.get('example_usage_label', '')}
expected_label: {row.get('expected_label', '')}
"""

        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
                text={"format": SCHEMA}
            )

            obj = json.loads(response.output_text)
            before_status = df.loc[idx, "validation_status"]

            if obj["is_valid"]:
                df.loc[idx, "validation_status"] = "verified"
            else:
                df.loc[idx, "validation_status"] = "invalid"

            if obj["correction_needed"]:
                corrected = obj["corrected_entry"]

                for col in [
                    "idiom_surface",
                    "idiom_in_example",
                    "idiom_in_example_arabic",
                    "idiom_in_example_meaning_en",
                    "idiom_in_example_meaning_arabic",
                    "explanation_en",
                    "explanation_ar",
                    "is_example_idiom",
                    "example_usage_label",
                    "expected_label",
                    "adversarial_type",
                ]:
                    if col in corrected:
                        df.loc[idx, col] = corrected[col]

                if str(df.loc[idx, "row_type"]) == "main_example":
                    if str(df.loc[idx, "example_usage_label"]).strip() == "borderline":
                        df.loc[idx, "example_usage_label"] = row.get("example_usage_label", "")
                    df.loc[idx, "expected_label"] = df.loc[idx, "example_usage_label"]

                if str(df.loc[idx, "row_type"]).startswith("adversarial_example"):
                    df.loc[idx, "example_usage_label"] = corrected.get(
                        "expected_label", df.loc[idx, "example_usage_label"]
                    )

                df.loc[idx, "validation_status"] = "corrected"
                corrected_indices.append(idx)

            verification_logs.append({
                "row_index": idx,
                "idiom_id": row.get("idiom_id", ""),
                "idiom_canonical": row.get("idiom_canonical", ""),
                "row_type": row.get("row_type", ""),
                "before_validation_status": before_status,
                "after_validation_status": df.loc[idx, "validation_status"],
                "correction_needed": obj.get("correction_needed", False),
                "is_valid": obj.get("is_valid", False),
                "before_idiom_surface": before_snapshot["idiom_surface"],
                "after_idiom_surface": df.loc[idx, "idiom_surface"],
                "before_idiom_in_example": before_snapshot["idiom_in_example"],
                "after_idiom_in_example": df.loc[idx, "idiom_in_example"],
                "before_example_usage_label": before_snapshot["example_usage_label"],
                "after_example_usage_label": df.loc[idx, "example_usage_label"],
                "before_expected_label": before_snapshot["expected_label"],
                "after_expected_label": df.loc[idx, "expected_label"],
            })

        except Exception as e:
            df.loc[idx, "validation_status"] = f"verification_error: {e}"
            verification_logs.append({
                "row_index": idx,
                "idiom_id": row.get("idiom_id", ""),
                "idiom_canonical": row.get("idiom_canonical", ""),
                "row_type": row.get("row_type", ""),
                "before_validation_status": row.get("validation_status", ""),
                "after_validation_status": df.loc[idx, "validation_status"],
                "correction_needed": False,
                "is_valid": False,
                "before_idiom_surface": before_snapshot["idiom_surface"],
                "after_idiom_surface": before_snapshot["idiom_surface"],
                "before_idiom_in_example": before_snapshot["idiom_in_example"],
                "after_idiom_in_example": before_snapshot["idiom_in_example"],
                "before_example_usage_label": before_snapshot["example_usage_label"],
                "after_example_usage_label": before_snapshot["example_usage_label"],
                "before_expected_label": before_snapshot["expected_label"],
                "after_expected_label": before_snapshot["expected_label"],
            })

        processed_index_set.add(idx)
        processed_counter += 1

        if processed_counter % save_every == 0:
            print(f"\n[INFO] Saving checkpoint after {processed_counter} newly processed rows...")
            save_progress(
                df=df,
                review_df_original=review_df_original,
                corrected_indices=corrected_indices,
                verification_logs=verification_logs,
                output_csv=output_csv,
                review_rows_csv=review_rows_csv,
                corrected_rows_csv=corrected_rows_csv,
                verification_log_csv=verification_log_csv,
                checkpoint_csv=checkpoint_csv,
                checkpoint_json=checkpoint_json,
                last_processed_idx=idx,
            )

    save_progress(
        df=df,
        review_df_original=review_df_original,
        corrected_indices=corrected_indices,
        verification_logs=verification_logs,
        output_csv=output_csv,
        review_rows_csv=review_rows_csv,
        corrected_rows_csv=corrected_rows_csv,
        verification_log_csv=verification_log_csv,
        checkpoint_csv=checkpoint_csv,
        checkpoint_json=checkpoint_json,
        last_processed_idx=-1,
    )

    print(f"Saved final verified dataset to: {output_csv}")
    print(f"Saved review rows to: {review_rows_csv}")
    print(f"Saved corrected rows to: {corrected_rows_csv}")
    print(f"Saved verification log to: {verification_log_csv}")

    print("\nVerification summary:")
    print(df["validation_status"].value_counts(dropna=False))

    corrected_ratio = (df["validation_status"] == "corrected").mean()
    print(f"Correction rate: {corrected_ratio:.2%}")

    return df, review_df_original, empty_corrected_df, empty_log_df


def parse_args():
    parser = argparse.ArgumentParser(description="Verify suspicious IdiomX rows v2 with checkpointing.")
    parser.add_argument("--sample", action="store_true", help="Use sample-mode default paths.")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input validated CSV.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to output final CSV.")
    parser.add_argument("--review-rows-csv", type=str, default=None, help="Path to save rows marked for review.")
    parser.add_argument("--corrected-rows-csv", type=str, default=None, help="Path to save corrected rows only.")
    parser.add_argument("--verification-log-csv", type=str, default=None, help="Path to save verification log.")
    parser.add_argument("--checkpoint-csv", type=str, default=None, help="Path to save checkpoint CSV.")
    parser.add_argument("--checkpoint-json", type=str, default=None, help="Path to save checkpoint JSON.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name to use for verification.")
    parser.add_argument("--save-every", type=int, default=25, help="Save progress every N processed rows.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    verify_suspicious_rows(
        input_csv=Path(args.input_csv) if args.input_csv else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        review_rows_csv=Path(args.review_rows_csv) if args.review_rows_csv else None,
        corrected_rows_csv=Path(args.corrected_rows_csv) if args.corrected_rows_csv else None,
        verification_log_csv=Path(args.verification_log_csv) if args.verification_log_csv else None,
        checkpoint_csv=Path(args.checkpoint_csv) if args.checkpoint_csv else None,
        checkpoint_json=Path(args.checkpoint_json) if args.checkpoint_json else None,
        use_sample=args.sample,
        model_name=args.model,
        save_every=args.save_every,
        resume=not args.no_resume,
    )