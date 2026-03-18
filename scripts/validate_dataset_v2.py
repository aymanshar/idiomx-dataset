from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2.csv"
OUTPUT_VALIDATED_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2_validated.csv"
OUTPUT_ISSUES_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2_issues.csv"

def validate_dataset(input_csv: Path = INPUT_CSV,
                     output_validated_csv: Path = OUTPUT_VALIDATED_CSV,
                     output_issues_csv: Path = OUTPUT_ISSUES_CSV):
    df = pd.read_csv(input_csv)
    statuses = []
    issues = []

    for i, row in df.iterrows():
        surface = str(row.get("idiom_surface", "")).strip()
        ex = str(row.get("idiom_in_example", "")).strip()
        ex_ar = str(row.get("idiom_in_example_arabic", "")).strip()
        mean_en = str(row.get("idiom_in_example_meaning_en", "")).strip()
        mean_ar = str(row.get("idiom_in_example_meaning_arabic", "")).strip()
        label = str(row.get("example_usage_label", "")).strip()

        problem = None

        surface_tokens = set(re.findall(r"\w+", surface.lower()))
        ex_tokens = set(re.findall(r"\w+", ex.lower()))
        missing_tokens = surface_tokens - ex_tokens

        if label not in {"idiomatic", "literal"}:
            problem = "invalid_example_usage_label"
        elif not surface:
            problem = "missing_surface"
        elif len(missing_tokens) > 1:
            problem = "surface_not_in_example"
        elif not ex_ar:
            problem = "missing_example_arabic"
        elif not mean_en:
            problem = "missing_example_meaning_en"
        elif not mean_ar:
            problem = "missing_example_meaning_arabic"

        if problem:
            statuses.append("needs_review")
            issues.append({
                "row_index": i,
                "problem": problem,
                "idiom_canonical": row.get("idiom_canonical", ""),
                "idiom_surface": surface,
                "idiom_in_example": ex
            })
        else:
            statuses.append("valid")

    df["validation_status"] = statuses
    output_validated_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_validated_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(issues).to_csv(output_issues_csv, index=False, encoding="utf-8-sig")

    print(f"Validated dataset saved to: {output_validated_csv}")
    print(f"Issues saved to: {output_issues_csv}")
    print(df["validation_status"].value_counts(dropna=False))

if __name__ == "__main__":
    validate_dataset()