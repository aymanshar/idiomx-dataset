from pathlib import Path
import json
import pandas as pd
import json
import re
from json_repair import repair_json

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = BASE_DIR / "data" / "raw" / "idiomx_dataset_v1.csv"
RESULTS_JSONL = BASE_DIR / "data" / "results" / "idiomx_results_v2.jsonl"
OUTPUT_CSV = BASE_DIR / "data" / "enriched" / "idiomx_enriched_v2.csv"

FINAL_COLUMNS = [
    "idiom",
    "meaning_en",
    "example",
    "source",
    "source_type",
    "pos",
    "tags",
    "idiom_confidence",
    "source_url",
    "idiom_canonical",
    "idiom_canonical_meaning",
    "idiom_canonical_meaning_arabic",
    "is_idiom",
    "ambiguity_flag",
    "idiom_compositionality_level",
    "idiom_register",
    "idiom_domain",
    "learner_difficulty",
    "idiom_surface",
    "idiom_in_example",
    "idiom_in_example_arabic",
    "idiom_in_example_meaning_en",
    "idiom_in_example_meaning_arabic",
    "is_example_idiom",
    "example_usage_label",
    "is_generated_example",
    "enrichment_model",
    "enrichment_version",
    "validation_status"
]

def repair_json(text):
    """
    Attempt to repair common LLM JSON formatting issues.
    """

    if text is None:
        return None

    # remove code fences
    text = text.replace("```json", "").replace("```", "")

    # remove leading / trailing spaces
    text = text.strip()

    # remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # remove junk before first brace
    if "{" in text:
        text = text[text.index("{"):]

    # remove junk after last brace
    if "}" in text:
        text = text[: text.rindex("}") + 1]

    return text

def extract_output_text(body: dict):
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    return content_item.get("text")
    return None

def merge_results(raw_csv: Path = RAW_CSV,
                  results_jsonl: Path = RESULTS_JSONL,
                  output_csv: Path = OUTPUT_CSV):
    raw_df = pd.read_csv(raw_csv)
    rows = []

    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            custom_id = record.get("custom_id", "")
            idx = int(custom_id.split("_")[-1])

            raw_row = raw_df.iloc[idx]
            body = record["response"]["body"]

            output_text = extract_output_text(body)
            if not output_text:
                print(f"Skipping {custom_id}: no output_text")
                continue

            try:
                obj = json.loads(output_text)

            except Exception:
                try:
                    repaired = repair_json(output_text)
                    obj = json.loads(repaired)
                    print(f"Repaired JSON for {custom_id}")

                except Exception as e:
                    print(f"Skipping {custom_id}: invalid JSON - {e}")
                    continue

            examples = obj.get("examples", [])
            if len(examples) != 8:
                print(f"Warning: {custom_id} returned {len(examples)} examples instead of 8")

            for ex in examples:
                row = {
                    "idiom": raw_row.get("idiom", ""),
                    "meaning_en": raw_row.get("meaning_en", ""),
                    "example": raw_row.get("example", ""),
                    "source": raw_row.get("source", ""),
                    "source_type": raw_row.get("source_type", ""),
                    "pos": raw_row.get("pos", ""),
                    "tags": raw_row.get("tags", ""),
                    "idiom_confidence": raw_row.get("idiom_confidence", ""),
                    "source_url": raw_row.get("source_url", ""),

                    "idiom_canonical": obj.get("idiom_canonical", ""),
                    "idiom_canonical_meaning": obj.get("idiom_canonical_meaning", ""),
                    "idiom_canonical_meaning_arabic": obj.get("idiom_canonical_meaning_arabic", ""),
                    "is_idiom": obj.get("is_idiom", None),
                    "ambiguity_flag": obj.get("ambiguity_flag", ""),
                    "idiom_compositionality_level": obj.get("idiom_compositionality_level", ""),
                    "idiom_register": obj.get("idiom_register", ""),
                    "idiom_domain": obj.get("idiom_domain", ""),
                    "learner_difficulty": obj.get("learner_difficulty", ""),

                    "idiom_surface": ex.get("idiom_surface", ""),
                    "idiom_in_example": ex.get("idiom_in_example", ""),
                    "idiom_in_example_arabic": ex.get("idiom_in_example_arabic", ""),
                    "idiom_in_example_meaning_en": ex.get("idiom_in_example_meaning_en", ""),
                    "idiom_in_example_meaning_arabic": ex.get("idiom_in_example_meaning_arabic", ""),
                    "is_example_idiom": ex.get("is_example_idiom", None),
                    "example_usage_label": ex.get("example_usage_label", ""),

                    "is_generated_example": 1,
                    "enrichment_model": body.get("model", "gpt-4.1-mini"),
                    "enrichment_version": "v2_4idiomatic_4literal",
                    "validation_status": "pending"
                }

                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=FINAL_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved merged v2 dataset to: {output_csv}")
    print(f"Total rows: {len(out_df)}")
    print(out_df["example_usage_label"].value_counts(dropna=False))
    return out_df

if __name__ == "__main__":
    merge_results()