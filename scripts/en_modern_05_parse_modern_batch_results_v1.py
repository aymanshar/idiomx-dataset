#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import argparse


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_JSONL = BASE_DIR / "data" / "results" / "idiomx_modern_results_v1.jsonl"
DEFAULT_OUTPUT_JSON = BASE_DIR / "data" / "results" / "idiomx_modern_enriched_full_v1.json"


def parse_batch_results(input_jsonl: Path, output_json: Path):
    input_jsonl = Path(input_jsonl)
    output_json = Path(output_json)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    parsed = []
    failed = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            custom_id = obj.get("custom_id", "")

            try:
                response_text = obj["response"]["body"]["output"][0]["content"][0]["text"]
                parsed_json = json.loads(response_text)
                parsed_json["_custom_id"] = custom_id
                parsed.append(parsed_json)
            except Exception:
                failed += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print("Saved parsed JSON:", output_json)
    print("Parsed rows:", len(parsed))
    print("Failed rows:", failed)

    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Parse modern batch JSONL results.")
    parser.add_argument("--input-jsonl", type=str, default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parse_batch_results(
        input_jsonl=Path(args.input_jsonl),
        output_json=Path(args.output_json),
    )