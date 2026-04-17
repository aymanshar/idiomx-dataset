#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.api_config import client

BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "synthetic_idiom_generation_batch_info_v1.json"
OUTPUT_JSONL = BASE_DIR / "data" / "generated" / "synthetic_idiom_generation_results_v1.jsonl"


def download_results():
    if not BATCH_INFO_FILE.exists():
        raise FileNotFoundError(f"Batch info file not found: {BATCH_INFO_FILE}")

    with open(BATCH_INFO_FILE, "r", encoding="utf-8") as f:
        info = json.load(f)

    batch_id = info["batch_id"]
    batch = client.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed yet. Current status: {batch.status}")

    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError("No output_file_id found for completed batch.")

    content = client.files.content(output_file_id)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "wb") as f:
        f.write(content.read())

    print(f"Downloaded results to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    download_results()