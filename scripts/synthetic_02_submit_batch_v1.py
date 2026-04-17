#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
synthetic_02_submit_batch_v1.py

Submit the synthetic idiom generation batch JSONL to the OpenAI Batch API.

Default input:
- data/batches/synthetic_idiom_generation_batch_v1.jsonl

Default output info:
- data/batches/synthetic_idiom_generation_batch_info_v1.json
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.api_config import client


DEFAULT_BATCH_FILE = BASE_DIR / "data" / "batches" / "synthetic_idiom_generation_batch_v1.jsonl"
DEFAULT_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "synthetic_idiom_generation_batch_info_v1.json"
DEFAULT_STAGE_NAME = "synthetic_missing_idiom_generation_v1"


def submit_batch(
    batch_file: Path,
    batch_info_file: Path,
    stage_name: str = DEFAULT_STAGE_NAME,
):
    batch_file = Path(batch_file)
    batch_info_file = Path(batch_info_file)

    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    # 1) Upload input file
    with open(batch_file, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch"
        )

    # 2) Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "stage_name": stage_name,
            "batch_file": str(batch_file.name),
        },
    )

    # 3) Save batch info locally
    batch_info = {
        "stage_name": stage_name,
        "batch_file": str(batch_file),
        "batch_file_name": batch_file.name,
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }

    batch_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_info_file, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, ensure_ascii=False, indent=2)

    print(f"Uploaded file ID: {uploaded.id}")
    print(f"Batch ID: {batch.id}")
    print(f"Batch status: {batch.status}")
    print(f"Batch info saved to: {batch_info_file}")

    return batch_info


def parse_args():
    parser = argparse.ArgumentParser(description="Submit synthetic idiom generation batch.")
    parser.add_argument("--batch-file", type=str, default=str(DEFAULT_BATCH_FILE), help="Path to input batch JSONL file.")
    parser.add_argument("--batch-info-file", type=str, default=str(DEFAULT_BATCH_INFO_FILE), help="Path to output batch info JSON.")
    parser.add_argument("--stage-name", type=str, default=DEFAULT_STAGE_NAME, help="Stage name to save in batch metadata.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    submit_batch(
        batch_file=Path(args.batch_file),
        batch_info_file=Path(args.batch_info_file),
        stage_name=args.stage_name,
    )