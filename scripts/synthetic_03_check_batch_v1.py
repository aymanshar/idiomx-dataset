#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check status of submitted batch
"""

from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.api_config import client


BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "synthetic_idiom_generation_batch_info_v1.json"


def check_batch():
    if not BATCH_INFO_FILE.exists():
        raise FileNotFoundError("Batch info file not found")

    with open(BATCH_INFO_FILE, "r", encoding="utf-8") as f:
        info = json.load(f)

    batch_id = info["batch_id"]

    batch = client.batches.retrieve(batch_id)

    print("\n===== BATCH STATUS =====")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    if batch.status == "completed":
        print("\n✅ Batch completed successfully")
        print(f"Output file ID: {batch.output_file_id}")

    elif batch.status in ["failed", "cancelled"]:
        print("\n❌ Batch failed or cancelled")

    else:
        print("\n⏳ Still processing...")

    return batch


if __name__ == "__main__":
    check_batch()