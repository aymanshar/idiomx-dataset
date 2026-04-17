#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client


BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_synth_batch_info_v1.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_synth_batch_sample_info_v1.json"


def get_mode_path(use_sample: bool = False) -> Path:
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_INFO_FILE


def load_batch_id(batch_info_file: Path) -> str:
    batch_info_file = Path(batch_info_file)

    if not batch_info_file.exists():
        raise FileNotFoundError(f"Batch info file not found: {batch_info_file}")

    with open(batch_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    batch_id = info.get("batch_id")
    if not batch_id:
        raise ValueError(f"'batch_id' not found in batch info file: {batch_info_file}")

    return batch_id


def check_batch(
    batch_id: Optional[str] = None,
    batch_info_file: Optional[Path] = None,
    use_sample: bool = False,
):

    batch_info_file = Path(batch_info_file) if batch_info_file is not None else get_mode_path(use_sample=use_sample)

    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as e:
        print(f"Failed to retrieve batch: {e}")
        return None

    print("\n=== BATCH STATUS ===")

    if batch.status == "completed":
        print("✅ Batch completed successfully.")
    elif batch.status == "failed":
        print("❌ Batch failed. Check error file.")
    elif batch.status == "cancelled":
        print("⚠️ Batch was cancelled.")
    else:
        print("⏳ Batch still in progress.")

    print("\n=== BASIC INFO ===")
    print("Batch ID:", batch.id)
    print("Status:", batch.status)
    print("Created at:", getattr(batch, "created_at", None))
    print("In progress at:", getattr(batch, "in_progress_at", None))
    print("Completed at:", getattr(batch, "completed_at", None))

    print("\n=== FILES ===")
    print("Output file ID:", getattr(batch, "output_file_id", None))
    print("Error file ID:", getattr(batch, "error_file_id", None))

    print("\n=== METADATA (IMPORTANT) ===")
    metadata = getattr(batch, "metadata", {})
    for k, v in metadata.items():
        print(f"{k}: {v}")

    print("\n============================\n")

    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Check status of a synthetic IdiomX batch job.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--batch-info-file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_batch(
        batch_id=args.batch_id,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        use_sample=args.sample,
    )