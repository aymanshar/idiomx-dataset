#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from typing import Optional
import argparse
import sys
from config.api_config import client


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# Full-mode defaults
DEFAULT_FULL_BATCH_FILE = BASE_DIR / "data" / "batches" / "idiomx_synth_batch_v1.jsonl"
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_synth_batch_info_v1.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_FILE = BASE_DIR / "data" / "sample" / "idiomx_synth_batch_sample_v1.jsonl"
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_synth_batch_sample_info_v1.json"


def get_mode_paths(use_sample: bool = False) -> tuple[Path, Path]:
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_FILE, DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_FILE, DEFAULT_FULL_BATCH_INFO_FILE


def submit_batch(
    batch_file: Optional[Path] = None,
    batch_info_file: Optional[Path] = None,
    use_sample: bool = False,
    stage_name: str = "idiomx_synth_pre_enrichment_to_enriched_v1",  # ✅ FIXED
) -> str:

    default_batch_file, default_batch_info_file = get_mode_paths(use_sample=use_sample)

    batch_file = Path(batch_file) if batch_file is not None else default_batch_file
    batch_info_file = Path(batch_info_file) if batch_info_file is not None else default_batch_info_file

    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    # Upload file
    with open(batch_file, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch"
        )

    # Create batch job
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "project": "IdiomX",
            "pipeline": "synthetic_enrichment",  # ✅ NEW
            "stage": stage_name,
            "version": "synth_v1",  # ✅ FIXED
            "mode": "sample" if use_sample else "full"
        }
    )

    batch_info = {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "batch_file": str(batch_file),
        "batch_info_file": str(batch_info_file),
        "mode": "sample" if use_sample else "full",
        "stage": stage_name,
        "version": "synth_v1"
    }

    batch_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_info_file, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)

    print("Uploaded file ID:", uploaded.id)
    print("Batch ID:", batch.id)
    print("Batch status:", batch.status)
    print("Batch info saved to:", batch_info_file)

    return batch.id


def parse_args():
    parser = argparse.ArgumentParser(description="Submit IdiomX synthetic enrichment batch.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-file", type=str, default=None)
    parser.add_argument("--batch-info-file", type=str, default=None)
    parser.add_argument(
        "--stage-name",
        type=str,
        default="idiomx_synth_pre_enrichment_to_enriched_v1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    submit_batch(
        batch_file=Path(args.batch_file) if args.batch_file else None,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        use_sample=args.sample,
        stage_name=args.stage_name,
    )