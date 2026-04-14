"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Batch monitoring utility for IdiomX LLM enrichment pipeline v2.

This script retrieves the status of previously submitted batch jobs,
allowing tracking of LLM generation tasks used for dataset augmentation
and enrichment.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode

License:
MIT License (see LICENSE file)

Citation:
If you use this code or dataset, please cite the IdiomX paper.
"""

from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client


BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_info_v2.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_sample_info_v2.json"


def get_mode_path(use_sample: bool = False) -> Path:
    """
    Return the appropriate batch info file path based on execution mode.
    """
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_INFO_FILE


def load_batch_id(batch_info_file: Path) -> str:
    """
    Load the batch ID from a stored batch info JSON file.
    """
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
    """
    Check the status of an existing LLM batch job.

    Retrieves batch metadata such as status, creation time, output file ID,
    and error file ID using the configured API client.

    Parameters
    ----------
    batch_id : Optional[str]
        Explicit batch ID. If None, it will be loaded from batch_info_file.
    batch_info_file : Optional[Path]
        Path to the batch info JSON file.
    use_sample : bool
        If True, use sample-mode default path when batch_info_file is not provided.

    Returns
    -------
    object
        Retrieved batch object from the API client.
    """
    batch_info_file = Path(batch_info_file) if batch_info_file is not None else get_mode_path(use_sample=use_sample)

    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as e:
        print(f"Failed to retrieve batch: {e}")
        return None

    if batch.status == "completed":
        print("Batch completed successfully.")
    elif batch.status == "failed":
        print("Batch failed. Check error file.")
    elif batch.status == "cancelled":
        print("Batch was cancelled.")
    else:
        print("Batch still in progress.")

    print("Batch ID:", batch.id)
    print("Status:", batch.status)
    print("Created at:", getattr(batch, "created_at", None))
    print("In progress at:", getattr(batch, "in_progress_at", None))
    print("Completed at:", getattr(batch, "completed_at", None))
    print("Output file ID:", getattr(batch, "output_file_id", None))
    print("Error file ID:", getattr(batch, "error_file_id", None))

    return batch


def parse_args():
    """
    Parse command-line arguments for batch status checking.
    """
    parser = argparse.ArgumentParser(description="Check status of an IdiomX batch job v2.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample-mode default batch info path.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Explicit batch ID. If omitted, it is loaded from the batch info file.",
    )
    parser.add_argument(
        "--batch-info-file",
        type=str,
        default=None,
        help="Path to batch info JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_batch(
        batch_id=args.batch_id,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        use_sample=args.sample,
    )