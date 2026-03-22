from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client

"""
Batch monitoring utility for IdiomX LLM enrichment pipeline.

This script retrieves the status of previously submitted batch jobs,
allowing tracking of large-scale LLM generation tasks used for
dataset augmentation and enrichment.
"""

"""
Check status of an existing IdiomX batch job.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode

Examples
--------
Notebook:
    check_batch(
        batch_info_file=DATA_DIR / "sample" / "idiomx_batch_sample_info.json"
    )

CMD (full):
    python scripts/check_existing_batch.py

CMD (sample):
    python scripts/check_existing_batch.py --sample
"""
# NOTE:
# This script is part of the large-scale LLM-based data enrichment pipeline,
# where idiom examples and annotations are generated asynchronously using batch jobs.

BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_info.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_sample_info.json"


def get_mode_path(use_sample: bool = False) -> Path:
    """
    Return the appropriate batch info file path based on execution mode.

    If sample mode is enabled, return the sample batch info file path;
    otherwise return the full dataset batch info path.
    """
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_INFO_FILE


def load_batch_id(batch_info_file: Path) -> str:
    """
    Load the batch ID from a stored batch info JSON file.

    Raises an error if the file is missing or does not contain a valid batch_id.
    """
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

    Supports both explicit batch IDs and loading from stored batch info files.

    Parameters
    ----------
    batch_id : Optional[str]
        Explicit batch ID. If None, it will be loaded from batch_info_file.
    batch_info_file : Optional[Path]
        Path to the batch info JSON file.
    use_sample : bool
        If True, use sample-mode default path when batch_info_file is not provided.
    """
    batch_info_file = Path(batch_info_file) if batch_info_file is not None else get_mode_path(use_sample=use_sample)

    # Load batch ID from file if not explicitly provided
    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    # Query the LLM provider API to retrieve batch job metadata
    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as e:
        print(f"Failed to retrieve batch: {e}")
        return None

    if batch.status == "completed":
        print("Batch completed successfully.")
    elif batch.status == "failed":
        print("Batch failed. Check error file.")
    else:
        print("Batch still in progress.")

    # Display key batch information for monitoring and debugging
    print("Batch ID:", batch.id)
    print("Status:", batch.status)
    print("Created at:", getattr(batch, "created_at", None))
    print("Output file ID:", getattr(batch, "output_file_id", None))
    print("Error file ID:", getattr(batch, "error_file_id", None))

    return batch


def parse_args():
    """
    Parse command-line arguments for batch status checking.

    Supports switching between sample and full modes, and overriding
    batch ID or batch info file paths.
    """
    parser = argparse.ArgumentParser(description="Check status of an IdiomX batch job.")

    # Configure CLI interface for flexible execution (sample/full modes)
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