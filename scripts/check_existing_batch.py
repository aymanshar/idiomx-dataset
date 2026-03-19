from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client

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

BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_info.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_sample_info.json"


def get_mode_path(use_sample: bool = False) -> Path:
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_INFO_FILE


def load_batch_id(batch_info_file: Path) -> str:
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
    Check the status of an existing batch job.

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

    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    batch = client.batches.retrieve(batch_id)

    print("Batch ID:", batch.id)
    print("Status:", batch.status)
    print("Created at:", getattr(batch, "created_at", None))
    print("Output file ID:", getattr(batch, "output_file_id", None))
    print("Error file ID:", getattr(batch, "error_file_id", None))

    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Check status of an IdiomX batch job.")
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