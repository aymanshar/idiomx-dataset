from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client

"""
Submit IdiomX batch enrichment job to OpenAI.

This script uploads a prepared JSONL request file, creates a paid batch job,
and saves the returned batch metadata locally for later monitoring and result retrieval.

WARNING:
This script uploads a JSONL batch file and creates a paid batch job.
Do not run unless you intentionally want to execute enrichment.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode
"""

# NOTE:
# This script is part of the scalable LLM enrichment pipeline, where
# schema-constrained prompts are submitted asynchronously as batch jobs
# to enrich the IdiomX dataset with bilingual meanings and example sentences.

BASE_DIR = Path(__file__).resolve().parents[1]

# Full-mode defaults
DEFAULT_FULL_BATCH_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch.jsonl"
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_info.json"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_sample.jsonl"
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_sample_info.json"


def get_mode_paths(use_sample: bool = False) -> tuple[Path, Path]:
    """
    Return the default batch input file and batch metadata file paths
    for either full mode or sample mode.
    """
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_FILE, DEFAULT_SAMPLE_BATCH_INFO_FILE
    return DEFAULT_FULL_BATCH_FILE, DEFAULT_FULL_BATCH_INFO_FILE


def submit_batch(
    batch_file: Optional[Path] = None,
    batch_info_file: Optional[Path] = None,
    use_sample: bool = False,
) -> str:
    """
    Submit a batch JSONL file to the API and save batch metadata locally.

    Uploads the batch input file, creates a batch job through the API,
    and stores the resulting batch identifiers for later monitoring and download.

    Parameters
    ----------
    batch_file : Optional[Path]
        Path to the batch JSONL file.
    batch_info_file : Optional[Path]
        Path where batch metadata JSON should be saved.
    use_sample : bool
        If True, use sample-mode defaults when explicit paths are not provided.

    Returns
    -------
    str
        The created batch ID.
    """
    # Resolve default batch file and metadata paths based on execution mode
    default_batch_file, default_batch_info_file = get_mode_paths(use_sample=use_sample)

    batch_file = Path(batch_file) if batch_file is not None else default_batch_file
    batch_info_file = Path(batch_info_file) if batch_info_file is not None else default_batch_info_file

    # Ensure the batch request file exists before attempting upload
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    # Upload the prepared JSONL file to the API as a batch input file
    try:
        with open(batch_file, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="batch"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to upload batch file {batch_file}: {e}")

    # Create the batch job using the uploaded file and attach project metadata
    try:
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "project": "IdiomX",
                "stage": "idiomx_pre_enrichment_to_enriched_full",
                "mode": "sample" if use_sample else "full"
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create batch job: {e}")

    # Store minimal batch metadata locally for later status checks and downloads
    batch_info = {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "batch_file": str(batch_file),
        "mode": "sample" if use_sample else "full"
    }

    # Save batch metadata JSON to disk for reproducibility and orchestration
    batch_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_info_file, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)

    print("Uploaded file ID:", uploaded.id)
    print("Batch ID:", batch.id)
    print("Batch status:", batch.status)
    print("Batch info saved to:", batch_info_file)

    return batch.id


def parse_args():
    """
    Parse command-line arguments for submitting an IdiomX batch job.
    """
    # Configure CLI interface for full-mode or sample-mode batch submission
    parser = argparse.ArgumentParser(description="Submit IdiomX enrichment batch.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample-mode default paths."
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Path to input batch JSONL file."
    )
    parser.add_argument(
        "--batch-info-file",
        type=str,
        default=None,
        help="Path to save batch metadata JSON."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    submit_batch(
        batch_file=Path(args.batch_file) if args.batch_file else None,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        use_sample=args.sample,
    )