"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
github: https://github.com/aymanshar/idiomx-dataset
Year: 2026

Description:
Download completed IdiomX batch results.

This script retrieves the output file associated with a completed LLM batch job
and saves the JSONL results locally for later parsing and dataset merging.

Supports:
- notebook execution with explicit paths
- command-line execution
- full mode and sample mode

Examples
Notebook:
    download_results(
        batch_info_file=DATA_DIR / "sample" / "idiomx_batch_v2_sample_info.json",
        output_path=DATA_DIR / "sample" / "idiomx_results_v2_sample.jsonl"
    )

CMD (full):
    python scripts/download_existing_batch_results.py

CMD (sample):
    python scripts/download_existing_batch_results.py --sample

CMD (explicit paths):
    python scripts/download_existing_batch_results.py ^
        --batch-info-file data/sample/idiomx_batch_v2_sample_info.json ^
        --output-path data/sample/idiomx_results_v2_sample.jsonl

Notes:
# This stage retrieves schema-constrained LLM outputs produced asynchronously
# through batch processing, enabling scalable enrichment of the IdiomX dataset.

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
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_v2_info.json"
DEFAULT_FULL_OUTPUT_PATH = BASE_DIR / "data" / "results" / "idiomx_results_v2.jsonl"

# Sample-mode defaults
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_batch_v2_sample_info.json"
DEFAULT_SAMPLE_OUTPUT_PATH = BASE_DIR / "data" / "sample" / "idiomx_results_v2_sample.jsonl"


def get_mode_paths(use_sample: bool = False) -> tuple[Path, Path]:
    """
    Return the default batch-info and result-output paths for either
    full mode or sample mode.
    """
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE, DEFAULT_SAMPLE_OUTPUT_PATH
    return DEFAULT_FULL_BATCH_INFO_FILE, DEFAULT_FULL_OUTPUT_PATH


def load_batch_id(batch_info_file: Path) -> str:
    """
    Load batch_id from the saved batch info JSON file.
    """
    if not batch_info_file.exists():
        raise FileNotFoundError(f"Batch info file not found: {batch_info_file}")

    with open(batch_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    batch_id = info.get("batch_id")
    if not batch_id:
        raise ValueError(f"No 'batch_id' found in: {batch_info_file}")

    return batch_id


def download_results(
    batch_id: Optional[str] = None,
    batch_info_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
    use_sample: bool = False,
) -> Path:
    """
    Download completed batch results from the API.

    Supports both explicit batch IDs and loading from stored batch metadata,
    and works in full-data or sample-data mode.

    Parameters
    ----------
    batch_id : Optional[str]
        Explicit batch id. If None, it will be loaded from batch_info_file.
    batch_info_file : Optional[Path]
        Path to the JSON file containing batch metadata.
    output_path : Optional[Path]
        Where to save the downloaded JSONL results.
    use_sample : bool
        If True, use sample-mode default paths when explicit paths are not provided.

    Returns
    -------
    Path
        Path to the downloaded output file.
    """
    default_batch_info_file, default_output_path = get_mode_paths(use_sample=use_sample)

    batch_info_file = Path(batch_info_file) if batch_info_file is not None else default_batch_info_file
    output_path = Path(output_path) if output_path is not None else default_output_path

    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    # Retrieve batch metadata from the API to verify completion and locate output file
    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve batch metadata for batch_id={batch_id}: {e}")

    # Only completed batches can be downloaded safely
    if batch.status != "completed":
        raise ValueError(f"Batch is not completed yet. Current status: {batch.status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise ValueError("No output_file_id found for completed batch.")

    # Download raw JSONL result content for the completed batch
    try:
        content = client.files.content(output_file_id)
    except Exception as e:
        raise RuntimeError(f"Failed to download output file {output_file_id}: {e}")

    # Save the downloaded batch results to disk for downstream parsing and merging
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(content.read())

    print(f"Downloaded results to: {output_path}")
    return output_path


def parse_args():
    """
    Parse command-line arguments for downloading completed batch results.
    """
    # Configure CLI interface for downloading batch results in full or sample mode
    parser = argparse.ArgumentParser(description="Download completed IdiomX batch results.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample-mode default paths.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Explicit batch id. If omitted, load it from batch-info JSON.",
    )
    parser.add_argument(
        "--batch-info-file",
        type=str,
        default=None,
        help="Path to batch info JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save downloaded JSONL results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download_results(
        batch_id=args.batch_id,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        output_path=Path(args.output_path) if args.output_path else None,
        use_sample=args.sample,
    )