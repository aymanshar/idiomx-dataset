"""
IdiomX Dataset Pipeline

Author: Ayman Ali Sharara
Project: IdiomX – Neural Understanding of English Idioms
Year: 2026

Description:
Download completed IdiomX batch results (v2).
"""

from pathlib import Path
import json
from typing import Optional
import argparse

from config.api_config import client


BASE_DIR = Path(__file__).resolve().parents[1]

# ✅ Correct v2 naming (aligned with previous scripts)

# Full mode
DEFAULT_FULL_BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_synth_batch_info_v1.json"
DEFAULT_FULL_OUTPUT_PATH = BASE_DIR / "data" / "results" / "idiomx_synth_results_v1.jsonl"

# Sample mode
DEFAULT_SAMPLE_BATCH_INFO_FILE = BASE_DIR / "data" / "sample" / "idiomx_synth_batch_sample_info_v1.json"
DEFAULT_SAMPLE_OUTPUT_PATH = BASE_DIR / "data" / "sample" / "idiomx_synth_results_sample_v1.jsonl"


def get_mode_paths(use_sample: bool = False) -> tuple[Path, Path]:
    if use_sample:
        return DEFAULT_SAMPLE_BATCH_INFO_FILE, DEFAULT_SAMPLE_OUTPUT_PATH
    return DEFAULT_FULL_BATCH_INFO_FILE, DEFAULT_FULL_OUTPUT_PATH


def load_batch_id(batch_info_file: Path) -> str:
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

    default_batch_info_file, default_output_path = get_mode_paths(use_sample=use_sample)

    batch_info_file = Path(batch_info_file) if batch_info_file else default_batch_info_file
    output_path = Path(output_path) if output_path else default_output_path

    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    # 🔍 Retrieve batch
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise ValueError(f"Batch not completed. Current status: {batch.status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise ValueError("No output_file_id found.")

    # ⬇️ Download
    content = client.files.content(output_file_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(content.read())

    print("Downloaded results to:", output_path)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Download IdiomX batch results v2.")

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--batch-info-file", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download_results(
        batch_id=args.batch_id,
        batch_info_file=Path(args.batch_info_file) if args.batch_info_file else None,
        output_path=Path(args.output_path) if args.output_path else None,
        use_sample=args.sample,
    )