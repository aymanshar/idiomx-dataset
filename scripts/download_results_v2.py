from pathlib import Path
import json
from typing import Optional
from llm_enrichment.config.api_config import client

BASE_DIR = Path(__file__).resolve().parents[1]
BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_v2_info.json"
OUTPUT_PATH = BASE_DIR / "data" / "results" / "idiomx_results_v2.jsonl"

def load_batch_id(batch_info_file: Path = BATCH_INFO_FILE) -> str:
    with open(batch_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info["batch_id"]

def download_results(batch_id: Optional[str] = None,
                     batch_info_file: Path = BATCH_INFO_FILE,
                     output_path: Path = OUTPUT_PATH):
    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise ValueError(f"Batch is not completed yet. Current status: {batch.status}")

    if not getattr(batch, "output_file_id", None):
        raise ValueError("No output_file_id found for completed batch.")

    content = client.files.content(batch.output_file_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(content.read())

    print(f"Downloaded results to: {output_path}")
    return output_path

if __name__ == "__main__":
    download_results()