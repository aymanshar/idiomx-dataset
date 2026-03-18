from pathlib import Path
import json
from typing import Optional
from llm_enrichment.config.api_config import client

BASE_DIR = Path(__file__).resolve().parents[1]
BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_v2_info.json"

def load_batch_id(batch_info_file: Path = BATCH_INFO_FILE) -> str:
    if not batch_info_file.exists():
        raise FileNotFoundError(f"Batch info file not found: {batch_info_file}")

    with open(batch_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    batch_id = info.get("batch_id")
    if not batch_id:
        raise ValueError("batch_id not found in batch info file")

    return batch_id

def check_batch(batch_id: Optional[str] = None, batch_info_file: Path = BATCH_INFO_FILE):
    if batch_id is None:
        batch_id = load_batch_id(batch_info_file)

    batch = client.batches.retrieve(batch_id)

    print("Batch ID:", batch.id)
    print("Status:", batch.status)
    print("Created at:", getattr(batch, "created_at", None))
    print("Output file ID:", getattr(batch, "output_file_id", None))
    print("Error file ID:", getattr(batch, "error_file_id", None))

    return batch

if __name__ == "__main__":
    check_batch()