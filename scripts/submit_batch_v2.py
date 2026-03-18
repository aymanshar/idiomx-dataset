from pathlib import Path
import json
from llm_enrichment.config.api_config import client

BASE_DIR = Path(__file__).resolve().parents[1]
BATCH_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_v2.jsonl"
BATCH_INFO_FILE = BASE_DIR / "data" / "batches" / "idiomx_batch_v2_info.json"

def submit_batch(batch_file: Path = BATCH_FILE, batch_info_file: Path = BATCH_INFO_FILE):
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    with open(batch_file, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch"
        )

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "project": "IdiomX",
            "stage": "v2_full_enrichment"
        }
    )

    batch_info = {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "batch_file": str(batch_file)
    }

    batch_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_info_file, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)

    print("Uploaded file ID:", uploaded.id)
    print("Batch ID:", batch.id)
    print("Batch status:", batch.status)
    print("Batch info saved to:", batch_info_file)

    return batch.id

if __name__ == "__main__":
    submit_batch()