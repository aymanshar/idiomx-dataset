#!/bin/bash

echo "======================================"
echo "IdiomX LLM Enrichment Pipeline"
echo "======================================"

echo "Creating conda environment..."

conda create -n idiomx python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate idiomx

echo "Installing python libraries..."

pip install -r scripts/requirements.txt

echo "======================================"
echo "Running LLM enrichment pipeline"
echo "======================================"

echo "Step 1: Preparing batch requests"
python scripts/en_01_prepare_enrichment_batch_requests_v2.py

echo "Step 2: Submitting batch job"
python scripts/en_02_submit_batch_v2.py

echo "Step 3: check existing batch status"
python scripts/en_03_check_existing_batch_v2.py

echo "Step 4: Downloading batch results"
python scripts/en_04_download_existing_batch_results_v2.py

echo "Step 5: Merging outputs"
python scripts/en_05_merge_batch_results_to_enriched_dataset_v2.py

echo "Step 6: Running validation"
python scripts/en_06_validate_dataset_v2.py

echo "Step 7: Verifying suspicious rows"
python scripts/en_07_verify_suspicious_rows_v2.py

echo "======================================"
echo "Pipeline completed successfully"
echo "======================================"

echo "Final dataset:"
echo "data/enriched/idiomx_core.csv"

echo "Computing dataset checksum..."
sha256sum data/enriched/idiomx_core.parquet