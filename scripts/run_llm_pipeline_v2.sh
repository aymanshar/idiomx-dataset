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
python scripts/prepare_enrichment_batch_requests.py

echo "Step 2: Submitting batch job"
python scripts/submit_batch.py

echo "Step 3: Downloading batch results"
python scripts/download_existing_batch_results.py

echo "Step 4: Merging outputs"
python scripts/merge_batch_results_to_enriched_dataset.py

echo "Step 5: Running validation"
python scripts/validate_dataset.py

echo "Step 6: Verifying suspicious rows"
python scripts/verify_suspicious_rows.py

echo "======================================"
echo "Pipeline completed successfully"
echo "======================================"

echo "Final dataset:"
echo "data/enriched/idiomx_core.csv"

echo "Computing dataset checksum..."
sha256sum data/enriched/idiomx_core.parquet