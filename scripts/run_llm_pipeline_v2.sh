#!/bin/bash

echo "======================================"
echo "IdiomX LLM Enrichment Pipeline v2"
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
python llm_enrichment/scripts/prepare_batch_requests_v2.py

echo "Step 2: Submitting batch job"
python llm_enrichment/scripts/submit_batch_v2.py

echo "Step 3: Downloading batch results"
python llm_enrichment/scripts/download_results_v2.py

echo "Step 4: Merging outputs"
python llm_enrichment/scripts/merge_results_v2.py

echo "Step 5: Running validation"
python llm_enrichment/scripts/validate_dataset_v2.py

echo "Step 6: Verifying suspicious rows"
python llm_enrichment/scripts/verify_suspicious_rows.py

echo "======================================"
echo "Pipeline completed successfully"
echo "======================================"

echo "Final dataset:"
echo "data/enriched/idiomx_enriched_final_v2.csv"

echo "Computing dataset checksum..."
sha256sum data/enriched/idiomx_enriched_final_v2.csv