## Data Construction Pipeline

IdiomX is built through **three modular pipelines**:

1. **Core Idiom Pipeline**
2. **Modern Idioms & Slang Pipeline**
3. **Synthetic Idiom Generation Pipeline**

These pipelines are modular and reproducible, and together produce the final unified dataset.

### Pipeline 1 — Core Idioms

- Extract idioms from Wiktionary (Kaikki) and WordNet  
- Normalize and deduplicate  
- Apply strict idiom filtering  
- Generate contextual examples using LLMs  
- Perform semantic validation  

#### Core collection steps

```bash
python -m scripts.collect_01_extract_idioms_from_kaikki
python -m scripts.collect_02_filter_strict_idioms
python -m scripts.collect_03_clean_idioms
python -m scripts.collect_04_build_high_precision_idioms
python -m scripts.collect_05_normalize_kaikki_high_precision
python -m scripts.collect_06_extract_wordnet_multiword_expressions
python -m scripts.collect_07_merge_wordnet_with_kaikki
python -m scripts.collect_08_filter_global_idioms
python -m scripts.collect_09_finalize_pre_enrichment_dataset
python -m scripts.collect_10_dataset_statistics
```

#### Description

* Extract idioms from Wiktionary (Kaikki) and WordNet
* Apply strict filtering
* Normalize canonical forms
* Merge sources
* Produce high-quality pre-enrichment dataset


#### Core enrichment steps

```bash 
python -m scripts.en_01_prepare_enrichment_batch_requests_v2
python -m scripts.en_02_submit_batch_v2
python -m scripts.en_03_check_existing_batch_v2
python -m scripts.en_04_download_existing_batch_results
python -m scripts.en_05_merge_batch_results_to_enriched_dataset_v2
python -m scripts.en_06_validate_dataset_v2
python -m scripts.en_07_verify_suspicious_rows_v2
python -m scripts.en_08_final_dataset_statistics_v2
```

---

### Pipeline 2 — Modern Idioms & Slang

- Sources: Urban Dictionary, Wiktionary slang, OpenSubtitles  
- Clean noisy user-generated content  
- Deduplicate at idiom level  
- Align to IdiomX schema  
- LLM enrichment and validation  

#### Modern Data Collection steps

```bash 
python -m scripts.collect_modern_01_extract_urban_dictionary_slang
python -m scripts.collect_modern_02_clean_urban_dictionary_source
python -m scripts.collect_modern_03_extract_wiktionary_slang
python -m scripts.collect_modern_04_clean_wiktionary_slang
python -m scripts.collect_modern_05_merge_sources_stage1_urban_wiktionary
python -m scripts.collect_modern_06_download_opensubtitles_source
python -m scripts.collect_modern_07_extract_opensubtitles_slang_candidates
python -m scripts.collect_modern_08_clean_opensubtitles_slang_candidates
python -m scripts.collect_modern_09_merge_sources_stage2_add_opensubtitles
python -m scripts.collect_modern_10_filter_global_modern_idioms
python -m scripts.collect_modern_11_finalize_modern_pre_enrichment_dataset
python -m scripts.collect_modern_12_compare_modern_with_main_idiomx
python -m scripts.collect_modern_13_dedup_modern_by_idiom_only_pre_llm
```

#### Modern LLM Enrichment

```bash 
python -m scripts.en_modern_01_prepare_enrichment_batch_requests_v1
python -m scripts.en_modern_02_submit_batch_v1
python -m scripts.en_modern_03_check_existing_batch_v1
python -m scripts.en_modern_04_download_existing_batch_results_v1
python -m scripts.en_modern_05_parse_modern_batch_results_v1
python -m scripts.en_modern_06_flatten_modern_enriched_results_v1
python -m scripts.en_modern_07_filter_valid_modern_idioms_v1
python -m scripts.en_modern_08_finalize_for_merge_v1
python -m scripts.en_modern_09_align_to_idiomx_final_schema_v1
python -m scripts.en_modern_09_validate_dataset_v1
python -m scripts.en_modern_10_finalize_for_merge_v1
python -m scripts.en_modern_11_align_schema_for_merge_v1
```
---
### Pipeline 3 — Synthetic Idiom Generation

- Generate missing idioms using LLMs  
- Deduplicate and filter weak candidates  
- Maintain blacklist of invalid patterns  
- Enrich and validate generated idioms  

#### Synthetic generation steps
```bash 
python -m scripts.synthetic_01_prepare_missing_idiom_generation_requests_v1 --candidates-per-category 300
python -m scripts.synthetic_02_submit_batch_v1
python -m scripts.synthetic_03_check_batch_v1
python -m scripts.synthetic_03b_list_batches_v1
python -m scripts.synthetic_cancel_batch
python -m scripts.synthetic_04_download_batch_results_v1
python -m scripts.synthetic_05_parse_clean_deduplicate_v1
python -m scripts.synthetic_06_merge_and_update_blacklist_v1
python -m scripts.synthetic_07_prepare_for_enrichment_v1
```

#### Synthetic enrichment steps

```bash 
python -m scripts.en_synth_01_prepare_enrichment_batch_requests_v1
python -m scripts.en_synth_02_submit_batch_v1
python -m scripts.en_synth_03_check_existing_batch_v1
python -m scripts.en_synth_04_download_existing_batch_results_v1
python -m scripts.en_synth_05_parse_synth_batch_results_v1
python -m scripts.en_synth_06_flatten_synth_enriched_results_v1
python -m scripts.en_synth_07_filter_valid_synth_idioms_v1
python -m scripts.en_synth_08_finalize_for_merge_v1
python -m scripts.en_synth_09_validate_dataset_v1
```

---
## Pipeline Notebooks

The dataset workflow is also documented in notebooks:

1. `01_data_collection.ipynb`
2. `02_data_enrichment_pipeline.ipynb`
3. `03_finalize_idiomx_dataset.ipynb`
4. `04_finalize_idiomx_modern_dataset_v1.ipynb`
5. `05_merge_idiomX_and_modern_idiom.ipynb`
6. `06_merge_idiomX_modern_and_synth.ipynb`

These correspond to:

| Step | Description |
| --- | --- |
| 01 | Data extraction and preprocessing |
| 02 | LLM enrichment and semantic augmentation |
| 03 | Final cleaning, validation, and dataset export |
| 04 | finalize idiomx modern dataset v1 |
| 05 | merge idiomX and modern |
| 06 | final merge idiomX main, modern and synth |

---