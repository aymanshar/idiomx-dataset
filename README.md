# IdiomX: Multilingual Idiom Understanding Dataset (EN–AR–FR)
  
---

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/aymansharara/IdiomX)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/aymansharara/idiomx)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19137833-blue)](https://doi.org/10.5281/zenodo.19137833)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset Size](https://img.shields.io/badge/Examples-196K+-informational)]()
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20AR-blue)]()
[![Tasks](https://img.shields.io/badge/Tasks-NLP%20%7C%20Translation%20%7C%20Classification-purple)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)]()


<p align="center">
  <img src="figures/IdiomX_Data_perep_Pipeline_v2.png" width="100%">
</p>

*Three-stage reproducible pipeline for collecting, enriching, validating, and generating idioms.*
---

**A Large-Scale Multilingual Dataset for Idiomatic Expression Understanding**

**Author:** Ayman Ali Sharara  

**Affiliation:**  
MSc Data Science & Machine Learning (SPOC S21)  
DSTI School of Engineering  
https://dsti.school/

**Project Context:**  
Deep Learning with Python  
Supervised by Prof. Hanna Abi Akl  

**Contact:**  
- Academic: ayman.sharara@edu.dsti.institute  
- Personal: aymanshar@gmail.com  

---

## Overview

**IdiomX** is a large-scale **multilingual dataset** for **idiomatic expression understanding in context**.

It is designed to support multiple NLP tasks, including:

- Idiom Detection (idiomatic vs. literal)
- Context → Idiom Retrieval (English)
- Arabic → English Idiom Retrieval
- Multilingual semantic understanding (EN–AR–FR)

Idioms are difficult for NLP systems because their meanings are often **non-compositional**. Expressions such as *“spill the beans”* or *“kick the bucket”* cannot be understood correctly from individual words alone. IdiomX is designed to help models learn this distinction from rich contextual examples.

> This repository contains the full data collection and enrichment pipeline. 
> For the dataset only, see Hugging Face.

---

## Why IdiomX Matters

Idiomatic language remains one of the most challenging phenomena for NLP systems, even with modern large language models.

IdiomX provides:
- large-scale contextual supervision
- controlled semantic variation
- cross-lingual alignment

This makes it a strong benchmark for evaluating real language understanding beyond surface-level text modeling.

---

## Dataset Scale

- ~196K contextualized examples
- ~12K+ unique idioms
- ~172K unique sentences
- ~14 examples per idiom
- Balanced labels (idiomatic / literal / borderline)
- Multilingual semantic alignment (EN–AR–FR)

---

## Key Features

- High semantic quality (validated pipeline)
- Balanced idiomatic vs literal examples
- Low duplication (reuse factor ≈ 1.04)
- Adversarial / hard-negative examples
- Multilingual alignment (EN–AR–FR)
- Modern idioms and slang coverage
- Synthetic expansion for missing idioms
- Fully reproducible pipeline


---

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

## Environment Setup

```bash

conda create -n idiomx python=3.11 -y
conda activate idiomx
pip install -r scripts/requirements.txt

```
---

## Final Dataset

All pipelines are merged into:

```
idiomx_full.parquet
```
---

## Dataset Schema

Each row represents a **contextualized idiom usage example**.

### Core Fields
- `idiom_id`
- `idiom_canonical`
- `example`
- `example_usage_label`
- `is_example_idiom`

### Semantics
- `idiom_canonical_meaning`
- `idiom_in_example_meaning_en`
- `idiom_in_example_meaning_arabic`
- `idiom_in_example_meaning_french`

### Quality
- `semantic_similarity_example_vs_meaning`
- `semantic_quality`

### Metadata
- `source`
- `source_type`
- `idiom_domain`
- `idiom_register`
- `compositionality`
- `learner_difficulty`

---

### Derived Features

| Feature | Description |
|--------|------------|
| sentence_length_chars | Number of characters |
| sentence_length_words | Number of words |
| semantic_similarity_example_vs_meaning | Embedding similarity |
| semantic_quality | High / Medium / Low |

---

## Dataset Statistics

| Metric | Value |
|--------|------|
| Total examples | ~196K |
| Unique idioms | ~12K+ |
| Unique sentences | ~190K |
| Avg examples per idiom | ~14 |
| Reuse factor | ~1.04 |
| Idiomatic examples | ~45–48% |
| Literal examples | ~45–48% |
| Borderline examples | ~5–8% |
| High-quality subset | ~123K |
| Languages | EN / AR / FR |

---

## Data Sources

- Wiktionary (Kaikki)
- WordNet
- Urban Dictionary
- OpenSubtitles
- LLM-based enrichment

All data undergo strict filtering and validation.

---

## Use Cases

IdiomX supports a wide range of NLP tasks:

* idiom detection
* contextual idiom understanding
* idiom retrieval
* cross-lingual semantic retrieval
* multilingual semantic modeling
* machine translation evaluation
* LLM fine-tuning
* semantic search

---

## Limitations

* Some examples are LLM-generated
* Minor annotation noise may still exist
* Idiomatic interpretation may vary across contexts
* Some multilingual fields may be more complete than others depending on source and enrichment stage

---

## License

- MIT License
- CC BY-SA 4.0 (Wiktionary-derived)
- WordNet License

---

## Reproducibility

All dataset construction steps are fully reproducible via:

- Python scripts (modular pipeline)
- Batch LLM enrichment
- Deterministic processing stages

Final outputs can be regenerated from raw sources using the provided scripts.
---

## Links

- HuggingFace: https://huggingface.co/datasets/aymansharara/IdiomX
- GitHub: https://github.com/aymanshar/idiomx-dataset
- Kaggle: https://www.kaggle.com/datasets/aymansharara/idiomx
- Zenodo: https://doi.org/10.5281/zenodo.19137833

---

## Paper

The full dataset paper is available here:

 `docs/IdiomX_Dataset_Paper_v7.pdf`

---

## Citation

If you use this dataset, please cite:

Sharara, Ayman Ali (2026). 
 
**IdiomX: A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding**.  
Zenodo. https://doi.org/10.5281/zenodo.19137833

```bibtex
@article{sharara2026idiomx,
  title={IdiomX: A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding},
  author={Sharara, Ayman Ali},
  year={2026},
  note={Dataset and paper available on GitHub and HuggingFace}
}
```