---
pretty_name: IdiomX
language:
- en
- ar
license: other
task_categories:
- text-classification
- translation
- text-generation
tags:
- idioms
- bilingual
- english
- arabic
- semantic-understanding
- figurative-language
size_categories:
- 100K<n<1M
---

# IdiomX  
### A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding

**Author:** Ayman Ali Sharara  
**Affiliation:** Independent Researcher, UAE  
**Email:** aymanshar@gmail.com  

---

## 📌 Overview

**IdiomX** is a large-scale, high-quality bilingual dataset designed for **idiomatic expression understanding**, including detection, interpretation, and cross-lingual analysis.

The dataset contains **over 123,000 contextualized examples** derived from approximately **15,000 English idioms**, enriched with semantic annotations and **English–Arabic translations**.

To the best of our knowledge, **IdiomX is the largest publicly available bilingual idiom dataset** with contextualized examples and semantic consistency validation.

---

## 📊 Dataset Statistics

| Metric | Value |
|------|------|
| Total examples | 123,336 |
| Unique idioms | 14,986 |
| Avg examples per idiom | 8.2 |
| Arabic coverage | 99.99% |
| Label balance | 50/50 |

---

## 🌍 Languages

- English 🇬🇧  
- Arabic 🇸🇦  

---

## 🧠 Features

Each record includes:

- `idiom_canonical`
- `idiom_surface`
- `idiom_in_example`
- `idiom_in_example_meaning_en`
- `idiom_in_example_meaning_arabic`
- `idiom_canonical_meaning`
- `idiom_canonical_meaning_arabic`
- `example_usage_label` (idiomatic / literal)
- `semantic_consistency`
- Additional linguistic features

---

## 🔍 Data Sources

This dataset is constructed from **high-quality lexical resources only**:

- **Wiktionary**
- **WordNet**

All other sources were excluded to ensure consistency and reliability.

---

## ⚙️ Dataset Construction

The dataset is built through a multi-stage pipeline:

1. **Data Collection**
   - Extract idioms from Wiktionary and WordNet

2. **Preprocessing**
   - Cleaning, normalization, deduplication

3. **LLM Enrichment**
   - Generate contextual examples
   - Generate English and Arabic meanings
   - Generate translations

4. **Validation**
   - Missing value analysis
   - Label consistency checks (>99.98%)
   - Semantic consistency scoring
   - Surface-form validation

---

## 📈 Validation Highlights

- Label consistency: **>99.98%**
- Arabic coverage: **~100%**
- Mean semantic consistency score: **~0.59**

---

## 📂 Files

- `idiomx_core.parquet` → main dataset (recommended)
- `idiomx_core.csv` → CSV version
- `dataset_statistics.json` → dataset summary statistics

---

## 🚀 Use Cases

IdiomX supports a wide range of NLP tasks:

- Idiom detection (idiomatic vs literal classification)
- Idiom interpretation and meaning retrieval
- Context-to-idiom generation
- Cross-lingual idiom translation
- Multilingual semantic understanding

---

## ⚠️ Limitations

- Some examples are generated using LLMs
- Minor annotation noise may exist (<0.01%)
- Idiomatic interpretation may vary across contexts

---

## 📄 Paper

The full dataset paper is available here:

👉 `docs/IdiomX_Dataset_Paper_v1.pdf`

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@article{sharara2026idiomx,
  title={IdiomX: A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding},
  author={Sharara, Ayman Ali},
  year={2026},
  note={Dataset and paper available on GitHub and HuggingFace}
}