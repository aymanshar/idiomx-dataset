# IdiomX Dataset 🧠

**IdiomX** is a large-scale, high-quality dataset for English idiom understanding, translation, and contextual modeling.

It is designed for research in:
- Idiom detection
- Context → idiom mapping
- Semantic understanding
- Cross-lingual idiom translation (English ↔ Arabic)

---

## 📊 Dataset Overview

| Property | Value |
|--------|------|
| Total Idioms | ~12,800 |
| Sources | Wiktionary (Kaikki), WordNet |
| Language | English |
| Format | CSV, Parquet |
| Enrichment | Optional (LLM-based) |

---

## 🧱 Dataset Structure

	data/
	├── raw/ # (ignored in git)
	├── processed/ # intermediate
	├── enriched/ # LLM outputs
	├── final/ # final datasets
	└── samples/ # small public samples
	

---

## 📌 Core Dataset Columns

| Column | Description |
|------|------------|
| idiom_id | Unique identifier |
| idiom_canonical | Base idiom |
| idiom_surface | Variant form |
| example | Example sentence |
| idiom_canonical_meaning | Meaning of idiom |
| source | Data source |
| pos | Part of speech |
| tags | Linguistic tags |
| idiom_confidence_score | Confidence score |
| record_origin | source / LLM |
| license_source | License origin |

---

## 📚 Data Sources

### Wiktionary (via Kaikki.org)
- License: CC BY-SA 4.0

### WordNet
- License: Princeton WordNet License

---

## ⚖️ Licensing

This dataset includes material from:

- Wiktionary (CC BY-SA 4.0)
- WordNet (Princeton License)

See:
- `DATASET_LICENSE.md`
- `THIRD_PARTY_NOTICES.md`

---

## 🔁 Reproducibility

Full pipeline included:

	scripts/
	notebooks/
	

Steps:
1. Extract idioms from Wiktionary
2. Clean + normalize
3. Merge WordNet
4. Build dataset
5. (Optional) LLM enrichment

---

## 📦 Available Files

### Final dataset
- `data/final/idiomx_core.csv`
- `data/final/idiomx_core.parquet`

### Sample
- `data/samples/idiomx_sample_1000.csv`

---

## 🚀 Use Cases

- NLP research
- Idiom detection models
- Transformer training (T5, BERT)
- Cross-lingual translation
- Semantic retrieval

---

## 📖 Citation

If you use this dataset:

@dataset{idiomx2026,
title={IdiomX: Neural Dataset for Idiom Understanding},
author={Ayman},
year={2026}
}


---

## 👨‍💻 Author

Developed by Ayman Sharara  
UAE 🇦🇪

---

## ⭐ Contributions

Pull requests are welcome for:
- Additional sources
- Better filtering
- Multilingual extensions
