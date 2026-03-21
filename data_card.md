# IdiomX Dataset Card

## Overview
IdiomX is a large-scale bilingual dataset for idiomatic expression understanding.

## Project Context
This dataset was developed as part of the MSc Data Science & Machine Learning program at DSTI School of Engineering, under the Deep Learning with Python course, supervised by Prof. Hanna Abi Akl.

## Statistics
- Total examples: 123,336
- Unique idioms: 14,986
- Languages: English, Arabic
- Label balance: 50/50
- Arabic coverage: 99.99%

## Languages
- English (primary)
- Arabic (optional enrichment)

## Fields
- idiom_canonical
- idiom_surface
- example
- idiom_canonical_meaning
- source
- pos
- tags

## Source Data
- Wiktionary (via Kaikki.org)
- WordNet (Princeton)

## Tasks Supported
- Idiom detection
- Literal vs idiomatic classification
- Cross-lingual idiom understanding

## License
- MT. License
- CC BY-SA 4.0 (Wiktionary-derived)
- WordNet License

## Intended Use
- Research
- Model training
- Semantic understanding

## Limitations
- Some examples missing
- WordNet portion is small

## Ethical Considerations
Dataset contains linguistic data only; no personal or sensitive information.

