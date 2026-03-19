# IdiomX Dataset Card

## Overview
IdiomX is a large-scale bilingual dataset for idiomatic expression understanding.

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

