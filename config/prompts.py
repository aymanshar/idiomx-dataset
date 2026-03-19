IDIOM_ENRICHMENT_PROMPT = """
You are an expert linguist specializing in English idioms.

Task:
Analyze the idiom and enrich the dataset.

Input:
idiom: "{idiom}"
existing_meaning: "{meaning}"

Steps:

1. Determine if the phrase is a real idiom.
2. Normalize the canonical idiom.
3. Provide the canonical English meaning.
4. Translate the canonical meaning to Arabic.
5. Generate 4 natural example sentences using the idiom.
6. Extract the idiom surface form used in each sentence.
7. Explain the meaning of the idiom in that example.
8. Translate the sentence and meaning to Arabic.

Return ONLY valid JSON with this schema:

{
"is_idiom": true/false,

"idiom_canonical": "...",

"idiom_canonical_meaning": "...",

"idiom_canonical_meaning_arabic": "...",

"examples":[

{
"idiom_surface":"...",
"idiom_in_example":"...",
"idiom_in_example_arabic":"...",
"idiom_in_example_meaning_en":"...",
"idiom_in_example_meaning_arabic":"..."
}

]

}

Generate exactly 4 examples.
No explanations.
Only JSON.
"""