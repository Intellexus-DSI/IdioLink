# Data Generation Pipeline

End-to-end pipeline for generating the IdioLink idiom retrieval dataset. Reproducing the full dataset requires API keys for Gemini and GPT-4o-mini.

## Pipeline Overview

### 1. MAGPIE Filtering (`magpie/filter_magpie.py`)

Filters the MAGPIE corpus (Haagsma et al., 2020) for idioms with ambiguous usage patterns.

**Criteria:**
- Minimum 30 occurrences in corpus
- Ambiguity range: 25%-75% idiomatic vs literal usage
- Minimum annotation confidence: 1.0

**Input:** `MAGPIE_SOURCE_IDIOMS.csv` (118 candidate idioms with validity flags)

**Output:** Filtered idiom list (107 valid idioms after manual review)

### 2. Variant Generation (`generation/data_pipeline.py`)

Generates sentence variants using Gemini 2.5 Pro via the Agno framework.

**Subject domains (10):**
- Politics, Sport, Technology, History, Medicine
- Culture, Entertainment, Food, Business, Environment

**Variant types per idiom per subject:**
- 4 literal variants (idiom words in physical/concrete sense)
- 2 figurative/idiomatic variants (standard metaphorical meaning)
- 2 simplified variants (paraphrase without idiom, same context as figurative)
- 2 sense variants (independent explanation of meaning)

**Total per idiom:** 10 variants x 10 subjects = 100 sentences

### 3. Annotation/Validation (`annotation/annotation_pipeline.py`)

LLM-based quality validation using Gemini 2.5 Pro (3x majority vote).

**Two-part validation:**
- Sentence validity: subject relevance, grammar, usage label correctness, no duplication
- Span validity: existence, correct boundaries, minimality, semantic accuracy

**Prompt templates:** `annotation/annotation_prompts.py`

### 4. Data Splitting (`annotation/split_data.py`)

Stratified split by idiom (PIE):
- **Train:** 22 idioms (for fine-tuning retrieval models)
- **Val:** 10 idioms (for hyperparameter tuning and early stopping)
- **Test:** 75 idioms (for evaluation)

### 5. Quality Tiers

- **Gold data:** Human-validated subset (manually reviewed for correctness)
- **Silver data:** LLM-validated only (both sentence and span marked valid by annotator)

## Utilities

- `utils/combine_variants.py` - Merge multiple generation runs, filter invalid idioms
- `utils/analyze_errors.py` - Compare LLM vs human annotations, compute precision/recall

## Reproduction

```bash
# 1. Filter MAGPIE corpus
python -m data_generation.magpie.filter_magpie MAGPIE_UNFILTERED.jsonl --output_dir output/

# 2. Generate variants (requires GEMINI_API_KEY)
python -m data_generation.generation.data_pipeline magpie/MAGPIE_SOURCE_IDIOMS.csv --output_dir output/

# 3. Annotate (requires GEMINI_API_KEY)
python -m data_generation.annotation.annotation_pipeline output/idiom_variants.json --output_dir output/

# 4. Split into train/val/test
python -m data_generation.annotation.split_data output/annotated.json --output_dir output/splits/
```

## References

- MAGPIE corpus: Haagsma et al. (2020). "MAGPIE: A Large Corpus of Potentially Idiomatic Expressions." LREC 2020.
