"""Prompt templates for LLM-based annotation of idiom variants.

Two-part validation:
- Part A: Sentence validity (subject relevance, coherence, usage label, duplication, context matching)
- Part B: Span validity (existence, boundaries, minimality, semantic accuracy, contiguity)
"""

ANNOTATION_PROMPT = """You are an expert linguistic annotator specializing in validating idiomatic expressions and their usage patterns.
Your task is to validate generated sentence data for quality, correctness, and adherence to specific guidelines.

You will receive a group of 10 sentences generated for a specific idiom and subject domain.
Each sentence has: idiom, subject domain, sentence, span, and usage label (literal/idiomatic/simplification/sense).

Validate EACH sentence with TWO SEPARATE JUDGMENTS:

**PART A: SENTENCE VALIDITY**

1. Subject Relevance - sentence must relate to specified domain
2. General Coherence - logical, grammatical, natural
3. Usage Label Correctness:
   - literal: idiom words in physical/concrete sense
   - idiomatic: standard figurative meaning, idiom must appear
   - simplification: idiom meaning without the idiom itself, same context as paired idiomatic
   - sense: explains idiom meaning without using it, independent context
4. No Duplication - idiom meaning should not appear twice
5. Context Matching (simplification only) - must reuse paired idiomatic variant's context

**PART B: SPAN VALIDITY**

1. Phrase Existence - span must exist in sentence
2. Correct Boundaries - idiom form for literal/idiomatic, meaning phrase for simplification/sense
3. Minimality - only necessary words, no extras
4. Semantic Accuracy - captures action/state, not interpretation
5. Contiguous Substring - single unbroken text span

For each sentence provide:
- llm_sentence_valid: 1 (valid) or 0 (invalid)
- llm_sentence_correction: fix suggestion if invalid, else ""
- sentence_issues: description if invalid, else ""
- llm_span_valid: 1 (valid) or 0 (invalid)
- llm_span_correction: corrected span if invalid, else ""
- span_issues: description if invalid, else ""

Evaluate sentence and span INDEPENDENTLY."""

ANNOTATION_EXAMPLES = """
Example 1 (both valid):
  Idiom: "spill the beans", Subject: "Politics", Usage: "idiomatic"
  Sentence: "The whistleblower decided to spill the beans about the corruption scandal."
  Span: "spill the beans"
  -> sentence_valid=1, span_valid=1

Example 2 (sentence invalid, span valid):
  Idiom: "spill the beans", Subject: "Medicine", Usage: "literal"
  Sentence: "The chef accidentally spilled the beans while preparing dinner."
  Span: "spilled the beans"
  -> sentence_valid=0 (subject mismatch: cooking not medicine), span_valid=1

Example 3 (sentence valid, span invalid):
  Idiom: "all along", Subject: "History", Usage: "sense"
  Sentence: "His analysis showed the empire's decline was an inevitable process from the start."
  Span: "inevitable process from the start"
  -> sentence_valid=1, span_valid=0 (too broad: should be "from the start")
"""
