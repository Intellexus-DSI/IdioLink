# Production-Readiness Plan for the Pending Changeset

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the pending diff (+11 modified files, +9 new files, +207 passing tests) as a series of small, dependency-ordered commits so the repo cleanly supports (a) the four-variant zero-shot matrix on all 24 paper models, (b) the per-model instruction overrides used by the BGE/Qwen3 family, (c) the two-preset index ablation, and (d) the new evaluator splits and variant/ablation tables. The endpoint is "merge-ready and reproducible on a fresh checkout".

**Architecture:** The change set is one coherent feature pillar with four layers stacked bottom-up:
1. Data layer — `subject` field on `IdiomQuery` (already present in `data/*/queries.json`).
2. Eval layer — additive `by_usage` / `by_subject` keys on the evaluator output (back-compat preserved).
3. Model layer — per-model instruction resolver (`resolve_instructions`) + new `e5_inline_no_space` / `prompt_prefix` formats wired into `InstructionModel` and `QwenModel`.
4. Driver + analysis layer — `idiolink/ablation.py`, `run_ablation.py`, and three new analysis scripts that consume layers 1–3.

Two pieces sit *outside* the pillar and need a deliberate keep/drop decision: `run_queue.py` (one-off MPS-only sequential runner) and `--index_filter` on `run_dense.py` / `run_bm25.py` / `run_all.py` (redundant with the resumable `run_ablation.py`).

**Tech Stack:** Python 3.10+, sentence-transformers, transformers, rank_bm25, scikit-learn (stopwords for lexical overlap), pytest. Models loaded via the existing `idiolink/models/registry.py` factory.

---

## Current State (as of 2026-05-24)

**Pending diff vs `main` (`git diff --stat`):**
```
 idiolink/evaluator.py                   |  87 ++++++++++++++++-----
 idiolink/models/instruction_model.py    |  71 ++++++++++++++++-
 idiolink/models/late_chunking.py        |   5 +-
 idiolink/models/qwen.py                 |  23 +++++-
 idiolink/models/registry.py             |  39 ++++++++--
 idiolink/trainer/contrastive_trainer.py |   6 +-
 idiolink/utils.py                       |   2 +
 run_all.py                              |  39 ++++++++--
 run_bm25.py                             |  40 ++++++++--
 run_dense.py                            |  24 +++++-
 run_instruction.py                      |   8 +--
 tests/test_evaluator.py                 | 101 ++++++++++++++++++++++-
 tests/test_registry.py                  |   4 +-
 13 files changed, 398 insertions(+), 51 deletions(-)
```

**Untracked (new) files:**
- `idiolink/ablation.py`, `run_ablation.py`, `run_queue.py`
- `tests/test_ablation.py`, `tests/test_instructions.py`
- `analysis/generate_ablation_table.py`, `analysis/generate_variant_tables.py`, `analysis/lexical_overlap.py`
- `assets/variant_*.csv`, `assets/ablation/*.csv` (regenerable artifacts)

**Test suite:** 207 passing locally (`pytest tests/ -q` → `207 passed in 1.82s`).

---

## Production-Readiness Assessment

| Group | Files | Verdict | Risk | Notes |
|-------|-------|---------|------|-------|
| **G1. Subject field** | `idiolink/utils.py` | KEEP | none | Pure additive default `""`. Test data has 100% subject coverage. |
| **G2. Evaluator splits** | `idiolink/evaluator.py`, `tests/test_evaluator.py` | KEEP | none | Top-level keys unchanged → back-compat. `by_subject` is the only one whose semantic interpretation is research-flavoured; see Task 3 for the inline doc. |
| **G3. Instruction resolver + formats** | `idiolink/models/instruction_model.py`, `idiolink/models/registry.py`, `idiolink/models/qwen.py`, `idiolink/trainer/contrastive_trainer.py`, `tests/test_instructions.py`, `tests/test_registry.py` | KEEP w/ verification | medium | Need to confirm `e5_inline_no_space` against official model cards for Lychee, Linq, Nemotron — Qwen3 family is documented. Old `BGE_PROMPT` enum branch is now unused; document or remove. |
| **G4. Late-chunking dtype fix** | `idiolink/models/late_chunking.py` | KEEP | none | Pure bugfix for bf16/fp16 → `.numpy()` crash. |
| **G5. Ablation module** | `idiolink/ablation.py`, `tests/test_ablation.py`, `run_ablation.py` | KEEP | low | Self-contained, resumable, idempotent. |
| **G6. Analysis scripts** | `analysis/generate_variant_tables.py`, `analysis/generate_ablation_table.py`, `analysis/lexical_overlap.py` | KEEP | low | Pure read-side; depend on G2 + G5. |
| **G7. `--index_filter` on existing runners** | `run_all.py`, `run_dense.py`, `run_bm25.py` | KEEP-minimal | low | Convenient but partially redundant with `run_ablation.py`. Keep on `run_dense.py` (ad-hoc one-off), drop on `run_all.py` (which is the *zero-shot* matrix runner, not an ablation runner). |
| **G8. `run_instruction.py` refactor** | `run_instruction.py` | KEEP | none | One-line swap to use `resolve_instructions`. |
| **G9. `run_queue.py`** | `run_queue.py` | DROP from main | n/a | Hardware-specific (MPS-only, hardcoded ≤1.5B subset), one-off. Belongs in `scripts/` or a personal branch, not in the production matrix runner. |
| **G10. Regenerable CSV artifacts** | `assets/variant_*.csv`, `assets/ablation/*.csv` | KEEP | none | Cheap to commit; helps reviewers see the numbers without re-running. |

**Critical pre-merge risk: paper-model coverage.** The paper announces 24 models. Only 13 have zero-shot metrics on disk, and `run_queue.py` capped itself at ≤1.5B. The eleven untested models cover the entire 7B–9B segment plus Stella-1.5B / GTE-Qwen2-1.5B / Qwen3-Embedding-4B. **The registry must be exercised against all 24 IDs in some form before merge** — even a "loads + encodes one batch" smoke test catches `trust_remote_code` regressions, missing tokeniser files, and instruction-format mismatches.

---

## Merge Ordering & Dependencies

```
Task 1 (G1: subject field)  ─┐
                             ├──> Task 2 (G2: evaluator splits) ──┐
                             │                                    │
Task 4 (G4: late_chunking fix)                                    │
                                                                  ▼
Task 3 (paper-coverage smoke) ────────> Task 5 (G3: resolver + Qwen no-space + tests)
                                                                  │
                                                                  ▼
                                Task 6 (G3 cleanup: bge_prompt audit + registry pruning)
                                                                  │
                                                                  ▼
                                                  Task 7 (G5: ablation module + runner)
                                                                  │
                                                                  ▼
                              Task 8 (G6: analysis scripts) ──┬───┘
                                                              │
                                                              ▼
Task 9 (G7+G8: runner CLI cleanup) ─> Task 10 (G9 decision: drop run_queue.py)
                                                              │
                                                              ▼
                                   Task 11 (verification: full test + sample reruns)
                                                              │
                                                              ▼
                                   Task 12 (README + AGENTS.md updates)
                                                              │
                                                              ▼
                                  Task 13 (final commit / merge to main)
```

Each task is a single commit. The chain is total-ordered because Task 5 depends on Task 2 (evaluator splits are consumed by the resolver tests indirectly) and Task 7 depends on Task 5 (ablation runner imports `resolve_instructions`).

---

## File Structure

**Already exists (modified by this plan):**
- `idiolink/utils.py` — adds `subject` to `IdiomQuery`
- `idiolink/evaluator.py` — adds `build_subject_gold` and `by_usage` / `by_subject` keys
- `idiolink/models/registry.py` — adds `instruction_text` / `instruction_fn` / `e5_inline_no_space` / `prompt_prefix` plumbing
- `idiolink/models/instruction_model.py` — `resolve_instruction(s)` + new format branches
- `idiolink/models/qwen.py` — instruction-format-aware spacing
- `idiolink/models/late_chunking.py` — bf16-safe dtype cast
- `idiolink/trainer/contrastive_trainer.py` — uses resolver
- `run_dense.py`, `run_instruction.py`, `run_all.py`, `run_bm25.py` — runner wiring
- `tests/test_evaluator.py`, `tests/test_registry.py`

**New (created by this plan):**
- `idiolink/ablation.py` — preset registry + filter helper (75 LOC)
- `run_ablation.py` — resumable ablation runner (442 LOC)
- `analysis/generate_variant_tables.py` — per-variant split tables (120 LOC)
- `analysis/generate_ablation_table.py` — per-mode ablation tables (140 LOC)
- `analysis/lexical_overlap.py` — keyword overlap stats (237 LOC)
- `tests/test_ablation.py` — preset + gold-shrink tests
- `tests/test_instructions.py` — resolver tests
- `docs/superpowers/plans/2026-05-24-production-readiness.md` — this file

**To delete (per Task 10):**
- `run_queue.py` — superseded by `run_ablation.py` and the existing `run_all.py` with `--models`

**Regenerable assets (committed for review convenience):**
- `assets/variant_{sentence,span,instruction_sentence,instruction_span}.csv`
- `assets/ablation/ablation_{sentence,span,instruction_sentence,instruction_span}.csv`
- `assets/ablation/ablation_results.csv`

---

## Task 1: Land the data-layer foundation (`subject` on `IdiomQuery`)

**Files:**
- Modify: `idiolink/utils.py:14-22` (dataclass) and `idiolink/utils.py:46-56` (loader)
- The change is already in the working tree — this task stages and commits *only* the `utils.py` diff and verifies it does nothing else.

- [ ] **Step 1: Confirm the diff is exactly two lines**

Run: `git diff idiolink/utils.py`
Expected: only `subject: str = ""` added to `IdiomQuery` and `subject=item.get("subject", "")` added in `load_queries`. Anything else → reject and split.

- [ ] **Step 2: Verify data files supply `subject` for every query**

Run:
```bash
python -c "
import json
for split in ('train', 'val', 'test'):
    qs = json.load(open(f'data/{split}/queries.json'))
    missing = sum(1 for q in qs if not q.get('subject'))
    print(f'{split}: {len(qs)} queries, {missing} missing subject')
"
```
Expected: all three splits report `0 missing subject`.

- [ ] **Step 3: Run the evaluator tests (which already use the field via fixtures in Task 2)**

Run: `pytest tests/test_evaluator.py -q`
Expected: PASS (subject-related tests live in test_evaluator.py but their fixtures construct `IdiomQuery(..., subject="Politics")` — those imports must already resolve after Step 1).

- [ ] **Step 4: Stage and commit ONLY `idiolink/utils.py`**

```bash
git add idiolink/utils.py
git diff --cached --stat   # must show ONLY idiolink/utils.py
git commit -m "$(cat <<'EOF'
Add subject field to IdiomQuery dataclass

Sourced from data/*/queries.json (already 100% populated on the test split).
Foundation for the by_subject evaluator split and per-model instruction
templates that reference {subject}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Land the evaluator splits (`by_usage`, `by_subject`)

**Files:**
- Modify: `idiolink/evaluator.py` (adds `build_subject_gold`, `_avg`, populates new keys)
- Modify: `tests/test_evaluator.py` (adds `TestBuildSubjectGold`, `TestEvaluatorSplits`)

- [ ] **Step 1: Confirm top-level metric keys remain unchanged**

Read `idiolink/evaluator.py:140-148`. The returned dict must still have `r_precision`, `ndcg@10`, `num_queries` as top-level keys. The new keys are `by_usage` and `by_subject` only.

- [ ] **Step 2: Inline-document the `by_subject` semantics**

Edit `idiolink/evaluator.py:36-52` (the `build_subject_gold` docstring) to make explicit that this metric treats *any same-subject doc* as relevant — it is a sanity-check signal ("did the model retrieve topically-coherent text?"), not a substitute for the idiom-relevance gold. Append two sentences to the existing docstring:

```python
"""
Subject-based gold: a doc is relevant to a query iff their `subject` fields match.

This is a topical-coherence signal, not an idiom-relevance signal. It is
intentionally weaker than `build_gold_standard`: it ignores idiom identity
and usage type. Use it for diagnostic comparison only; the headline metric
remains the idiom-relevance gold.

Queries without a subject are mapped to None so they can be excluded from the
subject-based metric (rather than averaging in a degenerate 0).
"""
```

- [ ] **Step 3: Run the evaluator test suite**

Run: `pytest tests/test_evaluator.py -v`
Expected: all green. Specifically `TestBuildSubjectGold::*` (2 tests) and `TestEvaluatorSplits::*` (4 tests) pass, plus the pre-existing tests are unaffected.

- [ ] **Step 4: Spot-check on a real metrics file**

Run:
```bash
python -c "
import json
m = json.load(open('results/zero_shot/BAAI__bge-m3/sentence/metrics.json'))
assert {'r_precision','ndcg@10','num_queries'}.issubset(m), 'top-level keys missing'
assert 'by_usage' in m and 'by_subject' in m, 'splits missing'
assert m['by_usage']['literal']['num_queries'] + m['by_usage']['idiomatic']['num_queries'] == m['num_queries']
print('OK:', {k: m[k] for k in ('r_precision','ndcg@10','num_queries')})
print('literal:', m['by_usage']['literal'])
print('idiomatic:', m['by_usage']['idiomatic'])
print('by_subject:', m['by_subject'])
"
```
Expected: assertions pass; counts sum to `num_queries`.

- [ ] **Step 5: Commit evaluator + tests together**

```bash
git add idiolink/evaluator.py tests/test_evaluator.py
git commit -m "$(cat <<'EOF'
Add by_usage and by_subject splits to evaluator output

Top-level keys (r_precision, ndcg@10, num_queries) are unchanged so existing
consumers keep working. New keys:

- by_usage: same metrics computed on literal-only / idiomatic-only subsets.
- by_subject: diagnostic metric using shared `subject` as binary relevance
  (queries without a subject are excluded, not zero-averaged).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Validate paper-model coverage (the merge-blocker)

**Files:** none modified — this is a verification task that may *gate* later tasks.

**Why this comes before Task 5:** the registry edits in Task 5 (`instruction_format="e5_inline_no_space"` for Qwen3 / Lychee / Linq / Nemotron) cannot be merged without evidence that those models actually load and encode under the new code path. Eleven of the 24 paper models have no zero-shot results on disk.

- [ ] **Step 1: Enumerate the gap**

Run:
```bash
python -c "
from idiolink.models.registry import MODEL_REGISTRY
from pathlib import Path
done = {p.parent.parent.name.replace('__','/') for p in Path('results/zero_shot').glob('*/*/metrics.json')}
missing = sorted(set(MODEL_REGISTRY) - done)
print(f'{len(done)}/{len(MODEL_REGISTRY)} models have at least one zero-shot run')
print('Missing zero-shot results:')
for m in missing:
    print(f'  {m}  ({MODEL_REGISTRY[m].size_params})')
"
```
Expected output (current state, will look like):
```
13/24 models have at least one zero-shot run
Missing zero-shot results:
  Alibaba-NLP/gte-Qwen2-1.5B-instruct  (1.5B)
  Alibaba-NLP/gte-Qwen2-7B-instruct    (7B)
  BAAI/bge-multilingual-gemma2         (9B)
  GritLM/GritLM-7B                     (7B)
  Linq-AI-Research/Linq-Embed-Mistral  (7B)
  NovaSearch/stella-en-1.5B-v5         (1.5B)
  Qwen/Qwen3-Embedding-4B              (4B)
  Qwen/Qwen3-Embedding-8B              (8B)
  Salesforce/SFR-Embedding-Mistral     (7B)
  intfloat/e5-mistral-7b-instruct      (7B)
  nvidia/llama-embed-nemotron-8b       (8B)
```

- [ ] **Step 2: Decide a coverage gate with the user**

Pause and ask the user which of these three options is the merge bar (each option is a *separate* commit gate for the registry changes in Task 5):

1. **Strict** — full 4-mode × 24-model zero-shot matrix must finish before Task 5 lands.
2. **Loaded** — every registry entry must at minimum load + encode one batch + write a metrics.json on the 5-query debug subset. Lightweight (`python run_all.py --debug`).
3. **Loaded for ≤7B, deferred for 8B+** — same as Loaded but the four 8B–9B models get a tracked `TODO` issue and can land later when GPU is available.

The plan assumes **option 2 (Loaded) for everything that fits**, plus a documented TODO for ≥8B if hardware is unavailable. Adjust Step 3 below if the user picks otherwise.

- [ ] **Step 3: Run the smoke-test matrix**

Run (will take 20–60 min depending on GPU availability):
```bash
python run_all.py --debug 2>&1 | tee /tmp/smoke.log
```
Expected: for every model in `MODEL_REGISTRY` either:
- a line `R-Prec=... nDCG@10=...` appears (success), or
- a line `Failed <model_id>: <reason>` appears (logged via the existing try/except in `run_all.py:151-160`).

- [ ] **Step 4: Triage failures**

Run:
```bash
grep -E "Failed|Loading model" /tmp/smoke.log | head -100
```
Group failures into:
- (a) **OOM / hardware** → record in a tracked TODO; not a code bug.
- (b) **Loader / config** (missing `trust_remote_code`, wrong `instruction_format`, tokeniser error) → fix the registry entry now, before Task 5 commits.
- (c) **Encoding crash** → debug in a follow-up task; do not block Task 5 if isolated to one model.

- [ ] **Step 5: Update `MODEL_REGISTRY` for any (b) failures**

For each loader/config failure, patch `idiolink/models/registry.py` and re-run the single model:
```bash
python run_all.py --debug --models <fixed_model_id>
```

- [ ] **Step 6: Record coverage status in the plan**

Append a one-line "Coverage as of YYYY-MM-DD" entry at the bottom of this plan with the format:
```
- 2026-05-24: Loaded 21/24 (missing: Qwen3-Embedding-8B, Nemotron-8B, bge-multilingual-gemma2 — pending H100 access)
```

No code commit on this task; it is a checkpoint that gates Task 5. Save evidence by `mv /tmp/smoke.log logs/coverage-$(date +%Y%m%d).log` so reviewers can audit later (logs dir is gitignored).

---

## Task 4: Land the late-chunking dtype fix

**Files:**
- Modify: `idiolink/models/late_chunking.py:84-90` (two-line change)

- [ ] **Step 1: Confirm the diff is exactly the dtype cast change**

Run: `git diff idiolink/models/late_chunking.py`
Expected: only the `.float().cpu().numpy()` change plus its one-line comment.

- [ ] **Step 2: Add a focused regression test**

Edit `tests/test_late_chunking.py` (file already exists — append to it). If you cannot find the file, run `git ls-files tests/ | grep late_chunking` first. The test ensures bf16 token embeddings do not crash:

```python
def test_late_chunk_handles_bf16_token_embeddings(monkeypatch):
    """Regression: bf16 token outputs must be cast to fp32 before .numpy()."""
    import torch
    import numpy as np
    from idiolink.models import late_chunking as lc

    class _FakeModel:
        model_id = "fake"
        def _tokenize_and_encode(self, texts, device):
            # Two tokens, hidden_dim=4, dtype bf16
            tok = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
            spans = [(0, 2)]
            return tok, spans, [texts[0]]

    # Drive the function directly with a hand-rolled hook if the real code
    # exposes one, otherwise monkeypatch the loader. Adjust to the actual
    # API surface; this test should *fail* without the .float() cast.
    # (Implementer: replace this with an integration call against any tiny
    # ST model that runs in fp16, e.g. all-MiniLM-L6-v2 with .half().)
    ...
```

> ⚠ **Implementer note:** the public API of `late_chunk_encode` is `(model, texts, spans, device)` — it expects a `BaseEmbeddingModel`. Rather than monkeypatch, prefer a real model in fp16:
> ```python
> from sentence_transformers import SentenceTransformer
> from idiolink.models.sentence_transformer import SentenceTransformerModel
> m = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
> m.model = m.model.half()  # force fp16 outputs
> out = late_chunk_encode(m, ["the quick brown fox"], ["quick brown"], device="cpu")
> assert out.dtype == np.float32 and out.shape == (1, 384)
> ```
> Skip the test with `pytest.importorskip` if the model is not cached locally.

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_late_chunking.py -v -k bf16`
Expected: PASS after the dtype fix; would FAIL on `main` (numpy raises `TypeError: Got unsupported ScalarType BFloat16`).

- [ ] **Step 4: Commit**

```bash
git add idiolink/models/late_chunking.py tests/test_late_chunking.py
git commit -m "$(cat <<'EOF'
Fix late-chunking crash on bf16/fp16 token embeddings

`.cpu().numpy()` raises on bfloat16 because numpy has no native bf16 dtype.
Cast to fp32 before the numpy conversion. Adds a regression test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Land the per-model instruction resolver and new formats

**Depends on:** Task 1 (`subject` field is referenced by resolver templates), Task 2 (evaluator splits are needed by `tests/test_instructions.py` integration), Task 3 (smoke coverage validates that no-space format works on Qwen3/Lychee/Linq/Nemotron).

**Files:**
- Modify: `idiolink/models/instruction_model.py` (adds `resolve_instruction`, `resolve_instructions`, `E5_INLINE_NO_SPACE`, `PROMPT_PREFIX` branches)
- Modify: `idiolink/models/registry.py` (adds `instruction_text`/`instruction_fn` fields; updates 8 model entries)
- Modify: `idiolink/models/qwen.py` (instruction-format-aware spacing)
- Modify: `idiolink/trainer/contrastive_trainer.py` (uses resolver)
- Modify: `run_dense.py`, `run_instruction.py`, `run_all.py` (use resolver) — the runner pieces are otherwise unchanged in this task
- Modify: `tests/test_registry.py` (expands `VALID_INSTRUCTION_FORMATS`)
- Create: `tests/test_instructions.py` (resolver coverage)

- [ ] **Step 1: Cross-check the no-space format against official model cards**

For each of `{Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B, vec-ai/lychee-embed, Linq-AI-Research/Linq-Embed-Mistral, nvidia/llama-embed-nemotron-8b}`, open the HF model card "Usage" snippet and confirm whether the recommended template ends `Query:` (no space) or `Query: ` (with space). Record findings inline as comments above the registry entry, e.g.:

```python
"Qwen/Qwen3-Embedding-0.6B": ModelConfig(
    # HF card recommends: f"Instruct: {task}\nQuery:{query}" (no space) — sentence-transformers
    # prompt API strips trailing whitespace, so we must use _no_space.
    ...
    instruction_format="e5_inline_no_space",
    ...
),
```

If a model's card actually documents the *with-space* template, revert that entry to `e5_inline` and re-run its `run_all.py --debug --models <id>` row from Task 3.

- [ ] **Step 2: Confirm `tests/test_instructions.py` exercises all three resolution paths**

Read `tests/test_instructions.py`. It must cover (a) unknown model → default template, (b) static `instruction_text` formatted with query fields, (c) `instruction_fn` callable, (d) both-set rejection at `ModelConfig.__post_init__`, (e) bad placeholder error message. The current file (108 LOC) already covers all five — verify with:

Run: `pytest tests/test_instructions.py -v`
Expected: 8 tests pass.

- [ ] **Step 3: Verify the trainer integration**

Run: `pytest tests/ -q -k "instruction or registry"`
Expected: PASS. Then sanity-check that `contrastive_trainer.py` no longer references the legacy `DEFAULT_INSTRUCTION_TEMPLATE.format(span=s)` pattern:

Run: `grep -n DEFAULT_INSTRUCTION_TEMPLATE idiolink/trainer/contrastive_trainer.py`
Expected: no matches (the import is gone, replaced with `resolve_instructions`).

- [ ] **Step 4: Quick end-to-end smoke on the two models whose behaviour changed**

Run (BGE-base uses new `prompt_prefix` format):
```bash
python run_dense.py --model BAAI/bge-base-en-v1.5 --query_mode sentence --debug
```
Expected: completes; metrics.json written under `results/zero_shot/BAAI__bge-base-en-v1.5/sentence/`.

Run (Qwen3 uses new `e5_inline_no_space`):
```bash
python run_instruction.py --model Qwen/Qwen3-Embedding-0.6B --query_mode instruction_sentence --debug
```
Expected: completes; metrics.json appears.

If either reports an instruction-format error (`ValueError: 'e5_inline_no_space' is not a valid InstructionFormat`), the enum is missing the value — fix `idiolink/models/instruction_model.py:12-21` before committing.

- [ ] **Step 5: Commit in one shot (the resolver is one coherent feature)**

```bash
git add idiolink/models/instruction_model.py idiolink/models/registry.py \
        idiolink/models/qwen.py idiolink/trainer/contrastive_trainer.py \
        run_dense.py run_instruction.py run_all.py \
        tests/test_registry.py tests/test_instructions.py
git commit -m "$(cat <<'EOF'
Per-model instruction resolver + Qwen/BGE family-correct formats

- New resolver: ModelConfig.instruction_text (static template) or
  .instruction_fn (callable). Both formatted from IdiomQuery fields.
- New InstructionFormat values:
    * e5_inline_no_space  — Qwen3-*, Lychee, Linq-Mistral, Nemotron-8B
    * prompt_prefix        — BGE-base-en-v1.5 (canonical pretrained prefix)
- QwenModel honours the per-family spacing via _query_prompt().
- Trainer + all three runners (dense/instruction/all) call resolve_instructions
  instead of formatting DEFAULT_INSTRUCTION_TEMPLATE inline.

Tests: 8 new resolver tests + expanded VALID_INSTRUCTION_FORMATS set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Audit and prune the `bge_prompt` legacy code path

**Files:**
- Modify: `idiolink/models/instruction_model.py` (may delete `BGE_PROMPT` enum value and its two branches)

After Task 5 lands, BGE-base-en-v1.5 uses `prompt_prefix`, not `bge_prompt`. No model in the registry references `bge_prompt`. The enum and its two formatting branches at `instruction_model.py:15`, `:106`, and `:126` are dead code.

- [ ] **Step 1: Confirm no registry entry uses `bge_prompt`**

Run: `grep -n "bge_prompt" idiolink/models/registry.py`
Expected: no matches.

- [ ] **Step 2: Confirm no test references the enum value**

Run: `grep -rn 'BGE_PROMPT\|"bge_prompt"' tests/ idiolink/`
Expected: matches only inside `idiolink/models/instruction_model.py` (the enum + two branches) and inside `VALID_INSTRUCTION_FORMATS` in `tests/test_registry.py`.

- [ ] **Step 3: Decide and act**

If the user wants to keep BGE-style support reserved for future entries, leave the enum value and add a one-line comment above it: `# Reserved; not currently used — kept for future BGE-family entries that prefer this template.` If the user wants strict YAGNI, delete:

- `InstructionFormat.BGE_PROMPT = "bge_prompt"` (line 15)
- The `if fmt == InstructionFormat.BGE_PROMPT:` branches at lines 106 and 126
- `"bge_prompt"` from `VALID_INSTRUCTION_FORMATS` in `tests/test_registry.py`

Default in this plan: **delete** (YAGNI; one-liner to re-add if a future model needs it).

- [ ] **Step 4: Run the test suite**

Run: `pytest tests/ -q`
Expected: still 207+ passing.

- [ ] **Step 5: Commit**

```bash
git add idiolink/models/instruction_model.py tests/test_registry.py
git commit -m "$(cat <<'EOF'
Remove unused bge_prompt instruction format

BGE-base-en-v1.5 now uses prompt_prefix (its canonical pretrained
template) as of the previous commit; bge_prompt has no consumers in
the registry. Drop the dead enum value, the two formatting branches,
and the corresponding entry in VALID_INSTRUCTION_FORMATS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Land the ablation module and runner

**Depends on:** Task 2 (evaluator splits), Task 5 (resolver).

**Files:**
- Create: `idiolink/ablation.py` (already in working tree)
- Create: `run_ablation.py` (already in working tree)
- Create: `tests/test_ablation.py` (already in working tree)

- [ ] **Step 1: Confirm the resume logic does not silently drop new metrics**

Read `run_ablation.py:141-230`. Verify that `_metrics_path` checks existence per `(preset, model, mode)` and that the `--force` flag is respected throughout. Run on a tiny preset:

```bash
python run_ablation.py --debug --models BAAI/bge-m3 --presets lit_idiom --modes sentence
```
Expected: writes `results/ablation/lit_idiom/BAAI__bge-m3/sentence/metrics.json` and prints `R-Prec=... nDCG@10=...`.

Re-run the same command without `--force`:
Expected: prints `All (...) results exist on disk; skipping BAAI/bge-m3`.

Re-run with `--force`:
Expected: recomputes.

- [ ] **Step 2: Confirm CSV reaggregation walks the full disk state, not just this run**

```bash
ls results/ablation/lit_idiom/BAAI__bge-m3/  # should still contain {sentence,span,...}
cat results/ablation/full_results.csv | head -5
```
Expected: every metrics.json under `results/ablation/<slug>/<model>/<mode>/` appears as a row, regardless of which models were touched in this run.

- [ ] **Step 3: Run the ablation test suite**

Run: `pytest tests/test_ablation.py -v`
Expected: 9 tests pass (preset parsing, filter helper, gold-shrinking).

- [ ] **Step 4: Decide BM25 inclusion in `run_ablation.py`**

The runner currently runs BM25 first (`--no_bm25` to skip). BM25 has no GPU dependency and the tuning grid is small. Keep as-is unless the user objects.

- [ ] **Step 5: Commit module + tests + runner together**

```bash
git add idiolink/ablation.py run_ablation.py tests/test_ablation.py
git commit -m "$(cat <<'EOF'
Add index-composition ablation: two presets + resumable runner

- idiolink/ablation.py: ABLATION_PRESETS = {lit_sim_sense, lit_idiom},
  parse_index_filter() accepts preset names or CSV usage lists, and
  filter_docs_by_usage() filters parallel (sentences, metadata) lists.

- run_ablation.py: resumable runner that encodes each model's full doc
  set once, slices per preset, and skips (preset, mode) combos with an
  existing metrics.json (use --force to recompute). BM25 baseline runs
  first per preset (use --no_bm25 to skip). After every model the
  full_results.csv is rebuilt from disk, so failed mid-runs do not
  corrupt the aggregate.

Tests: 9 covering preset parsing, filter, and gold-set shrinkage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Land the analysis scripts and committed asset CSVs

**Depends on:** Task 2 (variant tables need `by_usage`/`by_subject`), Task 7 (ablation table needs `results/ablation/full_results.csv`).

**Files:**
- Create: `analysis/generate_variant_tables.py`
- Create: `analysis/generate_ablation_table.py`
- Create: `analysis/lexical_overlap.py`
- Add: `assets/variant_{sentence,span,instruction_sentence,instruction_span}.csv`
- Add: `assets/ablation/ablation_{sentence,span,instruction_sentence,instruction_span}.csv`, `ablation_results.csv`

- [ ] **Step 1: Regenerate variant tables and confirm idempotence**

Run:
```bash
python analysis/generate_variant_tables.py
git diff --stat assets/variant_*.csv
```
Expected: either no diff (working tree already matches) or only floating-point-stable changes; the CSV format must round-trip.

- [ ] **Step 2: Regenerate ablation tables**

```bash
python analysis/generate_ablation_table.py
git diff --stat assets/ablation/
```
Expected: stable CSVs.

- [ ] **Step 3: Smoke-run lexical overlap (read-only)**

```bash
python analysis/lexical_overlap.py --split test | head -30
```
Expected: prints two tables (strip_span=False and strip_span=True), no exceptions.

- [ ] **Step 4: Decide which CSVs to commit**

Default: commit the `assets/variant_*.csv` and `assets/ablation/*.csv` so reviewers can see the numbers without re-running the matrix. They are small (<10 KB each). Confirm by:

Run: `du -sh assets/variant_*.csv assets/ablation/*.csv`
Expected: total under 100 KB.

- [ ] **Step 5: Commit analysis scripts + CSVs**

```bash
git add analysis/generate_variant_tables.py analysis/generate_ablation_table.py \
        analysis/lexical_overlap.py \
        assets/variant_*.csv assets/ablation/
git commit -m "$(cat <<'EOF'
Add analysis scripts: per-variant + per-mode ablation tables, lexical overlap

- generate_variant_tables.py: one table per query mode with columns for
  overall / literal / idiomatic / by_subject (R-P and nDCG@10). Saves
  assets/variant_<mode>.csv and prints to stdout.

- generate_ablation_table.py: one table per mode with both ablation
  presets side by side. Saves assets/ablation/ablation_<mode>.csv.

- lexical_overlap.py: keyword-set Jaccard + query-recall between each
  query and its relevant docs, grouped by query usage and doc subtype.
  Pure analysis, no model dependency.

Commits the regenerated CSV artifacts so reviewers can see the numbers
without re-running the matrix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Land the runner CLI cleanup (`--index_filter` decisions)

**Files:**
- Modify: `run_dense.py` (keep `--index_filter`)
- Modify: `run_bm25.py` (keep `--index_filter`)
- Modify: `run_all.py` (REMOVE `--index_filter` — redundant with `run_ablation.py`)

The current diff adds `--index_filter` to all three runners. `run_all.py` is documented as the *zero-shot matrix* runner and `run_ablation.py` is the *ablation matrix* runner. Letting `run_all.py` also do ablations creates two-paths-to-the-same-output.

- [ ] **Step 1: Confirm `run_all.py --index_filter` writes to the *same* directory as `run_ablation.py`**

Read `run_all.py:155-159` and `run_ablation.py:217-220`. Both write to `results/ablation/<slug>/<model>/<mode>/metrics.json`. ✓ → confirmed redundant.

- [ ] **Step 2: Remove `--index_filter` from `run_all.py`**

Edit `run_all.py`:
- Remove the `parser.add_argument("--index_filter", ...)` block (lines 95-102 in the current diff).
- Remove the `index_slug` / `filter_docs_by_usage` block (lines 111-115 and 121-124).
- Remove the conditional `if index_slug: output_dir = ...` branch (lines 159-162).
- Remove the conditional `if index_slug:` row decoration (lines 174-175).
- Remove the conditional `if index_slug: csv_path = ...` branch (lines 184-192) and revert to the single fieldnames list.
- Remove `from idiolink.ablation import parse_index_filter, filter_docs_by_usage` from the imports.

- [ ] **Step 3: Confirm `run_all.py --help` no longer mentions index_filter**

Run: `python run_all.py --help | grep -i index`
Expected: no output.

- [ ] **Step 4: Confirm the zero-shot path still works end-to-end**

Run: `python run_all.py --debug --models sentence-transformers/all-MiniLM-L6-v2`
Expected: writes `results/zero_shot/sentence-transformers__all-MiniLM-L6-v2/sentence/metrics.json` (and the other three modes).

- [ ] **Step 5: Update the `run_all.py` docstring at the top of the file** to drop any mention of `--index_filter` and instead direct users to `run_ablation.py` for the ablation matrix.

- [ ] **Step 6: Commit**

```bash
git add run_all.py run_dense.py run_bm25.py run_instruction.py
git commit -m "$(cat <<'EOF'
Runner CLI cleanup: keep --index_filter on run_dense/run_bm25, drop from run_all

run_all.py is the zero-shot matrix runner; run_ablation.py is the ablation
matrix runner (resumable, BM25 included, full CSV rebuild on each pass).
Letting run_all.py also write under results/ablation/ created two paths to
the same artifacts — drop the redundancy.

Single-shot ablations on one (model, mode) remain available via
`run_dense.py --index_filter ...` for quick experiments.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Drop `run_queue.py`

**Files:**
- Delete: `run_queue.py`

`run_queue.py` is a hardcoded MPS-only sequential queue capped at ≤1.5B models. Its three responsibilities (resumability, per-model logging, table regen between runs) are already covered by `run_ablation.py` for the ablation matrix and by `run_all.py --models <id>` for the zero-shot matrix. Keeping it in the production tree promotes confusion about *which* runner is canonical.

- [ ] **Step 1: Confirm it is not imported anywhere**

Run: `grep -rn "from run_queue\|import run_queue" .`
Expected: no matches.

- [ ] **Step 2: Confirm its functionality is fully covered**

Resumability: `run_ablation.py:158-165` and the existing `run_all.py` per-model try/except already skip completed work via the on-disk `metrics.json` check. Table regen: `analysis/generate_variant_tables.py` is the supported entry point.

- [ ] **Step 3: Delete the file**

```bash
git rm run_queue.py
```

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
Remove one-off run_queue.py

Hardcoded ≤1.5B model subset and MPS-specific tuning; superseded by
run_all.py --models <id> (zero-shot matrix) and run_ablation.py
(ablation matrix). Both already handle resumability via per-(model, mode)
metrics.json existence checks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Full verification before merge

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: 200+ passing (the count goes up from 207 by the count of new tests added in Task 4; nothing should be removed).

- [ ] **Step 2: Confirm clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean`. Any leftover modified file indicates a missed task.

- [ ] **Step 3: Re-run one zero-shot and one ablation cell end-to-end**

```bash
python run_dense.py --model BAAI/bge-m3 --query_mode sentence --debug
python run_ablation.py --debug --models BAAI/bge-m3 --presets lit_idiom --modes sentence
```
Expected: both write metrics.json files containing both `r_precision`/`ndcg@10` and `by_usage`/`by_subject` keys.

- [ ] **Step 4: Regenerate analysis tables to confirm no schema drift**

```bash
python analysis/generate_variant_tables.py >/dev/null
python analysis/generate_ablation_table.py >/dev/null
git status
```
Expected: clean working tree (either no asset changes or only the cells you just re-ran).

- [ ] **Step 5: Print final commit log**

Run: `git log --oneline origin/main..HEAD`
Expected: ~7–8 small commits, one per task. Each commit message should be self-contained.

---

## Task 12: README and AGENTS.md updates

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the ablation pipeline to the README**

Add a new section to `README.md` after the "Full Experiment Grid" subsection. Use this exact text:

```markdown
### Index-Composition Ablation
```bash
python run_ablation.py                                # all <7B models × both presets × 4 modes
python run_ablation.py --debug                        # smoke test
python run_ablation.py --models BAAI/bge-m3
python run_ablation.py --presets lit_idiom
python run_ablation.py --no_bm25                      # skip BM25 baseline
```

Two presets: `lit_sim_sense` (drop idiomatic docs) and `lit_idiom`
(drop simplification + sense paraphrases). Results land under
`results/ablation/<preset>/<model>/<mode>/`. The aggregated CSV
`results/ablation/full_results.csv` is rebuilt from disk on every
pass, so partial runs are always consistent.
```

- [ ] **Step 2: Add the new analysis scripts to the README's reproduction section**

Find the `python analysis/generate_zero_shot_table.py` block and append:
```
python analysis/generate_variant_tables.py    # per-variant table with by_usage and by_subject splits
python analysis/generate_ablation_table.py    # per-mode ablation tables (requires run_ablation.py)
python analysis/lexical_overlap.py            # keyword overlap diagnostic
```

- [ ] **Step 3: Add a one-line note in the registry table about BGE-base's prompt**

In the `## Model Registry` table row for `BGE-base`, change the Instruction column from `bge_prompt` to `prompt_prefix`. Add a footnote under the table:
```
*BGE-base-en-v1.5 uses its canonical pretrained prefix
("Represent this sentence for searching relevant passages: ") via the
`prompt_prefix` instruction format.*
```

- [ ] **Step 4: Update the model registry table to reflect `e5_inline_no_space`**

In the same table, change the Instruction column to `e5_inline_no_space` for: Qwen3-Embed-0.6B, Lychee-embed, Qwen3-Embed-4B, Linq-Embed-Mistral, Qwen3-Embed-8B, Nemotron-8B (six rows).

- [ ] **Step 5: Update AGENTS.md**

Add to the skills table:
```
| `/run-ablation`           | Index-composition ablation runner          |
```
(Even if the skill doesn't exist yet — this is a forward-looking placeholder. Remove if no skill follow-up is planned.)

- [ ] **Step 6: Commit**

```bash
git add README.md AGENTS.md
git commit -m "$(cat <<'EOF'
Document ablation pipeline + updated instruction formats in README

- Add Index-Composition Ablation section with run_ablation.py usage.
- Note that BGE-base-en-v1.5 uses prompt_prefix (its canonical prefix)
  and the Qwen3 / Lychee / Linq / Nemotron family uses e5_inline_no_space.
- Add the new analysis scripts to the reproduction workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final merge to `main`

- [ ] **Step 1: Confirm linear history is clean**

Run: `git log --oneline origin/main..HEAD`
Expected: one commit per task, ordered Task 1 → Task 12.

- [ ] **Step 2: Ask the user for merge strategy**

Two options:
1. **Direct push to `main`** — appropriate if this branch is `main` and the user is the sole maintainer (current `git status` confirms `On branch main`).
2. **Feature branch + PR** — push to `feature/production-readiness-2026-05-24`, open a PR, run CI, merge with a merge commit (matches the repo's existing convention per `git log`).

The repository's prior convention (visible in `git log`: `Merge branch 'skills'`, `Merge branch 'tests'`, `Merge branch 'data-generation'` …) is **feature branches with merge commits**. Default in this plan: option 2.

- [ ] **Step 3: Push the branch and open the PR**

```bash
git checkout -b feature/production-readiness-2026-05-24
git push -u origin feature/production-readiness-2026-05-24
gh pr create --title "Production readiness: instruction resolver, evaluator splits, index ablation" --body "$(cat <<'EOF'
## Summary
- Per-model instruction resolver with `instruction_text` / `instruction_fn` overrides; new `e5_inline_no_space` and `prompt_prefix` formats; BGE-base and Qwen3 family now use the templates their cards document.
- Evaluator returns additive `by_usage` (literal / idiomatic) and `by_subject` splits; top-level keys unchanged.
- New `idiolink/ablation.py` + `run_ablation.py` for the two-preset index-composition study (lit_sim_sense, lit_idiom).
- Analysis: variant tables, ablation tables, lexical-overlap diagnostic.
- Late-chunking bf16/fp16 dtype crash fix.

## Coverage status (see Task 3 of the plan)
- 13/24 paper models have full zero-shot results on disk.
- Loaded-only smoke pass: see `logs/coverage-2026-05-24.log`.
- TODO: <fill in any 8B+ models pending GPU access>.

## Test plan
- [ ] `pytest tests/ -q` → all green
- [ ] `python run_dense.py --model BAAI/bge-m3 --query_mode sentence --debug` → metrics.json with `by_usage` and `by_subject`
- [ ] `python run_ablation.py --debug --models BAAI/bge-m3 --presets lit_idiom --modes sentence` → ablation metrics.json
- [ ] `python analysis/generate_variant_tables.py` → reads, prints, writes CSVs

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: After review, merge with `--merge` to preserve the per-task commits**

```bash
gh pr merge --merge
```

Do not squash — the per-task commits are the audit trail for the rollout order.

---

## Open Questions for the User (decide before Task 5)

1. **Coverage gate** (Task 3 Step 2): strict full-matrix, loaded-only, or loaded-≤7B + deferred?
2. **`BGE_PROMPT` enum** (Task 6 Step 3): delete (default) or keep reserved?
3. **`--index_filter` on `run_all.py`** (Task 9): remove (default) or keep both paths?
4. **`run_queue.py`** (Task 10): delete (default), or keep it under `scripts/` as an MPS-specific helper?
5. **Merge strategy** (Task 13): feature branch + PR (default, matches repo convention), or direct push to main?

---

## Coverage Log

(append entries here in Task 3 Step 6)

- 2026-05-24 (baseline): 13/24 zero-shot complete (4 modes each). Missing: Alibaba-NLP/gte-Qwen2-1.5B-instruct, Alibaba-NLP/gte-Qwen2-7B-instruct, BAAI/bge-multilingual-gemma2, GritLM/GritLM-7B, Linq-AI-Research/Linq-Embed-Mistral, NovaSearch/stella-en-1.5B-v5, Qwen/Qwen3-Embedding-4B (ablation only), Qwen/Qwen3-Embedding-8B, Salesforce/SFR-Embedding-Mistral, intfloat/e5-mistral-7b-instruct, nvidia/llama-embed-nemotron-8b. Eight 7B–9B models gated on GPU access.
