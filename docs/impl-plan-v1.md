# IdioLink Repository Implementation Plan (v1.2)

## Context

The paper "IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions" (arXiv 2605.22247) introduces a retrieval benchmark (10,700 docs, 2,140 queries, 107 idioms) evaluating whether embedding models can bridge idiomatic and literal expressions. The working repo (`work_IdioLink`) has functional code but is messy. The goal: restructure into a clean repo that makes the published benchmark easy to run first, with data generation as an optional reproducibility appendix.

---

## Paper Experiment Matrix

Four zero-shot query configurations. Documents are **always indexed as full sentences**; only query encoding changes:

| Config | Query Encoding | Document Encoding |
|--------|---------------|-------------------|
| **Sentence** | Full sentence | Full sentence |
| **Span** | Idiom span only (late chunking) | Full sentence |
| **Instruction + Sentence** | Instruction prefix + full sentence | Full sentence |
| **Instruction + Span** | Instruction prefix + span (late chunking) | Full sentence |

Plus:
- **BM25** baseline (lexical, no embeddings)
- **Contrastive fine-tuning** (subset of models, 3 seeds)

---

## Model Registry (24 paper models)

All models evaluated in the paper, ordered by parameter count:

| # | Model | HF ID | Size |
|---|-------|--------|------|
| 1 | SBERT | `sentence-transformers/all-MiniLM-L6-v2` | 110M |
| 2 | Contriever | `facebook/contriever` | 110M |
| 3 | E5-base-v2 | `intfloat/e5-base-v2` | 110M |
| 4 | TART | `orionweller/tart-dual-contriever-msmarco` | 110M |
| 5 | BGE-base | `BAAI/bge-base-en-v1.5` | 326M |
| 6 | Instructor-base | `hkunlp/instructor-base` | 335M |
| 7 | Nomic-v2 | `nomic-ai/nomic-embed-text-v2-moe` | 475M |
| 8 | Multilingual-E5-large | `intfloat/multilingual-e5-large-instruct` | 560M |
| 9 | BGE-M3 | `BAAI/bge-m3` | 568M |
| 10 | Qwen3-Embed-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 600M |
| 11 | DRAMA-1B | `facebook/drama-1b` | 1B |
| 12 | Stella-1.5B | `NovaSearch/stella-en-1.5B-v5` | 1.5B |
| 13 | Instructor-xl | `hkunlp/instructor-xl` | 1.5B |
| 14 | Lychee-embed | `vec-ai/lychee-embed` | 1.5B |
| 15 | GTE-Qwen2-1.5B | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1.5B |
| 16 | Qwen3-Embed-4B | `Qwen/Qwen3-Embedding-4B` | 4B |
| 17 | Linq-Embed-Mistral | `Linq-AI-Research/Linq-Embed-Mistral` | 7B |
| 18 | SFR-Embedding-Mistral | `Salesforce/SFR-Embedding-Mistral` | 7B |
| 19 | E5-Mistral-7B | `intfloat/e5-mistral-7b-instruct` | 7B |
| 20 | GritLM-7B | `GritLM/GritLM-7B` | 7B |
| 21 | GTE-Qwen2-7B | `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7B |
| 22 | Qwen3-Embed-8B | `Qwen/Qwen3-Embedding-8B` | 8B |
| 23 | Nemotron-8B | `nvidia/llama-embed-nemotron-8b` | 8B |
| 24 | BGE-Gemma2 | `BAAI/bge-multilingual-gemma2` | 9B |

### Registry Config Fields (per model)

Each entry in `idiolink/models/registry.py` specifies:

```python
@dataclass
class ModelConfig:
    model_id: str                    # HuggingFace model ID
    model_class: str                 # "sentence_transformer" | "instruction" | "gritlm" | "qwen"
    size_params: int                 # Parameter count
    max_length: int                  # Max token length (512, 4096, 8192, etc.)
    instruction_format: str | None   # "e5_inline" | "bge_prompt" | "instructor_pairs" | "tart_sep" | None
    query_prefix: str | None         # e.g. "query: " for E5, "Represent this sentence: " for Instructor
    passage_prefix: str | None       # e.g. "passage: " for E5, None for most
    trust_remote_code: bool          # Required for Qwen, Nomic, etc.
    batch_size: int                  # Default batch size (smaller for 7B+ models)
    dtype: str                       # "float32" | "float16" | "bfloat16"
    supports_span_pooling: bool      # Whether late-chunking span extraction works with this architecture
```

---

## Evaluation Contract

**Primary metrics:** R-Precision and nDCG@10

**Relevance rules (per query):**
- **Literal query** → relevant docs = all *literal* docs for the same PIE (idiom)
- **Idiomatic query** → relevant docs = all *idiomatic*, *simplification*, and *sense* docs for the same PIE

**Parameters:** `top_k=100` for retrieval; metrics computed at k=10.

---

## Fine-Tuning Specification

**Models to fine-tune:** SBERT, DRAMA-1B, E5-base-v2, BGE-M3, Qwen3-Embedding-0.6B

**Training modes (each model trained separately per mode):**
- sentence (query=full sentence, positive=full sentence)
- span (query=span, positive=full sentence)
- instruction+sentence (query=instruction+sentence, positive=full sentence)
- instruction+span (query=instruction+span, positive=full sentence)

**Negative mining:**
- **Hard negatives:** same PIE, opposite usage type (e.g., literal query paired with idiomatic doc of same idiom)
- **Soft negatives:** different PIEs (in-batch negatives from other idioms)

**Protocol:**
- 3 seeds (42, 43, 44)
- InfoNCE loss, temperature=0.05
- AdamW, lr=2e-5, linear warmup (100 steps)
- Early stopping (patience=3, monitor val nDCG@10)
- Batch size: 32 (gradient accumulation as needed for larger models)

**Verification:** Each fine-tuned model must produce nDCG@10 on the test set; results aggregated across seeds (mean ± std).

---

## Target Directory Structure

```
IdioLink/
├── README.md
├── LICENSE
├── .gitignore
├── config.yaml                          # Central experiment config
├── keys.yaml.example                    # API keys (Gemini — for data gen only)
├── requirements.txt
│
├── run_baseline.py                      # Sentence & Span configs
├── run_instruction.py                   # Instruction+Sentence & Instruction+Span
├── run_bm25.py                          # BM25 lexical baseline
├── run_fine_tune.py                     # Contrastive fine-tuning
├── run_all.py                           # Full experiment matrix orchestrator
│
├── assets/                              # Paper figures
│   └── .gitkeep
│
├── data/                                # Benchmark splits (committed)
│   ├── README.md                        # Schema, stats, relevance rules
│   ├── train/
│   │   ├── indexes.json
│   │   ├── queries.json
│   │   └── triplets.jsonl
│   ├── val/
│   │   ├── indexes.json
│   │   ├── queries.json
│   │   └── triplets.jsonl
│   └── test/
│       ├── indexes.json
│       └── queries.json
│
├── idiolink/                            # Core library
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py                  # MODEL_REGISTRY: name → config
│   │   ├── base.py                      # BaseEmbeddingModel ABC
│   │   ├── sentence_transformer.py      # SentenceTransformer wrapper
│   │   ├── instruction_model.py         # Instruction-aware (E5, BGE, Instructor, TART)
│   │   ├── late_chunking.py             # Late chunking embedding + utils
│   │   ├── gritlm.py                    # GritLM-7B
│   │   └── qwen.py                      # Qwen3-Embedding family
│   ├── retriever.py                     # Dense retriever (index + retrieve)
│   ├── evaluator.py                     # R-Precision, nDCG@10, relevance rules
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── contrastive_trainer.py       # InfoNCE training loop
│   │   ├── datasets.py                  # Triplet dataset loader
│   │   └── losses.py                    # InfoNCE loss
│   └── utils.py                         # File I/O, config loading, device detection
│
├── analysis/                            # Paper table/figure reproduction
│   ├── README.md
│   ├── generate_zero_shot_table.py      # Paper Table 1: 24 models × 4 configs
│   ├── generate_finetuning_table.py     # Paper Table 2: fine-tuning results
│   ├── generate_dataset_stats.py        # Dataset statistics table
│   └── plot_performance.py              # Paper figures
│
├── data_generation/                     # Dataset reproduction (optional appendix)
│   ├── README.md
│   ├── magpie/
│   │   ├── filter_magpie.py
│   │   └── MAGPIE_SOURCE_IDIOMS.csv
│   ├── generation/
│   │   └── data_pipeline.py
│   ├── annotation/
│   │   ├── annotation_pipeline.py
│   │   ├── annotation_prompts.py
│   │   └── split_data.py
│   └── utils/
│       ├── analyze_errors.py
│       └── combine_variants.py
│
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py
│   ├── test_retriever.py
│   └── test_registry.py
│
└── results/                             # Gitignored
    └── README.md                        # Explains output structure
```

---

## config.yaml Design

```yaml
seed: 42
device: auto  # auto, cuda, mps, cpu
debug: false
debug_samples: 5  # queries used in debug/smoke-test mode

data:
  test_dir: data/test
  train_dir: data/train
  val_dir: data/val

retrieval:
  top_k: 100

evaluation:
  metrics: [r_precision, ndcg@10]
  # Relevance: literal query → literal docs (same PIE)
  #            idiomatic query → idiomatic + simplification + sense docs (same PIE)

# Default model (override with --model)
model: BAAI/bge-m3

# Experiment config (override with --query_mode)
experiment:
  query_mode: sentence        # sentence, span, instruction_sentence, instruction_span
  instruction_template: null  # auto-selected from registry when needed

# BM25 (used by run_bm25.py)
bm25:
  k1: 0.9
  b: 0.4
  query_mode: sentence  # sentence, span
  tune_on_val: true
  k1_grid: [0.6, 0.9, 1.2, 1.5, 2.0]
  b_grid: [0.2, 0.4, 0.6, 0.75, 0.9]
  tune_metric: ndcg@10

# Fine-tuning (used by run_fine_tune.py)
training:
  models:
    - sentence-transformers/all-MiniLM-L6-v2
    - facebook/drama-1b
    - intfloat/e5-base-v2
    - BAAI/bge-m3
    - Qwen/Qwen3-Embedding-0.6B
  modes: [sentence, span, instruction_sentence, instruction_span]
  seeds: [42, 43, 44]
  batch_size: 32
  max_epochs: 10
  learning_rate: 2.0e-5
  warmup_steps: 100
  temperature: 0.05
  early_stopping_patience: 3
  early_stopping_metric: ndcg@10

results_dir: results
```

---

## Branch Strategy & Implementation Phases

### Branch Dependency Graph

```
main
 └── foundation ─────────────────────────────┐
      ├── dense-baseline ──────────────┐     │
      │    ├── experiment-matrix ──┐   │     │
      │    │    └── model-registry │   │     │
      │    │                       │   │     │
      │    └── fine-tuning ────────┘   │     │
      │                                │     │
      ├── bm25 ────────────────────────┘     │
      ├── data-generation ───────────────────┘
      └── analysis ──────────────────────────┘
```

### Sequential Chain (merge in order)

| Branch | Depends On | Phase |
|--------|-----------|-------|
| `foundation` | `main` | 1 |
| `dense-baseline` | `foundation` | 2 |
| `experiment-matrix` | `dense-baseline` | 3 |
| `model-registry` | `experiment-matrix` | 4 |

### Parallel Branches (develop independently after their dependency)

| Branch | Depends On | Phase |
|--------|-----------|-------|
| `bm25` | `foundation` (only needs evaluator + data) | 5 |
| `fine-tuning` | `dense-baseline` (needs model ABC + retriever) | 6 |
| `analysis` | `foundation` (reads results JSON; can stub on synthetic data) | 7 |
| `data-generation` | `foundation` (zero code dependency; just needs repo skeleton) | 8 |

---

## Phase 1: Foundation — branch `foundation` (from `main`)

**Scope:** Repo skeleton, committed data, evaluator, config, utilities.

- Create directory structure, `.gitignore`, `requirements.txt`, `keys.yaml.example`
- Create `config.yaml` (full config, all sections)
- Commit `data/{train,val,test}/` splits from work repo
- Write `data/README.md` documenting schema and relevance rules
- Implement `idiolink/__init__.py`, `idiolink/utils.py` (config loading, file I/O, device detection)
- Implement `idiolink/evaluator.py`:
  - R-Precision computation
  - nDCG@10 computation
  - Relevance rule logic (literal→literal, idiomatic→{idiomatic,simplification,sense})
  - top_k=100 retrieval depth
- Write `tests/test_evaluator.py`
- Create `results/README.md`, `assets/.gitkeep`

**Merge criteria:** `python -m pytest tests/test_evaluator.py` passes.

---

## Phase 2: Dense Baseline — branch `dense-baseline` (from `foundation`)

**Scope:** One working end-to-end retrieval pipeline with sentence mode.

- Implement `idiolink/models/base.py` (ABC: `encode(texts) → np.ndarray`)
- Implement `idiolink/models/sentence_transformer.py` (wraps sentence-transformers)
- Implement `idiolink/retriever.py` (index + cosine similarity search)
- Create `run_baseline.py` — loads config, encodes queries/docs, retrieves, evaluates, writes metrics
- Write `tests/test_retriever.py`

**Merge criteria:** `python run_baseline.py --model intfloat/e5-base-v2` outputs `metrics.json` with R-Precision and nDCG@10.

---

## Phase 3: Experiment Matrix — branch `experiment-matrix` (from `dense-baseline`)

**Scope:** All 4 zero-shot query configurations working.

- Implement `idiolink/models/instruction_model.py`:
  - E5 inline (`"query: "` prefix)
  - BGE prompt kwarg
  - Instructor pairs `[instruction, text]`
  - TART `[SEP]` format
- Implement `idiolink/models/late_chunking.py`:
  - Full-context encoding → span token extraction → pooling
  - Merge strategies: max, mean
- Extend `run_baseline.py` to accept `--query_mode span`
- Create `run_instruction.py` supporting `instruction_sentence` and `instruction_span`

**Merge criteria:** All 4 configs produce distinct results for BGE-M3; `run_instruction.py --model BAAI/bge-m3 --query_mode instruction_span` outputs metrics.

---

## Phase 4: Model Registry — branch `model-registry` (from `experiment-matrix`)

**Scope:** Full 24-model support, orchestrator, smoke test.

- Implement `idiolink/models/registry.py` — `ModelConfig` dataclass for all 24 models
- Implement `idiolink/models/gritlm.py` (GritLM-specific encoding)
- Implement `idiolink/models/qwen.py` (Qwen3-Embedding family)
- Create `run_all.py`:
  - Iterates registry × 4 configs
  - `--debug` flag runs on debug_samples subset (smoke test)
  - `--models` flag to select subset
  - Writes per-model results + aggregated `full_results.csv`
- Write `tests/test_registry.py` (instantiation, embedding dims, instruction application)

**Merge criteria:** `python run_all.py --debug --debug_samples 5` passes 96/96 model×mode combinations.

---

## Phase 5: BM25 — branch `bm25` (from `foundation`)

**Scope:** Lexical baseline with validation-tuned hyperparameters. Independent of dense retrieval.

- Implement `run_bm25.py`:
  - rank-bm25 library
  - Query modes: sentence, span
  - Validation grid search: k1=[0.6, 0.9, 1.2, 1.5, 2.0], b=[0.2, 0.4, 0.6, 0.75, 0.9]
  - Selects best (k1, b) by validation nDCG@10
  - Reports test metrics with validation-selected parameters
  - Saves `results/bm25/tuning_results.json`

**Merge criteria:** `python run_bm25.py` produces metrics for both query modes; reports selected (k1, b).

---

## Phase 6: Fine-Tuning — branch `fine-tuning` (from `dense-baseline`)

**Scope:** Contrastive training for 5 models × 4 modes × 3 seeds.

- Implement `idiolink/trainer/losses.py` (InfoNCE with temperature)
- Implement `idiolink/trainer/datasets.py` (triplet loader, hard/soft negative mining)
- Implement `idiolink/trainer/contrastive_trainer.py` (training loop, early stopping, checkpointing)
- Create `run_fine_tune.py`:
  - Hard negatives: same PIE, opposite usage type
  - Soft negatives: other PIEs (in-batch)
  - InfoNCE, temp=0.05, patience=3 on val nDCG@10
  - After training, evaluates on test set
  - Outputs: `results/fine_tuning/{model}/{mode}/seed_{N}/metrics.json`

**Merge criteria:** `python run_fine_tune.py --model intfloat/e5-base-v2 --mode sentence --seeds 42` completes training loop + test evaluation, produces nDCG@10 and R-Precision.

---

## Phase 7: Analysis — branch `analysis` (from `foundation`)

**Scope:** Paper table/figure reproduction scripts. Can develop with synthetic/stubbed results.

- `analysis/generate_zero_shot_table.py` — reads `results/zero_shot/`, produces paper Table 1 (24 models × 4 configs)
- `analysis/generate_finetuning_table.py` — reads `results/fine_tuning/`, produces paper Table 2 (5 models × 4 modes × 3 seeds, mean±std)
- `analysis/generate_dataset_stats.py` — reads `data/`, produces dataset statistics (PIE counts, split sizes, usage distributions)
- `analysis/plot_performance.py` — paper figures
- Write top-level `README.md` (ID10M-JAM style: paper link, dataset table, quick start, experiments, reproduction, citation)
- Write `analysis/README.md`

**Merge criteria:** `python analysis/generate_zero_shot_table.py` produces CSV matching paper Table 1 format (can use stubbed metrics for structure validation).

---

## Phase 8: Data Generation — branch `data-generation` (from `foundation`)

**Scope:** Optional reproducibility appendix. Zero dependency on experiment code.

- Port `data_generation/` pipeline code:
  - `magpie/filter_magpie.py` — MAGPIE corpus filtering
  - `generation/data_pipeline.py` — LLM variant generation
  - `annotation/annotation_pipeline.py` + `annotation_prompts.py` — Gemini validation
  - `annotation/split_data.py` — stratified splitting
  - `utils/` — error analysis, combining utilities
- Write `data_generation/README.md` documenting full reproduction:
  - **MAGPIE filtering**: min_occurrences=30, ambiguity_range=25-75%, confidence≥1.0
  - **10 subject domains**: Politics, Sport, Business, Technology, Science, Education, Health, Entertainment, Environment, Daily Life
  - **Generation**: Gemini models for index/query sentence generation (4 variant types: literal, idiomatic, simplification, sense)
  - **Validation**: GPT-4o-mini ×3 (majority vote) for sentence/span validity
  - **PIE split**: 22 train / 10 val / 75 test idioms
  - **Test split quality**: gold (human-verified) vs silver (LLM-validated only)

**Merge criteria:** `data_generation/README.md` documents every step; scripts run without import errors (may require API keys for actual execution).

---

## What to Commit vs Gitignore

**Commit:**
- `data/` benchmark splits (indexes.json, queries.json, triplets — ~2-4 MB)
- `data_generation/` code + `MAGPIE_SOURCE_IDIOMS.csv` (pipeline, not intermediate outputs)
- `idiolink/` source, `analysis/`, `tests/`
- `assets/` paper figures
- Config files, READMEs

**Gitignore:**
- `results/` (regenerated by experiments)
- `models/` (fine-tuned checkpoints)
- `keys.yaml`, `.env`
- `__pycache__/`, `*.pyc`
- `wandb/`
- Large MAGPIE corpus files (instructions in README)

---

## Results Structure

```
results/
├── README.md
├── zero_shot/
│   └── {model_slug}/
│       ├── sentence/metrics.json
│       ├── span/metrics.json
│       ├── instruction_sentence/metrics.json
│       └── instruction_span/metrics.json
├── bm25/
│   ├── sentence/metrics.json
│   ├── span/metrics.json
│   └── tuning_results.json          # Validation grid search results
├── fine_tuning/
│   └── {model_slug}/
│       └── {mode}/
│           └── seed_{N}/
│               ├── training_history.json
│               ├── best_model/      (gitignored)
│               └── metrics.json     (test evaluation)
└── full_results.csv                 # Aggregated: model × config → R-Prec, nDCG@10
```

---

## Key Source Files to Port From

| Target | Source in work_IdioLink | Notes |
|--------|------------------------|-------|
| `idiolink/evaluator.py` | `src/evaluator/idiom_evaluator.py` + `metrics.py` | Simplify to R-Prec + nDCG@10 |
| `idiolink/models/sentence_transformer.py` | `src/embedding_models/baselines/sentence_transformer_embedding.py` | Keep clean |
| `idiolink/models/instruction_model.py` | `src/embedding_models/baselines/instruction_aware_sentence_transformer.py` + `src/query_processing/encode_request.py` | Merge instruction logic |
| `idiolink/models/late_chunking.py` | `src/embedding_models/late_chunking_embedding.py` + `late_chunking_utils.py` | Span-only mode |
| `idiolink/models/gritlm.py` | `src/embedding_models/gritlm_embedding.py` | As-is |
| `idiolink/models/qwen.py` | `src/embedding_models/qwen_alibaba_embedding.py` | As-is |
| `idiolink/retriever.py` | `src/embedding_retriever.py` | Strip late_chunking_retriever; LC is a model concern |
| `idiolink/trainer/` | `src/trainer/` | Keep: contrastive_trainer, datasets, losses |
| `run_baseline.py` | `scripts/base_exp.py` + `scripts/run_late_chunking.py` | Unified: sentence/span via flag |
| `run_instruction.py` | `scripts/run_instruction_exp_v2.py` + `scripts/run_instruction_late_chunking_exp.py` | Unified |
| `run_bm25.py` | `scripts/run_bm25_exp.py` | Minimal |
| `run_fine_tune.py` | `scripts/contrastive_training/train.py` + `prepare_data.py` | Paper protocol |
| `data_generation/` | `data_generation/` | Clean port, no intermediate files |

---

## Do NOT Port

- `src/embedding_models/baselines/openai_embedding.py` (not in paper experiments)
- `src/embedding_models/baselines/gecko_dual_encoder.py` (draft, not in paper)
- `src/embedding_models/baselines/dual_encoder_core.py` (gecko infrastructure)
- Debug scripts (`debug_*.py`, `test_fix.py`, `test_instructor_*.py`)
- Fix/summary markdown files
- Shell scripts (`dr-bash-s.sh`, `qwen-bash-s.sh`, `sapn1.sh`, etc.)
- Log files (`*.log`, `nohup.out`)
- `results.zip`, `train_infer_results.json`
- `run_train_and_infer.py` / `run_train_and_infer_old.py` (replaced by `run_fine_tune.py`)
- `scripts/run_instruction_exp.py` (v1, superseded by v2)
- `scripts/run_instructor_late_chunking_v2.py` (merged into instruction runner)
- `examples/` directory (training examples folded into `run_fine_tune.py` docs)

---

## Compatibility Smoke Test

Before running full experiments, run every model × every zero-shot mode on a tiny debug split (5 queries, 25 docs) to catch loading failures, OOM, tokenization issues, or instruction-format mismatches early.

```bash
python run_all.py --debug --debug_samples 5
```

This iterates all 24 models × 4 configs = 96 combinations on the debug subset. Any failure is logged with model ID + mode + traceback. Must pass 96/96 before proceeding to full runs.

Tests in `tests/test_registry.py` also verify:
- Every registry entry can instantiate its model class
- `encode()` returns correct embedding dimensions
- Instruction prefixes are applied for instruction modes
- Span pooling produces different embeddings than full-sentence for models with `supports_span_pooling=True`

---

## Verification Checklist

| Branch | Gate Command | Pass Criteria |
|--------|-------------|---------------|
| `foundation` | `python -m pytest tests/test_evaluator.py` | All tests pass; R-Prec and nDCG@10 correct on sample data |
| `dense-baseline` | `python run_baseline.py --model intfloat/e5-base-v2` | Outputs `metrics.json` with R-Precision and nDCG@10 |
| `experiment-matrix` | `python run_instruction.py --model BAAI/bge-m3 --query_mode instruction_span` | Outputs metrics; all 4 modes produce distinct scores |
| `model-registry` | `python run_all.py --debug --debug_samples 5` | 96/96 model×mode combos pass (smoke test) |
| `bm25` | `python run_bm25.py` | Metrics for sentence+span modes; reports tuned (k1, b) |
| `fine-tuning` | `python run_fine_tune.py --model intfloat/e5-base-v2 --mode sentence --seeds 42` | Training completes; outputs test nDCG@10 + R-Precision |
| `analysis` | `python analysis/generate_zero_shot_table.py` | Produces CSV matching paper Table 1 structure |
| `data-generation` | `python -c "import data_generation"` + README review | No import errors; README covers full pipeline |
