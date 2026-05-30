# IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions

**arXiv 2025** · [[Paper]](https://arxiv.org/abs/2605.22247) · [[Dataset (HuggingFace)](https://huggingface.co/datasets/Intellexus/IdioLink)]

A retrieval benchmark evaluating whether embedding models can bridge idiomatic and literal expressions — matching queries to documents that share the same conceptual meaning regardless of figurative vs. literal usage.

**107 idioms | 10,700 documents | 2,140 queries | 24 models | 4 query configurations**

---

## 🚀 Quick Start

```bash
git clone <repo-url>
cd IdioLink
pip install -r requirements.txt
python download_data.py
python run_dense.py --model sentence-transformers/all-MiniLM-L6-v2 --query_mode sentence
```

Expected output: R-Precision and nDCG@10 on the test set (1,500 queries, 7,500 documents).

> **Hardware:** Models with 7B+ parameters require a GPU with ≥24GB VRAM.

---

## 🛠️ Skills (Slash Commands)

This repo ships with Claude Code skills for streamlined experiment workflows. Type `/` in Claude Code to see them.

| Command | What it does | Example |
|---------|-------------|---------|
| `/onboard` | Install deps, verify env, explain project | `/onboard` |
| `/run-experiment` | Run one model × one mode | `/run-experiment bge-m3 sentence` |
| `/run-all` | Run full experiment grid | `/run-all --debug` |
| `/fine-tune` | Contrastive fine-tuning | `/fine-tune e5-base-v2 sentence` |
| `/evaluate` | View/regenerate metrics | `/evaluate bge-m3 span` |
| `/reproduce-paper` | Full paper reproduction | `/reproduce-paper` |
| `/debug-model` | Diagnose model issues | `/debug-model GritLM/GritLM-7B` |

Skills are defined in `.claude/skills/` and work for any collaborator who clones the repo.

---

## 📋 Benchmark Overview

**Research question:** Can embedding models retrieve documents that share conceptual meaning with a query containing a potentially idiomatic expression (PIE), regardless of literal vs. figurative usage?

### Relevance Rules

| Query type | Relevant documents |
|-----------|-------------------|
| **Literal** query | All **literal** docs for the same PIE |
| **Idiomatic** query | All **idiomatic** + **simplification** + **sense** docs for the same PIE |

### Metrics

- **R-Precision** — Precision at R, where R = number of relevant documents for the query
- **nDCG@10** — Normalized Discounted Cumulative Gain at rank 10

Each `metrics.json` also carries two diagnostic breakdowns:

- **`by_usage`** *(recommended diagnostic)* — the headline metrics restricted to literal-only and idiomatic-only queries. This is the primary breakdown for understanding model behavior across PIE usage types.
- **`by_subject`** *(supplementary sanity check)* — a topical-coherence signal using shared `subject` as binary relevance. Intentionally weaker than the idiom-relevance gold — it ignores idiom identity and usage type. Use for diagnostic comparison only, not as a headline metric.

### Document Usage Types

| Type | Description | Share |
|------|-------------|-------|
| literal | PIE used with its word-by-word meaning | 40% |
| idiomatic | PIE used figuratively | 20% |
| simplification | Paraphrase of the idiomatic meaning | 20% |
| sense | Dictionary-sense rephrasing | 20% |

---

## 📁 Dataset

| Split | PIEs | Documents | Queries |
|-------|------|-----------|---------|
| Train | 22 | 2,200 | 440 |
| Val | 10 | 1,000 | 200 |
| Test | 75 | 7,500 | 1,500 |
| **Total** | **107** | **10,700** | **2,140** |

- 10 subject domains (Politics, Sport, Business, Technology, Science, Education, Health, Entertainment, Environment, Daily Life)
- Quality tiers: gold (human-verified) and silver (LLM-validated)
- Dataset on HuggingFace: [Intellexus/IdioLink](https://huggingface.co/datasets/Intellexus/IdioLink)
- See [`data/README.md`](data/README.md) for full schema documentation

---

## ⚙️ Experiment Configurations

Four zero-shot query configurations. Documents are **always indexed as full sentences**; only query encoding changes.

| Config | Query Encoding | Runner |
|--------|---------------|--------|
| **sentence** | Full sentence | `run_dense.py` |
| **span** | Late-chunking on idiom span | `run_dense.py` |
| **instruction_sentence** | Instruction prefix + full sentence | `run_instruction.py` |
| **instruction_span** | Instruction prefix + late-chunking span | `run_instruction.py` |

**Instruction template:**
> "Based on the literal/idiomatic usage of the span '{span}' in the query, retrieve documents that contain a span conveying the same conceptual meaning."

Plus:
- **BM25** lexical baseline (`run_bm25.py`)
- **Contrastive fine-tuning** for the reported training models via `run_fine_tune.py`

---

## 🏃 Running Experiments

### BM25 Baseline
```bash
python run_bm25.py --query_mode sentence          # default params
python run_bm25.py --query_mode sentence --tune    # grid search on val
python run_bm25.py --query_mode span --tune
```

### Dense Retrieval (sentence / span)
```bash
python run_dense.py --model BAAI/bge-m3 --query_mode sentence
python run_dense.py --model BAAI/bge-m3 --query_mode span
```

### Instruction-Based Retrieval
```bash
python run_instruction.py --model intfloat/multilingual-e5-large-instruct --query_mode instruction_sentence
python run_instruction.py --model intfloat/multilingual-e5-large-instruct --query_mode instruction_span
```

### Full Experiment Grid
```bash
python run_all.py                          # all 24 models × 4 modes
python run_all.py --debug                  # smoke test (5 queries)
python run_all.py --models BAAI/bge-m3 intfloat/e5-base-v2  # subset
```

### Index-Composition Ablation

`run_ablation.py` runs the two-preset ablation (`lit_sim_sense` and `lit_idiom`) across the model matrix. It is resumable: per-(preset, model, mode) `metrics.json` files are written atomically, and re-running skips combos that already exist on disk. After every model, the aggregated CSV is rebuilt from disk so partial runs are always consistent.

```bash
python run_ablation.py                                # all <7B models × both presets × 4 modes
python run_ablation.py --debug                        # smoke test (5 queries)
python run_ablation.py --models BAAI/bge-m3
python run_ablation.py --presets lit_idiom
python run_ablation.py --no_bm25                      # skip BM25 baseline
python run_ablation.py --force                        # recompute even if metrics exist
```

Two presets:
- `lit_sim_sense` — drop idiomatic docs (keep literal + simplification + sense)
- `lit_idiom` — drop simplification + sense paraphrases (keep literal + idiomatic)

Results land under `results/ablation/<preset>/{<model_slug>|bm25}/<mode>/metrics.json` and the rebuilt aggregate is `results/ablation/full_results.csv`. For one-off ad-hoc ablations on a single (model, mode), `run_dense.py` and `run_bm25.py` accept the same `--index_filter <preset|csv-list>` flag and write to the same directory layout.

### Output Structure
```
results/
├── zero_shot/{model_slug}/{mode}/metrics.json
├── bm25/{mode}/metrics.json
├── ablation/{preset}/{model_slug}/{mode}/metrics.json
├── ablation/full_results.csv
├── fine_tuning/{model_slug}/{mode}/seed_{N}/metrics.json
└── full_results.csv
```

All CLI arguments override values from `config.yaml`.

---

## 📊 Main Results

*Results will be updated upon paper publication. Run `python run_all.py` to reproduce.*

| Model | Size | Sentence | Span | Instr+Sent | Instr+Span |
|-------|------|----------|------|-----------|-----------|
| SBERT | 110M | -- | -- | -- | -- |
| Contriever | 110M | -- | -- | -- | -- |
| E5-base-v2 | 110M | -- | -- | -- | -- |
| TART | 110M | -- | -- | -- | -- |
| BGE-base | 326M | -- | -- | -- | -- |
| Instructor-base | 335M | -- | -- | -- | -- |
| Nomic-v2 | 475M | -- | -- | -- | -- |
| Multilingual-E5-large | 560M | -- | -- | -- | -- |
| BGE-M3 | 568M | -- | -- | -- | -- |
| Qwen3-Embed-0.6B | 600M | -- | -- | -- | -- |
| DRAMA-1B | 1B | -- | -- | -- | -- |
| Stella-1.5B | 1.5B | -- | -- | -- | -- |
| Instructor-xl | 1.5B | -- | -- | -- | -- |
| Lychee-embed | 1.5B | -- | -- | -- | -- |
| GTE-Qwen2-1.5B | 1.5B | -- | -- | -- | -- |
| Qwen3-Embed-4B | 4B | -- | -- | -- | -- |
| Linq-Embed-Mistral | 7B | -- | -- | -- | -- |
| SFR-Embedding-Mistral | 7B | -- | -- | -- | -- |
| E5-Mistral-7B | 7B | -- | -- | -- | -- |
| GritLM-7B | 7B | -- | -- | -- | -- |
| GTE-Qwen2-7B | 7B | -- | -- | -- | -- |
| Qwen3-Embed-8B | 8B | -- | -- | -- | -- |
| Nemotron-8B | 8B | -- | -- | -- | -- |
| BGE-Gemma2 | 9B | -- | -- | -- | -- |

*Columns show nDCG@10. See full results with R-Precision in `results/full_results.csv`.*

---

## 🔧 Fine-Tuning

**Models:** SBERT, DRAMA-1B, E5-base-v2, BGE-M3, Qwen3-Embedding-0.6B

**Protocol:**
- InfoNCE loss, temperature = 0.05
- AdamW, lr = 2e-5, linear warmup (100 steps)
- Early stopping: patience = 3, monitor val nDCG@10
- Seeds: 42, 43, 44 (results reported as mean ± std)
- Hard negatives: same PIE, opposite usage type

### Run Fine-Tuning

```bash
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode sentence --seeds 42
python run_fine_tune.py --model Qwen/Qwen3-Embedding-0.6B --mode instruction_sentence --seeds 42 43 44
```

Outputs are written to `results/fine_tuning/<slug>/<mode>/seed_<n>/metrics.json`.
Each metrics file includes a `_trainer_version` stamp so stale checkpoints can be
identified when the trainer's encoding contract changes.

Batch size: pulled from `MODEL_REGISTRY[model_id].batch_size` by default (overrideable via `--batch_size` or `training.batch_size` in config.yaml). Per-model defaults prevent OOM on large models.

### Models supported for fine-tuning

The trainer supports `sentence_transformer`, `instruction`, and `qwen` wrapper classes. Two restrictions:

- **GritLM is zero-shot only** — the trainer raises an error for any `gritlm`-class model. GritLM-7B is therefore excluded from `config.yaml::training.models` by default.
- **`instructor_pairs` models** (`hkunlp/instructor-base`, `hkunlp/instructor-xl`) can be fine-tuned in `sentence` or `span` modes, but **NOT in `instruction_sentence` / `instruction_span` modes**. Zero-shot inference passes a list of `[instruction, text]` pairs to the model, which the trainer's tokenize+forward gradient-flow path cannot mirror byte-equivalently. The trainer raises a clear error in this case.

---

## 🔬 Reproduction Workflow

Full paper reproduction in 4 steps:

```bash
# 1. BM25 baselines (tuned on validation)
python run_bm25.py --query_mode sentence --tune
python run_bm25.py --query_mode span --tune

# 2. Zero-shot dense retrieval (all 24 models × 4 modes)
python run_all.py

# 3. Fine-tuning reported models/modes
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode sentence --seeds 42 43 44

# 4. Generate paper tables and figures
python analysis/generate_zero_shot_table.py
python analysis/generate_finetuning_table.py
python analysis/generate_dataset_stats.py
python analysis/plot_performance.py
python analysis/generate_variant_tables.py    # per-variant table with by_usage and by_subject splits
python analysis/generate_ablation_table.py    # per-mode ablation tables (requires run_ablation.py)
python analysis/lexical_overlap.py            # keyword overlap diagnostic
```

Or use the skill: `/reproduce-paper`

---

## 🗂️ Model Registry

All 24 evaluated models:

| # | Model | HF ID | Size | Class | Instruction |
|---|-------|--------|------|-------|-------------|
| 1 | SBERT | `sentence-transformers/all-MiniLM-L6-v2` | 110M | sentence_transformer | — |
| 2 | Contriever | `facebook/contriever` | 110M | sentence_transformer | — |
| 3 | E5-base-v2 | `intfloat/e5-base-v2` | 110M | sentence_transformer | e5_inline |
| 4 | TART | `orionweller/tart-dual-contriever-msmarco` | 110M | instruction | tart_sep |
| 5 | BGE-base | `BAAI/bge-base-en-v1.5` | 326M | instruction | prompt_prefix |
| 6 | Instructor-base | `hkunlp/instructor-base` | 335M | instruction | instructor_pairs |
| 7 | Nomic-v2 | `nomic-ai/nomic-embed-text-v2-moe` | 475M | instruction | nomic_prefix |
| 8 | Multilingual-E5-large | `intfloat/multilingual-e5-large-instruct` | 560M | instruction | e5_inline |
| 9 | BGE-M3 | `BAAI/bge-m3` | 568M | sentence_transformer | — |
| 10 | Qwen3-Embed-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 600M | qwen | e5_inline_no_space |
| 11 | DRAMA-1B | `facebook/drama-1b` | 1B | sentence_transformer | — |
| 12 | Stella-1.5B | `NovaSearch/stella-en-1.5B-v5` | 1.5B | sentence_transformer | — |
| 13 | Instructor-xl | `hkunlp/instructor-xl` | 1.5B | instruction | instructor_pairs |
| 14 | Lychee-embed | `vec-ai/lychee-embed` | 1.5B | instruction | e5_inline_no_space |
| 15 | GTE-Qwen2-1.5B | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1.5B | qwen | e5_inline |
| 16 | Qwen3-Embed-4B | `Qwen/Qwen3-Embedding-4B` | 4B | qwen | e5_inline_no_space |
| 17 | Linq-Embed-Mistral | `Linq-AI-Research/Linq-Embed-Mistral` | 7B | instruction | e5_inline_no_space |
| 18 | SFR-Embedding-Mistral | `Salesforce/SFR-Embedding-Mistral` | 7B | instruction | e5_inline |
| 19 | E5-Mistral-7B | `intfloat/e5-mistral-7b-instruct` | 7B | instruction | e5_inline |
| 20 | GritLM-7B | `GritLM/GritLM-7B` | 7B | gritlm | instructor_pairs |
| 21 | GTE-Qwen2-7B | `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7B | qwen | e5_inline |
| 22 | Qwen3-Embed-8B | `Qwen/Qwen3-Embedding-8B` | 8B | qwen | e5_inline_no_space |
| 23 | Nemotron-8B | `nvidia/llama-embed-nemotron-8b` | 8B | instruction | e5_inline_no_space |
| 24 | BGE-Gemma2 | `BAAI/bge-multilingual-gemma2` | 9B | instruction | bge_gemma |

*BGE-base-en-v1.5 uses its canonical pretrained prefix (`"Represent this sentence for searching relevant passages: "`) via the `prompt_prefix` instruction format. The Qwen3 family, Lychee, Linq, and Nemotron use the no-space `Instruct: {task}\nQuery:{query}` variant per their model cards — verify against each card's "Usage" section before paper publication.*

---

## 🏗️ Data Generation

The benchmark data is pre-generated in `data/`. To reproduce the dataset from scratch (requires Gemini API key):

1. MAGPIE corpus filtering (min_occurrences=30, ambiguity 25-75%)
2. Variant generation via Gemini (10 domains × 4 usage types)
3. LLM validation (3× majority vote)
4. Stratified splitting (22/10/75 PIEs for train/val/test)

See [`data_generation/README.md`](data_generation/README.md) for full instructions.

---

## 📜 Citation

If you use IdioLink in your research, please cite:

**BibTeX:**

```bibtex
@misc{hashiloni2026idiolinkretrievingmeaningwords,
  title={IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions},
  author={Kai Golan Hashiloni and Daniel Fadlon and Lior Livyatan and Ofri Hefetz and Jiahuan Pei and Kfir Bar},
  year={2026},
  eprint={2605.22247},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.22247},
}
```

**APA:**

Hashiloni, K. G., Fadlon, D., Livyatan, L., Hefetz, O., Pei, J., & Bar, K. (2026). *IdioLink: Retrieving meaning beyond words across idiomatic and literal expressions*. arXiv. https://arxiv.org/abs/2605.22247

*Paper under review. Citation will be updated upon publication.*

---

## 📄 License

Code: Apache 2.0. See [LICENSE](LICENSE).  
Data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Authors

Kai Golan Hashiloni et al. ([Intellexus Project](https://intellexus.net/))

## 📫 Contact

For questions or contributions: [kai.golanhashiloni@post.runi.ac.il](mailto:kai.golanhashiloni@post.runi.ac.il?subject=IdioLink) · [daniel.fadlon@post.runi.ac.il](mailto:daniel.fadlon@post.runi.ac.il?subject=IdioLink), [daniel.fadlon@post.runi.ac.il](mailto:daniel.fadlon@post.runi.ac.il?subject=IdioLink)
