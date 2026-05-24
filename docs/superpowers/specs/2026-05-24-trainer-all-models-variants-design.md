# Trainer support for all models × all variants — design

## Problem

`idiolink/trainer/contrastive_trainer.py` and `idiolink/trainer/datasets.py` reimplement instruction/prompt formatting badly, with four distinct train↔eval mismatches plus three orchestration gaps:

### Correctness bugs

1. **Trainer bypasses the registry.** `contrastive_trainer.py:73` does `SentenceTransformer(config.model_id, device=...)` directly. `trust_remote_code`, per-model `batch_size`, `instruction_format`, `QwenModel`/`GritLMModel` wrapper selection, and `query_prefix`/`passage_prefix` are all ignored. Qwen3-Embedding-0.6B (in `config.yaml::training.models`) silently loads without `trust_remote_code=True` and without `QwenModel._query_prompt`'s no-space handling.

2. **Hardcoded E5 template ignores `instruction_format`.** Both `TripletDataset.__getitem__` (datasets.py:49) and `ContrastiveTrainer._evaluate` (contrastive_trainer.py:144, 150) hardcode `f"Instruct: {instruction}\nQuery: {query}"` (with space). Wrong for `e5_inline_no_space` (Qwen3), `prompt_prefix` (BGE-base-en-v1.5), `bge_gemma`, `instructor_pairs`, `nomic_prefix`, `tart_sep`.

3. **Trainer bypasses `encode_queries`.** `_evaluate` constructs an `_STModelWrapper` whose `encode(texts)` calls plain `st_model.encode(texts)`, then `retriever.retrieve()` routes through that — so the wrapper's `encode_queries` (including Qwen's `prompt=` kwarg and space-aware `_query_prompt`) is never used.

4. **`span` / `instruction_span` replace query with bare span at training time.** `TripletDataset.__getitem__` (datasets.py:40-41) sets `query = item["query_span"]` for these modes — so training encodes `"behind bars"` while `_evaluate` uses late-chunked sentence pooling. Different inputs at train vs eval.

5. **Doc-side `passage_prefix` is dropped.** Models with `passage_prefix` set in the registry (e5-base-v2: `"passage: "`, nomic: `"search_document: "`) require that prefix on documents at encode time. Trainer's `_compute_loss` calls `self.st_model.tokenize(positives + negatives)` directly with no prefix applied. Zero-shot inference applies the prefix; training does not.

### Orchestration gaps

6. **No matrix runner.** `run_fine_tune.py` runs one model × one mode per invocation. The config declares a 5×4×3 = 60-run matrix (`training.models` × `training.modes` × `training.seeds`) but there's no driver, no resumability, no skip-existing, no aggregate CSV. README still says "run this loop yourself."

7. **Batch size ignores the registry.** `TrainingConfig.batch_size` defaults to 32 for every model. Qwen3-Embedding-0.6B registry sets batch_size=32, but larger models the user might add later (Qwen3-Embedding-4B registry: 8, GritLM-7B: 4) would OOM silently.

8. **The production-readiness plan's claim that the trainer "uses the resolver" is misleading.** It uses `resolve_instructions` to build instruction text, then re-wraps it with the hardcoded E5 template — same bug, different framing. The new design replaces both call sites.

## Goal

For every (model, mode) pair the trainer accepts, training-time encoding produces byte-identical inputs to what the zero-shot path produces — with gradients flowing back through model parameters. Eval inside the trainer uses the exact same helper as zero-shot.

Supported wrapper classes for training: `sentence_transformer`, `instruction`, `qwen`. `gritlm` fails fast with a clear error (GritLM library has no unified `.tokenize() + forward()` path matching our gradient-flow design and is not in `config.yaml::training.models`).

**Out of scope** (preserved as-is per "mirror zero-shot exactly"):
- `SentenceTransformerModel` not receiving `query_prefix`/`passage_prefix` from `load_model` (latent zero-shot bug, raised in the prior code review; doesn't affect any of the 5 default training models since only e5-base-v2 sets those and the trainer will inherit the same gap).
- `nomic_prefix` format discards the instruction at zero-shot; training will too.
- `_atomic_write_json` lacks `fsync` (unrelated to trainer).

## Design

### Architecture change

Single source of truth for instruction formatting: the per-model wrappers in `idiolink/models/`. Trainer is a thin client.

```
Before:                                After:
─────────────────                      ─────────────────
ContrastiveTrainer                     ContrastiveTrainer
  └─ raw SentenceTransformer             └─ wrapper = load_model(model_id)
  └─ hardcoded template                       (InstructionModel/QwenModel/STModel)
  └─ _STModelWrapper                     └─ _encode_with_grad(texts)
                                                wrapper.model.tokenize → forward
                                         └─ late_chunk_encode_with_grad(wrapper,...)
                                                for span / instruction_span

TripletDataset                         TripletDataset
  └─ hardcoded template                  └─ returns plain dict
  └─ swaps query→query_span                    {query, query_span, ...}
                                          (no formatting; trainer dispatches)

_evaluate                              _evaluate
  └─ hardcoded template                  └─ encode_queries_for_mode(wrapper,...)
  └─ retriever via _STModelWrapper            (same helper as run_dense/run_ablation)

run_fine_tune.py: 1 model × 1 mode     run_fine_tune.py: unchanged (single run)
                                       run_fine_tune_matrix.py (NEW)
                                          └─ models × modes × seeds loop
                                          └─ resumability via metrics.json check
                                          └─ per-model batch_size from registry
                                          └─ aggregated full_results.csv
```

### Components

**1. `idiolink/models/encode_helpers.py` (new, ~60 lines).** Move `encode_queries_for_mode` out of `run_ablation.py` into this module so trainer and run-scripts share one definition. Pure relocation; update imports in `run_ablation.py`, `run_dense.py`, `run_all.py`, `run_instruction.py`.

**2. `idiolink/models/late_chunking.py` (+~45 lines).** Add:

```python
def late_chunk_encode_with_grad(
    model: BaseEmbeddingModel,
    texts: List[str],
    spans: List[str],
    device: Optional[str] = None,
    prefer_last_span: bool = False,
) -> torch.Tensor:
    """Mirror of late_chunk_encode without torch.no_grad(). Returns a tensor
    with gradients flowing back through model.model._first_module().auto_model.
    """
```

Mirrors `late_chunk_encode` line-for-line except: no `torch.no_grad()` wrap; uses `torch.stack(embeddings)` instead of `np.array(...)`. Pattern matches reference repo's `encode_spans_for_training` (`work_IdioLink/scripts/contrastive_training/train.py:163`). Fallback path (`span not found` or `no matching tokens`) detaches and re-attaches via `_encode_with_grad([doc])[0]` so gradient flow is preserved through the fallback.

**3. `idiolink/models/base.py` (+~3 lines).** Add `passage_prefix: str = ""` as a default attribute on `BaseEmbeddingModel` so `getattr(wrapper, "passage_prefix", "")` is never `AttributeError` — and so future wrappers explicitly opt in.

**4. `idiolink/models/gritlm.py` (no code change).** `GritLMModel.model` is a `GritLM` instance, not `SentenceTransformer`; it has no `.tokenize() + forward()` path matching our gradient-flow design. Trainer detects `model_class == "gritlm"` at `__init__` and raises:
```
ValueError("Fine-tuning is not supported for gritlm-class models (GritLM/GritLM-7B). "
           "GritLM is zero-shot-only in this codebase. Remove it from "
           "training.models or use a different model.")
```
GritLM-7B is not in `config.yaml::training.models`, so this is defensive.

**5. `idiolink/trainer/datasets.py` (-~25/+~10 lines).** `TripletDataset.__getitem__` returns:

```python
return {
    "query": item["query"],                                    # plain sentence
    "query_span": item.get("query_span") or item.get("query_idiom") or item["query"],
    "query_idiom": item.get("query_idiom", ""),
    "query_usage": item.get("query_usage", ""),
    "query_subject": item.get("query_subject", ""),
    "positive": item["positive"],
    "negatives": item["negatives"][:self.max_negatives],
}
```

No more `mode`-conditional substitution; no more `f"Instruct: ..."`. Remove the `mode` parameter from `TripletDataset.__init__` entirely (drop the `mode=mode` argument at `run_fine_tune.py:48`). The dataset is mode-agnostic by construction now.

**6. `idiolink/trainer/contrastive_trainer.py` (-~120/+~110 lines).** Main rewrite:

- `__init__`:
  ```python
  from ..models.registry import MODEL_REGISTRY, load_model
  cfg = MODEL_REGISTRY.get(config.model_id)
  if cfg is not None and cfg.model_class == "gritlm":
      raise ValueError(...)  # see component 4
  self.model = load_model(config.model_id, device=self.device)
  assert hasattr(self.model.model, "tokenize"), (
      f"Trainer requires self.model.model to be a SentenceTransformer "
      f"(got {type(self.model.model).__name__}). "
      f"Model class '{cfg.model_class if cfg else '?'}' is not trainable."
  )
  ```
  Optimizer wraps `self.model.model.parameters()` (the underlying ST).

- `_encode_with_grad(texts: List[str]) -> torch.Tensor`:
  ```python
  features = self.model.model.tokenize(texts)
  features = {k: v.to(self.device) for k, v in features.items()}
  output = self.model.model(features)
  return output["sentence_embedding"]
  ```

- `_format_query_strings(plain_texts, batch_idiom_queries) -> List[str]`:
  - `sentence` / `span`: returns `plain_texts` unchanged.
  - `instruction_sentence` / `instruction_span`: resolves instructions via `resolve_instructions(self.model.model_id, batch_idiom_queries)`, then calls `self.model.format_queries_for_late_chunking(plain_texts, instructions)`.

- `_compute_loss(batch)`:
  1. Build `IdiomQuery` objects from batch fields (query/idiom/usage_type/span/subject) — used by `resolve_instructions`.
  2. Get formatted query strings via `_format_query_strings`.
  3. Query embeddings:
     - `sentence` / `instruction_sentence`: `_encode_with_grad(formatted)`.
     - `span` / `instruction_span`: `late_chunk_encode_with_grad(self.model, formatted, query_spans, device=self.device, prefer_last_span=(mode == "instruction_span"))`.
  4. Doc embeddings: apply `getattr(self.model, "passage_prefix", "")` to positives + flat_negatives, then `_encode_with_grad(...)`.
  5. Loss as today.

- `_evaluate`:
  ```python
  from ..models.encode_helpers import encode_queries_for_mode
  ...
  self.model.model.eval()
  retriever = DenseRetriever(self.model)         # real wrapper, not _STModelWrapper
  retriever.index(doc_sentences, doc_metadata)
  query_texts, query_embeddings = encode_queries_for_mode(
      self.model, self.config.mode, idiom_queries, self.device
  )
  results = retriever.retrieve(query_texts, top_k=100, query_embeddings=query_embeddings)
  evaluator = Evaluator(idiom_queries, doc_metadata)
  metrics = evaluator.evaluate(results)
  self.model.model.train()
  return metrics
  ```
  Deletes the entire `_STModelWrapper` class.

- `TrainingConfig.batch_size` semantics change:
  - New default sentinel: `batch_size: Optional[int] = None` (was `32`).
  - Resolution order: CLI/config override > registry's `MODEL_REGISTRY[model_id].batch_size` > literal default `32`.
  - Resolution happens in `ContrastiveTrainer.__init__`, with a logged line:
    ```
    INFO trainer: model=<id> batch_size=<n> (from <source>)
    ```

**7. `run_fine_tune.py` (+~5 lines).** Update batch_size resolution to use the same priority as the trainer. Otherwise unchanged — stays a single-run entry point for debugging.

**8. `run_fine_tune_matrix.py` (new, ~250 lines).** Mirror of `run_ablation.py`. Shares the `_atomic_write_json` helper — promote it from `run_ablation.py` into `idiolink/utils.py` (called `atomic_write_json`) and have both runners import it. Per-cell `metrics.json` schema gains one new key: `"_trainer_version": 2` (the resume check skips a cell only when the on-disk version matches the current `TRAINER_VERSION` constant; older checkpoints are auto-recomputed). Skeleton:

```python
"""Fine-tuning matrix runner: models × modes × seeds with resume + aggregate CSV."""

def _result_path(results_dir, model_id, mode, seed) -> Path:
    return results_dir / "fine_tuning" / model_slug(model_id) / mode / f"seed_{seed}" / "metrics.json"

def run_one(model_id, mode, seed, train_cfg, data_cfg, results_dir) -> Optional[dict]:
    path = _result_path(results_dir, model_id, mode, seed)
    if path.exists() and not force:
        return _load_existing_row(path)
    cfg = MODEL_REGISTRY.get(model_id)
    batch_size = train_cfg.get("batch_size") or (cfg.batch_size if cfg else 32)
    config = TrainingConfig(model_id=model_id, mode=mode, seed=seed, batch_size=batch_size, ...)
    try:
        test_metrics = run_single_seed(config, ..., mode)
    except Exception as e:
        logger.error(f"FAILED {model_id}/{mode}/seed={seed}: {e}")
        traceback.print_exc()
        return None
    return _flatten_for_csv(test_metrics, model_id, mode, seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Default: cfg['training']['models']")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=["sentence","span","instruction_sentence","instruction_span"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute (model, mode, seed) even if metrics.json exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the matrix without running.")
    args = parser.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    train_cfg = cfg["training"]
    models = args.models or train_cfg["models"]
    modes = args.modes or train_cfg["modes"]
    seeds = args.seeds or train_cfg["seeds"]

    rows = []
    for model_id in models:
        for mode in modes:
            for seed in seeds:
                row = run_one(model_id, mode, seed, train_cfg, cfg["data"],
                              Path(cfg["results_dir"]), force=args.force)
                if row:
                    rows.append(row)

    # Rebuild aggregate CSV from every metrics.json on disk (mirrors run_ablation pattern)
    rows_from_disk = _collect_all_rows_from_disk(Path(cfg["results_dir"]), models, modes, seeds)
    _write_aggregate_csv(Path(cfg["results_dir"]) / "fine_tuning" / "full_results.csv", rows_from_disk)
```

Output layout:
```
results/fine_tuning/
  full_results.csv                                  (aggregate, rebuilt every run)
  sentence-transformers__all-MiniLM-L6-v2/
    sentence/seed_42/{metrics.json, best_model/}
    sentence/seed_43/{...}
    instruction_sentence/seed_42/{...}
    ...
  Qwen__Qwen3-Embedding-0.6B/
    ...
```

Resume check: per-(model, mode, seed) — if `metrics.json` exists and `--force` not set, skip. Matches `run_ablation.py` semantics including the `_atomic_write_json` write pattern for `metrics.json` (apply the same atomic-write helper for consistency with the ablation runner).

### Data flow per mode

| mode | training input to model | eval input to model | match? |
|---|---|---|---|
| `sentence` | `_encode_with_grad([q.query for q in batch])` | `encode_queries_for_mode("sentence")` → `model.encode([q.query...])` | ✓ |
| `span` | `late_chunk_encode_with_grad(model, [q.query...], [q.query_span...])` | `encode_queries_for_mode("span")` → `late_chunk_encode(model, ..., ...)` | ✓ |
| `instruction_sentence` | `_encode_with_grad(model.format_queries_for_late_chunking(plain, instructions))` | `encode_queries_for_mode("instruction_sentence")` → `model.encode_queries(plain, spans, instructions)` | ✓ for STModel/Instruction/Qwen — wrappers produce the same strings both paths |
| `instruction_span` | `late_chunk_encode_with_grad(model, model.format_queries_for_late_chunking(plain, instructions), spans, prefer_last_span=True)` | `encode_queries_for_mode("instruction_span")` → same chain | ✓ |

The key invariant: `model.format_queries_for_late_chunking(plain, instructions)` is the canonical formatter. Training tokenizes its output for gradient flow. Eval passes the same output through `model.encode(...)` / `late_chunk_encode(...)`. Same strings either way.

### Error handling

- **`gritlm` model_class:** trainer raises at `__init__` with the message in component 4.
- **Wrapper missing `format_queries_for_late_chunking`:** trainer asserts `hasattr(self.model, "format_queries_for_late_chunking")` once at `__init__`. All 4 current wrappers implement it.
- **Wrapper missing `passage_prefix`:** `getattr(wrapper, "passage_prefix", "")` — defaulted on `BaseEmbeddingModel`.
- **`prefer_last_span` for `instruction_span`:** `rfind` so the span match lands on the query-side occurrence rather than the instruction-side `{span}` placeholder.
- **Matrix runner per-cell failure:** logged + traceback, loop continues. Per-cell `metrics.json` is the resume marker (only written on full success via `_atomic_write_json`).

### Testing

`tests/test_trainer.py` (new, ~250 lines):

**Per-(wrapper_class, mode) string-equivalence tests — 3 classes × 4 modes = 12 tests.**

Fixture: `FakeBaseModel` for each of `SentenceTransformerModel`, `InstructionModel`, `QwenModel`. The fake's `model.tokenize` records strings; `model(features)` returns `torch.randn(batch, dim, requires_grad=True)`.

Each test:
1. Build a fake wrapper of class X with a chosen `instruction_format` (registry value).
2. Build `TripletDataset` with 2 known triplets.
3. Build `ContrastiveTrainer` with mode=Y.
4. Call `_compute_loss(collate([dataset[0], dataset[1]]))`.
5. Assert captured query-tokenize strings equal what `encode_queries_for_mode(wrapper, Y, [idiom_query, idiom_query], device)` would produce after replacing its `model.encode`/`late_chunk_encode` calls with a string-capturing stub (a small helper in the test file that intercepts at the right boundary).
6. Assert captured doc-tokenize strings equal `passage_prefix + positive` and `passage_prefix + each_negative`.
7. Assert `loss.backward()` populates `.grad` on at least one model parameter.

**Smoke test (end-to-end, 1 test):**
- Real `sentence-transformers/all-MiniLM-L6-v2` (already cached for other tests).
- 4 hand-built triplets, mode=`instruction_sentence`, 1 epoch, `max_negatives=1`.
- Assert no exception, `loss.item()` finite, `best_model/` directory created.

**Matrix runner test (1 test):**
- Mock `run_single_seed` to return a known dict.
- Run `run_fine_tune_matrix.main` with `--models a b --modes sentence span --seeds 42 43` on a tmpdir.
- Assert all 8 `metrics.json` files exist and `full_results.csv` has 8 rows.
- Re-run without `--force`; assert `run_single_seed` is not called again (resume worked).

All CPU-only; no CUDA/MPS dependencies.

### Files changed

| file | added | removed |
|---|---|---|
| `idiolink/models/encode_helpers.py` (new) | ~60 | 0 |
| `idiolink/models/late_chunking.py` | ~45 | 0 |
| `idiolink/models/base.py` | ~3 | 0 |
| `idiolink/models/gritlm.py` | 0 | 0 (defensive guard lives in trainer) |
| `idiolink/utils.py` | ~12 (move `atomic_write_json` here) | 0 |
| `run_ablation.py` | 2 (imports) | ~60 (delete moved fn + atomic helper) |
| `run_dense.py`, `run_all.py`, `run_instruction.py` | 1 each (import) | 0 |
| `idiolink/trainer/datasets.py` | ~10 | ~25 |
| `idiolink/trainer/contrastive_trainer.py` | ~110 | ~120 |
| `run_fine_tune.py` | ~5 (batch_size resolve, drop `mode=` arg) | 0 |
| `run_fine_tune_matrix.py` (new) | ~250 | 0 |
| `tests/test_trainer.py` (new) | ~250 | 0 |
| `config.yaml` | 0 | 0 (batch_size sentinel handled in code) |
| `README.md` | ~20 (matrix runner usage + GritLM note) | ~10 (old manual loop instructions) |

Net: ~750 lines added, ~210 removed. Largest chunks are the new tests (~250) and the new matrix runner (~250).

### Risks

- **Late chunking with gradients can OOM** on larger models at batch_size > registry default. Mitigated by component 6's batch_size resolution priority (registry default kicks in when CLI/config doesn't override).
- **Tokenizer differences across wrappers** — Qwen with `trust_remote_code=True` may register a tokenizer whose `tokenize()` output differs from vanilla SentenceTransformer. Verified in the per-class tests by string-capture rather than tensor-shape assertions.
- **`encode_queries_for_mode` relocation** is a pure move but touches 4 import sites. No behavior change; CI catches breakage.
- **Backwards-compat for existing fine-tuned checkpoints**: any checkpoint saved before this fix was trained on the wrong prompt format. The fix invalidates them. Existing `results/fine_tuning/**/best_model/` should be regenerated. Document in commit message and matrix runner's `--force` flag. Matrix runner detects checkpoint mismatch by metadata version: write `_trainer_version: 2` into `metrics.json` and skip-on-resume only when the version matches.
- **Matrix runner's per-cell failure semantics**: a failure does NOT write `metrics.json`, so resume re-tries on next invocation indefinitely. Mitigation: log a clearly-labeled FAILED summary at end-of-run with the (model, mode, seed) tuples that errored, and document `--force` for clearing partial state.

### Open questions

- None blocking. The "5 models × 4 modes × 3 seeds" matrix is what `config.yaml::training` already declares; this design just makes it actually runnable.
