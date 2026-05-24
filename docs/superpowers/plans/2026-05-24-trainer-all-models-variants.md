# Trainer Support for All Models × All Variants — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ContrastiveTrainer` route through `load_model()` + per-model wrappers so the trainer matches zero-shot encoding byte-for-byte for every (model, mode), with gradients. Add a fine-tune matrix runner with resume.

**Architecture:** Single source of truth for instruction formatting = the per-model wrappers (`SentenceTransformerModel`, `InstructionModel`, `QwenModel`). Trainer is a thin client that:
1. Loads the wrapper via the registry-backed factory (`load_model`).
2. Tokenizes the wrapper's canonical query strings (from `format_queries_for_late_chunking`) through the underlying `SentenceTransformer.tokenize`+forward path to get gradients.
3. Evaluates by calling the same `encode_queries_for_mode` helper as the zero-shot scripts.

**Tech Stack:** Python 3.11+, PyTorch, sentence-transformers, transformers, pytest, dataclasses.

**Spec:** `docs/superpowers/specs/2026-05-24-trainer-all-models-variants-design.md` (committed as `3be204a`).

**Scope guardrails:** This plan is narrow and comparability-preserving. Pre-existing zero-shot quirks (e5-base-v2 dropping `query_prefix`, nomic dropping instruction, `_atomic_write_json` no fsync) stay as-is; they're tracked separately in the follow-up cleanup task.

---

## File map

**Create:**
- `idiolink/models/encode_helpers.py` — relocated `encode_queries_for_mode`.
- `run_fine_tune_matrix.py` — matrix runner with resume.
- `tests/test_trainer.py` — per-(class, mode) string-equivalence tests + smoke test.
- `tests/test_fine_tune_matrix.py` — resume/skip-existing test.

**Modify:**
- `idiolink/utils.py` — add `atomic_write_json`.
- `idiolink/models/base.py` — add `passage_prefix: str = ""` default.
- `idiolink/models/late_chunking.py` — add `late_chunk_encode_with_grad`.
- `idiolink/trainer/datasets.py` — strip formatting; drop `mode` param.
- `idiolink/trainer/contrastive_trainer.py` — full rewrite of `__init__`, `_compute_loss`, `_evaluate`; delete `_STModelWrapper`.
- `run_fine_tune.py` — batch-size resolution; drop `mode=mode` arg to `TripletDataset`.
- `run_ablation.py` — delete moved helpers; import from new locations.
- `run_dense.py`, `run_all.py`, `run_instruction.py` — import `encode_queries_for_mode` from new location (only if they currently import it).
- `README.md` — document matrix runner; note GritLM is zero-shot only.

---

## Task 1: Move `atomic_write_json` into `idiolink/utils.py`

**Files:**
- Modify: `idiolink/utils.py`
- Modify: `run_ablation.py` (delete local helper, import from utils)
- Test: `tests/test_utils.py` (new, or add to existing test file if one exists for utils)

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils.py`:

```python
"""Tests for idiolink.utils helpers."""

import json
from pathlib import Path

from idiolink.utils import atomic_write_json


def test_atomic_write_json_creates_file(tmp_path: Path):
    target = tmp_path / "metrics.json"
    atomic_write_json(target, {"r_precision": 0.5, "n": 10})
    assert target.exists()
    assert json.loads(target.read_text()) == {"r_precision": 0.5, "n": 10}


def test_atomic_write_json_overwrites_existing(tmp_path: Path):
    target = tmp_path / "metrics.json"
    target.write_text(json.dumps({"old": 1}))
    atomic_write_json(target, {"new": 2})
    assert json.loads(target.read_text()) == {"new": 2}


def test_atomic_write_json_uses_temp_file(tmp_path: Path):
    target = tmp_path / "metrics.json"
    atomic_write_json(target, {"x": 1})
    # tmp file should not linger
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'atomic_write_json' from 'idiolink.utils'`

- [ ] **Step 3: Add `atomic_write_json` to `idiolink/utils.py`**

Append to `idiolink/utils.py`:

```python
import os


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically: stage to .tmp then os.replace onto target.

    Survives mid-write interruption (Ctrl-C / OOM / kill) — without this, a
    truncated metrics.json silently passes the resume-check `path.exists()`
    and corrupts the aggregated results.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
```

(`Path`, `json`, `Any` are already imported at top of `utils.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Replace the local `_atomic_write_json` in `run_ablation.py`**

Find the `_atomic_write_json` function in `run_ablation.py` (around lines 50-61) and the `import os` inside it. Delete the function entirely. Replace all calls to `_atomic_write_json(...)` with `atomic_write_json(...)` (drop the leading underscore). Add the import:

```python
from idiolink.utils import (
    atomic_write_json,
    get_device,
    load_config,
    load_documents,
    load_queries,
    model_slug,
    set_seed,
)
```

(Insert `atomic_write_json` into the existing alphabetised import block.)

- [ ] **Step 6: Run the ablation tests to verify no regression**

Run: `pytest tests/test_ablation.py -v`
Expected: same number of tests passing as before.

- [ ] **Step 7: Commit**

```bash
git add idiolink/utils.py run_ablation.py tests/test_utils.py
git commit -m "Promote atomic_write_json from run_ablation to idiolink.utils"
```

---

## Task 2: Add `passage_prefix=""` default to `BaseEmbeddingModel`

**Files:**
- Modify: `idiolink/models/base.py`
- Test: `tests/test_registry.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_all_loaded_wrappers_expose_passage_prefix():
    """Every wrapper must have a `passage_prefix` attribute (defaults to '')
    so trainer can read it uniformly via getattr without try/except.
    """
    from idiolink.models.base import BaseEmbeddingModel

    class _Stub(BaseEmbeddingModel):
        def encode(self, texts):
            import numpy as np
            return np.zeros((len(texts), 4))

    stub = _Stub("stub")
    assert hasattr(stub, "passage_prefix")
    assert stub.passage_prefix == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_all_loaded_wrappers_expose_passage_prefix -v`
Expected: FAIL with `AttributeError` or assertion.

- [ ] **Step 3: Add the default to `BaseEmbeddingModel.__init__`**

In `idiolink/models/base.py`, modify `__init__`:

```python
def __init__(self, model_id: str):
    self.model_id = model_id
    self.embedding_dim: int = 0
    self.passage_prefix: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: all tests including the new one PASS.

- [ ] **Step 5: Commit**

```bash
git add idiolink/models/base.py tests/test_registry.py
git commit -m "Add passage_prefix='' default on BaseEmbeddingModel"
```

---

## Task 3: Create `encode_helpers.py` and move `encode_queries_for_mode`

**Files:**
- Create: `idiolink/models/encode_helpers.py`
- Modify: `run_ablation.py` (delete moved fn; import from new location)
- Modify: `run_dense.py`, `run_all.py`, `run_instruction.py` (update imports if they reference it)
- Test: `tests/test_encode_helpers.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_encode_helpers.py`:

```python
"""Smoke tests for encode_helpers relocation."""

def test_encode_queries_for_mode_importable_from_models():
    from idiolink.models.encode_helpers import encode_queries_for_mode
    assert callable(encode_queries_for_mode)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_encode_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the new module**

Create `idiolink/models/encode_helpers.py`:

```python
"""Per-mode query encoding helpers shared between zero-shot and trainer eval."""

from typing import List, Tuple

import numpy as np

from .late_chunking import late_chunk_encode
from .instruction_model import resolve_instructions
from ..utils import IdiomQuery


def encode_queries_for_mode(
    model,
    query_mode: str,
    idiom_queries: List[IdiomQuery],
    device: str,
) -> Tuple[List[str], np.ndarray]:
    """Encode queries for the given mode. Returns (query_texts, query_embeddings).

    The single source of truth for per-mode query encoding. Used by both the
    zero-shot scripts (run_dense, run_ablation, run_instruction) and the
    trainer's evaluation path so the two cannot drift.
    """
    spans = [q.span if q.span else q.query for q in idiom_queries]
    query_texts = [q.query for q in idiom_queries]
    instructions = resolve_instructions(model.model_id, idiom_queries)

    if query_mode == "sentence":
        return query_texts, model.encode(query_texts)
    if query_mode == "span":
        return query_texts, late_chunk_encode(model, query_texts, spans, device=device)
    if query_mode == "instruction_sentence":
        if hasattr(model, "encode_queries"):
            embs = model.encode_queries(query_texts, spans=spans, instruction=instructions)
        else:
            embs = model.encode(query_texts)
        return query_texts, embs
    if query_mode == "instruction_span":
        if hasattr(model, "encode_queries"):
            chunking_texts = (
                model.format_queries_for_late_chunking(query_texts, instructions)
                if hasattr(model, "format_queries_for_late_chunking")
                else query_texts
            )
            embs = late_chunk_encode(
                model, chunking_texts, spans, device=device, prefer_last_span=True,
            )
        else:
            embs = model.encode(query_texts)
        return query_texts, embs
    raise ValueError(f"Unknown query_mode: {query_mode}")
```

- [ ] **Step 4: Delete the duplicate in `run_ablation.py`**

In `run_ablation.py`:
1. Delete the `encode_queries_for_mode` function (around lines 85-114).
2. Replace the existing import block for the helpers with:
   ```python
   from idiolink.models.encode_helpers import encode_queries_for_mode
   ```
3. Drop the now-unused local imports: `from idiolink.models.instruction_model import resolve_instructions` and `from idiolink.models.late_chunking import late_chunk_encode` — both are now used only by encode_helpers (verify with grep before deleting).

- [ ] **Step 5: Update import sites**

Search and update imports:

```bash
grep -rn "encode_queries_for_mode" /Users/dani/Documents/my-research/26-may-empnlp-idiolink/IdioLink --include="*.py" --exclude-dir=__pycache__
```

For each match outside `run_ablation.py` and `idiolink/models/encode_helpers.py`, change the import to `from idiolink.models.encode_helpers import encode_queries_for_mode`.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: same pass count as before the task started; the new import test passes.

- [ ] **Step 7: Commit**

```bash
git add idiolink/models/encode_helpers.py run_ablation.py tests/test_encode_helpers.py
# Also git add any other run_*.py files modified for imports
git commit -m "Move encode_queries_for_mode into idiolink.models.encode_helpers"
```

---

## Task 4: Add `late_chunk_encode_with_grad` to `late_chunking.py`

**Files:**
- Modify: `idiolink/models/late_chunking.py`
- Test: `tests/test_late_chunking.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_late_chunking.py`:

```python
def test_late_chunk_encode_with_grad_returns_tensor_with_grad():
    """Gradient version returns a torch.Tensor (not ndarray) and preserves
    gradient flow through the underlying transformer parameters.
    """
    from idiolink.models.late_chunking import late_chunk_encode_with_grad

    class _Wrapper:
        model_id = "fake/model"

        def __init__(self):
            self.model = _FakeST()

        def encode(self, texts):
            # fallback path only — not exercised when span is found
            import numpy as np
            return np.zeros((len(texts), 8), dtype=np.float32)

    class _FakeST:
        class _FakeFirstModule:
            def __init__(self):
                self.auto_model = _FakeTransformerWithGrad()
                self.tokenizer = _FakeTokenizer()

        def __init__(self):
            self._fm = _FakeST._FakeFirstModule()

        def _first_module(self):
            return self._fm

    docs = ["The cat sat on the mat."]
    spans = ["cat"]
    out = late_chunk_encode_with_grad(_Wrapper(), docs, spans, device="cpu")
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad or out.grad_fn is not None
    assert out.shape == (1, 8)
```

Add a `_FakeTransformerWithGrad` near the existing `_FakeTransformer` class:

```python
class _FakeTransformerWithGrad:
    """Like _FakeTransformer but produces tensors with requires_grad=True."""

    def __init__(self, dtype=torch.float32, hidden_dim=8):
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self._param = torch.zeros(1, device="cpu", requires_grad=True)

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        return self

    def __call__(self, **encoding):
        seq_len = encoding["input_ids"].shape[1]
        # multiply by self._param so gradient flows through
        out = torch.randn(1, seq_len, self.hidden_dim, dtype=self.dtype) * (1 + self._param)
        return _FakeTransformerOutput(out)
```

(Reuse the existing `_FakeTokenizer` from `test_late_chunking.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_late_chunking.py::test_late_chunk_encode_with_grad_returns_tensor_with_grad -v`
Expected: FAIL with `ImportError: cannot import name 'late_chunk_encode_with_grad'`.

- [ ] **Step 3: Implement `late_chunk_encode_with_grad`**

Append to `idiolink/models/late_chunking.py`:

```python
def late_chunk_encode_with_grad(
    model: BaseEmbeddingModel,
    documents: List[str],
    spans: List[str],
    device: Optional[str] = None,
    prefer_last_span: bool = False,
) -> torch.Tensor:
    """Gradient-flow version of `late_chunk_encode`.

    Mirrors `late_chunk_encode` exactly except: no `torch.no_grad()` wrap,
    and returns a `torch.Tensor` on `device` instead of an ndarray. Used by
    the trainer's `_compute_loss` to keep gradients flowing back through the
    underlying transformer for span and instruction_span training modes.
    """
    if hasattr(model, "model") and hasattr(model.model, "_first_module"):
        st_model = model.model
        transformer = st_model._first_module().auto_model
        tokenizer = st_model._first_module().tokenizer
    elif hasattr(model, "model") and hasattr(model.model, "auto_model"):
        transformer = model.model.auto_model
        tokenizer = model.model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        transformer = AutoModel.from_pretrained(model.model_id)

    if device is None:
        device = next(transformer.parameters()).device
    else:
        device = torch.device(device)
        transformer = transformer.to(device)

    embeddings = []
    for doc, span in zip(documents, spans):
        span_start = doc.rfind(span) if prefer_last_span else doc.find(span)
        if span_start == -1:
            # Fallback: encode full document (no gradients through wrapper.encode,
            # but this path is rare — span not found in the formatted query).
            emb = torch.from_numpy(model.encode([doc])[0]).to(device).float()
            embeddings.append(emb)
            continue

        span_end = span_start + len(span)

        encoding = tokenizer(
            doc,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # NO torch.no_grad() — this is the key difference from late_chunk_encode
        outputs = transformer(**encoding)
        token_embeddings = outputs.last_hidden_state[0]

        span_indices = _find_span_tokens(offset_mapping, span_start, span_end)

        if not span_indices:
            emb = torch.from_numpy(model.encode([doc])[0]).to(device).float()
            embeddings.append(emb)
            continue

        span_embs = token_embeddings[span_indices]
        pooled = span_embs.mean(dim=0).float()
        embeddings.append(pooled)

    return torch.stack(embeddings)
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_late_chunking.py -v`
Expected: all tests including the new one PASS.

- [ ] **Step 5: Commit**

```bash
git add idiolink/models/late_chunking.py tests/test_late_chunking.py
git commit -m "Add late_chunk_encode_with_grad for span/instruction_span training"
```

---

## Task 5: Strip formatting from `TripletDataset`; drop `mode` param

**Files:**
- Modify: `idiolink/trainer/datasets.py`
- Modify: `run_fine_tune.py` (drop `mode=mode` kwarg)
- Test: `tests/test_finetuning_pipeline.py` or new `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_finetuning_pipeline.py` (or create `tests/test_dataset.py` if cleaner):

```python
import json
from pathlib import Path


def test_triplet_dataset_returns_plain_dict_no_instruction_wrapping(tmp_path: Path):
    """Dataset is mode-agnostic; returns plain fields, never applies the
    hardcoded `Instruct: ...\\nQuery: ...` template.
    """
    from idiolink.trainer import TripletDataset

    triplet_path = tmp_path / "triplets.jsonl"
    triplet_path.write_text(json.dumps({
        "query": "She kicked the bucket yesterday.",
        "query_span": "kicked the bucket",
        "query_idiom": "kick the bucket",
        "query_usage": "idiomatic",
        "query_subject": "death",
        "positive": "He died last week.",
        "negatives": ["He kicked the actual bucket over."],
    }) + "\n")

    ds = TripletDataset(str(triplet_path), max_negatives=5)
    assert len(ds) == 1
    item = ds[0]
    assert item["query"] == "She kicked the bucket yesterday."
    assert item["query_span"] == "kicked the bucket"
    assert item["query_idiom"] == "kick the bucket"
    assert item["query_usage"] == "idiomatic"
    assert item["query_subject"] == "death"
    assert item["positive"] == "He died last week."
    assert item["negatives"] == ["He kicked the actual bucket over."]
    # No instruction wrapping must have occurred
    assert "Instruct:" not in item["query"]
    assert "Query:" not in item["query"]


def test_triplet_dataset_no_longer_accepts_mode_param(tmp_path: Path):
    """`mode` parameter is removed; passing it should TypeError."""
    from idiolink.trainer import TripletDataset

    triplet_path = tmp_path / "t.jsonl"
    triplet_path.write_text(json.dumps({
        "query": "x", "positive": "y", "negatives": ["z"],
    }) + "\n")

    import pytest
    with pytest.raises(TypeError):
        TripletDataset(str(triplet_path), mode="instruction_sentence")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_finetuning_pipeline.py::test_triplet_dataset_returns_plain_dict_no_instruction_wrapping tests/test_finetuning_pipeline.py::test_triplet_dataset_no_longer_accepts_mode_param -v`
Expected: FAIL (first test: missing query_span/idiom in returned dict; second: TypeError not raised).

- [ ] **Step 3: Rewrite `TripletDataset`**

Replace `idiolink/trainer/datasets.py::TripletDataset` (lines 11-54) with:

```python
class TripletDataset(Dataset):
    """
    Dataset that loads pre-mined triplets from a JSONL file.

    Each line: {"query": ..., "positive": ..., "negatives": [...],
                "query_idiom": ..., "query_usage": ..., "query_span": ...,
                "query_subject": ...}

    Mode-agnostic: returns plain fields. The trainer applies per-model
    instruction formatting and per-mode span substitution at encode time.
    """

    def __init__(
        self,
        triplet_file: str,
        max_negatives: int = 5,
    ):
        self.max_negatives = max_negatives
        self.samples: List[Dict[str, Any]] = []
        with open(triplet_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        negatives = item["negatives"][: self.max_negatives]
        return {
            "query": item["query"],
            "query_span": item.get("query_span") or item.get("query_idiom") or item["query"],
            "query_idiom": item.get("query_idiom", ""),
            "query_usage": item.get("query_usage", ""),
            "query_subject": item.get("query_subject", ""),
            "positive": item["positive"],
            "negatives": negatives,
        }
```

- [ ] **Step 4: Update `run_fine_tune.py` to drop the `mode=` kwarg**

In `run_fine_tune.py`, find the `TripletDataset(...)` call (around line 45-49) and remove `mode=mode`:

```python
train_dataset = TripletDataset(
    triplet_file,
    max_negatives=config.max_negatives,
)
```

- [ ] **Step 5: Run dataset tests to verify they pass**

Run: `pytest tests/test_finetuning_pipeline.py -v`
Expected: 2 new tests pass; any pre-existing dataset tests that pass `mode=` need to be updated to drop it (search the test file and fix).

- [ ] **Step 6: Commit**

```bash
git add idiolink/trainer/datasets.py run_fine_tune.py tests/test_finetuning_pipeline.py
git commit -m "TripletDataset: drop mode-dependent formatting, return plain fields"
```

---

## Task 6: Rewrite `ContrastiveTrainer.__init__` — use `load_model`, batch_size resolution, gritlm guard

**Files:**
- Modify: `idiolink/trainer/contrastive_trainer.py`
- Modify: `run_fine_tune.py` (batch_size CLI default → None)
- Test: `tests/test_trainer.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_trainer.py`:

```python
"""Tests for ContrastiveTrainer's wiring to load_model + registry."""

import pytest
import torch

from idiolink.trainer import ContrastiveTrainer, TrainingConfig


def test_init_loads_wrapper_via_registry_for_qwen():
    """Trainer must use load_model so QwenModel + trust_remote_code take effect.
    Smoke: assert the wrapper type after init is QwenModel for a Qwen registry entry.
    Skipped if no network / model not cached.
    """
    from idiolink.models.qwen import QwenModel
    try:
        cfg = TrainingConfig(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            mode="sentence",
            seed=42,
            device="cpu",
            max_epochs=1,
        )
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Qwen model not available: {e}")
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model.model_id == "Qwen/Qwen3-Embedding-0.6B"


def test_init_raises_for_gritlm_class():
    """gritlm-class models cannot be trained by this trainer; fail fast."""
    cfg = TrainingConfig(
        model_id="GritLM/GritLM-7B",
        mode="sentence",
        seed=42,
        device="cpu",
        max_epochs=1,
    )
    with pytest.raises(ValueError, match="not supported for gritlm"):
        ContrastiveTrainer(cfg)


def test_batch_size_resolution_uses_registry_default_when_none():
    """If TrainingConfig.batch_size is None, trainer pulls from MODEL_REGISTRY.
    facebook/drama-1b has batch_size=16 in the registry.
    """
    cfg = TrainingConfig(
        model_id="facebook/drama-1b",
        mode="sentence",
        seed=42,
        device="cpu",
        batch_size=None,
        max_epochs=1,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")
    assert trainer.config.batch_size == 16


def test_batch_size_resolution_cli_override_wins():
    """If TrainingConfig.batch_size is set, it overrides the registry."""
    cfg = TrainingConfig(
        model_id="facebook/drama-1b",
        mode="sentence",
        seed=42,
        device="cpu",
        batch_size=4,
        max_epochs=1,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")
    assert trainer.config.batch_size == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v`
Expected: gritlm test fails (no ValueError); batch_size tests fail or skip.

- [ ] **Step 3: Rewrite `TrainingConfig` and `ContrastiveTrainer.__init__`**

In `idiolink/trainer/contrastive_trainer.py`:

1. Change `batch_size: int = 32` to `batch_size: Optional[int] = None` in `TrainingConfig` (and import `Optional` from typing if not already imported).

2. Replace the `ContrastiveTrainer.__init__` (lines 68-75) with:

```python
def __init__(self, config: TrainingConfig):
    from ..models.registry import MODEL_REGISTRY, load_model

    self.config = config
    self.device = get_device(config.device)
    set_seed(config.seed)

    # Registry-based load: honours model_class, trust_remote_code,
    # instruction_format, query_prefix, passage_prefix, batch_size.
    registry_cfg = MODEL_REGISTRY.get(config.model_id)
    if registry_cfg is not None and registry_cfg.model_class == "gritlm":
        raise ValueError(
            f"Fine-tuning is not supported for gritlm-class models "
            f"({config.model_id}). GritLM is zero-shot-only in this codebase. "
            f"Remove it from training.models or use a different model."
        )

    # Resolve batch_size: CLI/config > registry > hardcoded 32 fallback.
    if config.batch_size is None:
        config.batch_size = registry_cfg.batch_size if registry_cfg else 32
        source = "registry" if registry_cfg else "default"
    else:
        source = "config"
    logger.info(
        f"trainer: model={config.model_id} batch_size={config.batch_size} (from {source})"
    )

    self.model = load_model(config.model_id, device=self.device)
    if not hasattr(self.model.model, "tokenize"):
        raise ValueError(
            f"Trainer requires self.model.model to be a SentenceTransformer "
            f"(got {type(self.model.model).__name__}). "
            f"Model class '{registry_cfg.model_class if registry_cfg else '?'}' is not trainable."
        )
    if not hasattr(self.model, "format_queries_for_late_chunking"):
        raise ValueError(
            f"Trainer requires the wrapper to implement "
            f"format_queries_for_late_chunking (missing on {type(self.model).__name__})."
        )

    self.st_model = self.model.model  # underlying SentenceTransformer
    self.loss_fn = InfoNCELoss(temperature=config.temperature)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trainer.py -v`
Expected: gritlm test PASS; batch_size tests PASS (or SKIP if model not cached).

- [ ] **Step 5: Update `run_fine_tune.py` CLI default**

In `run_fine_tune.py`, change the `batch_size` resolution at line 139 from:

```python
batch_size = args.batch_size or train_cfg.get("batch_size", 32)
```

to:

```python
# None lets ContrastiveTrainer pull from MODEL_REGISTRY[model_id].batch_size
batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size")
```

And confirm `--batch_size` argparse default is `None` (it already is at line 116).

- [ ] **Step 6: Commit**

```bash
git add idiolink/trainer/contrastive_trainer.py run_fine_tune.py tests/test_trainer.py
git commit -m "ContrastiveTrainer.__init__: load via registry, gritlm guard, batch_size from registry"
```

---

## Task 7: Add `_encode_with_grad` and `_format_query_strings` helpers

**Files:**
- Modify: `idiolink/trainer/contrastive_trainer.py`
- Test: `tests/test_trainer.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trainer.py`:

```python
def test_format_query_strings_passes_through_for_sentence_mode():
    """For sentence/span modes, formatter is a no-op (returns input unchanged)."""
    from idiolink.utils import IdiomQuery

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="sentence", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    queries = ["the cat sat", "he kicked the bucket"]
    iqs = [IdiomQuery(query=q, idiom="", usage_type="literal", span="cat", subject="")
           for q in queries]
    out = trainer._format_query_strings(queries, iqs)
    assert out == queries


def test_format_query_strings_applies_wrapper_format_for_instruction_modes():
    """For instruction modes, formatter delegates to wrapper.format_queries_for_late_chunking."""
    from idiolink.utils import IdiomQuery

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="instruction_sentence", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    queries = ["the cat sat"]
    iqs = [IdiomQuery(query="the cat sat", idiom="cat", usage_type="literal",
                      span="cat", subject="")]
    out = trainer._format_query_strings(queries, iqs)
    assert len(out) == 1
    # SentenceTransformerModel applies generic `Instruct: ...\nQuery: ...` wrap
    assert out[0].startswith("Instruct:")
    assert "Query: the cat sat" in out[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v -k "format_query_strings"`
Expected: FAIL with `AttributeError: '_format_query_strings'`.

- [ ] **Step 3: Add the helpers to `ContrastiveTrainer`**

In `idiolink/trainer/contrastive_trainer.py`, add these methods to `ContrastiveTrainer` (between `__init__` and `_compute_loss`):

```python
def _encode_with_grad(self, texts: List[str]) -> torch.Tensor:
    """Tokenize + forward through the underlying SentenceTransformer with gradients."""
    features = self.st_model.tokenize(texts)
    features = {k: v.to(self.device) for k, v in features.items()}
    output = self.st_model(features)
    return output["sentence_embedding"]


def _format_query_strings(
    self,
    plain_texts: List[str],
    idiom_queries: List["IdiomQuery"],
) -> List[str]:
    """Return the strings the model would tokenize for queries, per mode.

    - sentence / span: identity (plain query text).
    - instruction_sentence / instruction_span: per-model instruction-formatted
      string via wrapper.format_queries_for_late_chunking, which is the same
      output zero-shot's encode_queries_for_mode uses.
    """
    if self.config.mode in ("sentence", "span"):
        return list(plain_texts)
    instructions = resolve_instructions(self.config.model_id, idiom_queries)
    return self.model.format_queries_for_late_chunking(plain_texts, instructions)
```

Add the imports at the top of the file:

```python
from ..models.instruction_model import resolve_instructions
```

(`IdiomQuery` is imported transitively via `..utils`; for the type-only forward reference in the signature, no new import is needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py -v -k "format_query_strings"`
Expected: both new tests PASS (or SKIP if MiniLM not cached).

- [ ] **Step 5: Commit**

```bash
git add idiolink/trainer/contrastive_trainer.py tests/test_trainer.py
git commit -m "ContrastiveTrainer: add _encode_with_grad and _format_query_strings helpers"
```

---

## Task 8: Rewrite `_compute_loss` — per-mode dispatch + passage_prefix

**Files:**
- Modify: `idiolink/trainer/contrastive_trainer.py`
- Test: `tests/test_trainer.py` (extend with per-mode behavior tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_trainer.py`:

```python
def test_compute_loss_applies_passage_prefix_to_docs():
    """If the wrapper has passage_prefix set, _compute_loss prepends it to
    positives and negatives before encoding.
    """
    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="sentence", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")
    trainer.model.passage_prefix = "passage: "

    captured = []
    original_encode = trainer._encode_with_grad
    def capture(texts):
        captured.append(list(texts))
        return original_encode(texts)
    trainer._encode_with_grad = capture

    batch = {
        "queries": ["q1", "q2"],
        "query_spans": ["span1", "span2"],
        "query_idioms": ["", ""],
        "query_usages": ["literal", "literal"],
        "query_subjects": ["", ""],
        "positives": ["doc1", "doc2"],
        "negatives": [["n1"], ["n2"]],
    }
    trainer._compute_loss(batch)

    # First call: queries (no prefix in sentence mode). Subsequent: docs with prefix.
    assert captured[0] == ["q1", "q2"]
    # Positives and negatives both should have the prefix
    for c in captured[1:]:
        assert all(t.startswith("passage: ") for t in c), f"missing prefix in {c}"


def test_compute_loss_span_mode_uses_late_chunk_with_grad():
    """span mode routes queries through late_chunk_encode_with_grad."""
    from unittest.mock import patch

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="span", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    with patch("idiolink.trainer.contrastive_trainer.late_chunk_encode_with_grad") as mock_lc:
        mock_lc.return_value = torch.zeros((2, trainer.model.embedding_dim), requires_grad=True)
        batch = {
            "queries": ["a b c d", "x y z"],
            "query_spans": ["b", "z"],
            "query_idioms": ["", ""],
            "query_usages": ["literal", "literal"],
            "query_subjects": ["", ""],
            "positives": ["p1", "p2"],
            "negatives": [[], []],
        }
        trainer._compute_loss(batch)
        mock_lc.assert_called_once()
        # prefer_last_span should be False for plain `span` mode
        kwargs = mock_lc.call_args.kwargs
        assert kwargs.get("prefer_last_span", False) is False


def test_compute_loss_instruction_span_uses_prefer_last_span_true():
    """instruction_span mode calls late_chunk_encode_with_grad with prefer_last_span=True."""
    from unittest.mock import patch

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="instruction_span", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    with patch("idiolink.trainer.contrastive_trainer.late_chunk_encode_with_grad") as mock_lc:
        mock_lc.return_value = torch.zeros((1, trainer.model.embedding_dim), requires_grad=True)
        batch = {
            "queries": ["the cat sat on the mat"],
            "query_spans": ["cat"],
            "query_idioms": [""],
            "query_usages": ["literal"],
            "query_subjects": [""],
            "positives": ["p1"],
            "negatives": [[]],
        }
        trainer._compute_loss(batch)
        mock_lc.assert_called_once()
        assert mock_lc.call_args.kwargs["prefer_last_span"] is True
```

- [ ] **Step 2: Update `collate_triplets` to surface the new fields**

In `idiolink/trainer/contrastive_trainer.py`, replace `collate_triplets` (lines 44-57) with:

```python
def collate_triplets(batch: List[Dict]) -> Dict[str, Any]:
    """Collate triplet dicts into batched lists of strings."""
    queries = [item["query"] for item in batch]
    query_spans = [item.get("query_span") or "" for item in batch]
    query_idioms = [item.get("query_idiom", "") for item in batch]
    query_usages = [item.get("query_usage", "") for item in batch]
    query_subjects = [item.get("query_subject", "") for item in batch]
    positives = [item["positive"] for item in batch]

    max_neg = max(len(item["negatives"]) for item in batch) if batch else 0
    negatives = []
    for item in batch:
        negs = item["negatives"]
        while len(negs) < max_neg:
            negs = negs + [negs[-1] if negs else ""]
        negatives.append(negs)

    return {
        "queries": queries,
        "query_spans": query_spans,
        "query_idioms": query_idioms,
        "query_usages": query_usages,
        "query_subjects": query_subjects,
        "positives": positives,
        "negatives": negatives,
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v -k "compute_loss"`
Expected: FAIL (current `_compute_loss` doesn't handle per-mode dispatch or passage_prefix).

- [ ] **Step 4: Rewrite `_compute_loss`**

In `idiolink/trainer/contrastive_trainer.py`, replace the existing `_compute_loss` (lines 84-113) with:

```python
def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
    """Compute InfoNCE loss for a batch with per-mode query encoding and
    doc-side passage_prefix application. Mirrors zero-shot inference path.
    """
    from ..utils import IdiomQuery

    queries = batch["queries"]
    query_spans = batch["query_spans"]
    positives = batch["positives"]
    negatives = batch["negatives"]
    batch_size = len(queries)

    # Build IdiomQuery objects for instruction resolution.
    iqs = [
        IdiomQuery(
            query=q,
            idiom=batch["query_idioms"][i],
            usage_type=batch["query_usages"][i],
            span=batch["query_spans"][i],
            subject=batch["query_subjects"][i],
        )
        for i, q in enumerate(queries)
    ]

    # Query embeddings — per mode
    formatted_queries = self._format_query_strings(queries, iqs)
    if self.config.mode == "sentence":
        query_emb = self._encode_with_grad(formatted_queries)
    elif self.config.mode == "instruction_sentence":
        query_emb = self._encode_with_grad(formatted_queries)
    elif self.config.mode == "span":
        query_emb = late_chunk_encode_with_grad(
            self.model, formatted_queries, query_spans, device=self.device,
            prefer_last_span=False,
        )
    elif self.config.mode == "instruction_span":
        query_emb = late_chunk_encode_with_grad(
            self.model, formatted_queries, query_spans, device=self.device,
            prefer_last_span=True,
        )
    else:
        raise ValueError(f"Unknown mode: {self.config.mode}")

    # Doc-side: apply wrapper's passage_prefix (defaults to "" via BaseEmbeddingModel)
    passage_prefix = getattr(self.model, "passage_prefix", "")
    if passage_prefix:
        positives = [passage_prefix + p for p in positives]

    pos_emb = self._encode_with_grad(positives)

    neg_emb = None
    has_negatives = negatives and len(negatives[0]) > 0
    if has_negatives:
        num_neg = len(negatives[0])
        flat_negatives = [neg for neg_list in negatives for neg in neg_list]
        if passage_prefix:
            flat_negatives = [passage_prefix + n for n in flat_negatives]
        neg_flat_emb = self._encode_with_grad(flat_negatives)
        neg_emb = neg_flat_emb.view(batch_size, num_neg, -1)

    return self.loss_fn(query_emb, pos_emb, neg_emb)
```

Add the import at the top of the file:

```python
from ..models.late_chunking import late_chunk_encode_with_grad
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py -v -k "compute_loss"`
Expected: 3 new tests PASS (or SKIP).

- [ ] **Step 6: Commit**

```bash
git add idiolink/trainer/contrastive_trainer.py tests/test_trainer.py
git commit -m "ContrastiveTrainer._compute_loss: per-mode dispatch + passage_prefix"
```

---

## Task 9: Rewrite `_evaluate` — use `encode_queries_for_mode`, delete `_STModelWrapper`

**Files:**
- Modify: `idiolink/trainer/contrastive_trainer.py`
- Test: `tests/test_trainer.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/test_trainer.py`:

```python
def test_evaluate_uses_encode_queries_for_mode():
    """_evaluate routes through encode_queries_for_mode (the same helper as
    zero-shot) — no hardcoded Instruct/Query wrapping.
    """
    from unittest.mock import patch
    import json
    from pathlib import Path

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="instruction_sentence", seed=42, device="cpu", max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    # Tiny fixtures
    tmp = Path(__file__).parent / "_tmp_eval_fixtures"
    tmp.mkdir(exist_ok=True)
    q_path = tmp / "queries.json"
    i_path = tmp / "indexes.json"
    q_path.write_text(json.dumps([
        {"sentence": "cat", "idiom": "cat", "usage": "literal", "span": "cat", "subject": ""},
    ]))
    i_path.write_text(json.dumps([
        {"sentence": "feline", "id": "d1", "idiom": "cat", "usage": "literal", "subject": ""},
    ]))

    import numpy as np
    with patch(
        "idiolink.trainer.contrastive_trainer.encode_queries_for_mode"
    ) as mock_eqfm:
        mock_eqfm.return_value = (["cat"], np.zeros((1, trainer.model.embedding_dim), dtype=np.float32))
        trainer._evaluate(str(q_path), str(i_path))
        mock_eqfm.assert_called_once()
        args = mock_eqfm.call_args.args
        assert args[1] == "instruction_sentence"


def test_st_model_wrapper_class_is_deleted():
    """The internal _STModelWrapper class is removed (replaced by direct wrapper use)."""
    import idiolink.trainer.contrastive_trainer as ct_mod
    assert not hasattr(ct_mod, "_STModelWrapper")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v -k "evaluate_uses_encode_queries or st_model_wrapper"`
Expected: FAIL.

- [ ] **Step 3: Rewrite `_evaluate` and delete `_STModelWrapper`**

In `idiolink/trainer/contrastive_trainer.py`:

1. Replace `_evaluate` (lines 115-171) with:

```python
def _evaluate(
    self,
    queries_file: str,
    indexes_file: str,
) -> Dict[str, Any]:
    """Run evaluation via the same per-mode encoding helper as zero-shot."""
    self.st_model.eval()
    retriever = DenseRetriever(self.model)

    query_sentences, idiom_queries = load_queries(queries_file)
    doc_sentences, doc_metadata = load_documents(indexes_file)

    retriever.index(doc_sentences, doc_metadata)
    query_texts, query_embeddings = encode_queries_for_mode(
        self.model, self.config.mode, idiom_queries, self.device,
    )

    results = retriever.retrieve(query_texts, top_k=100, query_embeddings=query_embeddings)
    # Re-key by original sentence in case query_texts differ from query_sentences
    # (instruction modes prepend prompts). DenseRetriever keys by the strings
    # passed to retrieve, not by the encoded form; the formatted strings ARE
    # what zero-shot uses too, so the keys here match the wrapped form.
    # For backward compatibility with evaluator API (which keys by q.query),
    # map back to plain sentences:
    if query_texts != query_sentences:
        results = {
            original: results[encoded]
            for original, encoded in zip(query_sentences, query_texts)
        }

    evaluator = Evaluator(idiom_queries, doc_metadata)
    metrics = evaluator.evaluate(results)
    self.st_model.train()
    return metrics
```

2. Delete the `_STModelWrapper` class entirely (currently at lines 285-299).

3. Add the import at the top of the file:

```python
from ..models.encode_helpers import encode_queries_for_mode
```

- [ ] **Step 4: Update the return type annotation**

Change `_evaluate`'s return annotation from `Dict[str, float]` to `Dict[str, Any]` (since `Evaluator.evaluate` now returns nested dicts under `by_usage`/`by_subject`).

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_trainer.py -v`
Expected: all tests PASS (or SKIP for ones needing real models).

- [ ] **Step 6: Run the full test suite for regression**

Run: `pytest tests/ -v`
Expected: no regressions in test count vs. main branch.

- [ ] **Step 7: Commit**

```bash
git add idiolink/trainer/contrastive_trainer.py tests/test_trainer.py
git commit -m "ContrastiveTrainer._evaluate: route through encode_queries_for_mode, delete _STModelWrapper"
```

---

## Task 10: Add `_trainer_version` stamp to saved metrics

**Files:**
- Modify: `idiolink/trainer/contrastive_trainer.py`
- Test: `tests/test_trainer.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/test_trainer.py`:

```python
def test_save_metrics_writes_trainer_version(tmp_path: Path):
    """Saved metrics.json contains _trainer_version stamp so matrix runner
    can invalidate pre-fix checkpoints."""
    import json
    from idiolink.trainer.contrastive_trainer import TRAINER_VERSION

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="sentence", seed=42, device="cpu", max_epochs=1,
        output_dir=str(tmp_path),
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")

    trainer.save_metrics({"r_precision": 0.5})
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["_trainer_version"] == TRAINER_VERSION
    assert saved["r_precision"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py::test_save_metrics_writes_trainer_version -v`
Expected: FAIL with `ImportError` or `KeyError`.

- [ ] **Step 3: Add `TRAINER_VERSION` and update `save_metrics`**

At the top of `idiolink/trainer/contrastive_trainer.py` (after the imports):

```python
TRAINER_VERSION = 2  # Bump when the trainer's encoding contract changes.
                     # v2: per-model wrapper + passage_prefix + late_chunk gradient flow.
```

Replace `save_metrics` (lines 277-282) with:

```python
def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
    """Save metrics to JSON file in output directory (atomic write + version stamp)."""
    from ..utils import atomic_write_json
    output_path = Path(self.config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamped = {**metrics, "_trainer_version": TRAINER_VERSION}
    atomic_write_json(output_path / filename, stamped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py::test_save_metrics_writes_trainer_version -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add idiolink/trainer/contrastive_trainer.py tests/test_trainer.py
git commit -m "Trainer: stamp metrics.json with _trainer_version, use atomic_write_json"
```

---

## Task 11: Per-(wrapper_class, mode) string-equivalence tests

**Files:**
- Modify: `tests/test_trainer.py` (add the test matrix)

- [ ] **Step 1: Add the test matrix**

Append to `tests/test_trainer.py`:

```python
# ---------------------------------------------------------------------------
# Per-(wrapper_class, mode) string equivalence matrix.
# Verifies trainer's training-time query strings match zero-shot's exactly.
# ---------------------------------------------------------------------------

WRAPPER_CASES = [
    # (wrapper_class_name, model_id_in_registry, expected_class)
    ("SentenceTransformerModel", "sentence-transformers/all-MiniLM-L6-v2", "SentenceTransformerModel"),
    ("InstructionModel",          "BAAI/bge-base-en-v1.5",                  "InstructionModel"),
    ("QwenModel",                 "Qwen/Qwen3-Embedding-0.6B",              "QwenModel"),
]

MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]


@pytest.mark.parametrize("class_name,model_id,_expected", WRAPPER_CASES)
@pytest.mark.parametrize("mode", MODES)
def test_train_query_strings_match_zero_shot(class_name, model_id, _expected, mode):
    """For every (wrapper_class, mode): the strings the trainer would tokenize
    for queries equal the strings zero-shot's encode_queries_for_mode would
    pass to the model. Asserted by intercepting at the formatter boundary.
    """
    from idiolink.utils import IdiomQuery
    from idiolink.models.encode_helpers import encode_queries_for_mode

    cfg = TrainingConfig(
        model_id=model_id, mode=mode, seed=42, device="cpu",
        max_epochs=1, batch_size=2,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError, ValueError) as e:
        pytest.skip(f"Model not available or unsupported: {e}")

    iqs = [
        IdiomQuery(query="the cat sat", idiom="cat", usage_type="literal",
                   span="cat", subject="x"),
        IdiomQuery(query="he kicked the bucket", idiom="kick the bucket",
                   usage_type="idiomatic", span="kicked the bucket", subject="death"),
    ]
    plain = [q.query for q in iqs]

    # Train-time formatting:
    train_strings = trainer._format_query_strings(plain, iqs)

    # Zero-shot formatting: intercept what would be passed downstream.
    if mode in ("sentence", "span"):
        # sentence: model.encode(plain); span: late_chunk_encode(model, plain, spans)
        expected = plain
    elif mode == "instruction_sentence":
        # encode_queries internally formats; we compare the formatter output
        from idiolink.models.instruction_model import resolve_instructions
        instructions = resolve_instructions(model_id, iqs)
        if hasattr(trainer.model, "format_queries_for_late_chunking"):
            expected = trainer.model.format_queries_for_late_chunking(plain, instructions)
        else:
            expected = plain
    elif mode == "instruction_span":
        from idiolink.models.instruction_model import resolve_instructions
        instructions = resolve_instructions(model_id, iqs)
        expected = trainer.model.format_queries_for_late_chunking(plain, instructions)
    else:
        raise AssertionError(f"unhandled mode {mode}")

    assert train_strings == expected, (
        f"\n  class={class_name} mode={mode}\n"
        f"  train:    {train_strings}\n"
        f"  expected: {expected}\n"
    )
```

- [ ] **Step 2: Run the test matrix**

Run: `pytest tests/test_trainer.py -v -k "train_query_strings_match_zero_shot"`
Expected: 12 tests, all PASS or SKIP (skips OK when models aren't cached locally).

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer.py
git commit -m "Add per-(wrapper_class, mode) string-equivalence test matrix"
```

---

## Task 12: End-to-end smoke test on MiniLM

**Files:**
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Add the smoke test**

Append to `tests/test_trainer.py`:

```python
def test_end_to_end_one_step_minilm_instruction_sentence(tmp_path: Path):
    """One full train step on MiniLM, instruction_sentence mode. Asserts no
    exception, finite loss, gradients populated. Slow (~30s with model load).
    """
    import json
    from pathlib import Path as _P
    from torch.utils.data import DataLoader

    from idiolink.trainer import TripletDataset
    from idiolink.trainer.contrastive_trainer import collate_triplets

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="instruction_sentence",
        seed=42, device="cpu",
        max_epochs=1, batch_size=2, max_negatives=1,
        output_dir=str(tmp_path),
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"MiniLM not available: {e}")

    triplet_path = tmp_path / "tri.jsonl"
    triplets = [
        {"query": "the cat sat", "query_span": "cat", "query_idiom": "cat",
         "query_usage": "literal", "query_subject": "",
         "positive": "feline rest", "negatives": ["he kicked the bucket"]},
        {"query": "she dropped the ball", "query_span": "dropped the ball",
         "query_idiom": "drop the ball", "query_usage": "idiomatic", "query_subject": "",
         "positive": "she failed at it", "negatives": ["the ball fell"]},
    ]
    with open(triplet_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    ds = TripletDataset(str(triplet_path), max_negatives=1)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_triplets)

    trainer.st_model.train()
    batch = next(iter(loader))
    loss = trainer._compute_loss(batch)

    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    # At least one parameter must have a non-None grad
    had_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in trainer.st_model.parameters() if p.requires_grad)
    assert had_grad, "No gradients flowed back to any model parameter"
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_trainer.py::test_end_to_end_one_step_minilm_instruction_sentence -v`
Expected: PASS (or SKIP if MiniLM is not cached and offline).

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer.py
git commit -m "Add end-to-end smoke test: 1 trainer step on MiniLM, instruction_sentence"
```

---

## Task 13: `run_fine_tune_matrix.py` — matrix runner with resume

**Files:**
- Create: `run_fine_tune_matrix.py`
- Test: `tests/test_fine_tune_matrix.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_fine_tune_matrix.py`:

```python
"""Tests for run_fine_tune_matrix.py: resume + force + aggregate CSV."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_config(tmp_path: Path, train_dir: Path) -> Path:
    cfg = {
        "device": "cpu",
        "seed": 42,
        "results_dir": str(tmp_path / "results"),
        "data": {
            "train_dir": str(train_dir),
            "val_dir": str(train_dir),
            "test_dir": str(train_dir),
        },
        "training": {
            "models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "modes": ["sentence"],
            "seeds": [42, 43],
            "batch_size": 4,
            "max_epochs": 1,
            "learning_rate": 2e-5,
            "warmup_steps": 0,
            "temperature": 0.05,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
        },
        "retrieval": {"top_k": 10},
    }
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _write_data(train_dir: Path):
    """Write minimal triplets + queries + indexes for the runner."""
    train_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (train_dir / f"triplets_{split}_full.jsonl").write_text(
            json.dumps({"query": "x", "positive": "y", "negatives": ["z"]}) + "\n"
        )
    (train_dir / "queries.json").write_text(json.dumps([
        {"sentence": "x", "idiom": "x", "usage": "literal", "span": "x", "subject": ""},
    ]))
    (train_dir / "indexes.json").write_text(json.dumps([
        {"sentence": "y", "id": "d1", "idiom": "x", "usage": "literal", "subject": ""},
    ]))


def test_matrix_runner_writes_per_cell_metrics_and_aggregate(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner
    with patch.object(runner, "run_single_seed", return_value={"r_precision": 0.5, "ndcg@10": 0.6}):
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()

    results = Path(json.loads(cfg_path.read_text() if cfg_path.suffix == ".json"
                              else __import__("yaml").safe_dump(__import__("yaml").safe_load(cfg_path.read_text()))) or "")
    # Easier: read tmp_path / "results"
    rd = tmp_path / "results" / "fine_tuning"
    # 1 model x 1 mode x 2 seeds = 2 metrics.json
    files = list(rd.rglob("metrics.json"))
    assert len(files) == 2
    agg = rd / "full_results.csv"
    assert agg.exists()
    rows = list(csv.DictReader(open(agg)))
    assert len(rows) == 2


def test_matrix_runner_skips_existing_metrics(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner

    # First invocation: runs both seeds.
    with patch.object(runner, "run_single_seed", return_value={"r_precision": 0.5, "ndcg@10": 0.6}) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()
        assert mock_run.call_count == 2

    # Second invocation: should skip both (metrics.json exists with current TRAINER_VERSION)
    with patch.object(runner, "run_single_seed", return_value={"r_precision": 0.5, "ndcg@10": 0.6}) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()
        assert mock_run.call_count == 0


def test_matrix_runner_force_recomputes(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner

    with patch.object(runner, "run_single_seed", return_value={"r_precision": 0.5, "ndcg@10": 0.6}):
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()

    with patch.object(runner, "run_single_seed", return_value={"r_precision": 0.7, "ndcg@10": 0.8}) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path), "--force"]
        runner.main()
        assert mock_run.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fine_tune_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_fine_tune_matrix'`.

- [ ] **Step 3: Create `run_fine_tune_matrix.py`**

Create `/Users/dani/Documents/my-research/26-may-empnlp-idiolink/IdioLink/run_fine_tune_matrix.py`:

```python
"""Fine-tuning matrix runner: models × modes × seeds with resume + aggregate CSV.

Mirror of run_ablation.py for the training side. Resumes by checking that
per-(model, mode, seed) metrics.json exists with a current _trainer_version
stamp. Per-model batch_size pulled from registry unless overridden.

Usage:
    python run_fine_tune_matrix.py                       # full matrix from config
    python run_fine_tune_matrix.py --models <id> ...
    python run_fine_tune_matrix.py --modes sentence span
    python run_fine_tune_matrix.py --seeds 42 43 44
    python run_fine_tune_matrix.py --force               # recompute all
    python run_fine_tune_matrix.py --dry_run             # print matrix, exit
"""

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from idiolink.trainer import ContrastiveTrainer, TrainingConfig, TripletDataset
from idiolink.trainer.contrastive_trainer import TRAINER_VERSION
from idiolink.models.registry import MODEL_REGISTRY
from idiolink.utils import atomic_write_json, load_config, model_slug, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CSV_FIELDS = [
    "model", "mode", "seed",
    "r_precision", "ndcg@10",
    "num_queries",
    "_trainer_version",
]


def _metrics_path(results_dir: Path, model_id: str, mode: str, seed: int) -> Path:
    return results_dir / "fine_tuning" / model_slug(model_id) / mode / f"seed_{seed}" / "metrics.json"


def _is_complete(path: Path) -> bool:
    """True iff metrics.json exists AND has the current _trainer_version."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return data.get("_trainer_version") == TRAINER_VERSION


def _resolve_batch_size(args_batch_size, train_cfg, model_id) -> Optional[int]:
    """CLI > config > registry (resolved in trainer)."""
    if args_batch_size is not None:
        return args_batch_size
    if train_cfg.get("batch_size") is not None:
        return train_cfg.get("batch_size")
    # None lets ContrastiveTrainer pull from MODEL_REGISTRY[model_id].batch_size
    return None


def get_triplet_file(data_dir: str, mode: str, split: str) -> str:
    suffix = "span" if "span" in mode else "full"
    return str(Path(data_dir) / f"triplets_{split}_{suffix}.jsonl")


def run_single_seed(
    config: TrainingConfig,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mode: str,
) -> Dict:
    """Train and evaluate for a single seed. Returns test metrics dict."""
    set_seed(config.seed)

    triplet_file = get_triplet_file(train_dir, mode, "train")
    logger.info(f"Loading training triplets from: {triplet_file}")
    train_dataset = TripletDataset(triplet_file, max_negatives=config.max_negatives)
    logger.info(f"Training samples: {len(train_dataset)}")

    val_queries = str(Path(val_dir) / "queries.json")
    val_indexes = str(Path(val_dir) / "indexes.json")
    test_queries = str(Path(test_dir) / "queries.json")
    test_indexes = str(Path(test_dir) / "indexes.json")

    trainer = ContrastiveTrainer(config)
    logger.info(f"Training model: {config.model_id} | seed: {config.seed} | mode: {mode}")
    val_metrics = trainer.train(train_dataset, val_queries, val_indexes)
    logger.info(f"Best val metrics: {val_metrics}")

    test_metrics = trainer.evaluate_test(test_queries, test_indexes)
    logger.info(f"Test metrics: {test_metrics}")

    trainer.save_metrics(test_metrics)
    return test_metrics


def _flatten_for_csv(test_metrics: Dict, model_id: str, mode: str, seed: int) -> Dict:
    return {
        "model": model_id,
        "mode": mode,
        "seed": seed,
        "r_precision": test_metrics.get("r_precision", 0.0),
        "ndcg@10": test_metrics.get("ndcg@10", 0.0),
        "num_queries": test_metrics.get("num_queries", 0),
        "_trainer_version": TRAINER_VERSION,
    }


def _collect_all_rows_from_disk(
    results_dir: Path,
    models: List[str],
    modes: List[str],
    seeds: List[int],
) -> List[dict]:
    """Walk results/fine_tuning/ and rebuild rows from every metrics.json."""
    out: List[dict] = []
    for model_id in models:
        for mode in modes:
            for seed in seeds:
                mp = _metrics_path(results_dir, model_id, mode, seed)
                if not mp.exists():
                    continue
                try:
                    metrics = json.loads(mp.read_text())
                    out.append(_flatten_for_csv(metrics, model_id, mode, seed))
                except Exception as e:
                    logger.warning(f"Could not read {mp}: {e}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning matrix (models x modes x seeds)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Default: cfg['training']['models']")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=["sentence", "span", "instruction_sentence", "instruction_span"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute (model, mode, seed) cells even if metrics.json exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the matrix and exit without training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    results_dir = Path(cfg["results_dir"])

    models = args.models or train_cfg["models"]
    modes = args.modes or train_cfg["modes"]
    seeds = args.seeds or train_cfg["seeds"]

    logger.info(f"Matrix: {len(models)} models x {len(modes)} modes x {len(seeds)} seeds "
                f"= {len(models)*len(modes)*len(seeds)} cells")
    logger.info(f"Models: {models}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Seeds: {seeds}")

    if args.dry_run:
        for m in models:
            for mo in modes:
                for s in seeds:
                    mp = _metrics_path(results_dir, m, mo, s)
                    status = "DONE" if _is_complete(mp) else "PENDING"
                    logger.info(f"  [{status}] {m} / {mo} / seed={s} -> {mp}")
        return

    failed: List[tuple] = []
    for model_id in models:
        for mode in modes:
            for seed in seeds:
                mp = _metrics_path(results_dir, model_id, mode, seed)
                if not args.force and _is_complete(mp):
                    logger.info(f"  SKIP existing: {model_id} / {mode} / seed={seed}")
                    continue

                output_dir = str(mp.parent)
                training_config = TrainingConfig(
                    model_id=model_id,
                    batch_size=_resolve_batch_size(args.batch_size, train_cfg, model_id),
                    lr=train_cfg.get("learning_rate", 2e-5),
                    max_epochs=train_cfg.get("max_epochs", 10),
                    warmup_steps=train_cfg.get("warmup_steps", 100),
                    temperature=train_cfg.get("temperature", 0.05),
                    early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
                    early_stopping_metric=train_cfg.get("early_stopping_metric", "ndcg@10"),
                    output_dir=output_dir,
                    seed=seed,
                    device=cfg.get("device", "auto"),
                    mode=mode,
                )
                try:
                    run_single_seed(
                        training_config,
                        data_cfg["train_dir"],
                        data_cfg["val_dir"],
                        data_cfg["test_dir"],
                        mode,
                    )
                except Exception as e:
                    logger.error(f"FAILED {model_id} / {mode} / seed={seed}: {e}")
                    traceback.print_exc()
                    failed.append((model_id, mode, seed))

    # Rebuild aggregate CSV from every metrics.json on disk
    rows = _collect_all_rows_from_disk(results_dir, models, modes, seeds)
    if rows:
        agg_path = results_dir / "fine_tuning" / "full_results.csv"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Aggregated results saved to {agg_path} ({len(rows)} rows)")
    else:
        logger.warning("No fine-tuning results to aggregate.")

    if failed:
        logger.error(f"\n{len(failed)} cells failed:")
        for m, mo, s in failed:
            logger.error(f"  {m} / {mo} / seed={s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_fine_tune_matrix.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add run_fine_tune_matrix.py tests/test_fine_tune_matrix.py
git commit -m "Add run_fine_tune_matrix.py: models x modes x seeds with resume + force"
```

---

## Task 14: README update — matrix runner usage + GritLM note

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the fine-tuning section**

```bash
grep -n -i "fine.tun\|fine.tun" /Users/dani/Documents/my-research/26-may-empnlp-idiolink/IdioLink/README.md | head -20
```

- [ ] **Step 2: Update the fine-tuning section**

Replace the existing fine-tuning usage block in `README.md` with:

```markdown
## Fine-tuning

### Single run (debugging)

```bash
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode sentence --seeds 42
```

### Full matrix (5 models × 4 modes × 3 seeds = 60 runs)

```bash
python run_fine_tune_matrix.py                       # everything from config.yaml
python run_fine_tune_matrix.py --models <id> ...     # subset of models
python run_fine_tune_matrix.py --modes sentence span # subset of modes
python run_fine_tune_matrix.py --seeds 42            # subset of seeds
python run_fine_tune_matrix.py --dry_run             # see what would run
python run_fine_tune_matrix.py --force               # recompute already-done cells
```

Resumability: each `(model, mode, seed)` cell writes `results/fine_tuning/<slug>/<mode>/seed_<n>/metrics.json` with a `_trainer_version` stamp. Re-running skips cells whose metrics.json exists with the current trainer version; bump `TRAINER_VERSION` in `idiolink/trainer/contrastive_trainer.py` to invalidate stale checkpoints.

Batch size: pulled from `MODEL_REGISTRY[model_id].batch_size` by default (overrideable via `--batch_size` or `training.batch_size` in config.yaml). Per-model defaults prevent OOM on large models.

### Models supported for fine-tuning

The trainer supports `sentence_transformer`, `instruction`, and `qwen` wrapper classes. **GritLM is zero-shot only** — the trainer raises an error if asked to fine-tune a `gritlm`-class model. (GritLM-7B is therefore excluded from `config.yaml::training.models` by default.)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: document matrix runner usage and GritLM training restriction"
```

---

## Task 15: Final verification — full test suite + smoke-run the matrix runner

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v 2>&1 | tail -40`
Expected: all tests PASS or SKIP. Note any new failures introduced (compare to pre-Task-1 baseline).

- [ ] **Step 2: Dry-run the matrix runner**

Run: `python run_fine_tune_matrix.py --dry_run 2>&1 | tail -30`
Expected: prints "60 cells" matrix; each row labeled PENDING (since no results yet). No exceptions.

- [ ] **Step 3: Smoke-run one cell**

Run: `python run_fine_tune_matrix.py --models sentence-transformers/all-MiniLM-L6-v2 --modes sentence --seeds 42 2>&1 | tail -20`
Expected: completes one training run, writes `results/fine_tuning/sentence-transformers__all-MiniLM-L6-v2/sentence/seed_42/metrics.json` with `_trainer_version: 2`. `results/fine_tuning/full_results.csv` has 1 row.

- [ ] **Step 4: Verify resume works**

Run the same command again: `python run_fine_tune_matrix.py --models sentence-transformers/all-MiniLM-L6-v2 --modes sentence --seeds 42 2>&1 | tail -10`
Expected: logs `SKIP existing: sentence-transformers/all-MiniLM-L6-v2 / sentence / seed=42`. No re-training.

- [ ] **Step 5: Verify --force re-runs**

Run: `python run_fine_tune_matrix.py --models sentence-transformers/all-MiniLM-L6-v2 --modes sentence --seeds 42 --force 2>&1 | tail -10`
Expected: re-runs the training. metrics.json is rewritten.

- [ ] **Step 6: Commit final verification artifacts (if any new ones beyond results/)**

If `results/` is gitignored (it is in most projects), nothing to commit. If anything else was incidentally modified, review with `git status` and commit if relevant.

```bash
git status
# If there are unstaged changes that are part of this work, stage and commit them.
# Otherwise nothing to do.
```

---

## Self-review checklist

After all 15 tasks complete:

- [ ] **Spec coverage:**
  - Component 1 (encode_helpers) → Task 3 ✓
  - Component 2 (late_chunk_encode_with_grad) → Task 4 ✓
  - Component 3 (passage_prefix default) → Task 2 ✓
  - Component 4 (gritlm guard) → Task 6 ✓
  - Component 5 (TripletDataset strip) → Task 5 ✓
  - Component 6 (trainer rewrite) → Tasks 6, 7, 8, 9 ✓
  - Component 7 (run_fine_tune.py batch_size) → Task 6 step 5 ✓
  - Component 8 (matrix runner) → Task 13 ✓
  - atomic_write_json shared → Task 1 ✓
  - _trainer_version → Task 10 ✓
  - Per-(class, mode) tests → Task 11 ✓
  - Smoke test → Task 12 ✓
  - README → Task 14 ✓

- [ ] **No placeholders** — every code block is concrete; no "implement X here".

- [ ] **Type consistency** — `TrainingConfig.batch_size: Optional[int]` everywhere; `Dict[str, Any]` for the new evaluator return; `_trainer_version` (int) consistent across save_metrics and matrix runner.

- [ ] **Out-of-scope items NOT touched** — confirm no edits to `SentenceTransformerModel.__init__` (e5 prefix gap), `InstructionFormat.NOMIC_PREFIX` (instruction discard), `_atomic_write_json` body changes (fsync). All three remain on the follow-up tracker (Task #7 in the session task list).
