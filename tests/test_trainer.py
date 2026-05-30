"""Tests for ContrastiveTrainer's wiring to load_model + registry."""

from pathlib import Path

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


def test_init_raises_for_instructor_pairs_in_instruction_modes():
    """instructor_pairs models can't be byte-equivalent to zero-shot in
    instruction modes (zero-shot passes list-of-pairs to ST.encode, the
    training path can only feed concatenated strings). Fail fast.
    """
    for mode in ("instruction_sentence", "instruction_span"):
        cfg = TrainingConfig(
            model_id="hkunlp/instructor-base",
            mode=mode, seed=42, device="cpu", max_epochs=1,
        )
        with pytest.raises(ValueError, match="instructor_pairs"):
            ContrastiveTrainer(cfg)


def test_init_allows_instructor_pairs_in_non_instruction_modes():
    """sentence/span modes don't use instructions, so instructor_pairs models
    are still trainable in those modes.
    """
    cfg = TrainingConfig(
        model_id="hkunlp/instructor-base",
        mode="sentence", seed=42, device="cpu", max_epochs=1,
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"Model not available: {e}")
    assert trainer.model.model_id == "hkunlp/instructor-base"


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


def test_instruction_sentence_prompt_models_use_prompt_api_for_queries():
    """Prompt-prefix models must train through ST's prompt path, not by
    concatenating prompt text into the query string. Qwen/BGE pooling can depend
    on prompt metadata that only SentenceTransformer.preprocess(prompt=...)
    preserves.
    """
    from types import SimpleNamespace

    from idiolink.utils import IdiomQuery

    queries = ["the cat sat", "he kicked the bucket"]
    iqs = [
        IdiomQuery(query=queries[0], idiom="cat", usage_type="literal",
                   span="cat", subject=""),
        IdiomQuery(query=queries[1], idiom="kick the bucket",
                   usage_type="idiomatic", span="kicked the bucket", subject=""),
    ]

    cases = [
        (
            "Qwen/Qwen3-Embedding-0.6B",
            SimpleNamespace(
                _query_prompt=lambda instruction: f"Instruct: {instruction}\nQuery:"
            ),
        ),
        (
            "BAAI/bge-base-en-v1.5",
            SimpleNamespace(instruction_format=SimpleNamespace(value="prompt_prefix")),
        ),
    ]

    for model_id, fake_wrapper in cases:
        trainer = object.__new__(ContrastiveTrainer)
        trainer.config = TrainingConfig(
            model_id=model_id,
            mode="instruction_sentence",
            seed=42,
            device="cpu",
            max_epochs=1,
            batch_size=2,
        )
        trainer.model = fake_wrapper

        captured = []

        def encode_with_prompt(texts, prompt):
            captured.append((list(texts), prompt))
            return torch.ones((len(texts), 2), requires_grad=True)

        def encode_without_prompt(_texts):
            raise AssertionError(f"{model_id} fell back to concatenated query strings")

        trainer._encode_with_prompt_grad = encode_with_prompt
        trainer._encode_with_grad = encode_without_prompt

        embeddings = trainer._encode_instruction_sentence_with_grad(queries, iqs)

        assert embeddings.shape == (2, 2)
        assert [text for texts, _ in captured for text in texts] == queries
        assert all("Instruct:" not in text for texts, _ in captured for text in texts)
        assert all("Query:" not in text for texts, _ in captured for text in texts)
        assert all(prompt for _, prompt in captured)
        if model_id.startswith("Qwen/"):
            assert all(prompt.startswith("Instruct: ") for _, prompt in captured)
            assert all(prompt.endswith("\nQuery:") for _, prompt in captured)
        else:
            assert captured == [
                (queries, "Represent this sentence for searching relevant passages: ")
            ]


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


def test_save_metrics_writes_exact_metrics(tmp_path: Path):
    """Saved metrics.json contains the provided metrics without metadata stamps."""
    import json

    trainer = object.__new__(ContrastiveTrainer)
    trainer.config = TrainingConfig(output_dir=str(tmp_path))

    trainer.save_metrics({"r_precision": 0.5})
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == {"r_precision": 0.5}


class _TinySentenceTransformer(torch.nn.Module):
    """Small differentiable stand-in for SentenceTransformer sanity checks."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, dim)
        self.tokenize_calls = []
        self.preprocess_calls = []

    def _ids(self, texts):
        rows = []
        for text in texts:
            values = [(ord(ch) % 255) + 1 for ch in text[:12]]
            values = values or [0]
            rows.append(values + [0] * (12 - len(values)))
        return torch.tensor(rows, dtype=torch.long)

    def tokenize(self, texts):
        self.tokenize_calls.append(list(texts))
        return {"input_ids": self._ids(texts)}

    def preprocess(self, texts, prompt):
        self.preprocess_calls.append((list(texts), prompt))
        features = self.tokenize(texts)
        features["prompt_ids"] = self._ids([prompt] * len(texts))
        features["prompt_text"] = prompt
        return features

    def forward(self, features):
        token_emb = self.embedding(features["input_ids"]).mean(dim=1)
        prompt_ids = features.get("prompt_ids")
        if prompt_ids is not None:
            token_emb = token_emb + 0.1 * self.embedding(prompt_ids).mean(dim=1)
        return {"sentence_embedding": token_emb}


def _make_tiny_trainer(model_id: str, mode: str, wrapper_attrs: dict):
    from types import SimpleNamespace
    from idiolink.trainer.losses import InfoNCELoss

    st_model = _TinySentenceTransformer()
    attrs = {
        "model": st_model,
        "model_id": model_id,
        "passage_prefix": "",
        "format_queries_for_late_chunking": lambda texts, instructions: [
            f"Instruct: {inst}\nQuery: {text}"
            for text, inst in zip(
                texts,
                instructions if isinstance(instructions, list)
                else [instructions] * len(texts),
            )
        ],
    }
    attrs.update(wrapper_attrs)
    wrapper = SimpleNamespace(**attrs)

    trainer = object.__new__(ContrastiveTrainer)
    trainer.config = TrainingConfig(
        model_id=model_id,
        mode=mode,
        seed=42,
        device="cpu",
        batch_size=2,
        max_epochs=1,
    )
    trainer.device = torch.device("cpu")
    trainer.model = wrapper
    trainer.st_model = st_model
    trainer.loss_fn = InfoNCELoss(temperature=0.2)
    return trainer


@pytest.mark.parametrize(
    "name,model_id,mode,wrapper_attrs,expect_prompt,expect_passage_prefix",
    [
        (
            "plain_sentence_transformer",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence",
            {},
            False,
            "",
        ),
        (
            "e5_passage_prefix",
            "intfloat/e5-base-v2",
            "sentence",
            {"passage_prefix": "passage: "},
            False,
            "passage: ",
        ),
        (
            "bge_prompt_prefix",
            "BAAI/bge-base-en-v1.5",
            "instruction_sentence",
            {"instruction_format": type("Fmt", (), {"value": "prompt_prefix"})()},
            True,
            "",
        ),
        (
            "qwen_prompt_method",
            "Qwen/Qwen3-Embedding-0.6B",
            "instruction_sentence",
            {"_query_prompt": lambda instruction: f"Instruct: {instruction}\nQuery:"},
            True,
            "",
        ),
    ],
)
def test_sanity_optimizer_step_for_small_wrapper_contracts(
    name,
    model_id,
    mode,
    wrapper_attrs,
    expect_prompt,
    expect_passage_prefix,
):
    """Run one optimizer-backed training step across representative wrappers."""
    trainer = _make_tiny_trainer(model_id, mode, wrapper_attrs)
    optimizer = torch.optim.AdamW(trainer.st_model.parameters(), lr=1e-2)

    batch = {
        "queries": ["the cat sat", "she dropped the ball"],
        "query_spans": ["cat", "dropped the ball"],
        "query_idioms": ["cat", "drop the ball"],
        "query_usages": ["literal", "idiomatic"],
        "query_subjects": ["animal", "failure"],
        "positives": ["a feline rested", "she made a mistake"],
        "negatives": [["the bucket fell"], ["an unrelated sentence"]],
    }

    before = [p.detach().clone() for p in trainer.st_model.parameters()]
    optimizer.zero_grad()
    loss = trainer._compute_loss(batch)
    assert torch.isfinite(loss), f"{name} produced a non-finite loss"
    loss.backward()
    optimizer.step()

    after = list(trainer.st_model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(after, before)), (
        f"{name} did not update any parameters"
    )

    if expect_prompt:
        assert trainer.st_model.preprocess_calls, f"{name} did not use prompt preprocessing"
        query_texts = [text for texts, _ in trainer.st_model.preprocess_calls for text in texts]
        assert query_texts == batch["queries"]
        assert all("Instruct:" not in text and "Query:" not in text for text in query_texts)
    else:
        assert not trainer.st_model.preprocess_calls

    if expect_passage_prefix:
        encoded_texts = [text for call in trainer.st_model.tokenize_calls for text in call]
        docs = batch["positives"] + [neg for negs in batch["negatives"] for neg in negs]
        for doc in docs:
            assert expect_passage_prefix + doc in encoded_texts


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


def test_evaluate_test_uses_reloaded_best_model(tmp_path: Path):
    """evaluate_test must route encoding through the RELOADED best-model
    weights, not the pre-trained model. After T9 deleted _STModelWrapper,
    rebinding self.st_model is not enough — self.model.model also needs the
    swap, since the wrapper holds its own reference.
    """
    from unittest.mock import patch, MagicMock

    cfg = TrainingConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        mode="sentence", seed=42, device="cpu", max_epochs=1, batch_size=2,
        output_dir=str(tmp_path),
    )
    try:
        trainer = ContrastiveTrainer(cfg)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"MiniLM not available: {e}")

    # Create a fake "best model" directory; SentenceTransformer load will be patched
    best_dir = tmp_path / "best_model"
    best_dir.mkdir()
    # Just need it to exist; we patch the actual SentenceTransformer constructor
    (best_dir / "config.json").write_text("{}")

    fake_reloaded = MagicMock(name="reloaded_ST")

    # Capture wrapper.model.model at the moment _evaluate would run. The point
    # of the regression test is: by the time _evaluate fires, the wrapper must
    # already be pointing at the reloaded instance — otherwise encoding routes
    # through the pre-trained weights.
    observed: dict = {}

    def fake_evaluate(_q, _i):
        observed["st_model"] = trainer.st_model
        observed["wrapper_inner"] = trainer.model.model
        return {"ndcg@10": 0.0}

    with patch(
        "idiolink.trainer.contrastive_trainer.SentenceTransformer",
        return_value=fake_reloaded,
    ):
        with patch.object(trainer, "_evaluate", side_effect=fake_evaluate):
            trainer.evaluate_test("dummy_q.json", "dummy_i.json")

    # After evaluate_test triggered _evaluate: BOTH the trainer's st_model AND
    # the wrapper's internal model attribute should point to the reloaded
    # instance.
    assert observed["st_model"] is fake_reloaded, "trainer.st_model not swapped"
    assert observed["wrapper_inner"] is fake_reloaded, (
        "wrapper.model.model not swapped — evaluate_test would silently use "
        "the pre-trained model"
    )


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
