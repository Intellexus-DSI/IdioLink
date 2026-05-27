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
