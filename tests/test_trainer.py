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
