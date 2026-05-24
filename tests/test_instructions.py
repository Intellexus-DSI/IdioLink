"""Tests for the per-model instruction resolver."""

import pytest

from idiolink.models.instruction_model import (
    DEFAULT_INSTRUCTION_TEMPLATE,
    resolve_instruction,
    resolve_instructions,
)
from idiolink.models.registry import MODEL_REGISTRY, ModelConfig
from idiolink.utils import IdiomQuery


@pytest.fixture
def sample_query():
    return IdiomQuery(
        query="The team broke the ice at the meeting.",
        idiom="break the ice",
        usage_type="idiomatic",
        span="break the ice",
        subject="Politics",
    )


@pytest.fixture(autouse=True)
def isolate_registry():
    """Snapshot/restore MODEL_REGISTRY around each test so registrations don't leak."""
    saved = dict(MODEL_REGISTRY)
    yield
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(saved)


class TestResolveInstruction:
    def test_unknown_model_falls_back_to_default(self, sample_query):
        got = resolve_instruction("nonexistent/model", sample_query)
        assert got == DEFAULT_INSTRUCTION_TEMPLATE.format(span=sample_query.span)

    def test_model_without_overrides_uses_default(self, sample_query):
        got = resolve_instruction("sentence-transformers/all-MiniLM-L6-v2", sample_query)
        assert got == DEFAULT_INSTRUCTION_TEMPLATE.format(span=sample_query.span)

    def test_static_template_is_formatted_with_query_fields(self, sample_query):
        MODEL_REGISTRY["dummy/static"] = ModelConfig(
            model_id="dummy/static",
            model_class="sentence_transformer",
            size_params="0",
            instruction_text="Retrieve docs about '{span}' (idiom: {idiom}, subject: {subject}).",
        )
        got = resolve_instruction("dummy/static", sample_query)
        assert got == (
            "Retrieve docs about 'break the ice' "
            "(idiom: break the ice, subject: Politics)."
        )

    def test_static_template_without_placeholders_is_returned_verbatim(self, sample_query):
        MODEL_REGISTRY["dummy/fixed"] = ModelConfig(
            model_id="dummy/fixed",
            model_class="sentence_transformer",
            size_params="0",
            instruction_text="Represent this sentence for searching relevant passages: ",
        )
        got = resolve_instruction("dummy/fixed", sample_query)
        assert got == "Represent this sentence for searching relevant passages: "

    def test_builder_function_is_called_with_query(self, sample_query):
        captured = []

        def builder(q):
            captured.append(q)
            return f"<dynamic for {q.idiom} / {q.usage_type}>"

        MODEL_REGISTRY["dummy/dynamic"] = ModelConfig(
            model_id="dummy/dynamic",
            model_class="sentence_transformer",
            size_params="0",
            instruction_fn=builder,
        )
        got = resolve_instruction("dummy/dynamic", sample_query)
        assert got == "<dynamic for break the ice / idiomatic>"
        assert captured == [sample_query]

    def test_setting_both_text_and_fn_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="instruction_text.*instruction_fn"):
            ModelConfig(
                model_id="dummy/both",
                model_class="sentence_transformer",
                size_params="0",
                instruction_text="some text",
                instruction_fn=lambda q: "also a fn",
            )

    def test_unknown_placeholder_in_template_raises_clearly(self, sample_query):
        MODEL_REGISTRY["dummy/bad"] = ModelConfig(
            model_id="dummy/bad",
            model_class="sentence_transformer",
            size_params="0",
            instruction_text="Refers to {nonexistent_field}.",
        )
        with pytest.raises(ValueError, match="dummy/bad.*nonexistent_field"):
            resolve_instruction("dummy/bad", sample_query)


class TestResolveInstructionsBatch:
    def test_batch_returns_one_per_query(self, sample_query):
        qs = [sample_query, sample_query]
        got = resolve_instructions("sentence-transformers/all-MiniLM-L6-v2", qs)
        assert len(got) == 2
        assert got[0] == got[1]


class TestRegistryOverrides:
    def test_bge_base_en_v1_5_has_canonical_prompt(self, sample_query):
        """BGE-base-en-v1.5 should use its trained query-prefix, not generic Instruct/Query wrapping."""
        cfg = MODEL_REGISTRY["BAAI/bge-base-en-v1.5"]
        assert cfg.instruction_text is not None
        assert "Represent this sentence" in cfg.instruction_text
