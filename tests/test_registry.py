"""Tests for model registry configuration and factory."""

import pytest
from idiolink.models.registry import MODEL_REGISTRY, ModelConfig, load_model


EXPECTED_MODEL_COUNT = 24

VALID_MODEL_CLASSES = {"sentence_transformer", "instruction", "gritlm", "qwen"}
VALID_INSTRUCTION_FORMATS = {
    "e5_inline", "e5_inline_no_space", "instructor_pairs",
    "tart_sep", "nomic_prefix", "bge_gemma", "prompt_prefix", None,
}


class TestRegistryCompleteness:
    def test_registry_has_24_models(self):
        assert len(MODEL_REGISTRY) == EXPECTED_MODEL_COUNT

    def test_all_known_models_present(self):
        expected_ids = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "facebook/contriever",
            "intfloat/e5-base-v2",
            "orionweller/tart-dual-contriever-msmarco",
            "BAAI/bge-base-en-v1.5",
            "hkunlp/instructor-base",
            "nomic-ai/nomic-embed-text-v2-moe",
            "intfloat/multilingual-e5-large-instruct",
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-0.6B",
            "facebook/drama-1b",
            "NovaSearch/stella-en-1.5B-v5",
            "hkunlp/instructor-xl",
            "vec-ai/lychee-embed",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "Qwen/Qwen3-Embedding-4B",
            "Linq-AI-Research/Linq-Embed-Mistral",
            "Salesforce/SFR-Embedding-Mistral",
            "intfloat/e5-mistral-7b-instruct",
            "GritLM/GritLM-7B",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "Qwen/Qwen3-Embedding-8B",
            "nvidia/llama-embed-nemotron-8b",
            "BAAI/bge-multilingual-gemma2",
        ]
        for model_id in expected_ids:
            assert model_id in MODEL_REGISTRY, f"Missing model: {model_id}"


class TestModelConfigValidation:
    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_config_is_model_config(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert isinstance(config, ModelConfig)

    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_model_class_valid(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert config.model_class in VALID_MODEL_CLASSES

    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_instruction_format_valid(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert config.instruction_format in VALID_INSTRUCTION_FORMATS

    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_model_id_matches_key(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert config.model_id == model_id

    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_batch_size_positive(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert config.batch_size > 0

    @pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
    def test_size_params_not_empty(self, model_id):
        config = MODEL_REGISTRY[model_id]
        assert config.size_params != ""


class TestLoadModel:
    def test_load_sentence_transformer_returns_correct_class(self):
        import sys
        from unittest.mock import patch, MagicMock

        # Mock the sentence_transformers module to avoid transformers dependency
        mock_st_module = MagicMock()
        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_st_instance
        mock_st_module.SentenceTransformer = mock_st_class

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            # Force reimport
            if "idiolink.models.sentence_transformer" in sys.modules:
                del sys.modules["idiolink.models.sentence_transformer"]
            from idiolink.models.sentence_transformer import SentenceTransformerModel
            model = load_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            assert isinstance(model, SentenceTransformerModel)
            assert hasattr(model, "encode_queries")
            assert hasattr(model, "format_queries_for_late_chunking")

    def test_load_instruction_returns_correct_class(self):
        import sys
        from unittest.mock import patch, MagicMock

        mock_st_module = MagicMock()
        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_st_instance
        mock_st_module.SentenceTransformer = mock_st_class

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            if "idiolink.models.instruction_model" in sys.modules:
                del sys.modules["idiolink.models.instruction_model"]
            from idiolink.models.instruction_model import InstructionModel
            model = load_model("BAAI/bge-base-en-v1.5", device="cpu")
            assert isinstance(model, InstructionModel)

    def test_load_qwen_returns_correct_class(self):
        import sys
        from unittest.mock import patch, MagicMock

        mock_st_module = MagicMock()
        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 1024
        mock_st_class.return_value = mock_st_instance
        mock_st_module.SentenceTransformer = mock_st_class

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            if "idiolink.models.qwen" in sys.modules:
                del sys.modules["idiolink.models.qwen"]
            from idiolink.models.qwen import QwenModel
            model = load_model("Qwen/Qwen3-Embedding-0.6B", device="cpu")
            assert isinstance(model, QwenModel)

    def test_load_gritlm_returns_correct_class(self):
        import sys
        from unittest.mock import patch, MagicMock

        mock_gritlm_module = MagicMock()
        mock_grit_class = MagicMock()
        mock_grit_instance = MagicMock()
        mock_grit_instance.embed.return_value = [MagicMock(shape=(4096,))]
        mock_grit_class.return_value = mock_grit_instance
        mock_gritlm_module.GritLM = mock_grit_class

        with patch.dict(sys.modules, {"gritlm": mock_gritlm_module}):
            if "idiolink.models.gritlm" in sys.modules:
                del sys.modules["idiolink.models.gritlm"]
            from idiolink.models.gritlm import GritLMModel
            model = load_model("GritLM/GritLM-7B", device="cpu")
            assert isinstance(model, GritLMModel)

    def test_load_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent/model", device="cpu")


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
