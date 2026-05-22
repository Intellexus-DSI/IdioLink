"""Model registry: configuration and factory for all 24 embedding models."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import BaseEmbeddingModel


@dataclass
class ModelConfig:
    """Configuration for a single embedding model."""
    model_id: str
    model_class: str  # sentence_transformer, instruction, gritlm, qwen
    size_params: str  # e.g. "110M", "7B"
    max_length: int = 512
    instruction_format: Optional[str] = None  # e5_inline, bge_prompt, instructor_pairs, tart_sep, nomic_prefix, bge_gemma
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False
    batch_size: int = 32
    dtype: str = "float32"
    supports_span_pooling: bool = True


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "sentence-transformers/all-MiniLM-L6-v2": ModelConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_class="sentence_transformer",
        size_params="110M",
    ),
    "facebook/contriever": ModelConfig(
        model_id="facebook/contriever",
        model_class="sentence_transformer",
        size_params="110M",
    ),
    "intfloat/e5-base-v2": ModelConfig(
        model_id="intfloat/e5-base-v2",
        model_class="sentence_transformer",
        size_params="110M",
        instruction_format="e5_inline",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "orionweller/tart-dual-contriever-msmarco": ModelConfig(
        model_id="orionweller/tart-dual-contriever-msmarco",
        model_class="instruction",
        size_params="110M",
        instruction_format="tart_sep",
    ),
    "BAAI/bge-base-en-v1.5": ModelConfig(
        model_id="BAAI/bge-base-en-v1.5",
        model_class="instruction",
        size_params="326M",
        instruction_format="bge_prompt",
    ),
    "hkunlp/instructor-base": ModelConfig(
        model_id="hkunlp/instructor-base",
        model_class="instruction",
        size_params="335M",
        instruction_format="instructor_pairs",
    ),
    "nomic-ai/nomic-embed-text-v2-moe": ModelConfig(
        model_id="nomic-ai/nomic-embed-text-v2-moe",
        model_class="instruction",
        size_params="475M",
        instruction_format="nomic_prefix",
        trust_remote_code=True,
    ),
    "intfloat/multilingual-e5-large-instruct": ModelConfig(
        model_id="intfloat/multilingual-e5-large-instruct",
        model_class="instruction",
        size_params="560M",
        instruction_format="e5_inline",
    ),
    "BAAI/bge-m3": ModelConfig(
        model_id="BAAI/bge-m3",
        model_class="sentence_transformer",
        size_params="568M",
    ),
    "Qwen/Qwen3-Embedding-0.6B": ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_class="qwen",
        size_params="600M",
        instruction_format="e5_inline",
        trust_remote_code=True,
    ),
    "facebook/drama-1b": ModelConfig(
        model_id="facebook/drama-1b",
        model_class="sentence_transformer",
        size_params="1B",
        batch_size=16,
    ),
    "NovaSearch/stella-en-1.5B-v5": ModelConfig(
        model_id="NovaSearch/stella-en-1.5B-v5",
        model_class="sentence_transformer",
        size_params="1.5B",
        trust_remote_code=True,
        batch_size=16,
    ),
    "hkunlp/instructor-xl": ModelConfig(
        model_id="hkunlp/instructor-xl",
        model_class="instruction",
        size_params="1.5B",
        instruction_format="instructor_pairs",
        batch_size=16,
    ),
    "vec-ai/lychee-embed": ModelConfig(
        model_id="vec-ai/lychee-embed",
        model_class="instruction",
        size_params="1.5B",
        instruction_format="e5_inline",
        batch_size=16,
    ),
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": ModelConfig(
        model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        model_class="qwen",
        size_params="1.5B",
        instruction_format="e5_inline",
        trust_remote_code=True,
        batch_size=16,
    ),
    "Qwen/Qwen3-Embedding-4B": ModelConfig(
        model_id="Qwen/Qwen3-Embedding-4B",
        model_class="qwen",
        size_params="4B",
        instruction_format="e5_inline",
        trust_remote_code=True,
        batch_size=8,
    ),
    "Linq-AI-Research/Linq-Embed-Mistral": ModelConfig(
        model_id="Linq-AI-Research/Linq-Embed-Mistral",
        model_class="instruction",
        size_params="7B",
        instruction_format="e5_inline",
        batch_size=4,
    ),
    "Salesforce/SFR-Embedding-Mistral": ModelConfig(
        model_id="Salesforce/SFR-Embedding-Mistral",
        model_class="instruction",
        size_params="7B",
        instruction_format="e5_inline",
        batch_size=4,
    ),
    "intfloat/e5-mistral-7b-instruct": ModelConfig(
        model_id="intfloat/e5-mistral-7b-instruct",
        model_class="instruction",
        size_params="7B",
        instruction_format="e5_inline",
        batch_size=4,
    ),
    "GritLM/GritLM-7B": ModelConfig(
        model_id="GritLM/GritLM-7B",
        model_class="gritlm",
        size_params="7B",
        instruction_format="instructor_pairs",
        batch_size=4,
    ),
    "Alibaba-NLP/gte-Qwen2-7B-instruct": ModelConfig(
        model_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
        model_class="qwen",
        size_params="7B",
        instruction_format="e5_inline",
        trust_remote_code=True,
        batch_size=4,
    ),
    "Qwen/Qwen3-Embedding-8B": ModelConfig(
        model_id="Qwen/Qwen3-Embedding-8B",
        model_class="qwen",
        size_params="8B",
        instruction_format="e5_inline",
        trust_remote_code=True,
        batch_size=4,
    ),
    "nvidia/llama-embed-nemotron-8b": ModelConfig(
        model_id="nvidia/llama-embed-nemotron-8b",
        model_class="instruction",
        size_params="8B",
        instruction_format="e5_inline",
        trust_remote_code=True,
        batch_size=4,
    ),
    "BAAI/bge-multilingual-gemma2": ModelConfig(
        model_id="BAAI/bge-multilingual-gemma2",
        model_class="instruction",
        size_params="9B",
        instruction_format="bge_gemma",
        batch_size=4,
    ),
}


def load_model(model_id: str, device: Optional[str] = None) -> BaseEmbeddingModel:
    """Instantiate the correct model class from the registry."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_id]

    if config.model_class == "sentence_transformer":
        from .sentence_transformer import SentenceTransformerModel
        return SentenceTransformerModel(
            model_id=config.model_id,
            device=device,
            batch_size=config.batch_size,
            trust_remote_code=config.trust_remote_code,
        )
    elif config.model_class == "instruction":
        from .instruction_model import InstructionModel
        return InstructionModel(
            model_id=config.model_id,
            instruction_format=config.instruction_format or "plain",
            device=device,
            batch_size=config.batch_size,
            trust_remote_code=config.trust_remote_code,
            query_prefix=config.query_prefix,
            passage_prefix=config.passage_prefix,
        )
    elif config.model_class == "gritlm":
        from .gritlm import GritLMModel
        return GritLMModel(
            model_id=config.model_id,
            device=device,
            batch_size=config.batch_size,
        )
    elif config.model_class == "qwen":
        from .qwen import QwenModel
        return QwenModel(
            model_id=config.model_id,
            device=device,
            batch_size=config.batch_size,
        )
    else:
        raise ValueError(f"Unknown model_class: {config.model_class}")
