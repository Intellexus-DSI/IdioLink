"""Qwen embedding model wrapper (Qwen3-Embedding, GTE-Qwen2 families)."""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel


class QwenModel(BaseEmbeddingModel):
    """Wraps Qwen/GTE-Qwen2 models using SentenceTransformer with trust_remote_code."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        super().__init__(model_id)
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_id, device=device, trust_remote_code=True
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode documents without instruction."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    def format_queries_for_late_chunking(
        self,
        texts: List[str],
        instructions: Union[str, List[str]],
    ) -> List[str]:
        """Return plain-text instructed queries suitable for token-level span pooling."""
        if isinstance(instructions, str):
            instructions = [instructions] * len(texts)
        return [
            f"Instruct: {instruction}\nQuery: {text}"
            for text, instruction in zip(texts, instructions)
        ]

    def encode_queries(
        self,
        texts: List[str],
        spans: Optional[List[str]] = None,
        instruction: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Encode queries with instruction as prompt kwarg."""
        kwargs = {}
        if isinstance(instruction, list):
            if len(set(instruction)) != 1:
                encoded = np.vstack([
                    self.model.encode(
                        [text],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        prompt=f"Instruct: {inst}\nQuery: ",
                    )
                    for text, inst in zip(texts, instruction)
                ])
                return encoded.astype(np.float32)
            instruction = instruction[0]
        if instruction:
            kwargs["prompt"] = f"Instruct: {instruction}\nQuery: "
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            **kwargs,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings
