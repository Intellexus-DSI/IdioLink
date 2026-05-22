"""SentenceTransformer wrapper for standard embedding models."""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel


class SentenceTransformerModel(BaseEmbeddingModel):
    """Wraps any HuggingFace SentenceTransformer model."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        trust_remote_code: bool = False,
    ):
        super().__init__(model_id)
        kwargs = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        self.model = SentenceTransformer(model_id, device=device, **kwargs)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _format_instruction(self, text: str, instruction: str) -> str:
        return f"Instruct: {instruction}\nQuery: {text}" if instruction else text

    def format_queries_for_late_chunking(
        self,
        texts: List[str],
        instructions: Union[str, List[str]],
    ) -> List[str]:
        """Return plain-text instructed queries suitable for token-level span pooling."""
        if isinstance(instructions, str):
            instructions = [instructions] * len(texts)
        return [
            self._format_instruction(text, instruction)
            for text, instruction in zip(texts, instructions)
        ]

    def encode_queries(
        self,
        texts: List[str],
        spans: Optional[List[str]] = None,
        instruction: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Encode queries with generic inline instruction formatting."""
        if instruction is None:
            instruction = ""
        if isinstance(instruction, str):
            instruction = [instruction] * len(texts)
        return self.encode([
            self._format_instruction(text, inst)
            for text, inst in zip(texts, instruction)
        ])

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings
