"""Abstract base class for all embedding models."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbeddingModel(ABC):
    """All embedding models must implement encode()."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.embedding_dim: int = 0

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings. Returns (N, dim) array."""
        ...

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
