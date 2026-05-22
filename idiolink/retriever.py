"""Dense retriever: index documents and retrieve by cosine similarity."""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models.base import BaseEmbeddingModel


class DenseRetriever:
    """Indexes documents and retrieves by cosine similarity."""

    def __init__(self, model: BaseEmbeddingModel):
        self.model = model
        self.doc_embeddings: np.ndarray = np.array([])
        self.doc_metadata: List[Dict[str, Any]] = []

    def index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Encode and store document embeddings."""
        self.doc_embeddings = self.model.encode(documents)
        self.doc_metadata = metadata

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 100,
        query_embeddings: np.ndarray = None,
    ) -> Dict[str, List[str]]:
        """
        Retrieve top-k docs for each query.

        Args:
            queries: query sentences (used as keys in output)
            top_k: number of results per query
            query_embeddings: pre-computed query embeddings (optional)

        Returns:
            Dict mapping query text -> ranked list of doc IDs
        """
        if query_embeddings is None:
            query_embeddings = self.model.encode(queries)

        sims = cosine_similarity(query_embeddings, self.doc_embeddings)
        results = {}
        for i, query in enumerate(queries):
            top_indices = np.argsort(sims[i])[::-1][:top_k]
            results[query] = [self.doc_metadata[idx]["id"] for idx in top_indices]
        return results
