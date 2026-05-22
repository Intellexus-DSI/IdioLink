"""Tests for the retriever module."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from idiolink.retriever import DenseRetriever
from idiolink.models.base import BaseEmbeddingModel


class MockModel(BaseEmbeddingModel):
    """Mock embedding model that returns predictable embeddings."""

    def __init__(self):
        super().__init__("mock-model")
        self.embedding_dim = 3

    def encode(self, texts):
        embeddings = []
        for i, text in enumerate(texts):
            if "query_match" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "doc_match" in text:
                embeddings.append([0.9, 0.1, 0.0])
            elif "doc_other" in text:
                embeddings.append([0.0, 0.0, 1.0])
            else:
                embeddings.append([0.5, 0.5, 0.0])
        return np.array(embeddings, dtype=np.float32)


class TestDenseRetriever:
    def test_index_and_retrieve(self):
        model = MockModel()
        retriever = DenseRetriever(model)

        docs = ["doc_match A", "doc_other B", "doc_match C"]
        metadata = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
        retriever.index(docs, metadata)

        results = retriever.retrieve(["query_match"], top_k=2)
        assert "query_match" in results
        retrieved_ids = results["query_match"]
        assert len(retrieved_ids) == 2
        # doc_match docs should rank higher than doc_other
        assert "d2" not in retrieved_ids

    def test_top_k_limits_results(self):
        model = MockModel()
        retriever = DenseRetriever(model)

        docs = ["doc_match"] * 10
        metadata = [{"id": f"d{i}"} for i in range(10)]
        retriever.index(docs, metadata)

        results = retriever.retrieve(["query_match"], top_k=3)
        assert len(results["query_match"]) == 3

    def test_multiple_queries(self):
        model = MockModel()
        retriever = DenseRetriever(model)

        docs = ["doc_match", "doc_other"]
        metadata = [{"id": "d1"}, {"id": "d2"}]
        retriever.index(docs, metadata)

        results = retriever.retrieve(["query_match", "doc_other"], top_k=2)
        assert len(results) == 2

    def test_precomputed_embeddings(self):
        model = MockModel()
        retriever = DenseRetriever(model)

        docs = ["doc_match", "doc_other"]
        metadata = [{"id": "d1"}, {"id": "d2"}]
        retriever.index(docs, metadata)

        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        results = retriever.retrieve(["q"], top_k=2, query_embeddings=query_emb)
        assert results["q"][0] == "d1"
