"""Tests for the evaluator module."""

import pytest
from idiolink.evaluator import Evaluator, build_gold_standard, ndcg_at_k, r_precision
from idiolink.utils import IdiomQuery


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    # 4 literal docs for "break the ice"
    for i in range(4):
        docs.append({
            "id": f"lit_{i}",
            "sentence": f"Literal sentence {i}",
            "idiom": "break the ice",
            "usage": "literal",
        })
    # 3 idiomatic + 2 simplification + 1 sense = 6 non-literal docs
    for i in range(3):
        docs.append({
            "id": f"idiom_{i}",
            "sentence": f"Idiomatic sentence {i}",
            "idiom": "break the ice",
            "usage": "idiomatic",
        })
    for i in range(2):
        docs.append({
            "id": f"simp_{i}",
            "sentence": f"Simplification sentence {i}",
            "idiom": "break the ice",
            "usage": "simplification",
        })
    docs.append({
        "id": "sense_0",
        "sentence": "Sense sentence 0",
        "idiom": "break the ice",
        "usage": "sense",
    })
    # Docs for a different idiom (should not be relevant)
    docs.append({
        "id": "other_lit_0",
        "sentence": "Other literal",
        "idiom": "hit the road",
        "usage": "literal",
    })
    return docs


@pytest.fixture
def sample_queries():
    return [
        IdiomQuery(query="Breaking ice literally", idiom="break the ice", usage_type="literal"),
        IdiomQuery(query="Breaking ice idiomatically", idiom="break the ice", usage_type="idiomatic"),
    ]


class TestBuildGoldStandard:
    def test_literal_query_gets_literal_docs(self, sample_queries, sample_documents):
        gold = build_gold_standard(sample_queries, sample_documents)
        literal_q = sample_queries[0]
        assert gold[literal_q.query] == {"lit_0", "lit_1", "lit_2", "lit_3"}

    def test_idiomatic_query_gets_nonliteral_docs(self, sample_queries, sample_documents):
        gold = build_gold_standard(sample_queries, sample_documents)
        idiomatic_q = sample_queries[1]
        expected = {"idiom_0", "idiom_1", "idiom_2", "simp_0", "simp_1", "sense_0"}
        assert gold[idiomatic_q.query] == expected

    def test_different_idiom_not_included(self, sample_queries, sample_documents):
        gold = build_gold_standard(sample_queries, sample_documents)
        for q in sample_queries:
            assert "other_lit_0" not in gold[q.query]


class TestNDCG:
    def test_perfect_ranking(self):
        gold = {"a", "b", "c"}
        retrieved = ["a", "b", "c", "d", "e"]
        assert ndcg_at_k(gold, retrieved, 10) == pytest.approx(1.0)

    def test_empty_gold(self):
        assert ndcg_at_k(set(), ["a", "b"], 10) == 0.0

    def test_no_relevant_retrieved(self):
        gold = {"a", "b"}
        retrieved = ["c", "d", "e"]
        assert ndcg_at_k(gold, retrieved, 10) == 0.0

    def test_partial_ranking(self):
        gold = {"a", "b"}
        retrieved = ["x", "a", "y", "b"]
        score = ndcg_at_k(gold, retrieved, 10)
        assert 0 < score < 1.0


class TestRPrecision:
    def test_perfect_retrieval(self):
        gold = {"a", "b", "c"}
        retrieved = ["a", "b", "c", "d"]
        assert r_precision(gold, retrieved) == pytest.approx(1.0)

    def test_no_relevant(self):
        gold = {"a", "b", "c"}
        retrieved = ["x", "y", "z", "w"]
        assert r_precision(gold, retrieved) == pytest.approx(0.0)

    def test_partial(self):
        gold = {"a", "b", "c", "d"}
        retrieved = ["a", "x", "b", "y", "c"]
        # R=4, top-4 = [a, x, b, y] -> 2 hits -> 2/4 = 0.5
        assert r_precision(gold, retrieved) == pytest.approx(0.5)

    def test_empty_gold(self):
        assert r_precision(set(), ["a", "b"]) == 0.0


class TestEvaluator:
    def test_full_evaluation(self, sample_queries, sample_documents):
        evaluator = Evaluator(sample_queries, sample_documents)

        # Perfect retrieval for literal, imperfect for idiomatic
        results = {
            sample_queries[0].query: ["lit_0", "lit_1", "lit_2", "lit_3"],
            sample_queries[1].query: ["idiom_0", "other_lit_0", "simp_0", "sense_0", "idiom_1", "idiom_2", "simp_1"],
        }
        metrics = evaluator.evaluate(results)
        assert "r_precision" in metrics
        assert "ndcg@10" in metrics
        assert metrics["num_queries"] == 2
        assert 0 <= metrics["r_precision"] <= 1
        assert 0 <= metrics["ndcg@10"] <= 1

    def test_empty_results(self, sample_queries, sample_documents):
        evaluator = Evaluator(sample_queries, sample_documents)
        results = {q.query: [] for q in sample_queries}
        metrics = evaluator.evaluate(results)
        assert metrics["r_precision"] == 0.0
        assert metrics["ndcg@10"] == 0.0
