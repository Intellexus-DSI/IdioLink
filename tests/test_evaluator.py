"""Tests for the evaluator module."""

import pytest
from idiolink.evaluator import (
    Evaluator,
    _avg,
    build_gold_standard,
    build_subject_gold,
    ndcg_at_k,
    r_precision,
)
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


# ---------- Splits: by_usage and by_subject ----------


@pytest.fixture
def subject_documents():
    """Docs with subjects across two idioms and two subjects."""
    docs = []
    # break the ice / Politics
    for i in range(2):
        docs.append({"id": f"bti_pol_lit_{i}", "sentence": "x", "idiom": "break the ice", "usage": "literal", "subject": "Politics"})
    for i in range(2):
        docs.append({"id": f"bti_pol_idi_{i}", "sentence": "x", "idiom": "break the ice", "usage": "idiomatic", "subject": "Politics"})
    # break the ice / Sports
    for i in range(2):
        docs.append({"id": f"bti_spo_lit_{i}", "sentence": "x", "idiom": "break the ice", "usage": "literal", "subject": "Sports"})
    for i in range(2):
        docs.append({"id": f"bti_spo_idi_{i}", "sentence": "x", "idiom": "break the ice", "usage": "idiomatic", "subject": "Sports"})
    return docs


@pytest.fixture
def subject_queries():
    return [
        IdiomQuery(query="Q_lit_pol", idiom="break the ice", usage_type="literal", subject="Politics"),
        IdiomQuery(query="Q_idi_spo", idiom="break the ice", usage_type="idiomatic", subject="Sports"),
    ]


class TestBuildSubjectGold:
    def test_subject_gold_matches_subject(self, subject_queries, subject_documents):
        sgold = build_subject_gold(subject_queries, subject_documents)
        # Politics query -> all 4 Politics docs (regardless of usage)
        assert sgold["Q_lit_pol"] == {"bti_pol_lit_0", "bti_pol_lit_1", "bti_pol_idi_0", "bti_pol_idi_1"}
        # Sports query -> all 4 Sports docs
        assert sgold["Q_idi_spo"] == {"bti_spo_lit_0", "bti_spo_lit_1", "bti_spo_idi_0", "bti_spo_idi_1"}

    def test_query_without_subject_is_none(self, subject_documents):
        q = IdiomQuery(query="no_subj", idiom="break the ice", usage_type="literal", subject="")
        sgold = build_subject_gold([q], subject_documents)
        assert sgold["no_subj"] is None

    def test_query_subject_absent_from_index_is_none(self):
        """Subject present on query but not on any indexed doc → None (excluded), not empty set."""
        docs = [
            {"id": "d1", "sentence": "x", "idiom": "x", "usage": "literal", "subject": "Sports"},
        ]
        qs = [IdiomQuery(query="q1", idiom="x", usage_type="literal", subject="Politics")]
        sgold = build_subject_gold(qs, docs)
        assert sgold["q1"] is None, (
            "Query with a subject not represented in the index should be excluded "
            "(returned as None) so it isn't counted as a zero-precision query."
        )


class TestEvaluatorSplits:
    def test_by_usage_separates_literal_and_idiomatic(self, subject_queries, subject_documents):
        evaluator = Evaluator(subject_queries, subject_documents)
        # Perfect retrieval for literal query (all 4 literal docs for the idiom),
        # zero for idiomatic (returns only literal docs).
        results = {
            "Q_lit_pol": ["bti_pol_lit_0", "bti_pol_lit_1", "bti_spo_lit_0", "bti_spo_lit_1"],
            "Q_idi_spo": ["bti_pol_lit_0", "bti_pol_lit_1", "bti_spo_lit_0", "bti_spo_lit_1"],
        }
        metrics = evaluator.evaluate(results)
        assert metrics["by_usage"]["literal"]["num_queries"] == 1
        assert metrics["by_usage"]["idiomatic"]["num_queries"] == 1
        assert metrics["by_usage"]["literal"]["r_precision"] == pytest.approx(1.0)
        assert metrics["by_usage"]["idiomatic"]["r_precision"] == pytest.approx(0.0)

    def test_by_subject_uses_subject_as_gold(self, subject_queries, subject_documents):
        evaluator = Evaluator(subject_queries, subject_documents)
        # For the literal Politics query, retrieve any Politics doc (incl. idiomatic) -> all hit
        # For the idiomatic Sports query, retrieve Politics docs -> all miss
        results = {
            "Q_lit_pol": ["bti_pol_idi_0", "bti_pol_idi_1", "bti_pol_lit_0", "bti_pol_lit_1"],
            "Q_idi_spo": ["bti_pol_idi_0", "bti_pol_idi_1", "bti_pol_lit_0", "bti_pol_lit_1"],
        }
        metrics = evaluator.evaluate(results)
        assert metrics["by_subject"]["num_queries"] == 2
        # First query: R=4, top-4 are all Politics -> 1.0; Second: 0.0 -> mean 0.5
        assert metrics["by_subject"]["r_precision"] == pytest.approx(0.5)

    def test_by_subject_skips_queries_without_subject(self, subject_documents):
        queries = [
            IdiomQuery(query="has_subj", idiom="break the ice", usage_type="literal", subject="Politics"),
            IdiomQuery(query="no_subj", idiom="break the ice", usage_type="literal", subject=""),
        ]
        evaluator = Evaluator(queries, subject_documents)
        results = {
            "has_subj": ["bti_pol_lit_0", "bti_pol_lit_1", "bti_pol_idi_0", "bti_pol_idi_1"],
            "no_subj": [],
        }
        metrics = evaluator.evaluate(results)
        assert metrics["by_subject"]["num_queries"] == 1
        assert metrics["by_subject"]["r_precision"] == pytest.approx(1.0)

    def test_top_level_metrics_unchanged_by_splits(self, sample_queries, sample_documents):
        """Existing top-level keys must remain present and well-formed."""
        evaluator = Evaluator(sample_queries, sample_documents)
        results = {q.query: [] for q in sample_queries}
        metrics = evaluator.evaluate(results)
        assert set(["r_precision", "ndcg@10", "num_queries"]).issubset(metrics.keys())
        assert metrics["by_subject"]["num_queries"] == 0  # no subject on the legacy fixtures


class TestAvgHelper:
    def test_empty_returns_zero(self):
        assert _avg([]) == 0.0

    def test_non_empty_returns_mean(self):
        assert _avg([1.0, 3.0]) == 2.0
