"""Tests for index ablation: filter helper + gold-shrinking behavior."""

import pytest

from idiolink.ablation import (
    ABLATION_PRESETS,
    filter_docs_by_usage,
    parse_index_filter,
)
from idiolink.evaluator import Evaluator, build_gold_standard
from idiolink.utils import IdiomQuery


# ---------- Filter helper ----------


def _docs():
    return [
        {"id": "lit_0", "sentence": "L0", "idiom": "x", "usage": "literal"},
        {"id": "lit_1", "sentence": "L1", "idiom": "x", "usage": "literal"},
        {"id": "idi_0", "sentence": "I0", "idiom": "x", "usage": "idiomatic"},
        {"id": "sim_0", "sentence": "S0", "idiom": "x", "usage": "simplification"},
        {"id": "sen_0", "sentence": "N0", "idiom": "x", "usage": "sense"},
    ]


def _split(docs):
    return [d["sentence"] for d in docs], docs


class TestParseIndexFilter:
    def test_preset_names(self):
        slug, keep = parse_index_filter("lit_sim_sense")
        assert slug == "lit_sim_sense"
        assert keep == {"literal", "simplification", "sense"}

        slug, keep = parse_index_filter("lit_idiom")
        assert slug == "lit_idiom"
        assert keep == {"literal", "idiomatic"}

    def test_csv_list(self):
        slug, keep = parse_index_filter("literal,idiomatic")
        assert keep == {"literal", "idiomatic"}
        assert slug == "idiomatic_literal"  # sorted

    def test_unknown_usage_raises(self):
        with pytest.raises(ValueError):
            parse_index_filter("literal,bogus")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_index_filter("")


class TestFilterDocsByUsage:
    def test_lit_sim_sense_drops_idiomatic(self):
        sents, meta = _split(_docs())
        s, m = filter_docs_by_usage(sents, meta, ABLATION_PRESETS["lit_sim_sense"])
        kept_ids = {d["id"] for d in m}
        assert kept_ids == {"lit_0", "lit_1", "sim_0", "sen_0"}
        assert len(s) == len(m) == 4

    def test_lit_idiom_drops_paraphrases(self):
        sents, meta = _split(_docs())
        s, m = filter_docs_by_usage(sents, meta, ABLATION_PRESETS["lit_idiom"])
        kept_ids = {d["id"] for d in m}
        assert kept_ids == {"lit_0", "lit_1", "idi_0"}
        assert len(s) == len(m) == 3

    def test_empty_keep_returns_empty(self):
        sents, meta = _split(_docs())
        s, m = filter_docs_by_usage(sents, meta, set())
        assert s == [] and m == []

    def test_preserves_parallel_order(self):
        sents, meta = _split(_docs())
        s, m = filter_docs_by_usage(sents, meta, {"literal"})
        # Sentences and metadata must stay aligned by index.
        for sent, md in zip(s, m):
            assert sent == md["sentence"]


# ---------- Gold-standard shrinks correctly under ablation ----------


@pytest.fixture
def ablation_docs():
    """4 literal + 3 idiomatic + 2 simplification + 1 sense for one idiom."""
    docs = []
    for i in range(4):
        docs.append({"id": f"lit_{i}", "sentence": "x", "idiom": "x", "usage": "literal"})
    for i in range(3):
        docs.append({"id": f"idi_{i}", "sentence": "x", "idiom": "x", "usage": "idiomatic"})
    for i in range(2):
        docs.append({"id": f"sim_{i}", "sentence": "x", "idiom": "x", "usage": "simplification"})
    docs.append({"id": "sen_0", "sentence": "x", "idiom": "x", "usage": "sense"})
    return docs


@pytest.fixture
def ablation_queries():
    return [
        IdiomQuery(query="QL", idiom="x", usage_type="literal"),
        IdiomQuery(query="QI", idiom="x", usage_type="idiomatic"),
    ]


class TestEvaluatorUnderAblation:
    def test_lit_sim_sense_idiomatic_gold_drops_idiomatic_docs(
        self, ablation_docs, ablation_queries
    ):
        sents = [d["sentence"] for d in ablation_docs]
        filt_sents, filt_meta = filter_docs_by_usage(
            sents, ablation_docs, ABLATION_PRESETS["lit_sim_sense"]
        )
        gold = build_gold_standard(ablation_queries, filt_meta)
        # Literal query gold is unchanged (literal docs still present).
        assert gold["QL"] == {"lit_0", "lit_1", "lit_2", "lit_3"}
        # Idiomatic query gold = sim + sense only.
        assert gold["QI"] == {"sim_0", "sim_1", "sen_0"}

    def test_lit_idiom_idiomatic_gold_drops_paraphrases(
        self, ablation_docs, ablation_queries
    ):
        sents = [d["sentence"] for d in ablation_docs]
        filt_sents, filt_meta = filter_docs_by_usage(
            sents, ablation_docs, ABLATION_PRESETS["lit_idiom"]
        )
        gold = build_gold_standard(ablation_queries, filt_meta)
        assert gold["QL"] == {"lit_0", "lit_1", "lit_2", "lit_3"}
        assert gold["QI"] == {"idi_0", "idi_1", "idi_2"}

    def test_r_precision_uses_shrunk_R(self, ablation_docs, ablation_queries):
        """R adjusts automatically because r_precision uses len(gold_ids)."""
        sents = [d["sentence"] for d in ablation_docs]
        filt_sents, filt_meta = filter_docs_by_usage(
            sents, ablation_docs, ABLATION_PRESETS["lit_idiom"]
        )
        evaluator = Evaluator(ablation_queries, filt_meta)
        # Perfect retrieval for the idiomatic query: return only the 3 idi docs.
        results = {
            "QL": ["lit_0", "lit_1", "lit_2", "lit_3"],
            "QI": ["idi_0", "idi_1", "idi_2"],
        }
        metrics = evaluator.evaluate(results)
        # Both queries: perfect; R-Prec should be 1.0 overall and per usage.
        assert metrics["r_precision"] == pytest.approx(1.0)
        assert metrics["by_usage"]["literal"]["r_precision"] == pytest.approx(1.0)
        assert metrics["by_usage"]["idiomatic"]["r_precision"] == pytest.approx(1.0)
        # Sanity: under this preset the idiomatic gold has 3 docs, not 6.
        # Returning all 3 in top-3 -> R-Prec = 3/3 = 1.0 (already asserted above).

    def test_full_index_idiomatic_gold_unchanged(self, ablation_docs, ablation_queries):
        """Sanity: with no filter, idiomatic gold = idi + sim + sen as before."""
        gold = build_gold_standard(ablation_queries, ablation_docs)
        assert gold["QI"] == {"idi_0", "idi_1", "idi_2", "sim_0", "sim_1", "sen_0"}
