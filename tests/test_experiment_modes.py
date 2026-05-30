"""Tests that experiment modes follow the paper's query/document encoding contract."""

from dataclasses import dataclass

import numpy as np

import run_all
from idiolink.models import encode_helpers
from idiolink.utils import IdiomQuery


@dataclass
class RecordingModel:
    model_id: str = "recording-model"

    def __post_init__(self):
        self.encoded_batches = []
        self.encode_query_calls = []
        self.format_calls = []

    def encode(self, texts):
        self.encoded_batches.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    def encode_queries(self, texts, spans=None, instruction=None):
        self.encode_query_calls.append({
            "texts": list(texts),
            "spans": list(spans),
            "instruction": list(instruction) if isinstance(instruction, list) else instruction,
        })
        return np.ones((len(texts), 3), dtype=np.float32)

    def format_queries_for_late_chunking(self, texts, instructions):
        self.format_calls.append({
            "texts": list(texts),
            "instructions": list(instructions),
        })
        return [
            f"Instruct: {instruction}\nQuery: {text}"
            for text, instruction in zip(texts, instructions)
        ]


def _fixture_data():
    queries = [
        IdiomQuery(
            query="The mayor knew the answer all along.",
            idiom="all along",
            usage_type="idiomatic",
            span="all along",
        ),
        IdiomQuery(
            query="The banners hung all along the street.",
            idiom="all along",
            usage_type="literal",
            span="all along",
        ),
    ]
    docs = [
        {"id": "d1", "idiom": "all along", "usage": "idiomatic"},
        {"id": "d2", "idiom": "all along", "usage": "literal"},
        {"id": "d3", "idiom": "all along", "usage": "sense"},
    ]
    doc_sentences = [
        "The official had known the answer from the beginning.",
        "The decorations were placed all along the road.",
        "The plan was hidden from the start.",
    ]
    return queries, doc_sentences, docs


def test_span_mode_uses_late_chunking_with_full_query_context(monkeypatch):
    calls = []

    def fake_late_chunk_encode(model, documents, spans, device=None, prefer_last_span=False):
        calls.append({
            "documents": list(documents),
            "spans": list(spans),
            "prefer_last_span": prefer_last_span,
        })
        return np.ones((len(documents), 3), dtype=np.float32)

    monkeypatch.setattr(encode_helpers, "late_chunk_encode", fake_late_chunk_encode)

    queries, doc_sentences, docs = _fixture_data()
    model = RecordingModel()
    run_all.run_single(model, "span", queries, doc_sentences, docs, 2, "cpu")

    assert calls
    assert calls[0]["documents"] == [q.query for q in queries]
    assert calls[0]["spans"] == [q.span for q in queries]
    assert calls[0]["prefer_last_span"] is False


def test_instruction_sentence_uses_per_query_instructions():
    queries, doc_sentences, docs = _fixture_data()
    model = RecordingModel()

    run_all.run_single(
        model,
        "instruction_sentence",
        queries,
        doc_sentences,
        docs,
        2,
        "cpu",
    )

    call = model.encode_query_calls[0]
    assert call["texts"] == [q.query for q in queries]
    assert len(call["instruction"]) == len(queries)
    assert all("all along" in instruction for instruction in call["instruction"])


def test_instruction_span_formats_queries_before_late_chunking(monkeypatch):
    calls = []

    def fake_late_chunk_encode(model, documents, spans, device=None, prefer_last_span=False):
        calls.append({
            "documents": list(documents),
            "spans": list(spans),
            "prefer_last_span": prefer_last_span,
        })
        return np.ones((len(documents), 3), dtype=np.float32)

    monkeypatch.setattr(encode_helpers, "late_chunk_encode", fake_late_chunk_encode)

    queries, doc_sentences, docs = _fixture_data()
    model = RecordingModel()
    run_all.run_single(
        model,
        "instruction_span",
        queries,
        doc_sentences,
        docs,
        2,
        "cpu",
    )

    assert model.format_calls
    assert calls[0]["prefer_last_span"] is True
    assert calls[0]["documents"][0].startswith("Instruct:")
    assert queries[0].query in calls[0]["documents"][0]
