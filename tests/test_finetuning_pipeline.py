"""Tests for fine-tuning data shaping and result aggregation contracts."""

import json
from pathlib import Path

import pytest

from analysis.generate_finetuning_table import collect_results
from idiolink.trainer.datasets import TripletDataset


def _write_triplets(path: Path):
    sample = {
        "query": "The plan was cut and dried before the meeting.",
        "query_span": "cut and dried",
        "positive": "The decision had already been made.",
        "negatives": ["The herbs were cut and dried.", "Another unrelated document."],
        "query_idiom": "cut and dried",
        "query_usage": "idiomatic",
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")


def test_triplet_dataset_loads_basic_fields(tmp_path):
    """Dataset loads triplets with their plain fields; formatting is the
    trainer's job, not the dataset's."""
    triplets = tmp_path / "triplets.jsonl"
    _write_triplets(triplets)

    item = TripletDataset(str(triplets))[0]

    assert item["query"] == "The plan was cut and dried before the meeting."
    assert item["query_span"] == "cut and dried"
    assert item["positive"] == "The decision had already been made."
    assert len(item["negatives"]) == 2


def test_triplet_dataset_returns_plain_dict_no_instruction_wrapping(tmp_path: Path):
    """Dataset is mode-agnostic; returns plain fields, never applies the
    hardcoded `Instruct: ...\\nQuery: ...` template.
    """
    from idiolink.trainer import TripletDataset

    triplet_path = tmp_path / "triplets.jsonl"
    triplet_path.write_text(json.dumps({
        "query": "She kicked the bucket yesterday.",
        "query_span": "kicked the bucket",
        "query_idiom": "kick the bucket",
        "query_usage": "idiomatic",
        "query_subject": "death",
        "positive": "He died last week.",
        "negatives": ["He kicked the actual bucket over."],
    }) + "\n")

    ds = TripletDataset(str(triplet_path), max_negatives=5)
    assert len(ds) == 1
    item = ds[0]
    assert item["query"] == "She kicked the bucket yesterday."
    assert item["query_span"] == "kicked the bucket"
    assert item["query_idiom"] == "kick the bucket"
    assert item["query_usage"] == "idiomatic"
    assert item["query_subject"] == "death"
    assert item["positive"] == "He died last week."
    assert item["negatives"] == ["He kicked the actual bucket over."]
    # No instruction wrapping must have occurred
    assert "Instruct:" not in item["query"]
    assert "Query:" not in item["query"]


def test_triplet_dataset_no_longer_accepts_mode_param(tmp_path: Path):
    """`mode` parameter is removed; passing it should TypeError."""
    from idiolink.trainer import TripletDataset

    triplet_path = tmp_path / "t.jsonl"
    triplet_path.write_text(json.dumps({
        "query": "x", "positive": "y", "negatives": ["z"],
    }) + "\n")

    import pytest
    with pytest.raises(TypeError):
        TripletDataset(str(triplet_path), mode="instruction_sentence")


def test_finetuning_table_reads_nested_test_metrics(tmp_path):
    model = "sentence-transformers/all-MiniLM-L6-v2"
    metrics_path = (
        tmp_path
        / "fine_tuning"
        / "sentence-transformers__all-MiniLM-L6-v2"
        / "sentence"
        / "seed_42"
        / "metrics.json"
    )
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"test": {"r_precision": 0.5, "ndcg@10": 0.75}}, f)

    results = collect_results(tmp_path, [model], ["sentence"], [42])

    assert results[model]["sentence"]["r_precision"]["mean"] == pytest.approx(0.5)
    assert results[model]["sentence"]["ndcg@10"]["mean"] == pytest.approx(0.75)
