"""Tests for fine-tuning data shaping and result aggregation contracts."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

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


def test_triplet_dataset_sentence_mode_uses_full_query(tmp_path):
    triplets = tmp_path / "triplets.jsonl"
    _write_triplets(triplets)

    item = TripletDataset(str(triplets), mode="sentence")[0]

    assert item["query"] == "The plan was cut and dried before the meeting."
    assert item["positive"] == "The decision had already been made."
    assert len(item["negatives"]) == 2


@pytest.mark.parametrize("mode", ["span", "instruction_span"])
def test_triplet_dataset_span_modes_use_query_span(tmp_path, mode):
    triplets = tmp_path / "triplets.jsonl"
    _write_triplets(triplets)

    item = TripletDataset(str(triplets), mode=mode)[0]

    assert "cut and dried" in item["query"]
    if mode == "span":
        assert item["query"] == "cut and dried"
    else:
        assert item["query"].startswith("Instruct:")


def test_triplet_dataset_instruction_sentence_keeps_full_query_with_instruction(tmp_path):
    triplets = tmp_path / "triplets.jsonl"
    _write_triplets(triplets)

    item = TripletDataset(str(triplets), mode="instruction_sentence")[0]

    assert item["query"].startswith("Instruct:")
    assert "The plan was cut and dried before the meeting." in item["query"]
    assert "cut and dried" in item["query"]


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

