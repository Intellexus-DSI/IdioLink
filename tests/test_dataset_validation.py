"""Release validation tests for the committed IdioLink benchmark data."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EXPECTED_SPLITS = {
    "train": {"docs": 2200, "queries": 440, "idioms": 22},
    "val": {"docs": 1000, "queries": 200, "idioms": 10},
    "test": {"docs": 7500, "queries": 1500, "idioms": 75},
}

REQUIRED_FIELDS = {"id", "sentence", "idiom", "span", "subject", "usage", "is_gold"}
DOC_USAGES = {"literal", "idiomatic", "simplification", "sense"}
QUERY_USAGES = {"literal", "idiomatic"}


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_idioms(split: str) -> set[str]:
    indexes = _load_json(DATA_DIR / split / "indexes.json")
    queries = _load_json(DATA_DIR / split / "queries.json")
    return {item["idiom"] for item in indexes} | {item["idiom"] for item in queries}


def test_split_counts_match_paper():
    total_docs = 0
    total_queries = 0
    total_idioms = set()

    for split, expected in EXPECTED_SPLITS.items():
        indexes = _load_json(DATA_DIR / split / "indexes.json")
        queries = _load_json(DATA_DIR / split / "queries.json")
        idioms = {item["idiom"] for item in indexes} | {item["idiom"] for item in queries}

        assert len(indexes) == expected["docs"]
        assert len(queries) == expected["queries"]
        assert len(idioms) == expected["idioms"]

        total_docs += len(indexes)
        total_queries += len(queries)
        total_idioms.update(idioms)

    assert total_docs == 10700
    assert total_queries == 2140
    assert len(total_idioms) == 107


def test_required_schema_fields_and_usage_values():
    for split in EXPECTED_SPLITS:
        indexes = _load_json(DATA_DIR / split / "indexes.json")
        queries = _load_json(DATA_DIR / split / "queries.json")

        for item in indexes:
            assert REQUIRED_FIELDS <= item.keys()
            assert item["usage"] in DOC_USAGES
            assert item["span"] in item["sentence"]

        for item in queries:
            assert REQUIRED_FIELDS <= item.keys()
            assert item["usage"] in QUERY_USAGES
            assert item["span"] in item["sentence"]


def test_no_idiom_overlap_across_splits():
    train_idioms = _split_idioms("train")
    val_idioms = _split_idioms("val")
    test_idioms = _split_idioms("test")

    assert train_idioms.isdisjoint(val_idioms)
    assert train_idioms.isdisjoint(test_idioms)
    assert val_idioms.isdisjoint(test_idioms)


def test_triplet_files_exist_and_match_expected_shapes():
    expected_files = [
        DATA_DIR / "train" / "triplets_train_full.jsonl",
        DATA_DIR / "train" / "triplets_train_span.jsonl",
        DATA_DIR / "val" / "triplets_val_full.jsonl",
        DATA_DIR / "val" / "triplets_val_span.jsonl",
        DATA_DIR / "test" / "triplets_test_span.jsonl",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing triplet file: {path}"
        with open(path, encoding="utf-8") as f:
            first = json.loads(next(f))
        assert {"query", "positive", "negatives", "query_idiom", "query_usage"} <= first.keys()
        assert isinstance(first["negatives"], list)
        assert first["negatives"]

