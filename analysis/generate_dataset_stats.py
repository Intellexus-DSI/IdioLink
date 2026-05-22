"""Generate dataset statistics across train/val/test splits."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.utils import load_config


def load_split_data(split_dir: Path):
    """Load indexes and queries for a single split."""
    indexes = []
    queries = []
    idx_path = split_dir / "indexes.json"
    qry_path = split_dir / "queries.json"

    if idx_path.exists():
        with open(idx_path) as f:
            indexes = json.load(f)
    if qry_path.exists():
        with open(qry_path) as f:
            queries = json.load(f)
    return indexes, queries


def compute_stats(indexes: list, queries: list) -> dict:
    """Compute statistics for one split."""
    idioms_idx = set(item["idiom"] for item in indexes)
    idioms_qry = set(item["idiom"] for item in queries)
    all_idioms = idioms_idx | idioms_qry

    usage_idx = Counter(item["usage"] for item in indexes)
    usage_qry = Counter(item["usage"] for item in queries)

    gold_idx = sum(1 for item in indexes if item.get("is_gold", False))
    gold_qry = sum(1 for item in queries if item.get("is_gold", False))

    return {
        "num_pies": len(all_idioms),
        "num_docs": len(indexes),
        "num_queries": len(queries),
        "doc_usage": dict(usage_idx),
        "query_usage": dict(usage_qry),
        "gold_docs": gold_idx,
        "gold_queries": gold_qry,
    }


def print_stats(all_stats: dict):
    """Print formatted statistics table."""
    splits = list(all_stats.keys())

    print(f"{'Statistic':<25}", end="")
    for split in splits:
        print(f"{split:>12}", end="")
    print()
    print("-" * (25 + 12 * len(splits)))

    fields = [
        ("PIEs (idioms)", "num_pies"),
        ("Documents", "num_docs"),
        ("Queries", "num_queries"),
        ("Gold docs", "gold_docs"),
        ("Gold queries", "gold_queries"),
    ]

    for label, key in fields:
        print(f"{label:<25}", end="")
        for split in splits:
            print(f"{all_stats[split][key]:>12}", end="")
        print()

    print()
    print("Document usage distribution:")
    all_usages = sorted(
        set(u for s in all_stats.values() for u in s["doc_usage"].keys())
    )
    for usage in all_usages:
        print(f"  {usage:<23}", end="")
        for split in splits:
            count = all_stats[split]["doc_usage"].get(usage, 0)
            print(f"{count:>12}", end="")
        print()

    print()
    print("Query usage distribution:")
    all_usages = sorted(
        set(u for s in all_stats.values() for u in s["query_usage"].keys())
    )
    for usage in all_usages:
        print(f"  {usage:<23}", end="")
        for split in splits:
            count = all_stats[split]["query_usage"].get(usage, 0)
            print(f"{count:>12}", end="")
        print()


def main():
    cfg = load_config()

    splits = {
        "train": Path(cfg["data"]["train_dir"]),
        "val": Path(cfg["data"]["val_dir"]),
        "test": Path(cfg["data"]["test_dir"]),
    }

    all_stats = {}
    for name, path in splits.items():
        indexes, queries = load_split_data(path)
        all_stats[name] = compute_stats(indexes, queries)

    print("IdioLink Dataset Statistics")
    print("=" * 60)
    print()
    print_stats(all_stats)

    # Summary totals
    print()
    print("-" * 60)
    total_pies = set()
    total_docs = 0
    total_queries = 0
    for name, stats in all_stats.items():
        split_dir = splits[name]
        with open(split_dir / "indexes.json") as f:
            data = json.load(f)
        for item in data:
            total_pies.add(item["idiom"])
        total_docs += stats["num_docs"]
        total_queries += stats["num_queries"]

    print(f"{'Total unique PIEs':<25}{len(total_pies):>12}")
    print(f"{'Total documents':<25}{total_docs:>12}")
    print(f"{'Total queries':<25}{total_queries:>12}")


if __name__ == "__main__":
    main()
