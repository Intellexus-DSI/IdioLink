"""Stratified splitting of annotated data into train/val/test sets.

Split by idiom (PIE):
- Train: 22 idioms
- Val: 10 idioms
- Test: 75 idioms

Ensures balanced representation of usage types across splits.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_annotated_data(json_file: str) -> List[dict]:
    """Load annotated data from JSON."""
    with open(json_file, "r") as f:
        return json.load(f)


def get_idiom_stats(data: List[dict]) -> Dict[str, dict]:
    """Compute per-idiom statistics."""
    stats = defaultdict(lambda: {"total": 0, "valid": 0, "usages": Counter()})

    for item in data:
        idiom = item["idiom"]
        stats[idiom]["total"] += 1
        stats[idiom]["usages"][item["usage"]] += 1
        if item.get("llm_sentence_valid") == 1 and item.get("llm_span_valid") == 1:
            stats[idiom]["valid"] += 1

    return dict(stats)


def stratified_split(
    data: List[dict],
    train_idioms: int = 22,
    val_idioms: int = 10,
    test_idioms: int = 75,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split data by idiom with stratification.

    Args:
        data: Full annotated dataset.
        train_idioms: Number of idioms for training.
        val_idioms: Number of idioms for validation.
        test_idioms: Number of idioms for testing.
        seed: Random seed for reproducibility.

    Returns:
        (train_data, val_data, test_data) tuples.
    """
    random.seed(seed)

    # Get unique idioms
    all_idioms = sorted(set(item["idiom"] for item in data))
    total_needed = train_idioms + val_idioms + test_idioms

    if len(all_idioms) < total_needed:
        print(f"Warning: only {len(all_idioms)} idioms available, need {total_needed}")
        # Proportional split
        ratio_train = train_idioms / total_needed
        ratio_val = val_idioms / total_needed
        train_idioms = int(len(all_idioms) * ratio_train)
        val_idioms = int(len(all_idioms) * ratio_val)
        test_idioms = len(all_idioms) - train_idioms - val_idioms

    # Shuffle and split
    shuffled = all_idioms.copy()
    random.shuffle(shuffled)

    train_set = set(shuffled[:train_idioms])
    val_set = set(shuffled[train_idioms : train_idioms + val_idioms])
    test_set = set(shuffled[train_idioms + val_idioms : train_idioms + val_idioms + test_idioms])

    train_data = [item for item in data if item["idiom"] in train_set]
    val_data = [item for item in data if item["idiom"] in val_set]
    test_data = [item for item in data if item["idiom"] in test_set]

    return train_data, val_data, test_data


def print_split_stats(train: List[dict], val: List[dict], test: List[dict]):
    """Print statistics for each split."""
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        idioms = set(item["idiom"] for item in split)
        usages = Counter(item["usage"] for item in split)
        print(f"\n{name}:")
        print(f"  Idioms: {len(idioms)}")
        print(f"  Sentences: {len(split)}")
        for usage, count in sorted(usages.items()):
            print(f"  {usage}: {count}")


def save_splits(
    train: List[dict],
    val: List[dict],
    test: List[dict],
    output_dir: str,
):
    """Save splits to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}_data.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name}: {path} ({len(data)} sentences)")


def split_annotated_data(
    input_file: str,
    output_dir: str = "output",
    train_idioms: int = 22,
    val_idioms: int = 10,
    test_idioms: int = 75,
    seed: int = 42,
):
    """Main split function.

    Args:
        input_file: Path to annotated JSON data.
        output_dir: Output directory for split files.
        train_idioms: Number of PIEs for training set.
        val_idioms: Number of PIEs for validation set.
        test_idioms: Number of PIEs for test set.
        seed: Random seed.
    """
    data = load_annotated_data(input_file)
    print(f"Loaded {len(data)} sentences")

    train, val, test = stratified_split(
        data, train_idioms, val_idioms, test_idioms, seed
    )

    print_split_stats(train, val, test)
    save_splits(train, val, test, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split annotated data into train/val/test")
    parser.add_argument("input_file", help="Annotated JSON file")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--train_idioms", type=int, default=22)
    parser.add_argument("--val_idioms", type=int, default=10)
    parser.add_argument("--test_idioms", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_annotated_data(
        args.input_file,
        args.output_dir,
        args.train_idioms,
        args.val_idioms,
        args.test_idioms,
        args.seed,
    )
