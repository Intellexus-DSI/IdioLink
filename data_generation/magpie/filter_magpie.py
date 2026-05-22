"""Filter MAGPIE corpus for ambiguous idioms.

Criteria:
- min_occurrences: minimum number of occurrences in corpus (default: 30)
- ambiguity_range: percentage range for idiomatic/literal balance (default: 25-75%)
- min_confidence: minimum annotation confidence (default: 1.0)

Input: MAGPIE JSONL corpus (one entry per line with keys: idiom, label, confidence, id)
Output: Filtered JSONL with ambiguous idioms and a CSV of selected idiom IDs.
"""

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_magpie_corpus(input_file: str) -> Tuple[Counter, dict, dict]:
    """Load MAGPIE corpus and compute per-idiom statistics.

    Returns:
        idiom_counts: Counter of total occurrences per idiom
        idiom_labels: dict mapping idiom -> {label: count}
        idiom_entries: dict mapping idiom -> list of entries
    """
    idiom_counts = Counter()
    idiom_labels = defaultdict(lambda: {"i": 0, "l": 0, "f": 0, "o": 0, "?": 0})
    idiom_entries = defaultdict(list)

    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            idiom = entry["idiom"]
            label = entry["label"]
            idiom_counts[idiom] += 1
            idiom_labels[idiom][label] += 1
            idiom_entries[idiom].append(entry)

    return idiom_counts, dict(idiom_labels), dict(idiom_entries)


def filter_ambiguous_idioms(
    idiom_counts: Counter,
    idiom_labels: dict,
    idiom_entries: dict,
    min_occurrences: int = 30,
    ambiguity_range: int = 25,
    min_confidence: float = 1.0,
) -> Dict[str, dict]:
    """Filter idioms that are ambiguous (balanced literal/idiomatic usage).

    Returns dict mapping idiom -> {total, i, l, i_pct, l_pct, matching_entries, entries}
    """
    frequent_idioms = [
        idiom for idiom, count in idiom_counts.items() if count >= min_occurrences
    ]

    ambiguous = {}
    for idiom in frequent_idioms:
        count = idiom_counts[idiom]
        labels = idiom_labels[idiom]
        idiomatic_pct = (labels["i"] / count) * 100
        literal_pct = (labels["l"] / count) * 100

        in_range = (ambiguity_range <= idiomatic_pct <= (100 - ambiguity_range)) or (
            ambiguity_range <= literal_pct <= (100 - ambiguity_range)
        )
        if not in_range:
            continue

        matching = [
            e
            for e in idiom_entries[idiom]
            if e["label"] in ["i", "l"] and e["confidence"] >= min_confidence
        ]

        ambiguous[idiom] = {
            "total": count,
            "i": labels["i"],
            "l": labels["l"],
            "i_pct": idiomatic_pct,
            "l_pct": literal_pct,
            "matching_entries": len(matching),
            "entries": matching,
        }

    return ambiguous


def save_filtered_output(
    ambiguous: dict,
    output_dir: str,
    min_occurrences: int = 30,
    ambiguity_range: int = 25,
    min_confidence: float = 1.0,
) -> Tuple[Path, Path]:
    """Save filtered corpus and idiom list.

    Returns paths to (jsonl_file, csv_file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"min{min_occurrences}_range{ambiguity_range}-{100 - ambiguity_range}_conf{min_confidence}"

    # Collect and sort all matching entries
    all_entries = []
    for idiom_data in ambiguous.values():
        all_entries.extend(idiom_data["entries"])
    all_entries.sort(key=lambda x: (x["idiom"], x["label"], x.get("id", "")))

    # Write JSONL
    jsonl_path = output_dir / f"MAGPIE_ambiguous_{suffix}.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Write idiom CSV with IDs
    unique_idioms = sorted(set(entry["idiom"] for entry in all_entries))
    csv_path = output_dir / f"MAGPIE_ambiguous_{suffix}_idioms.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "idiom"])
        for idx, idiom in enumerate(unique_idioms, start=1):
            writer.writerow([f"{idx:03d}", idiom])

    return jsonl_path, csv_path


def filter_magpie(
    input_file: str,
    output_dir: str = ".",
    min_occurrences: int = 30,
    ambiguity_range: int = 25,
    min_confidence: float = 1.0,
) -> Dict[str, dict]:
    """Main filtering function.

    Args:
        input_file: Path to MAGPIE JSONL corpus.
        output_dir: Directory for output files.
        min_occurrences: Minimum corpus occurrences for an idiom.
        ambiguity_range: Percent range for idiomatic/literal balance (e.g. 25 means 25%-75%).
        min_confidence: Minimum annotation confidence score.

    Returns:
        Dictionary of ambiguous idioms with their statistics.
    """
    idiom_counts, idiom_labels, idiom_entries = load_magpie_corpus(input_file)

    ambiguous = filter_ambiguous_idioms(
        idiom_counts,
        idiom_labels,
        idiom_entries,
        min_occurrences=min_occurrences,
        ambiguity_range=ambiguity_range,
        min_confidence=min_confidence,
    )

    print(f"Total unique idioms in corpus: {len(idiom_counts)}")
    print(f"Idioms with {min_occurrences}+ occurrences: {len([i for i, c in idiom_counts.items() if c >= min_occurrences])}")
    print(f"Ambiguous idioms found: {len(ambiguous)}")
    print(f"Total matching entries: {sum(d['matching_entries'] for d in ambiguous.values())}")

    if ambiguous:
        save_filtered_output(
            ambiguous, output_dir, min_occurrences, ambiguity_range, min_confidence
        )

    return ambiguous


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter MAGPIE corpus for ambiguous idioms")
    parser.add_argument("input_file", help="Path to MAGPIE JSONL corpus")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--min_occurrences", type=int, default=30)
    parser.add_argument("--ambiguity_range", type=int, default=25)
    parser.add_argument("--min_confidence", type=float, default=1.0)
    args = parser.parse_args()

    filter_magpie(
        args.input_file,
        args.output_dir,
        args.min_occurrences,
        args.ambiguity_range,
        args.min_confidence,
    )
