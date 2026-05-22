"""Combine idiom variant JSON files from multiple generation runs.

Handles:
- Loading and merging multiple JSON files
- Filtering out invalid idioms (valid=0 in source CSV)
- Deduplication by entry ID
- Sorting by structured ID components
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_invalid_idiom_ids(csv_path: str) -> List[str]:
    """Load IDs of idioms marked as invalid (valid=0) from source CSV."""
    invalid_ids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["valid"] == "0":
                invalid_ids.append(row["id"])
    return invalid_ids


def extract_id_components(entry_id: str) -> Optional[Tuple[str, int, str, int]]:
    """Parse entry ID into (idiom_id, subject_num, usage_type, seq_num).

    ID format: XXX_sYY_ZZZ_NN (e.g., 001_s01_lit_01)
    """
    parts = entry_id.split("_")
    if len(parts) == 4:
        try:
            idiom_id = parts[0]
            subject_num = int(parts[1][1:])  # Remove 's' prefix
            usage_type = parts[2]
            seq_num = int(parts[3])
            return (idiom_id, subject_num, usage_type, seq_num)
        except (ValueError, IndexError):
            return None
    return None


def sort_key(entry: dict) -> tuple:
    """Generate sort key for an entry based on its ID."""
    components = extract_id_components(entry["id"])
    if components:
        return components
    return (entry["id"], 0, "", 0)


def filter_invalid_idioms(
    data: List[dict], invalid_ids: List[str]
) -> Tuple[List[dict], Dict[str, int]]:
    """Remove entries belonging to invalid idioms.

    Returns filtered data and counts of removed entries per idiom ID.
    """
    filtered = []
    removed_counts = defaultdict(int)

    for entry in data:
        idiom_id = entry["id"][:3]
        if idiom_id in invalid_ids:
            removed_counts[idiom_id] += 1
        else:
            filtered.append(entry)

    return filtered, dict(removed_counts)


def combine_variant_files(
    json_files: List[str],
    idioms_csv: Optional[str] = None,
    output_file: Optional[str] = None,
) -> List[dict]:
    """Combine multiple variant JSON files into one.

    Args:
        json_files: List of paths to JSON files to combine.
        idioms_csv: Optional CSV with valid column to filter invalid idioms.
        output_file: Optional output path. If None, returns data without saving.

    Returns:
        Combined and sorted list of entries.
    """
    all_data = []
    seen_ids = set()

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            if entry["id"] not in seen_ids:
                all_data.append(entry)
                seen_ids.add(entry["id"])

    print(f"Combined {len(all_data)} unique entries from {len(json_files)} files")

    # Filter invalid idioms if CSV provided
    if idioms_csv:
        invalid_ids = load_invalid_idiom_ids(idioms_csv)
        all_data, removed = filter_invalid_idioms(all_data, invalid_ids)
        if removed:
            total_removed = sum(removed.values())
            print(f"Removed {total_removed} entries from {len(removed)} invalid idioms")

    # Sort by ID
    all_data.sort(key=sort_key)

    # Save if output path given
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)
        print(f"Saved {len(all_data)} entries to {output_file}")

    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine idiom variant files")
    parser.add_argument("files", nargs="+", help="JSON files to combine")
    parser.add_argument("--csv", help="MAGPIE_SOURCE_IDIOMS.csv for filtering")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    combine_variant_files(args.files, idioms_csv=args.csv, output_file=args.output)
