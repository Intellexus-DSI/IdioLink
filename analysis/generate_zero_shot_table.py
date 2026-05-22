"""Generate results table for zero-shot retrieval experiments."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.utils import load_config

MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]
METRICS = ["r_precision", "ndcg@10"]


def collect_results(results_dir: Path):
    """Collect all zero-shot results from the results directory."""
    zs_dir = results_dir / "zero_shot"
    if not zs_dir.exists():
        print(f"No zero-shot results directory found at {zs_dir}")
        return {}

    results = {}
    for model_dir in sorted(zs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        results[model_name] = {}
        for mode in MODES:
            metrics_file = model_dir / mode / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    results[model_name][mode] = json.load(f)
    return results


def format_table(results: dict) -> list:
    """Format results as table rows."""
    header = ["model"]
    for mode in MODES:
        for metric in METRICS:
            header.append(f"{mode}/{metric}")

    rows = [header]
    for model_name in sorted(results.keys()):
        row = [model_name]
        for mode in MODES:
            for metric in METRICS:
                mode_data = results[model_name].get(mode)
                if mode_data and metric in mode_data:
                    row.append(f"{mode_data[metric]:.4f}")
                else:
                    row.append("-")
        rows.append(row)
    return rows


def print_table(rows: list):
    """Print formatted table to stdout."""
    if not rows:
        print("No results found.")
        return

    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    header = rows[0]
    print("  ".join(h.ljust(w) for h, w in zip(header, col_widths)))
    print("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def save_csv(rows: list, output_path: Path):
    """Save results table to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nSaved CSV to: {output_path}")


def main():
    cfg = load_config()
    results_dir = Path(cfg["results_dir"])
    results = collect_results(results_dir)

    if not results:
        print("No zero-shot results found.")
        return

    print(f"Found results for {len(results)} models\n")
    rows = format_table(results)
    print_table(rows)
    save_csv(rows, Path("assets") / "zero_shot_results.csv")


if __name__ == "__main__":
    main()
