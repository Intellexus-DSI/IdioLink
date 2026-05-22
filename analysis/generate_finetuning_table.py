"""Generate results table for fine-tuning experiments (mean +/- std across seeds)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.utils import load_config, model_slug

METRICS = ["r_precision", "ndcg@10"]


def collect_results(results_dir: Path, models: list, modes: list, seeds: list):
    """Collect fine-tuning results across seeds."""
    ft_dir = results_dir / "fine_tuning"
    if not ft_dir.exists():
        print(f"No fine-tuning results directory found at {ft_dir}")
        return {}

    results = {}
    for model_id in models:
        slug = model_slug(model_id)
        results[model_id] = {}
        for mode in modes:
            seed_values = {m: [] for m in METRICS}
            for seed in seeds:
                metrics_file = ft_dir / slug / mode / f"seed_{seed}" / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        data = json.load(f)
                    metric_data = data.get("test", data)
                    for m in METRICS:
                        if m in metric_data:
                            seed_values[m].append(metric_data[m])

            if any(seed_values[m] for m in METRICS):
                results[model_id][mode] = {}
                for m in METRICS:
                    vals = seed_values[m]
                    if vals:
                        results[model_id][mode][m] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "n": len(vals),
                        }
    return results


def format_table(results: dict, modes: list) -> list:
    """Format results as table rows with mean +/- std."""
    header = ["model"]
    for mode in modes:
        for metric in METRICS:
            header.append(f"{mode}/{metric}")

    rows = [header]
    for model_id in sorted(results.keys()):
        row = [model_id]
        for mode in modes:
            mode_data = results[model_id].get(mode, {})
            for metric in METRICS:
                if metric in mode_data:
                    m = mode_data[metric]
                    row.append(f"{m['mean']:.4f}+/-{m['std']:.4f}")
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
    models = cfg["training"]["models"]
    modes = cfg["training"]["modes"]
    seeds = cfg["training"]["seeds"]

    results = collect_results(results_dir, models, modes, seeds)

    if not results:
        print("No fine-tuning results found.")
        return

    print(f"Found results for {len(results)} models\n")
    rows = format_table(results, modes)
    print_table(rows)
    save_csv(rows, Path("assets") / "finetuning_results.csv")


if __name__ == "__main__":
    main()
