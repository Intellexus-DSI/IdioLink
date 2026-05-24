"""Generate one results table per query variant, broken down by the evaluator splits.

For each of the 4 query modes (sentence, span, instruction_sentence, instruction_span),
emit a table whose rows are models and whose columns are R-Precision and nDCG@10 on the
four splits: overall, literal, idiomatic, by_subject. Saves a CSV per variant under
`assets/variant_<mode>.csv` and prints to stdout.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.utils import load_config

MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]
SPLITS = ["overall", "literal", "idiomatic", "by_subject"]
METRICS = ["r_precision", "ndcg@10"]
METRIC_LABEL = {"r_precision": "R-P", "ndcg@10": "nDCG@10"}


def extract_split(metrics: dict, split: str, metric: str):
    """Pull a single (split, metric) value from a metrics.json dict. Returns None if missing."""
    if not metrics:
        return None
    if split == "overall":
        return metrics.get(metric)
    if split in ("literal", "idiomatic"):
        return metrics.get("by_usage", {}).get(split, {}).get(metric)
    if split == "by_subject":
        return metrics.get("by_subject", {}).get(metric)
    return None


def collect_results(results_dir: Path):
    """Returns {model_name: {mode: metrics_dict}}."""
    zs_dir = results_dir / "zero_shot"
    if not zs_dir.exists():
        return {}
    out = {}
    for model_dir in sorted(zs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        out[model_name] = {}
        for mode in MODES:
            f = model_dir / mode / "metrics.json"
            if f.exists():
                with open(f) as fh:
                    out[model_name][mode] = json.load(fh)
    return out


def build_variant_table(results: dict, mode: str):
    """Return (header_rows, data_rows) for one variant.

    header_rows is two lines for grouped columns (R-P group + nDCG group).
    """
    top_header = ["model"]
    sub_header = [""]
    for metric in METRICS:
        for split in SPLITS:
            top_header.append(METRIC_LABEL[metric])
            sub_header.append(split)

    rows = []
    for model_name in sorted(results.keys()):
        row = [model_name]
        metrics = results[model_name].get(mode)
        for metric in METRICS:
            for split in SPLITS:
                v = extract_split(metrics, split, metric)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
        rows.append(row)
    return top_header, sub_header, rows


def print_table(top: list, sub: list, rows: list):
    cols = [top, sub] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(top))]
    def fmt(row):
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))
    print(fmt(top))
    print(fmt(sub))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def save_csv(top: list, sub: list, rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(top)
        writer.writerow(sub)
        writer.writerows(rows)


def main():
    cfg = load_config()
    results_dir = Path(cfg["results_dir"])
    results = collect_results(results_dir)
    if not results:
        print("No zero-shot results found.")
        return

    print(f"Found results for {len(results)} model(s)")
    for mode in MODES:
        print(f"\n=== Variant: {mode} ===")
        top, sub, rows = build_variant_table(results, mode)
        print_table(top, sub, rows)
        out_path = Path("assets") / f"variant_{mode}.csv"
        save_csv(top, sub, rows, out_path)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
