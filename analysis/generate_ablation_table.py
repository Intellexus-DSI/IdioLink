"""Generate per-mode ablation tables.

One table per query mode (sentence, span, instruction_sentence, instruction_span).
Each table has one row per embedding model with both ablation indices
(`lit_sim_sense` and `lit_idiom`) side by side, each broken into
overall / literal-queries / idiomatic-queries (R-Prec, nDCG@10).

BM25 has no instruction modes, so the BM25 row always uses its `sentence`-mode
numbers regardless of which table is being rendered.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.ablation import ABLATION_PRESETS
from idiolink.utils import load_config

MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]
INDICES = ["lit_sim_sense", "lit_idiom"]
BM25_FALLBACK_MODE = "sentence"

METRIC_KEYS = [
    ("R-P", "r_precision"),
    ("nDCG@10", "ndcg@10"),
    ("lit R-P", "r_precision_literal"),
    ("lit nDCG", "ndcg@10_literal"),
    ("idi R-P", "r_precision_idiomatic"),
    ("idi nDCG", "ndcg@10_idiomatic"),
]


def load_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        print(f"No ablation results CSV at {csv_path}")
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(v) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def build_lookup(rows: list) -> dict:
    """Return lookup[(model, index, mode)] -> row_dict."""
    return {(r["model"], r["index"], r["query_mode"]): r for r in rows}


def row_for(lookup: dict, model: str, index: str, mode: str) -> dict | None:
    if model == "bm25":
        return lookup.get((model, index, BM25_FALLBACK_MODE))
    return lookup.get((model, index, mode))


def print_mode_table(mode: str, models: list, lookup: dict):
    print(f"\n=== Query mode: {mode} ===")
    print("(BM25 row uses sentence-mode numbers; it has no instruction modes.)\n")

    header_top = ["model"]
    header_bot = [""]
    for idx in INDICES:
        for label, _ in METRIC_KEYS:
            header_top.append(f"{idx}/{label}")
            header_bot.append("")

    rows = [header_top]
    for model in models:
        row = [model]
        for idx in INDICES:
            r = row_for(lookup, model, idx, mode)
            for _, key in METRIC_KEYS:
                row.append(fmt(r.get(key)) if r else "-")
        rows.append(row)
    _print_table(rows)


def _print_table(rows: list):
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    print("  ".join(h.ljust(w) for h, w in zip(rows[0], widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows[1:]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def save_per_mode_csv(mode: str, models: list, lookup: dict, out_dir: Path):
    """Save a wide CSV per mode (one row per model, 12 metric cols + model)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ablation_{mode}.csv"
    header = ["model"] + [f"{idx}__{label.replace(' ', '_')}" for idx in INDICES for label, _ in METRIC_KEYS]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for model in models:
            row = [model]
            for idx in INDICES:
                r = row_for(lookup, model, idx, mode)
                for _, key in METRIC_KEYS:
                    row.append(fmt(r.get(key)) if r else "")
            w.writerow(row)
    print(f"Saved {out}")


def main():
    cfg = load_config()
    results_dir = Path(cfg["results_dir"])
    csv_path = results_dir / "ablation" / "full_results.csv"

    rows = load_rows(csv_path)
    if not rows:
        return

    lookup = build_lookup(rows)
    # Collect all models present in the data; sort dense models alphabetically
    # with bm25 first for visibility.
    all_models = sorted({r["model"] for r in rows})
    dense_models = [m for m in all_models if m != "bm25"]
    models = (["bm25"] if "bm25" in all_models else []) + dense_models

    print(f"Found {len(rows)} ablation rows across {len(all_models)} model labels.")
    for slug in INDICES:
        keep = sorted(ABLATION_PRESETS[slug])
        print(f"  {slug}: keep {keep}")

    assets_dir = Path("assets") / "ablation"
    for mode in MODES:
        print_mode_table(mode, models, lookup)
        save_per_mode_csv(mode, models, lookup, assets_dir)


if __name__ == "__main__":
    main()
