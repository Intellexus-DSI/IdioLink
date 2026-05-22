"""Generate performance comparison plots."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from idiolink.utils import load_config

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("matplotlib and numpy are required: pip install matplotlib numpy")

MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]
ASSETS_DIR = Path("assets")


def collect_zero_shot_results(results_dir: Path) -> dict:
    """Collect zero-shot results."""
    zs_dir = results_dir / "zero_shot"
    if not zs_dir.exists():
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


def short_name(model_id: str) -> str:
    """Shorten model ID for plot labels."""
    parts = model_id.split("/")
    return parts[-1] if len(parts) > 1 else model_id


def plot_sentence_ndcg(results: dict, output_path: Path):
    """Bar chart: nDCG@10 by model for sentence mode."""
    models = []
    scores = []
    for model_name in sorted(results.keys()):
        data = results[model_name].get("sentence", {})
        if "ndcg@10" in data:
            models.append(short_name(model_name))
            scores.append(data["ndcg@10"])

    if not models:
        print("No sentence mode results to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.5), 6))
    x = np.arange(len(models))
    bars = ax.bar(x, scores, color="steelblue", width=0.6)
    ax.set_xlabel("Model")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Zero-Shot nDCG@10 - Sentence Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_top_models_grouped(results: dict, output_path: Path, top_n: int = 5):
    """Grouped bar chart: 4 modes for top-N models by sentence nDCG@10."""
    # Rank by sentence ndcg
    ranked = []
    for model_name, mode_data in results.items():
        score = mode_data.get("sentence", {}).get("ndcg@10", 0)
        ranked.append((model_name, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_models = [m for m, _ in ranked[:top_n]]

    if not top_models:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_models))
    width = 0.18
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, mode in enumerate(MODES):
        scores = []
        for model_name in top_models:
            val = results[model_name].get(mode, {}).get("ndcg@10", 0)
            scores.append(val)
        offset = (i - 1.5) * width
        ax.bar(x + offset, scores, width, label=mode, color=colors[i])

    ax.set_xlabel("Model")
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"Top-{top_n} Models: nDCG@10 Across Query Modes")
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in top_models], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    cfg = load_config()
    results_dir = Path(cfg["results_dir"])
    results = collect_zero_shot_results(results_dir)

    if not results:
        print("No zero-shot results found. Run experiments first.")
        return

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found results for {len(results)} models\n")
    plot_sentence_ndcg(results, ASSETS_DIR / "zero_shot_sentence_ndcg.png")
    plot_top_models_grouped(results, ASSETS_DIR / "top_models_by_mode.png")


if __name__ == "__main__":
    main()
