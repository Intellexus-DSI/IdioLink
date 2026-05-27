"""CLI for contrastive fine-tuning of embedding models on IdioLink data."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from idiolink.trainer import ContrastiveTrainer, TrainingConfig, TripletDataset
from idiolink.utils import load_config, model_slug, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_triplet_file(data_dir: str, mode: str, split: str) -> str:
    """Resolve triplet file path based on mode and split."""
    # mode: sentence | span | instruction_sentence | instruction_span
    # For span/instruction_span modes, use span triplets; else full
    if "span" in mode:
        suffix = "span"
    else:
        suffix = "full"
    return str(Path(data_dir) / f"triplets_{split}_{suffix}.jsonl")


def run_single_seed(
    config: TrainingConfig,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mode: str,
) -> dict:
    """Train and evaluate for a single seed. Returns test metrics."""
    set_seed(config.seed)

    # Load training dataset
    triplet_file = get_triplet_file(train_dir, mode, "train")
    logger.info(f"Loading training triplets from: {triplet_file}")
    train_dataset = TripletDataset(
        triplet_file,
        max_negatives=config.max_negatives,
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    # Validation and test paths
    val_queries = str(Path(val_dir) / "queries.json")
    val_indexes = str(Path(val_dir) / "indexes.json")
    test_queries = str(Path(test_dir) / "queries.json")
    test_indexes = str(Path(test_dir) / "indexes.json")

    # Validation triplets for reference
    val_triplet_file = get_triplet_file(val_dir, mode, "val")

    # Train
    trainer = ContrastiveTrainer(config)
    logger.info(f"Training model: {config.model_id} | seed: {config.seed} | mode: {mode}")
    val_metrics = trainer.train(train_dataset, val_queries, val_indexes)
    logger.info(f"Best val metrics: {val_metrics}")

    # Test evaluation
    test_metrics = trainer.evaluate_test(test_queries, test_indexes)
    logger.info(f"Test metrics: {test_metrics}")

    # Combine and save
    all_metrics = {
        "model_id": config.model_id,
        "mode": mode,
        "seed": config.seed,
        "val": val_metrics,
        "test": test_metrics,
        **test_metrics,
    }
    trainer.save_metrics(all_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for IdioLink")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sentence",
        choices=["sentence", "span", "instruction_sentence", "instruction_span"],
        help="Query mode for training",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds for repeated runs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})

    # Resolve parameters (CLI > config > defaults)
    model_id = args.model or train_cfg.get("models", ["sentence-transformers/all-MiniLM-L6-v2"])[0]
    seeds = args.seeds or train_cfg.get("seeds", [42, 43, 44])
    # None lets ContrastiveTrainer pull from MODEL_REGISTRY[model_id].batch_size
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size")
    lr = args.lr or train_cfg.get("learning_rate", 2e-5)
    max_epochs = args.max_epochs or train_cfg.get("max_epochs", 10)
    warmup_steps = train_cfg.get("warmup_steps", 100)
    temperature = train_cfg.get("temperature", 0.05)
    patience = train_cfg.get("early_stopping_patience", 3)
    es_metric = train_cfg.get("early_stopping_metric", "ndcg@10")
    device = cfg.get("device", "auto")

    # Data directories
    data_cfg = cfg.get("data", {})
    train_dir = data_cfg.get("train_dir", "data/train")
    val_dir = data_cfg.get("val_dir", "data/val")
    test_dir = data_cfg.get("test_dir", "data/test")
    results_dir = cfg.get("results_dir", "results")

    mode = args.mode
    slug = model_slug(model_id)

    logger.info(f"Model: {model_id}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Batch size: {batch_size}, LR: {lr}, Max epochs: {max_epochs}")

    all_test_metrics = []

    for seed in seeds:
        output_dir = str(
            Path(results_dir) / "fine_tuning" / slug / mode / f"seed_{seed}"
        )

        training_config = TrainingConfig(
            model_id=model_id,
            batch_size=batch_size,
            lr=lr,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            temperature=temperature,
            early_stopping_patience=patience,
            early_stopping_metric=es_metric,
            output_dir=output_dir,
            seed=seed,
            device=device,
            mode=mode,
        )

        test_metrics = run_single_seed(
            training_config, train_dir, val_dir, test_dir, mode
        )
        all_test_metrics.append(test_metrics)

    # Summary across seeds
    if len(all_test_metrics) > 1:
        print("\n" + "=" * 60)
        print(f"SUMMARY: {model_id} | mode={mode} | seeds={seeds}")
        print("=" * 60)
        for metric_name in ["ndcg@10", "r_precision"]:
            values = [m.get(metric_name, 0.0) for m in all_test_metrics]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric_name}: {mean:.4f} +/- {std:.4f}")
        print("=" * 60)

        # Save summary
        summary_dir = Path(results_dir) / "fine_tuning" / slug / mode
        summary = {
            "model_id": model_id,
            "mode": mode,
            "seeds": seeds,
            "test_metrics": all_test_metrics,
            "summary": {
                metric: {
                    "mean": float(np.mean([m.get(metric, 0) for m in all_test_metrics])),
                    "std": float(np.std([m.get(metric, 0) for m in all_test_metrics])),
                }
                for metric in ["ndcg@10", "r_precision"]
            },
        }
        with open(summary_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print(f"\nTest metrics (seed={seeds[0]}): {all_test_metrics[0]}")


if __name__ == "__main__":
    main()
