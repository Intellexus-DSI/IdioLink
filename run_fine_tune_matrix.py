"""Fine-tuning matrix runner: models × modes × seeds with resume + aggregate CSV.

Mirror of run_ablation.py for the training side. Resumes by checking that
per-(model, mode, seed) metrics.json exists with a current _trainer_version
stamp. Per-model batch_size pulled from registry unless overridden.

Usage:
    python run_fine_tune_matrix.py                       # full matrix from config
    python run_fine_tune_matrix.py --models <id> ...
    python run_fine_tune_matrix.py --modes sentence span
    python run_fine_tune_matrix.py --seeds 42 43 44
    python run_fine_tune_matrix.py --force               # recompute all
    python run_fine_tune_matrix.py --dry_run             # print matrix, exit
"""

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from idiolink.trainer import ContrastiveTrainer, TrainingConfig, TripletDataset
from idiolink.trainer.contrastive_trainer import TRAINER_VERSION
from idiolink.models.registry import MODEL_REGISTRY
from idiolink.utils import atomic_write_json, load_config, model_slug, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CSV_FIELDS = [
    "model", "mode", "seed",
    "r_precision", "ndcg@10",
    "num_queries",
    "_trainer_version",
]


def _metrics_path(results_dir: Path, model_id: str, mode: str, seed: int) -> Path:
    return results_dir / "fine_tuning" / model_slug(model_id) / mode / f"seed_{seed}" / "metrics.json"


def _is_complete(path: Path) -> bool:
    """True iff metrics.json exists AND has the current _trainer_version."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return data.get("_trainer_version") == TRAINER_VERSION


def _resolve_batch_size(args_batch_size, train_cfg, model_id) -> Optional[int]:
    """CLI > config > registry (resolved in trainer)."""
    if args_batch_size is not None:
        return args_batch_size
    if train_cfg.get("batch_size") is not None:
        return train_cfg.get("batch_size")
    # None lets ContrastiveTrainer pull from MODEL_REGISTRY[model_id].batch_size
    return None


def get_triplet_file(data_dir: str, mode: str, split: str) -> str:
    suffix = "span" if "span" in mode else "full"
    return str(Path(data_dir) / f"triplets_{split}_{suffix}.jsonl")


def run_single_seed(
    config: TrainingConfig,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mode: str,
) -> Dict:
    """Train and evaluate for a single seed. Returns test metrics dict."""
    set_seed(config.seed)

    triplet_file = get_triplet_file(train_dir, mode, "train")
    logger.info(f"Loading training triplets from: {triplet_file}")
    train_dataset = TripletDataset(triplet_file, max_negatives=config.max_negatives)
    logger.info(f"Training samples: {len(train_dataset)}")

    val_queries = str(Path(val_dir) / "queries.json")
    val_indexes = str(Path(val_dir) / "indexes.json")
    test_queries = str(Path(test_dir) / "queries.json")
    test_indexes = str(Path(test_dir) / "indexes.json")

    trainer = ContrastiveTrainer(config)
    logger.info(f"Training model: {config.model_id} | seed: {config.seed} | mode: {mode}")
    val_metrics = trainer.train(train_dataset, val_queries, val_indexes)
    logger.info(f"Best val metrics: {val_metrics}")

    test_metrics = trainer.evaluate_test(test_queries, test_indexes)
    logger.info(f"Test metrics: {test_metrics}")

    trainer.save_metrics(test_metrics)
    return test_metrics


def _flatten_for_csv(test_metrics: Dict, model_id: str, mode: str, seed: int) -> Dict:
    return {
        "model": model_id,
        "mode": mode,
        "seed": seed,
        "r_precision": test_metrics.get("r_precision", 0.0),
        "ndcg@10": test_metrics.get("ndcg@10", 0.0),
        "num_queries": test_metrics.get("num_queries", 0),
        "_trainer_version": TRAINER_VERSION,
    }


def _collect_all_rows_from_disk(
    results_dir: Path,
    models: List[str],
    modes: List[str],
    seeds: List[int],
) -> List[dict]:
    """Walk results/fine_tuning/ and rebuild rows from every metrics.json."""
    out: List[dict] = []
    for model_id in models:
        for mode in modes:
            for seed in seeds:
                mp = _metrics_path(results_dir, model_id, mode, seed)
                if not mp.exists():
                    continue
                try:
                    metrics = json.loads(mp.read_text())
                    out.append(_flatten_for_csv(metrics, model_id, mode, seed))
                except Exception as e:
                    logger.warning(f"Could not read {mp}: {e}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning matrix (models x modes x seeds)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Default: cfg['training']['models']")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=["sentence", "span", "instruction_sentence", "instruction_span"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute (model, mode, seed) cells even if metrics.json exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the matrix and exit without training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    results_dir = Path(cfg["results_dir"])

    models = args.models or train_cfg["models"]
    modes = args.modes or train_cfg["modes"]
    seeds = args.seeds or train_cfg["seeds"]

    logger.info(f"Matrix: {len(models)} models x {len(modes)} modes x {len(seeds)} seeds "
                f"= {len(models)*len(modes)*len(seeds)} cells")
    logger.info(f"Models: {models}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Seeds: {seeds}")

    if args.dry_run:
        for m in models:
            for mo in modes:
                for s in seeds:
                    mp = _metrics_path(results_dir, m, mo, s)
                    status = "DONE" if _is_complete(mp) else "PENDING"
                    logger.info(f"  [{status}] {m} / {mo} / seed={s} -> {mp}")
        return

    failed: List[tuple] = []
    for model_id in models:
        for mode in modes:
            for seed in seeds:
                mp = _metrics_path(results_dir, model_id, mode, seed)
                if not args.force and _is_complete(mp):
                    logger.info(f"  SKIP existing: {model_id} / {mode} / seed={seed}")
                    continue

                output_dir = str(mp.parent)
                training_config = TrainingConfig(
                    model_id=model_id,
                    batch_size=_resolve_batch_size(args.batch_size, train_cfg, model_id),
                    lr=train_cfg.get("learning_rate", 2e-5),
                    max_epochs=train_cfg.get("max_epochs", 10),
                    warmup_steps=train_cfg.get("warmup_steps", 100),
                    temperature=train_cfg.get("temperature", 0.05),
                    early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
                    early_stopping_metric=train_cfg.get("early_stopping_metric", "ndcg@10"),
                    output_dir=output_dir,
                    seed=seed,
                    device=cfg.get("device", "auto"),
                    mode=mode,
                )
                try:
                    run_single_seed(
                        training_config,
                        data_cfg["train_dir"],
                        data_cfg["val_dir"],
                        data_cfg["test_dir"],
                        mode,
                    )
                except Exception as e:
                    logger.error(f"FAILED {model_id} / {mode} / seed={seed}: {e}")
                    traceback.print_exc()
                    failed.append((model_id, mode, seed))

    # Rebuild aggregate CSV from every metrics.json on disk
    rows = _collect_all_rows_from_disk(results_dir, models, modes, seeds)
    if rows:
        agg_path = results_dir / "fine_tuning" / "full_results.csv"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Aggregated results saved to {agg_path} ({len(rows)} rows)")
    else:
        logger.warning("No fine-tuning results to aggregate.")

    if failed:
        logger.error(f"\n{len(failed)} cells failed:")
        for m, mo, s in failed:
            logger.error(f"  {m} / {mo} / seed={s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
