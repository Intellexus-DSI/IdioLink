"""Tests for run_fine_tune_matrix.py: resume + force + aggregate CSV."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_config(tmp_path: Path, train_dir: Path) -> Path:
    cfg = {
        "device": "cpu",
        "seed": 42,
        "results_dir": str(tmp_path / "results"),
        "data": {
            "train_dir": str(train_dir),
            "val_dir": str(train_dir),
            "test_dir": str(train_dir),
        },
        "training": {
            "models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "modes": ["sentence"],
            "seeds": [42, 43],
            "batch_size": 4,
            "max_epochs": 1,
            "learning_rate": 2e-5,
            "warmup_steps": 0,
            "temperature": 0.05,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
        },
        "retrieval": {"top_k": 10},
    }
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _write_data(train_dir: Path):
    """Write minimal triplets + queries + indexes for the runner."""
    train_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (train_dir / f"triplets_{split}_full.jsonl").write_text(
            json.dumps({"query": "x", "positive": "y", "negatives": ["z"]}) + "\n"
        )
    (train_dir / "queries.json").write_text(json.dumps([
        {"sentence": "x", "idiom": "x", "usage": "literal", "span": "x", "subject": ""},
    ]))
    (train_dir / "indexes.json").write_text(json.dumps([
        {"sentence": "y", "id": "d1", "idiom": "x", "usage": "literal", "subject": ""},
    ]))


def _fake_run_single_seed_factory(metrics: dict):
    """Build a side_effect callable that mimics real run_single_seed by writing
    metrics.json into the TrainingConfig.output_dir (the real run does this
    via trainer.save_metrics). Returns the metrics dict the runner expects.
    """
    from idiolink.utils import atomic_write_json
    from idiolink.trainer.contrastive_trainer import TRAINER_VERSION

    def _impl(config, train_dir, val_dir, test_dir, mode):
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "metrics.json", {**metrics, "_trainer_version": TRAINER_VERSION})
        return dict(metrics)

    return _impl


def test_matrix_runner_writes_per_cell_metrics_and_aggregate(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner
    metrics = {"r_precision": 0.5, "ndcg@10": 0.6, "num_queries": 1}

    with patch.object(runner, "run_single_seed", side_effect=_fake_run_single_seed_factory(metrics)):
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()

    rd = tmp_path / "results" / "fine_tuning"
    # 1 model x 1 mode x 2 seeds = 2 metrics.json
    files = list(rd.rglob("metrics.json"))
    assert len(files) == 2, f"expected 2 metrics.json files, found {files}"
    agg = rd / "full_results.csv"
    assert agg.exists(), "aggregate CSV missing"
    rows = list(csv.DictReader(open(agg)))
    assert len(rows) == 2
    # Each row must reflect the mock metrics
    for row in rows:
        assert float(row["r_precision"]) == 0.5
        assert float(row["ndcg@10"]) == 0.6


def test_matrix_runner_skips_existing_metrics(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner
    metrics = {"r_precision": 0.5, "ndcg@10": 0.6, "num_queries": 1}

    # First invocation: runs both seeds, writes metrics.json each.
    with patch.object(runner, "run_single_seed",
                      side_effect=_fake_run_single_seed_factory(metrics)) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()
        assert mock_run.call_count == 2

    # Second invocation: should skip both (metrics.json exists with current TRAINER_VERSION).
    with patch.object(runner, "run_single_seed",
                      side_effect=_fake_run_single_seed_factory(metrics)) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()
        assert mock_run.call_count == 0


def test_matrix_runner_force_recomputes(tmp_path: Path):
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner
    first = {"r_precision": 0.5, "ndcg@10": 0.6, "num_queries": 1}
    second = {"r_precision": 0.7, "ndcg@10": 0.8, "num_queries": 1}

    with patch.object(runner, "run_single_seed",
                      side_effect=_fake_run_single_seed_factory(first)):
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()

    with patch.object(runner, "run_single_seed",
                      side_effect=_fake_run_single_seed_factory(second)) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path), "--force"]
        runner.main()
        assert mock_run.call_count == 2

    # The rewrite must reflect the second metrics
    agg = tmp_path / "results" / "fine_tuning" / "full_results.csv"
    rows = list(csv.DictReader(open(agg)))
    for row in rows:
        assert float(row["r_precision"]) == 0.7


def test_matrix_runner_skips_stale_trainer_version(tmp_path: Path):
    """If metrics.json exists but _trainer_version is older, recompute."""
    train_dir = tmp_path / "data"
    _write_data(train_dir)
    cfg_path = _write_config(tmp_path, train_dir)

    import run_fine_tune_matrix as runner
    from idiolink.trainer.contrastive_trainer import TRAINER_VERSION

    # Pre-seed a stale metrics.json for seed=42
    stale_path = (
        tmp_path / "results" / "fine_tuning"
        / "sentence-transformers__all-MiniLM-L6-v2" / "sentence" / "seed_42" / "metrics.json"
    )
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text(json.dumps({
        "r_precision": 0.0, "ndcg@10": 0.0, "_trainer_version": TRAINER_VERSION - 1,
    }))

    metrics = {"r_precision": 0.5, "ndcg@10": 0.6, "num_queries": 1}
    with patch.object(runner, "run_single_seed",
                      side_effect=_fake_run_single_seed_factory(metrics)) as mock_run:
        sys.argv = ["run_fine_tune_matrix.py", "--config", str(cfg_path)]
        runner.main()
        # Both cells should run: stale seed=42 (version mismatch) + new seed=43.
        assert mock_run.call_count == 2
