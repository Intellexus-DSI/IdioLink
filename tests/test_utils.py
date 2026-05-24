"""Tests for idiolink.utils helpers."""

import json
from pathlib import Path

from idiolink.utils import atomic_write_json


def test_atomic_write_json_creates_file(tmp_path: Path):
    target = tmp_path / "metrics.json"
    atomic_write_json(target, {"r_precision": 0.5, "n": 10})
    assert target.exists()
    assert json.loads(target.read_text()) == {"r_precision": 0.5, "n": 10}


def test_atomic_write_json_overwrites_existing(tmp_path: Path):
    target = tmp_path / "metrics.json"
    target.write_text(json.dumps({"old": 1}))
    atomic_write_json(target, {"new": 2})
    assert json.loads(target.read_text()) == {"new": 2}


def test_atomic_write_json_uses_temp_file(tmp_path: Path):
    target = tmp_path / "metrics.json"
    atomic_write_json(target, {"x": 1})
    # tmp file should not linger
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
