"""Utility functions for config loading, file I/O, and device detection."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import yaml
import torch


@dataclass
class IdiomQuery:
    query: str
    idiom: str
    usage_type: str  # "literal" or "idiomatic"
    span: str = ""
    subject: str = ""


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_json(file_path: str) -> Any:
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(query_file: str) -> Tuple[List[str], List[IdiomQuery]]:
    """Load queries from JSON file. Returns (query_sentences, IdiomQuery objects)."""
    data = load_json(query_file)
    query_strings = []
    idiom_queries = []
    for item in data:
        query_str = item["sentence"]
        query_strings.append(query_str)
        idiom_queries.append(IdiomQuery(
            query=query_str,
            idiom=item["idiom"],
            usage_type=item["usage"],
            span=item.get("span", ""),
            subject=item.get("subject", ""),
        ))
    return query_strings, idiom_queries


def load_documents(index_file: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load index documents. Returns (sentences, metadata_list)."""
    data = load_json(index_file)
    sentences = []
    metadata = []
    for item in data:
        sentences.append(item["sentence"])
        metadata.append({k: v for k, v in item.items() if k != "sentence"})
    return sentences, metadata


def get_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return model_id.replace("/", "__")
