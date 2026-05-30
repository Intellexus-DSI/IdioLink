"""Datasets for contrastive fine-tuning of embedding models."""

import json
from typing import Any, Dict, List

from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """
    Dataset that loads pre-mined triplets from a JSONL file.

    Each line: {"query": ..., "positive": ..., "negatives": [...],
                "query_idiom": ..., "query_usage": ..., "query_span": ...,
                "query_subject": ...}

    Mode-agnostic: returns plain fields. The trainer applies per-model
    instruction formatting and per-mode span substitution at encode time.
    """

    def __init__(
        self,
        triplet_file: str,
        max_negatives: int = 5,
    ):
        self.max_negatives = max_negatives
        self.samples: List[Dict[str, Any]] = []
        with open(triplet_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        negatives = item["negatives"][: self.max_negatives]
        return {
            "query": item["query"],
            "query_span": item.get("query_span") or item.get("query_idiom") or item["query"],
            "query_idiom": item.get("query_idiom", ""),
            "query_usage": item.get("query_usage", ""),
            "query_subject": item.get("query_subject", ""),
            "positive": item["positive"],
            "negatives": negatives,
        }
