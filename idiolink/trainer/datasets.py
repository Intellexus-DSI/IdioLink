"""Datasets for contrastive fine-tuning of embedding models."""

import json
import random
from pathlib import Path
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


class DynamicTripletDataset(Dataset):
    """
    Generates triplets on-the-fly from queries.json + indexes.json.

    Hard negatives: same idiom, opposite usage type.
    Soft negatives: different idioms (sampled).
    """

    def __init__(
        self,
        queries_file: str,
        indexes_file: str,
        num_hard_negatives: int = 2,
        num_soft_negatives: int = 3,
        seed: int = 42,
    ):
        self.num_hard_negatives = num_hard_negatives
        self.num_soft_negatives = num_soft_negatives
        self.rng = random.Random(seed)

        # Load queries
        with open(queries_file, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        # Load index documents
        with open(indexes_file, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Organize documents by idiom and usage
        self.idiom_usage_docs: Dict[str, Dict[str, List[str]]] = {}
        self.all_sentences: List[str] = []
        for doc in self.documents:
            idiom = doc["idiom"]
            usage = doc["usage"]
            sentence = doc["sentence"]
            self.all_sentences.append(sentence)
            self.idiom_usage_docs.setdefault(idiom, {}).setdefault(usage, []).append(sentence)

        # Build idiom list for soft negatives
        self.idioms = list(self.idiom_usage_docs.keys())

    def set_epoch(self, epoch: int):
        """Reset RNG for per-epoch randomization."""
        self.rng = random.Random(42 + epoch)

    def __len__(self) -> int:
        return len(self.queries)

    def _get_opposite_usage(self, usage: str) -> List[str]:
        """Return usage types that are 'opposite' for hard negatives."""
        if usage == "literal":
            return ["idiomatic", "simplification", "sense"]
        else:
            return ["literal"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query_item = self.queries[idx]
        query_sentence = query_item["sentence"]
        query_idiom = query_item["idiom"]
        query_usage = query_item["usage"]

        # Positive: same idiom, same usage type
        same_usage_docs = self.idiom_usage_docs.get(query_idiom, {}).get(query_usage, [])
        if same_usage_docs:
            positive = self.rng.choice(same_usage_docs)
        else:
            # Fallback: any doc from same idiom
            all_same_idiom = []
            for docs in self.idiom_usage_docs.get(query_idiom, {}).values():
                all_same_idiom.extend(docs)
            positive = self.rng.choice(all_same_idiom) if all_same_idiom else query_sentence

        # Hard negatives: same idiom, opposite usage
        hard_negatives = []
        opposite_usages = self._get_opposite_usage(query_usage)
        for opp_usage in opposite_usages:
            hard_negatives.extend(
                self.idiom_usage_docs.get(query_idiom, {}).get(opp_usage, [])
            )
        if len(hard_negatives) > self.num_hard_negatives:
            hard_negatives = self.rng.sample(hard_negatives, self.num_hard_negatives)

        # Soft negatives: different idioms
        soft_negatives = []
        other_idioms = [i for i in self.idioms if i != query_idiom]
        sampled_idioms = self.rng.sample(
            other_idioms, min(self.num_soft_negatives, len(other_idioms))
        )
        for idiom in sampled_idioms:
            idiom_docs = []
            for docs in self.idiom_usage_docs.get(idiom, {}).values():
                idiom_docs.extend(docs)
            if idiom_docs:
                soft_negatives.append(self.rng.choice(idiom_docs))

        negatives = hard_negatives + soft_negatives
        return {
            "query": query_sentence,
            "positive": positive,
            "negatives": negatives,
        }
