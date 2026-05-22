"""Evaluation module: R-Precision and nDCG@10 with idiom-specific relevance rules."""

from typing import Dict, List, Set, Any
import numpy as np
from .utils import IdiomQuery


def build_gold_standard(
    queries: List[IdiomQuery],
    documents: List[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """
    Build gold standard mapping: query_sentence -> set of relevant doc IDs.

    Relevance rules:
      - Literal query -> all literal docs for the same idiom
      - Idiomatic query -> all idiomatic + simplification + sense docs for the same idiom
    """
    idiom_docs: Dict[str, List[Dict[str, Any]]] = {}
    for doc in documents:
        idiom_docs.setdefault(doc["idiom"], []).append(doc)

    gold = {}
    for q in queries:
        relevant = set()
        for doc in idiom_docs.get(q.idiom, []):
            doc_usage = doc.get("usage", "").lower()
            if q.usage_type == "literal" and doc_usage == "literal":
                relevant.add(doc["id"])
            elif q.usage_type == "idiomatic" and doc_usage in ("idiomatic", "simplification", "sense"):
                relevant.add(doc["id"])
        gold[q.query] = relevant
    return gold


def ndcg_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Compute NDCG@k with binary relevance."""
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in gold_ids:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def r_precision(gold_ids: Set[str], retrieved_ids: List[str]) -> float:
    """Compute R-Precision: precision at R where R = |relevant docs|."""
    r = len(gold_ids)
    if r == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:r] if doc_id in gold_ids)
    return hits / r


class Evaluator:
    """Evaluates retrieval results against idiom-specific gold standard."""

    def __init__(self, queries: List[IdiomQuery], documents: List[Dict[str, Any]]):
        self.queries = queries
        self.gold = build_gold_standard(queries, documents)

    def evaluate(self, results: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate retrieval results.

        Args:
            results: mapping of query_sentence -> list of retrieved doc IDs (ranked)

        Returns:
            Dict with r_precision and ndcg@10 (averaged over all queries)
        """
        r_precs = []
        ndcgs = []
        for q in self.queries:
            gold_ids = self.gold.get(q.query, set())
            retrieved = results.get(q.query, [])
            r_precs.append(r_precision(gold_ids, retrieved))
            ndcgs.append(ndcg_at_k(gold_ids, retrieved, 10))

        return {
            "r_precision": float(np.mean(r_precs)),
            "ndcg@10": float(np.mean(ndcgs)),
            "num_queries": len(self.queries),
        }
