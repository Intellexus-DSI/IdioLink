"""Evaluation module: R-Precision and nDCG@10 with idiom-specific relevance rules."""

from typing import Dict, List, Optional, Set, Any
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


def build_subject_gold(
    queries: List[IdiomQuery],
    documents: List[Dict[str, Any]],
) -> Dict[str, Optional[Set[str]]]:
    """
    Subject-based gold: a doc is relevant to a query iff their `subject` fields match.

    This is a topical-coherence signal, not an idiom-relevance signal. It is
    intentionally weaker than `build_gold_standard`: it ignores idiom identity
    and usage type. Use it for diagnostic comparison only; the headline metric
    remains the idiom-relevance gold.

    Queries without a subject are mapped to None so they can be excluded from the
    subject-based metric (rather than averaging in a degenerate 0).
    """
    subject_docs: Dict[str, Set[str]] = {}
    for doc in documents:
        subj = doc.get("subject", "")
        if subj:
            subject_docs.setdefault(subj, set()).add(doc["id"])
    gold: Dict[str, Optional[Set[str]]] = {}
    for q in queries:
        gold[q.query] = subject_docs.get(q.subject, set()) if q.subject else None
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


def _avg(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


class Evaluator:
    """Evaluates retrieval results against idiom-specific gold standard."""

    def __init__(self, queries: List[IdiomQuery], documents: List[Dict[str, Any]]):
        self.queries = queries
        self.gold = build_gold_standard(queries, documents)
        self.subject_gold = build_subject_gold(queries, documents)

    def evaluate(self, results: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate retrieval results.

        Returns a dict with the overall idiom-relevance metrics at the top level
        (unchanged for backward compatibility), plus:
          - `by_usage`: same metrics computed on literal-only and idiomatic-only subsets.
          - `by_subject`: same metrics with the binary gold defined by matching `subject`.
        """
        r_precs: List[float] = []
        ndcgs: List[float] = []
        bucket_rp: Dict[str, List[float]] = {"literal": [], "idiomatic": []}
        bucket_nd: Dict[str, List[float]] = {"literal": [], "idiomatic": []}
        subj_rp: List[float] = []
        subj_nd: List[float] = []

        for q in self.queries:
            retrieved = results.get(q.query, [])

            gold_ids = self.gold.get(q.query, set())
            rp = r_precision(gold_ids, retrieved)
            nd = ndcg_at_k(gold_ids, retrieved, 10)
            r_precs.append(rp)
            ndcgs.append(nd)

            if q.usage_type in bucket_rp:
                bucket_rp[q.usage_type].append(rp)
                bucket_nd[q.usage_type].append(nd)

            sgold = self.subject_gold.get(q.query)
            if sgold is not None:
                subj_rp.append(r_precision(sgold, retrieved))
                subj_nd.append(ndcg_at_k(sgold, retrieved, 10))

        return {
            "r_precision": _avg(r_precs),
            "ndcg@10": _avg(ndcgs),
            "num_queries": len(self.queries),
            "by_usage": {
                "literal": {
                    "r_precision": _avg(bucket_rp["literal"]),
                    "ndcg@10": _avg(bucket_nd["literal"]),
                    "num_queries": len(bucket_rp["literal"]),
                },
                "idiomatic": {
                    "r_precision": _avg(bucket_rp["idiomatic"]),
                    "ndcg@10": _avg(bucket_nd["idiomatic"]),
                    "num_queries": len(bucket_rp["idiomatic"]),
                },
            },
            "by_subject": {
                "r_precision": _avg(subj_rp),
                "ndcg@10": _avg(subj_nd),
                "num_queries": len(subj_rp),
            },
        }
