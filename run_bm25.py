"""
BM25 lexical baseline for IdioLink retrieval.

Usage:
    python run_bm25.py --query_mode sentence
    python run_bm25.py --query_mode span --tune
    python run_bm25.py --tune --config config.yaml
"""

import argparse
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from idiolink.utils import load_config, load_queries, load_documents, set_seed, IdiomQuery
from idiolink.evaluator import Evaluator

# Simple tokenizer: words including contractions (e.g., don't, it's)
TOKENIZER = re.compile(r"\b\w+(?:'\w+)?\b")


def tokenize(text: str) -> list:
    """Tokenize text into lowercase word tokens."""
    return TOKENIZER.findall(text.lower())


def get_query_texts(idiom_queries: list, query_mode: str) -> list:
    """Extract query text based on mode: sentence or span."""
    if query_mode == "sentence":
        return [q.query for q in idiom_queries]
    elif query_mode == "span":
        return [q.span if q.span else q.query for q in idiom_queries]
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")


def run_bm25(
    doc_sentences: list,
    doc_metadata: list,
    idiom_queries: list,
    query_mode: str,
    k1: float,
    b: float,
    top_k: int = 100,
) -> dict:
    """
    Run BM25 retrieval and return results dict mapping query.query -> list of doc IDs.
    """
    # Tokenize all documents
    tokenized_corpus = [tokenize(sent) for sent in doc_sentences]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Get query texts based on mode
    query_texts = get_query_texts(idiom_queries, query_mode)

    # Retrieve for each query
    results = {}
    for q, qt in zip(idiom_queries, query_texts):
        tokenized_query = tokenize(qt)
        scores = bm25.get_scores(tokenized_query)
        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        # Map indices to doc IDs
        doc_ids = [doc_metadata[i]["id"] for i in top_indices]
        results[q.query] = doc_ids

    return results


def build_evaluator_docs(doc_sentences: list, doc_metadata: list) -> list:
    """Reconstruct full document dicts for the Evaluator."""
    docs = []
    for sent, meta in zip(doc_sentences, doc_metadata):
        doc = {"sentence": sent, **meta}
        docs.append(doc)
    return docs


def run_tuning(cfg: dict, query_mode: str, top_k: int) -> tuple:
    """
    Grid search over k1 and b on the validation set.
    Returns (best_k1, best_b, tuning_results_list).
    """
    bm25_cfg = cfg["bm25"]
    k1_grid = bm25_cfg["k1_grid"]
    b_grid = bm25_cfg["b_grid"]
    tune_metric = bm25_cfg["tune_metric"]

    # Load validation data
    val_dir = Path(cfg["data"]["val_dir"])
    val_doc_sentences, val_doc_metadata = load_documents(str(val_dir / "indexes.json"))
    _, val_idiom_queries = load_queries(str(val_dir / "queries.json"))

    val_docs = build_evaluator_docs(val_doc_sentences, val_doc_metadata)
    evaluator = Evaluator(val_idiom_queries, val_docs)

    print(f"Tuning on validation set ({len(val_idiom_queries)} queries, {len(val_doc_sentences)} docs)")
    print(f"Grid: k1={k1_grid}, b={b_grid}")
    print(f"Optimizing: {tune_metric}")

    best_score = -1.0
    best_k1 = k1_grid[0]
    best_b = b_grid[0]
    tuning_results = []

    for k1, b in product(k1_grid, b_grid):
        results = run_bm25(
            val_doc_sentences, val_doc_metadata, val_idiom_queries,
            query_mode, k1, b, top_k,
        )
        metrics = evaluator.evaluate(results)
        score = metrics[tune_metric]
        tuning_results.append({
            "k1": k1,
            "b": b,
            "metrics": metrics,
        })
        if score > best_score:
            best_score = score
            best_k1 = k1
            best_b = b

    print(f"Best: k1={best_k1}, b={best_b}, {tune_metric}={best_score:.4f}")
    return best_k1, best_b, tuning_results


def main():
    parser = argparse.ArgumentParser(description="BM25 lexical baseline")
    parser.add_argument("--query_mode", type=str, default=None, choices=["sentence", "span"])
    parser.add_argument("--tune", action="store_true", help="Run grid search on validation set")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    bm25_cfg = cfg["bm25"]
    query_mode = args.query_mode or bm25_cfg["query_mode"]
    top_k = cfg["retrieval"]["top_k"]

    print(f"BM25 Baseline")
    print(f"Query mode: {query_mode}")

    # Tuning phase
    k1 = bm25_cfg["k1"]
    b = bm25_cfg["b"]

    if args.tune:
        k1, b, tuning_results = run_tuning(cfg, query_mode, top_k)

        # Save tuning results
        output_dir = Path(cfg["results_dir"]) / "bm25" / query_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "tuning_results.json", "w") as f:
            json.dump(tuning_results, f, indent=2)
        print(f"Tuning results saved to: {output_dir / 'tuning_results.json'}")
    else:
        print(f"Using default params: k1={k1}, b={b}")

    # Test phase
    print(f"\nRunning on test set with k1={k1}, b={b}...")
    test_dir = Path(cfg["data"]["test_dir"])
    doc_sentences, doc_metadata = load_documents(str(test_dir / "indexes.json"))
    _, idiom_queries = load_queries(str(test_dir / "queries.json"))

    results = run_bm25(doc_sentences, doc_metadata, idiom_queries, query_mode, k1, b, top_k)

    # Evaluate
    test_docs = build_evaluator_docs(doc_sentences, doc_metadata)
    evaluator = Evaluator(idiom_queries, test_docs)
    metrics = evaluator.evaluate(results)

    print(f"\nResults (BM25 / {query_mode}):")
    print(f"  R-Precision: {metrics['r_precision']:.4f}")
    print(f"  nDCG@10:     {metrics['ndcg@10']:.4f}")
    print(f"  Queries:     {metrics['num_queries']}")
    print(f"  k1={k1}, b={b}")

    # Add params to saved metrics
    metrics["k1"] = k1
    metrics["b"] = b
    metrics["query_mode"] = query_mode

    # Save metrics
    output_dir = Path(cfg["results_dir"]) / "bm25" / query_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
