"""
Dense retrieval: sentence and span query modes.

Usage:
    python run_dense.py --model intfloat/e5-base-v2
    python run_dense.py --model BAAI/bge-m3 --query_mode span
"""

import argparse
import json
from pathlib import Path

from idiolink.utils import (
    load_config,
    load_queries,
    load_documents,
    get_device,
    set_seed,
    model_slug,
)
from idiolink.models.registry import load_model
from idiolink.models.late_chunking import late_chunk_encode
from idiolink.retriever import DenseRetriever
from idiolink.evaluator import Evaluator
from idiolink.ablation import parse_index_filter, filter_docs_by_usage


def get_query_texts(idiom_queries: list, query_mode: str) -> list:
    """Return the full query context used as the result key for dense modes."""
    if query_mode == "sentence":
        return [q.query for q in idiom_queries]
    elif query_mode == "span":
        return [q.query for q in idiom_queries]
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval baseline")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--query_mode", type=str, default=None, choices=["sentence", "span"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--index_filter",
        type=str,
        default=None,
        help="Restrict the index to a subset of doc usage types. Accepts a preset "
             "name (lit_sim_sense, lit_idiom) or a comma-separated list "
             "(e.g. literal,idiomatic).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    model_id = args.model or cfg["model"]
    query_mode = args.query_mode or cfg["experiment"]["query_mode"]
    device = get_device(cfg["device"])
    top_k = cfg["retrieval"]["top_k"]

    print(f"Model: {model_id}")
    print(f"Query mode: {query_mode}")
    print(f"Device: {device}")

    index_slug = None
    if args.index_filter:
        index_slug, keep = parse_index_filter(args.index_filter)
        print(f"Index filter: {index_slug} (keep {sorted(keep)})")

    # Load data
    test_dir = Path(cfg["data"]["test_dir"])
    doc_sentences, doc_metadata = load_documents(str(test_dir / "indexes.json"))
    query_sentences, idiom_queries = load_queries(str(test_dir / "queries.json"))

    if args.index_filter:
        before = len(doc_sentences)
        doc_sentences, doc_metadata = filter_docs_by_usage(doc_sentences, doc_metadata, keep)
        print(f"Filtered docs: {before} -> {len(doc_sentences)}")

    if args.debug:
        n = cfg.get("debug_samples", 5)
        idiom_queries = idiom_queries[:n]
        query_sentences = query_sentences[:n]

    # Build model and retriever
    model = load_model(model_id, device=device)
    retriever = DenseRetriever(model)

    print(f"Indexing {len(doc_sentences)} documents...")
    retriever.index(doc_sentences, doc_metadata)

    # Encode queries
    query_texts = get_query_texts(idiom_queries, query_mode)
    query_embeddings = None
    if query_mode == "span":
        spans = [q.span if q.span else q.query for q in idiom_queries]
        query_embeddings = late_chunk_encode(model, query_texts, spans, device=device)
    print(f"Retrieving for {len(query_texts)} queries...")
    results = retriever.retrieve(query_texts, top_k=top_k, query_embeddings=query_embeddings)

    # Remap results keys to original sentence for evaluator
    mapped_results = {}
    for q, qt in zip(idiom_queries, query_texts):
        mapped_results[q.query] = results[qt]

    # Evaluate
    evaluator = Evaluator(idiom_queries, [{"id": m["id"], **m} for m in doc_metadata])
    metrics = evaluator.evaluate(mapped_results)

    print(f"\nResults ({model_id} / {query_mode}):")
    print(f"  R-Precision: {metrics['r_precision']:.4f}")
    print(f"  nDCG@10:     {metrics['ndcg@10']:.4f}")
    print(f"  Queries:     {metrics['num_queries']}")

    # Save
    if index_slug:
        output_dir = Path(cfg["results_dir"]) / "ablation" / index_slug / model_slug(model_id) / query_mode
    else:
        output_dir = Path(cfg["results_dir"]) / "zero_shot" / model_slug(model_id) / query_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
