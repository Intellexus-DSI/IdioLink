"""
Instruction-based dense retrieval: instruction_sentence and instruction_span modes.

Usage:
    python run_instruction.py --model intfloat/multilingual-e5-large-instruct --query_mode instruction_sentence
    python run_instruction.py --model BAAI/bge-base-en-v1.5 --query_mode instruction_span
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
from idiolink.models.registry import MODEL_REGISTRY, load_model
from idiolink.models.encode_helpers import encode_queries_for_mode
from idiolink.retriever import DenseRetriever
from idiolink.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Instruction-based dense retrieval")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--query_mode", type=str, default="instruction_sentence",
        choices=["instruction_sentence", "instruction_span"],
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    model_id = args.model or cfg["model"]
    query_mode = args.query_mode
    device = get_device(cfg["device"])
    top_k = cfg["retrieval"]["top_k"]

    print(f"Model: {model_id}")
    print(f"Query mode: {query_mode}")
    print(f"Device: {device}")

    # Load data
    test_dir = Path(cfg["data"]["test_dir"])
    doc_sentences, doc_metadata = load_documents(str(test_dir / "indexes.json"))
    query_sentences, idiom_queries = load_queries(str(test_dir / "queries.json"))

    if args.debug:
        n = cfg.get("debug_samples", 5)
        idiom_queries = idiom_queries[:n]
        query_sentences = query_sentences[:n]

    # Load model from registry
    model = load_model(model_id, device=device)
    retriever = DenseRetriever(model)

    print(f"Indexing {len(doc_sentences)} documents...")
    retriever.index(doc_sentences, doc_metadata)

    query_texts, query_embeddings = encode_queries_for_mode(
        model, query_mode, idiom_queries, device,
    )

    # Retrieve
    print(f"Retrieving for {len(query_texts)} queries...")
    results = retriever.retrieve(query_texts, top_k=top_k, query_embeddings=query_embeddings)

    # Remap results keys to original query sentence for evaluator
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
    output_dir = Path(cfg["results_dir"]) / "zero_shot" / model_slug(model_id) / query_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
