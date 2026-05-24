"""
Run all models across all query modes and aggregate results.

Usage:
    python run_all.py
    python run_all.py --debug
    python run_all.py --models BAAI/bge-m3 intfloat/e5-base-v2
"""

import argparse
import json
import csv
import logging
import traceback
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
from idiolink.models.instruction_model import resolve_instructions
from idiolink.models.late_chunking import late_chunk_encode
from idiolink.retriever import DenseRetriever
from idiolink.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERY_MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]


def run_single(model_id, model, query_mode, idiom_queries, doc_sentences, doc_metadata, top_k, device):
    """Run a single model x mode combination. Returns metrics dict or None on failure."""
    retriever = DenseRetriever(model)
    retriever.index(doc_sentences, doc_metadata)

    spans = [q.span if q.span else q.query for q in idiom_queries]
    query_sentences = [q.query for q in idiom_queries]
    instructions = resolve_instructions(model_id, idiom_queries)

    if query_mode == "sentence":
        query_texts = query_sentences
        query_embeddings = model.encode(query_texts)
    elif query_mode == "span":
        query_texts = query_sentences
        query_embeddings = late_chunk_encode(model, query_texts, spans, device=device)
    elif query_mode == "instruction_sentence":
        query_texts = query_sentences
        if hasattr(model, "encode_queries"):
            query_embeddings = model.encode_queries(query_texts, spans=spans, instruction=instructions)
        else:
            query_embeddings = model.encode(query_texts)
    elif query_mode == "instruction_span":
        query_texts = query_sentences
        if hasattr(model, "encode_queries"):
            chunking_texts = (
                model.format_queries_for_late_chunking(query_texts, instructions)
                if hasattr(model, "format_queries_for_late_chunking")
                else query_texts
            )
            query_embeddings = late_chunk_encode(
                model,
                chunking_texts,
                spans,
                device=device,
                prefer_last_span=True,
            )
        else:
            query_embeddings = model.encode(query_texts)
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")

    results = retriever.retrieve(query_texts, top_k=top_k, query_embeddings=query_embeddings)

    # Remap
    mapped_results = {}
    for q, qt in zip(idiom_queries, query_texts):
        mapped_results[q.query] = results[qt]

    evaluator = Evaluator(idiom_queries, [{"id": m["id"], **m} for m in doc_metadata])
    return evaluator.evaluate(mapped_results)


def main():
    parser = argparse.ArgumentParser(description="Run all models x modes")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--models", nargs="+", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    top_k = cfg["retrieval"]["top_k"]

    # Load data
    test_dir = Path(cfg["data"]["test_dir"])
    doc_sentences, doc_metadata = load_documents(str(test_dir / "indexes.json"))
    query_sentences, idiom_queries = load_queries(str(test_dir / "queries.json"))

    if args.debug:
        n = cfg.get("debug_samples", 5)
        idiom_queries = idiom_queries[:n]
        query_sentences = query_sentences[:n]

    # Select models
    model_ids = args.models if args.models else list(MODEL_REGISTRY.keys())

    all_results = []
    results_dir = Path(cfg["results_dir"])

    for model_id in model_ids:
        logger.info(f"Loading model: {model_id}")
        try:
            model = load_model(model_id, device=device)
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            traceback.print_exc()
            continue

        for query_mode in QUERY_MODES:
            logger.info(f"  Running {model_id} / {query_mode}")
            try:
                metrics = run_single(
                    model_id, model, query_mode,
                    idiom_queries, doc_sentences, doc_metadata, top_k, device,
                )
            except Exception as e:
                logger.error(f"  Failed {model_id}/{query_mode}: {e}")
                traceback.print_exc()
                continue

            # Save per-model results
            output_dir = results_dir / "zero_shot" / model_slug(model_id) / query_mode
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            row = {
                "model": model_id,
                "query_mode": query_mode,
                "r_precision": metrics["r_precision"],
                "ndcg@10": metrics["ndcg@10"],
                "num_queries": metrics["num_queries"],
            }
            all_results.append(row)
            logger.info(f"    R-Prec={metrics['r_precision']:.4f} nDCG@10={metrics['ndcg@10']:.4f}")

        # Free model memory
        del model

    # Write aggregated CSV
    if all_results:
        csv_path = results_dir / "full_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "query_mode", "r_precision", "ndcg@10", "num_queries"])
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Aggregated results saved to {csv_path}")
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
