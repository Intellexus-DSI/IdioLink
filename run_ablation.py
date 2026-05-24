"""
Index ablation runner.

Re-runs zero-shot dense retrieval (and BM25) against two reduced indices:
  - lit_sim_sense: keep {literal, simplification, sense}  (drop idiomatic)
  - lit_idiom:     keep {literal, idiomatic}              (drop sim + sense)

For each model the full document set is encoded once and then sliced per
preset, so per-(preset, mode) cost is one query encoding + one cosine-sim
sweep, not a full re-encode.

Usage:
    python run_ablation.py                                # <7B models, both presets, all 4 modes
    python run_ablation.py --debug                        # smoke test
    python run_ablation.py --models BAAI/bge-m3
    python run_ablation.py --presets lit_idiom
    python run_ablation.py --no_bm25                      # skip BM25 baseline
"""

import argparse
import csv
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from idiolink.ablation import ABLATION_PRESETS, filter_docs_by_usage
from idiolink.evaluator import Evaluator
from idiolink.models.instruction_model import resolve_instructions
from idiolink.models.late_chunking import late_chunk_encode
from idiolink.models.registry import MODEL_REGISTRY, load_model
from idiolink.retriever import DenseRetriever
from idiolink.utils import (
    get_device,
    load_config,
    load_documents,
    load_queries,
    model_slug,
    set_seed,
)
from run_bm25 import build_evaluator_docs, run_bm25, run_tuning

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: stage to .tmp and os.replace() onto the target.

    Survives mid-write interruption (Ctrl-C / OOM / kill) — without this, a
    truncated metrics.json silently passes the resume-check `path.exists()`
    and corrupts the aggregated full_results.csv.
    """
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

QUERY_MODES = ["sentence", "span", "instruction_sentence", "instruction_span"]
BM25_MODES = ["sentence", "span"]


def parse_size_b(s: str) -> float:
    """Parse a size string like '110M' or '1.5B' into billions of params."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 1000.0
    if s.endswith("B"):
        return float(s[:-1])
    return float(s)


def select_models(max_size_b: float = 7.0) -> List[str]:
    """Return registry model IDs with size strictly less than `max_size_b` billion params."""
    return [
        mid for mid, cfg in MODEL_REGISTRY.items()
        if parse_size_b(cfg.size_params) < max_size_b
    ]


def encode_queries_for_mode(model, query_mode: str, idiom_queries, device: str):
    """Encode queries for the given mode. Returns (query_texts, query_embeddings)."""
    spans = [q.span if q.span else q.query for q in idiom_queries]
    query_texts = [q.query for q in idiom_queries]
    instructions = resolve_instructions(model.model_id, idiom_queries)

    if query_mode == "sentence":
        return query_texts, model.encode(query_texts)
    if query_mode == "span":
        return query_texts, late_chunk_encode(model, query_texts, spans, device=device)
    if query_mode == "instruction_sentence":
        if hasattr(model, "encode_queries"):
            embs = model.encode_queries(query_texts, spans=spans, instruction=instructions)
        else:
            embs = model.encode(query_texts)
        return query_texts, embs
    if query_mode == "instruction_span":
        if hasattr(model, "encode_queries"):
            chunking_texts = (
                model.format_queries_for_late_chunking(query_texts, instructions)
                if hasattr(model, "format_queries_for_late_chunking")
                else query_texts
            )
            embs = late_chunk_encode(
                model, chunking_texts, spans, device=device, prefer_last_span=True,
            )
        else:
            embs = model.encode(query_texts)
        return query_texts, embs
    raise ValueError(f"Unknown query_mode: {query_mode}")


def flatten_metrics(metrics: dict, model_id: str, query_mode: str, index_slug: str) -> dict:
    """Flatten the evaluator output into a single CSV row."""
    by_usage = metrics.get("by_usage", {})
    lit = by_usage.get("literal", {})
    idi = by_usage.get("idiomatic", {})
    return {
        "model": model_id,
        "query_mode": query_mode,
        "index": index_slug,
        "r_precision": metrics.get("r_precision", 0.0),
        "ndcg@10": metrics.get("ndcg@10", 0.0),
        "r_precision_literal": lit.get("r_precision", 0.0),
        "ndcg@10_literal": lit.get("ndcg@10", 0.0),
        "r_precision_idiomatic": idi.get("r_precision", 0.0),
        "ndcg@10_idiomatic": idi.get("ndcg@10", 0.0),
        "num_queries": metrics.get("num_queries", 0),
        "num_queries_literal": lit.get("num_queries", 0),
        "num_queries_idiomatic": idi.get("num_queries", 0),
    }


CSV_FIELDS = [
    "model", "query_mode", "index",
    "r_precision", "ndcg@10",
    "r_precision_literal", "ndcg@10_literal",
    "r_precision_idiomatic", "ndcg@10_idiomatic",
    "num_queries", "num_queries_literal", "num_queries_idiomatic",
]


def _metrics_path(results_dir: Path, slug: str, model_id: str, mode: str) -> Path:
    return results_dir / "ablation" / slug / model_slug(model_id) / mode / "metrics.json"


def _bm25_metrics_path(results_dir: Path, slug: str, mode: str) -> Path:
    return results_dir / "ablation" / slug / "bm25" / mode / "metrics.json"


def run_dense_for_model(
    model_id: str,
    model,
    presets: Dict[str, Set[str]],
    query_modes: List[str],
    idiom_queries,
    doc_sentences,
    doc_metadata,
    top_k: int,
    device: str,
    results_dir: Path,
    force: bool = False,
) -> List[dict]:
    """Encode full docs once, then run all (preset, mode) combos for this model."""
    rows: List[dict] = []

    # Resume: figure out which (preset, mode) combos are still missing.
    pending: Dict[str, List[str]] = {}
    for slug in presets:
        missing = [m for m in query_modes if force or not _metrics_path(results_dir, slug, model_id, m).exists()]
        if missing:
            pending[slug] = missing
    if not pending:
        logger.info(f"  All ({len(presets)} presets x {len(query_modes)} modes) results exist; skipping {model_id}")
        return rows

    # Encode the full doc set once; we'll slice this per preset.
    logger.info(f"  Encoding {len(doc_sentences)} docs (full set, one-time per model)")
    full_doc_embeddings = model.encode(doc_sentences)

    # Pre-build preset slices into (embeddings, metadata, evaluator) for the
    # presets that still have pending modes.
    preset_views = {}
    for slug, keep in presets.items():
        if slug not in pending:
            continue
        idx = [i for i, m in enumerate(doc_metadata) if m.get("usage") in keep]
        if not idx:
            logger.warning(f"  Preset {slug} has 0 docs after filtering; skipping")
            continue
        emb = full_doc_embeddings[idx]
        meta = [doc_metadata[i] for i in idx]
        evaluator = Evaluator(idiom_queries, [{"id": m["id"], **m} for m in meta])
        preset_views[slug] = (emb, meta, evaluator)
        logger.info(f"  Preset {slug}: {len(meta)} docs (pending modes: {pending[slug]})")

    # Pre-compute query encodings only for modes that are still pending across any preset.
    needed_modes = sorted({m for ms in pending.values() for m in ms})
    mode_queries = {}
    for mode in needed_modes:
        logger.info(f"  Encoding queries for mode={mode}")
        try:
            mode_queries[mode] = encode_queries_for_mode(model, mode, idiom_queries, device)
        except Exception as e:
            logger.error(f"    Query encoding failed for {model_id}/{mode}: {e}")
            traceback.print_exc()

    # For each (preset, mode): set up retriever with pre-computed embeddings and run.
    for slug, (emb, meta, evaluator) in preset_views.items():
        retriever = DenseRetriever(model)
        retriever.doc_embeddings = emb
        retriever.doc_metadata = meta
        for mode in pending[slug]:
            if mode not in mode_queries:
                continue  # encoding failed earlier
            qt, qe = mode_queries[mode]
            try:
                results = retriever.retrieve(qt, top_k=top_k, query_embeddings=qe)
                mapped = {q.query: results[t] for q, t in zip(idiom_queries, qt)}
                metrics = evaluator.evaluate(mapped)
            except Exception as e:
                logger.error(f"    Failed {model_id}/{slug}/{mode}: {e}")
                traceback.print_exc()
                continue

            # Persist per-run metrics.json
            out_dir = results_dir / "ablation" / slug / model_slug(model_id) / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(out_dir / "metrics.json", metrics)

            row = flatten_metrics(metrics, model_id, mode, slug)
            rows.append(row)
            logger.info(
                f"    {slug}/{mode}: R-Prec={row['r_precision']:.4f} "
                f"nDCG@10={row['ndcg@10']:.4f} "
                f"(lit R-P={row['r_precision_literal']:.4f}, "
                f"idi R-P={row['r_precision_idiomatic']:.4f})"
            )
    return rows


def run_bm25_for_preset(
    cfg: dict,
    slug: str,
    keep: Set[str],
    query_modes: List[str],
    test_doc_sentences,
    test_doc_metadata,
    idiom_queries,
    top_k: int,
    results_dir: Path,
    force: bool = False,
) -> List[dict]:
    """Run BM25 (tuned on val) for one preset across the requested modes."""
    rows: List[dict] = []

    # Resume: skip modes already on disk.
    pending = [m for m in query_modes if force or not _bm25_metrics_path(results_dir, slug, m).exists()]
    if not pending:
        logger.info(f"  BM25 {slug}: all modes done; skipping")
        return rows

    # Filter test docs once for this preset.
    ts, tm = filter_docs_by_usage(test_doc_sentences, test_doc_metadata, keep)
    evaluator = Evaluator(idiom_queries, [{"id": m["id"], **m} for m in tm])

    for mode in pending:
        try:
            k1, b, tuning_results = run_tuning(cfg, mode, top_k, keep=keep)
        except Exception as e:
            logger.error(f"  BM25 tuning failed for {slug}/{mode}: {e}")
            traceback.print_exc()
            continue

        try:
            results = run_bm25(ts, tm, idiom_queries, mode, k1, b, top_k)
            metrics = evaluator.evaluate(results)
        except Exception as e:
            logger.error(f"  BM25 retrieval failed for {slug}/{mode}: {e}")
            traceback.print_exc()
            continue

        metrics["k1"] = k1
        metrics["b"] = b
        metrics["query_mode"] = mode
        metrics["index_filter"] = slug

        out_dir = results_dir / "ablation" / slug / "bm25" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out_dir / "metrics.json", metrics)
        _atomic_write_json(out_dir / "tuning_results.json", tuning_results)

        row = flatten_metrics(metrics, "bm25", mode, slug)
        rows.append(row)
        logger.info(
            f"  bm25/{slug}/{mode}: R-Prec={row['r_precision']:.4f} "
            f"nDCG@10={row['ndcg@10']:.4f} (k1={k1}, b={b})"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run index-composition ablations")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="Model IDs to run. Default: all registry models with size < 7B.",
    )
    parser.add_argument(
        "--presets", nargs="+", type=str, default=None,
        choices=list(ABLATION_PRESETS.keys()),
        help="Which ablation presets to run. Default: all.",
    )
    parser.add_argument(
        "--modes", nargs="+", type=str, default=None, choices=QUERY_MODES,
        help="Which query modes to run. Default: all 4.",
    )
    parser.add_argument(
        "--no_bm25", action="store_true",
        help="Skip the BM25 baseline pass.",
    )
    parser.add_argument(
        "--max_size_b", type=float, default=7.0,
        help="Default model filter: include models with size_params < this value (in B). Default 7.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute (preset, mode) combos even if metrics.json already exists.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = get_device(cfg["device"])
    top_k = cfg["retrieval"]["top_k"]
    results_dir = Path(cfg["results_dir"])

    presets = {
        name: ABLATION_PRESETS[name]
        for name in (args.presets if args.presets else ABLATION_PRESETS.keys())
    }
    query_modes = args.modes if args.modes else QUERY_MODES
    model_ids = args.models if args.models else select_models(args.max_size_b)

    logger.info(f"Device: {device}")
    logger.info(f"Presets: {list(presets.keys())}")
    logger.info(f"Modes: {query_modes}")
    logger.info(f"Models ({len(model_ids)}): {model_ids}")

    # Load test data once.
    test_dir = Path(cfg["data"]["test_dir"])
    doc_sentences, doc_metadata = load_documents(str(test_dir / "indexes.json"))
    _, idiom_queries = load_queries(str(test_dir / "queries.json"))

    if args.debug:
        n = cfg.get("debug_samples", 5)
        idiom_queries = idiom_queries[:n]

    # BM25 baseline (cheap, do it first so we have it even if dense runs fail).
    if not args.no_bm25:
        for slug, keep in presets.items():
            logger.info(f"BM25 baseline / preset={slug}")
            run_bm25_for_preset(
                cfg, slug, keep, BM25_MODES,
                doc_sentences, doc_metadata, idiom_queries,
                top_k, results_dir, force=args.force,
            )

    # Dense models.
    for model_id in model_ids:
        logger.info(f"Loading model: {model_id}")
        # If everything for this model already exists on disk, skip loading entirely.
        if not args.force:
            all_done = all(
                _metrics_path(results_dir, slug, model_id, mode).exists()
                for slug in presets
                for mode in query_modes
            )
            if all_done:
                logger.info(f"  All results exist on disk; skipping {model_id}")
                continue

        try:
            model = load_model(model_id, device=device)
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            traceback.print_exc()
            continue

        try:
            run_dense_for_model(
                model_id, model, presets, query_modes,
                idiom_queries, doc_sentences, doc_metadata,
                top_k, device, results_dir, force=args.force,
            )
        finally:
            del model

    # Rebuild the aggregated CSV from every metrics.json on disk so the file
    # is always complete, regardless of what was (re)run this pass.
    rows_from_disk = collect_all_rows_from_disk(results_dir, presets, query_modes)
    if rows_from_disk:
        csv_path = results_dir / "ablation" / "full_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows_from_disk)
        logger.info(f"Aggregated results saved to {csv_path} ({len(rows_from_disk)} rows)")
    else:
        logger.warning("No ablation results generated.")


def collect_all_rows_from_disk(
    results_dir: Path,
    presets: Dict[str, Set[str]],
    query_modes: List[str],
) -> List[dict]:
    """Walk results/ablation/<slug>/{model|bm25}/<mode>/metrics.json and rebuild rows."""
    out: List[dict] = []
    ab_dir = results_dir / "ablation"
    if not ab_dir.exists():
        return out
    for slug in presets:
        slug_dir = ab_dir / slug
        if not slug_dir.exists():
            continue
        for sub in sorted(slug_dir.iterdir()):
            if not sub.is_dir():
                continue
            model_label = "bm25" if sub.name == "bm25" else sub.name.replace("__", "/")
            for mode in query_modes:
                mp = sub / mode / "metrics.json"
                if not mp.exists():
                    continue
                try:
                    with open(mp) as f:
                        metrics = json.load(f)
                    out.append(flatten_metrics(metrics, model_label, mode, slug))
                except Exception as e:
                    logger.warning(f"Could not read {mp}: {e}")
    return out


if __name__ == "__main__":
    main()
