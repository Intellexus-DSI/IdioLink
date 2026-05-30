"""Per-mode query encoding helpers shared between zero-shot and trainer eval."""

from typing import List, Tuple

import numpy as np

from .late_chunking import late_chunk_encode
from .instruction_model import resolve_instructions
from ..utils import IdiomQuery


def encode_queries_for_mode(
    model,
    query_mode: str,
    idiom_queries: List[IdiomQuery],
    device: str,
) -> Tuple[List[str], np.ndarray]:
    """Encode queries for the given mode. Returns (query_texts, query_embeddings).

    The single source of truth for per-mode query encoding. Used by both the
    zero-shot scripts (run_dense, run_ablation, run_instruction) and the
    trainer's evaluation path so the two cannot drift.
    """
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
