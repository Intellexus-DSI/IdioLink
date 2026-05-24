"""Late chunking: span-level token pooling from full-document embeddings."""

from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .base import BaseEmbeddingModel


def _find_span_tokens(offset_mapping: List, span_start: int, span_end: int) -> List[int]:
    """Find token indices whose character offsets overlap with the span."""
    indices = []
    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end == 0:
            continue  # skip special tokens
        if tok_end > span_start and tok_start < span_end:
            indices.append(idx)
    return indices


def late_chunk_encode(
    model: BaseEmbeddingModel,
    documents: List[str],
    spans: List[str],
    device: Optional[str] = None,
    prefer_last_span: bool = False,
) -> np.ndarray:
    """
    Encode documents using late chunking: get full-doc token embeddings,
    then mean-pool only the span's tokens.
    """
    # Get the underlying transformer model
    if hasattr(model, "model") and hasattr(model.model, "_first_module"):
        st_model = model.model
        transformer = st_model._first_module().auto_model
        tokenizer = st_model._first_module().tokenizer
    elif hasattr(model, "model") and hasattr(model.model, "auto_model"):
        transformer = model.model.auto_model
        tokenizer = model.model.tokenizer
    else:
        # Fallback: load tokenizer/model directly
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        transformer = AutoModel.from_pretrained(model.model_id)

    if device is None:
        device = next(transformer.parameters()).device
    else:
        device = torch.device(device)
        transformer = transformer.to(device)

    embeddings = []
    for doc, span in zip(documents, spans):
        span_start = doc.rfind(span) if prefer_last_span else doc.find(span)
        if span_start == -1:
            # Fallback: encode full document
            emb = model.encode([doc])[0]
            embeddings.append(emb)
            continue

        span_end = span_start + len(span)

        # Tokenize with offset mapping
        encoding = tokenizer(
            doc,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Get token-level embeddings
        with torch.no_grad():
            outputs = transformer(**encoding)
            token_embeddings = outputs.last_hidden_state[0]  # (seq_len, dim)

        # Find span token indices
        span_indices = _find_span_tokens(offset_mapping, span_start, span_end)

        if not span_indices:
            emb = model.encode([doc])[0]
            embeddings.append(emb)
            continue

        # Mean-pool span tokens. Cast to fp32 before .numpy() because numpy
        # has no native bf16/fp16 dtype.
        span_embs = token_embeddings[span_indices]
        pooled = span_embs.mean(dim=0).float().cpu().numpy()
        embeddings.append(pooled)

    return np.array(embeddings, dtype=np.float32)
