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


def _grad_fallback_full_doc_embedding(transformer, tokenizer, doc: str, device) -> torch.Tensor:
    """Tokenize + forward + mean-pool the full doc, preserving gradients.

    Used when the span isn't found in the doc or no tokens match — same
    fall-back semantics as `late_chunk_encode` but without breaking the
    gradient graph (the no-grad version uses model.encode() which goes
    through ST's torch.no_grad path).
    """
    enc = tokenizer(
        doc, return_tensors="pt", truncation=True, max_length=512,
    )
    # Strip offset_mapping if returned (some tokenizers include it by default).
    enc.pop("offset_mapping", None)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = transformer(**enc)
    return out.last_hidden_state[0].mean(dim=0).float()


def late_chunk_encode_with_grad(
    model: BaseEmbeddingModel,
    documents: List[str],
    spans: List[str],
    device: Optional[str] = None,
    prefer_last_span: bool = False,
) -> torch.Tensor:
    """Gradient-flow version of `late_chunk_encode`.

    Mirrors `late_chunk_encode` exactly except: no `torch.no_grad()` wrap,
    and returns a `torch.Tensor` on `device` instead of an ndarray. The
    fallback path (span not found / no matching tokens) also keeps gradients
    by tokenize+forward directly, not via model.encode (which is no-grad).
    """
    if hasattr(model, "model") and hasattr(model.model, "_first_module"):
        st_model = model.model
        transformer = st_model._first_module().auto_model
        tokenizer = st_model._first_module().tokenizer
    elif hasattr(model, "model") and hasattr(model.model, "auto_model"):
        transformer = model.model.auto_model
        tokenizer = model.model.tokenizer
    else:
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
            embeddings.append(_grad_fallback_full_doc_embedding(transformer, tokenizer, doc, device))
            continue

        span_end = span_start + len(span)

        encoding = tokenizer(
            doc,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # NO torch.no_grad() — this is the key difference from late_chunk_encode
        outputs = transformer(**encoding)
        token_embeddings = outputs.last_hidden_state[0]

        span_indices = _find_span_tokens(offset_mapping, span_start, span_end)

        if not span_indices:
            embeddings.append(_grad_fallback_full_doc_embedding(transformer, tokenizer, doc, device))
            continue

        span_embs = token_embeddings[span_indices]
        pooled = span_embs.mean(dim=0).float()
        embeddings.append(pooled)

    return torch.stack(embeddings)
