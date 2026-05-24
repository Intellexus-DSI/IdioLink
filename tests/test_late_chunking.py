"""Regression tests for late_chunking edge cases."""

import numpy as np
import pytest
import torch

from idiolink.models.late_chunking import late_chunk_encode


# ---------------------------------------------------------------------------
# Fake transformer / tokenizer infrastructure
# ---------------------------------------------------------------------------

class _FakeTransformerOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeTransformer:
    """Pretends to be a transformers AutoModel that returns bf16 token states."""

    def __init__(self, dtype=torch.bfloat16, hidden_dim=8):
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self._param = torch.zeros(1, device="cpu")

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        return self

    def __call__(self, **encoding):
        seq_len = encoding["input_ids"].shape[1]
        token_embeds = torch.randn(1, seq_len, self.hidden_dim).to(self.dtype)
        return _FakeTransformerOutput(token_embeds)


class _FakeTokenizerEncoding(dict):
    """Dict-like encoding that supports .pop() and .items()."""

    def pop(self, key, *args):
        return super().pop(key, *args)


class _FakeTokenizer:
    """
    Minimal tokenizer that late_chunk_encode can use.

    It produces a fixed-length sequence of tokens and constructs a
    character-level offset_mapping so that span detection works.
    """

    def __call__(self, text, return_offsets_mapping=False, return_tensors=None,
                 truncation=True, max_length=512):
        # Build a simple whitespace-based "tokenization"
        words = text.split()
        # Build offset mapping: track character positions per word
        offsets = []
        pos = 0
        # CLS token at position 0
        offsets.append((0, 0))
        for word in words:
            start = text.find(word, pos)
            end = start + len(word)
            offsets.append((start, end))
            pos = end
        # SEP token
        offsets.append((0, 0))

        seq_len = len(offsets)
        input_ids = torch.ones(1, seq_len, dtype=torch.long)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        offset_tensor = torch.tensor(offsets, dtype=torch.long).unsqueeze(0)  # (1, seq_len, 2)

        enc = _FakeTokenizerEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        if return_offsets_mapping:
            enc["offset_mapping"] = offset_tensor
        return enc


class _FakeSTModel:
    """Pretends to be a sentence-transformers model exposing auto_model + tokenizer."""

    def __init__(self, transformer, tokenizer):
        self.auto_model = transformer
        self.tokenizer = tokenizer


class _FakeIdiolinkModel:
    """Pretends to be a BaseEmbeddingModel — only the attributes late_chunk_encode reads."""

    def __init__(self, transformer, tokenizer):
        self.model_id = "fake/bf16-model"
        self.model = _FakeSTModel(transformer, tokenizer)

    def encode(self, texts):
        hidden_dim = self.model.auto_model.hidden_dim
        return np.zeros((len(texts), hidden_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_model(dtype=torch.bfloat16, hidden_dim=8):
    transformer = _FakeTransformer(dtype=dtype, hidden_dim=hidden_dim)
    tokenizer = _FakeTokenizer()
    return _FakeIdiolinkModel(transformer, tokenizer)


def test_late_chunk_encode_handles_bf16_token_embeddings():
    """Regression: bf16 outputs must be cast to fp32 before .numpy()."""
    model = _make_model(dtype=torch.bfloat16, hidden_dim=8)

    docs = ["the quick brown fox jumps over the lazy dog"]
    spans = ["brown fox"]

    out = late_chunk_encode(model, docs, spans, device="cpu")

    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    assert out.shape == (1, 8), f"Expected shape (1, 8), got {out.shape}"
    assert np.isfinite(out).all(), "Output contains non-finite values"


def test_late_chunk_encode_handles_fp16_token_embeddings():
    """fp16 has the same numpy incompatibility; ensure the fix covers it too."""
    model = _make_model(dtype=torch.float16, hidden_dim=8)

    docs = ["idioms are tricky phrases"]
    spans = ["tricky phrases"]

    out = late_chunk_encode(model, docs, spans, device="cpu")

    assert out.dtype == np.float32
    assert out.shape == (1, 8)
    assert np.isfinite(out).all()


def test_late_chunk_encode_span_not_found_fallback():
    """When span is not in document, model.encode() fallback is used."""
    model = _make_model(dtype=torch.bfloat16, hidden_dim=8)

    docs = ["hello world"]
    spans = ["missing phrase"]  # not in doc

    out = late_chunk_encode(model, docs, spans, device="cpu")

    # Fallback returns zeros (from _FakeIdiolinkModel.encode)
    assert out.dtype == np.float32
    assert out.shape == (1, 8)
    assert (out == 0).all()
