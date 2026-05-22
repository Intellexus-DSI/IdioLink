"""Instruction-aware embedding model supporting multiple instruction formats."""

from enum import Enum
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel


class InstructionFormat(Enum):
    E5_INLINE = "e5_inline"
    BGE_PROMPT = "bge_prompt"
    INSTRUCTOR_PAIRS = "instructor_pairs"
    TART_SEP = "tart_sep"
    NOMIC_PREFIX = "nomic_prefix"
    BGE_GEMMA = "bge_gemma"
    PLAIN = "plain"


DEFAULT_INSTRUCTION_TEMPLATE = (
    "Based on the literal/idiomatic usage of the span '{span}' in the query, "
    "retrieve documents that contain a span conveying the same conceptual meaning."
)


class InstructionModel(BaseEmbeddingModel):
    """Wraps SentenceTransformer with instruction-aware query encoding."""

    def __init__(
        self,
        model_id: str,
        instruction_format: str = "e5_inline",
        device: Optional[str] = None,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        super().__init__(model_id)
        self.instruction_format = InstructionFormat(instruction_format)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        kwargs = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        self.model = SentenceTransformer(model_id, device=device, **kwargs)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _format_query(self, text: str, instruction: str) -> str:
        """Format a single query text with its instruction."""
        fmt = self.instruction_format
        if fmt == InstructionFormat.E5_INLINE:
            return f"Instruct: {instruction}\nQuery: {text}"
        elif fmt == InstructionFormat.TART_SEP:
            return f"{instruction} [SEP] {text}"
        elif fmt == InstructionFormat.NOMIC_PREFIX:
            return f"search_query: {text}"
        elif fmt == InstructionFormat.BGE_GEMMA:
            return f"<instruct>{instruction}\n<query>{text}"
        elif fmt == InstructionFormat.PLAIN:
            return text
        # For BGE_PROMPT and INSTRUCTOR_PAIRS, formatting is handled in encode_queries
        return text

    def format_queries_for_late_chunking(
        self,
        texts: List[str],
        instructions: Union[str, List[str]],
    ) -> List[str]:
        """Return plain-text instructed queries suitable for token-level span pooling."""
        if isinstance(instructions, str):
            instructions = [instructions] * len(texts)

        fmt = self.instruction_format
        if fmt == InstructionFormat.INSTRUCTOR_PAIRS:
            return [f"{inst}\nQuery: {text}" for text, inst in zip(texts, instructions)]
        if fmt == InstructionFormat.BGE_PROMPT:
            return [f"Instruct: {inst}\nQuery: {text}" for text, inst in zip(texts, instructions)]
        return [self._format_query(text, inst) for text, inst in zip(texts, instructions)]

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode documents (no instruction)."""
        if self.passage_prefix:
            texts = [self.passage_prefix + t for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    def encode_queries(
        self,
        texts: List[str],
        spans: Optional[List[str]] = None,
        instruction: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Encode queries with instruction-aware formatting."""
        if instruction is None:
            instruction = ""
        if isinstance(instruction, str):
            instructions = [instruction] * len(texts)
        else:
            instructions = instruction

        fmt = self.instruction_format

        if fmt == InstructionFormat.BGE_PROMPT:
            if len(set(instructions)) == 1:
                prompt_name = f"Instruct: {instructions[0]}\nQuery: "
                encoded = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=len(texts) > 100,
                    convert_to_numpy=True,
                    prompt=prompt_name,
                )
            else:
                encoded = np.vstack([
                    self.model.encode(
                        [text],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        prompt=f"Instruct: {inst}\nQuery: ",
                    )
                    for text, inst in zip(texts, instructions)
                ])
        elif fmt == InstructionFormat.INSTRUCTOR_PAIRS:
            pairs = [[inst, t] for t, inst in zip(texts, instructions)]
            encoded = self.model.encode(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
        else:
            formatted = [self._format_query(t, inst) for t, inst in zip(texts, instructions)]
            if self.query_prefix and fmt == InstructionFormat.PLAIN:
                formatted = [self.query_prefix + t for t in formatted]
            encoded = self.model.encode(
                formatted,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )

        if encoded.dtype != np.float32:
            encoded = encoded.astype(np.float32)
        return encoded
