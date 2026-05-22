"""GritLM embedding model wrapper."""

from typing import List, Optional, Union
import numpy as np

from .base import BaseEmbeddingModel


class GritLMModel(BaseEmbeddingModel):
    """Wraps the GritLM library for embedding with instruction support."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        super().__init__(model_id)
        from gritlm import GritLM

        self.batch_size = batch_size
        self.model = GritLM(model_id, torch_dtype="auto", mode="embedding")
        self.embedding_dim = self.model.embed(["test"])[0].shape[-1] if hasattr(self.model, "embed") else 4096

    def _format_instruction(self, instruction: str) -> str:
        if instruction:
            return f"<|user|>\n{instruction}\n<|embed|>\n"
        return "<|embed|>\n"

    def format_queries_for_late_chunking(
        self,
        texts: List[str],
        instructions: Union[str, List[str]],
    ) -> List[str]:
        """Return plain-text instructed queries suitable for token-level span pooling."""
        if isinstance(instructions, str):
            instructions = [instructions] * len(texts)
        return [
            f"<|user|>\n{instruction}\n<|embed|>\n{text}"
            for text, instruction in zip(texts, instructions)
        ]

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode documents without instruction."""
        instruction = self._format_instruction("")
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embs = self.model.encode(batch, instruction=instruction)
            if not isinstance(embs, np.ndarray):
                embs = np.array(embs)
            all_embeddings.append(embs)
        result = np.concatenate(all_embeddings, axis=0)
        return result.astype(np.float32)

    def encode_queries(
        self,
        texts: List[str],
        spans: Optional[List[str]] = None,
        instruction: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Encode queries with instruction formatting."""
        if instruction is None:
            instruction = ""
        if isinstance(instruction, list) and len(set(instruction)) != 1:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_instructions = instruction[i : i + self.batch_size]
                for text, inst in zip(batch, batch_instructions):
                    embs = self.model.encode([text], instruction=self._format_instruction(inst))
                    if not isinstance(embs, np.ndarray):
                        embs = np.array(embs)
                    all_embeddings.append(embs)
            result = np.concatenate(all_embeddings, axis=0)
            return result.astype(np.float32)
        if isinstance(instruction, list):
            instruction = instruction[0]
        formatted_instr = self._format_instruction(instruction)
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embs = self.model.encode(batch, instruction=formatted_instr)
            if not isinstance(embs, np.ndarray):
                embs = np.array(embs)
            all_embeddings.append(embs)
        result = np.concatenate(all_embeddings, axis=0)
        return result.astype(np.float32)
