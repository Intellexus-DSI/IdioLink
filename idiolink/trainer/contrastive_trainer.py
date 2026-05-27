"""Contrastive trainer for fine-tuning sentence embedding models with InfoNCE."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from ..evaluator import Evaluator
from ..retriever import DenseRetriever
from ..utils import load_documents, load_queries, set_seed, get_device
from ..models.encode_helpers import encode_queries_for_mode
from ..models.instruction_model import resolve_instructions
from ..models.late_chunking import late_chunk_encode_with_grad
from .losses import InfoNCELoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for contrastive fine-tuning."""

    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: Optional[int] = None
    lr: float = 2e-5
    max_epochs: int = 10
    warmup_steps: int = 100
    temperature: float = 0.05
    early_stopping_patience: int = 3
    early_stopping_metric: str = "ndcg@10"
    output_dir: str = "results/fine_tuning"
    seed: int = 42
    device: str = "auto"
    max_negatives: int = 5
    mode: str = "sentence"


def collate_triplets(batch: List[Dict]) -> Dict[str, Any]:
    """Collate triplet dicts into batched lists of strings."""
    queries = [item["query"] for item in batch]
    query_spans = [item.get("query_span") or "" for item in batch]
    query_idioms = [item.get("query_idiom", "") for item in batch]
    query_usages = [item.get("query_usage", "") for item in batch]
    query_subjects = [item.get("query_subject", "") for item in batch]
    positives = [item["positive"] for item in batch]

    max_neg = max(len(item["negatives"]) for item in batch) if batch else 0
    negatives = []
    for item in batch:
        negs = item["negatives"]
        while len(negs) < max_neg:
            negs = negs + [negs[-1] if negs else ""]
        negatives.append(negs)

    return {
        "queries": queries,
        "query_spans": query_spans,
        "query_idioms": query_idioms,
        "query_usages": query_usages,
        "query_subjects": query_subjects,
        "positives": positives,
        "negatives": negatives,
    }


class ContrastiveTrainer:
    """
    Fine-tunes a SentenceTransformer model using InfoNCE contrastive loss.

    Uses a manual training loop with the underlying transformer for full control
    over the loss function, hard negatives, and early stopping.
    """

    def __init__(self, config: TrainingConfig):
        from ..models.registry import MODEL_REGISTRY, load_model

        self.config = config
        self.device = get_device(config.device)
        set_seed(config.seed)

        # Registry-based load: honours model_class, trust_remote_code,
        # instruction_format, query_prefix, passage_prefix, batch_size.
        registry_cfg = MODEL_REGISTRY.get(config.model_id)
        if registry_cfg is not None and registry_cfg.model_class == "gritlm":
            raise ValueError(
                f"Fine-tuning is not supported for gritlm-class models "
                f"({config.model_id}). GritLM is zero-shot-only in this codebase. "
                f"Remove it from training.models or use a different model."
            )

        # instructor_pairs models can't be byte-equivalent to zero-shot in
        # instruction modes: zero-shot encode_queries passes a list of [inst, text]
        # pairs to SentenceTransformer.encode for the InstructorEmbedding pooling.
        # We can only feed concatenated strings through tokenize+forward at train
        # time. Fail fast for instruction modes; allow plain sentence/span.
        if (
            registry_cfg is not None
            and registry_cfg.instruction_format == "instructor_pairs"
            and config.mode in ("instruction_sentence", "instruction_span")
        ):
            raise ValueError(
                f"Fine-tuning is not supported for instructor_pairs models in "
                f"instruction modes ({config.model_id}, mode={config.mode}). "
                f"Zero-shot inference uses list-of-pairs ST.encode which the trainer's "
                f"gradient-flow tokenize+forward path cannot mirror byte-equivalently. "
                f"Use mode=sentence or mode=span, or pick a different model."
            )

        # Resolve batch_size: CLI/config > registry > hardcoded 32 fallback.
        if config.batch_size is None:
            config.batch_size = registry_cfg.batch_size if registry_cfg else 32
            source = "registry" if registry_cfg else "default"
        else:
            source = "config"
        logger.info(
            f"trainer: model={config.model_id} batch_size={config.batch_size} (from {source})"
        )

        self.model = load_model(config.model_id, device=self.device)
        if not hasattr(self.model.model, "tokenize"):
            raise ValueError(
                f"Trainer requires self.model.model to be a SentenceTransformer "
                f"(got {type(self.model.model).__name__}). "
                f"Model class '{registry_cfg.model_class if registry_cfg else '?'}' is not trainable."
            )
        if not hasattr(self.model, "format_queries_for_late_chunking"):
            raise ValueError(
                f"Trainer requires the wrapper to implement "
                f"format_queries_for_late_chunking (missing on {type(self.model).__name__})."
            )

        self.st_model = self.model.model  # underlying SentenceTransformer
        self.loss_fn = InfoNCELoss(temperature=config.temperature)

    def _encode_with_grad(self, texts: List[str]) -> torch.Tensor:
        """Tokenize + forward through the underlying SentenceTransformer with gradients."""
        features = self.st_model.tokenize(texts)
        # Newer SentenceTransformer versions may include non-tensor metadata
        # (e.g. a `modality` string). Only move tensors to device.
        features = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in features.items()
        }
        output = self.st_model(features)
        return output["sentence_embedding"]

    def _format_query_strings(
        self,
        plain_texts: List[str],
        idiom_queries: List["IdiomQuery"],
    ) -> List[str]:
        """Return the strings the model would tokenize for queries, per mode.

        - sentence / span: identity (plain query text).
        - instruction_sentence / instruction_span: per-model instruction-formatted
          string via wrapper.format_queries_for_late_chunking, which is the same
          output zero-shot's encode_queries_for_mode uses.
        """
        if self.config.mode in ("sentence", "span"):
            return list(plain_texts)
        instructions = resolve_instructions(self.config.model_id, idiom_queries)
        return self.model.format_queries_for_late_chunking(plain_texts, instructions)

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute InfoNCE loss for a batch with per-mode query encoding and
        doc-side passage_prefix application. Mirrors zero-shot inference path.
        """
        from ..utils import IdiomQuery

        queries = batch["queries"]
        query_spans = batch["query_spans"]
        positives = batch["positives"]
        negatives = batch["negatives"]
        batch_size = len(queries)

        # Build IdiomQuery objects for instruction resolution.
        iqs = [
            IdiomQuery(
                query=q,
                idiom=batch["query_idioms"][i],
                usage_type=batch["query_usages"][i],
                span=batch["query_spans"][i],
                subject=batch["query_subjects"][i],
            )
            for i, q in enumerate(queries)
        ]

        # Query embeddings — per mode
        formatted_queries = self._format_query_strings(queries, iqs)
        if self.config.mode in ("sentence", "instruction_sentence"):
            query_emb = self._encode_with_grad(formatted_queries)
        elif self.config.mode in ("span", "instruction_span"):
            query_emb = late_chunk_encode_with_grad(
                self.model, formatted_queries, query_spans, device=self.device,
                prefer_last_span=(self.config.mode == "instruction_span"),
            )
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Doc-side: apply wrapper's passage_prefix (defaults to "" via BaseEmbeddingModel)
        passage_prefix = getattr(self.model, "passage_prefix", "")
        if passage_prefix:
            positives = [passage_prefix + p for p in positives]

        pos_emb = self._encode_with_grad(positives)

        neg_emb = None
        has_negatives = negatives and len(negatives[0]) > 0
        if has_negatives:
            num_neg = len(negatives[0])
            flat_negatives = [neg for neg_list in negatives for neg in neg_list]
            if passage_prefix:
                flat_negatives = [passage_prefix + n for n in flat_negatives]
            neg_flat_emb = self._encode_with_grad(flat_negatives)
            neg_emb = neg_flat_emb.view(batch_size, num_neg, -1)

        return self.loss_fn(query_emb, pos_emb, neg_emb)

    def _evaluate(
        self,
        queries_file: str,
        indexes_file: str,
    ) -> Dict[str, Any]:
        """Run evaluation via the same per-mode encoding helper as zero-shot."""
        self.st_model.eval()
        retriever = DenseRetriever(self.model)

        query_sentences, idiom_queries = load_queries(queries_file)
        doc_sentences, doc_metadata = load_documents(indexes_file)

        retriever.index(doc_sentences, doc_metadata)
        query_texts, query_embeddings = encode_queries_for_mode(
            self.model, self.config.mode, idiom_queries, self.device,
        )

        results = retriever.retrieve(query_texts, top_k=100, query_embeddings=query_embeddings)
        # Re-key by original sentence in case query_texts differ from query_sentences
        # (instruction modes prepend prompts). DenseRetriever keys by the strings
        # passed to retrieve, not by the encoded form; the formatted strings ARE
        # what zero-shot uses too, so the keys here match the wrapped form.
        # For backward compatibility with evaluator API (which keys by q.query),
        # map back to plain sentences:
        if query_texts != query_sentences:
            results = {
                original: results[encoded]
                for original, encoded in zip(query_sentences, query_texts)
            }

        evaluator = Evaluator(idiom_queries, doc_metadata)
        metrics = evaluator.evaluate(results)
        self.st_model.train()
        return metrics

    def train(
        self,
        train_dataset,
        val_queries_file: str,
        val_indexes_file: str,
    ) -> Dict[str, float]:
        """
        Train the model with early stopping.

        Returns:
            Best validation metrics dict.
        """
        self.st_model.train()
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_triplets,
            drop_last=True,
        )

        # Optimizer and scheduler
        optimizer = AdamW(self.st_model.parameters(), lr=self.config.lr)
        total_steps = len(dataloader) * self.config.max_epochs
        warmup_steps = min(self.config.warmup_steps, total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(
                0.0,
                (total_steps - step) / max(1, total_steps - warmup_steps),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Early stopping state
        best_metric = -float("inf")
        patience_counter = 0
        best_metrics: Dict[str, float] = {}
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        global_step = 0
        for epoch in range(self.config.max_epochs):
            # Support dynamic datasets with per-epoch randomization
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)

            epoch_losses = []
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.st_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs} - Loss: {avg_loss:.4f}")

            # Validation
            val_metrics = self._evaluate(val_queries_file, val_indexes_file)
            current_metric = val_metrics[self.config.early_stopping_metric]
            logger.info(
                f"Epoch {epoch + 1} - Val {self.config.early_stopping_metric}: "
                f"{current_metric:.4f}"
            )

            if current_metric > best_metric:
                best_metric = current_metric
                best_metrics = val_metrics
                best_metrics["epoch"] = epoch + 1
                best_metrics["train_loss"] = avg_loss
                patience_counter = 0
                # Save best model
                self.st_model.save(str(output_path / "best_model"))
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience={self.config.early_stopping_patience})"
                    )
                    break

        return best_metrics

    def evaluate_test(
        self,
        test_queries_file: str,
        test_indexes_file: str,
    ) -> Dict[str, float]:
        """Load best model and evaluate on test set."""
        output_path = Path(self.config.output_dir)
        best_model_path = output_path / "best_model"
        if best_model_path.exists():
            self.st_model = SentenceTransformer(
                str(best_model_path), device=self.device
            )
        return self._evaluate(test_queries_file, test_indexes_file)

    def save_metrics(self, metrics: Dict[str, float], filename: str = "metrics.json"):
        """Save metrics to JSON file in output directory."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / filename, "w") as f:
            json.dump(metrics, f, indent=2)
