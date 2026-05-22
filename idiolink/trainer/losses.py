"""InfoNCE contrastive loss for embedding fine-tuning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with in-batch negatives and optional hard negatives.

    For a batch of (query, positive) pairs, all other positives serve as
    in-batch negatives. If hard negatives are provided, they are appended
    to the negative set for each query.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query_emb: (B, D) query embeddings
            pos_emb: (B, D) positive embeddings
            neg_emb: optional (B, num_neg, D) hard negative embeddings

        Returns:
            Scalar loss
        """
        # L2 normalize
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)

        # In-batch similarity: (B, B) — each query against all positives
        logits = torch.matmul(query_emb, pos_emb.t()) / self.temperature

        # If hard negatives provided, append them
        if neg_emb is not None:
            neg_emb = F.normalize(neg_emb, p=2, dim=-1)
            # (B, num_neg) similarity of each query with its hard negatives
            hard_neg_logits = torch.bmm(
                neg_emb, query_emb.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            # Append hard negatives: logits becomes (B, B + num_neg)
            logits = torch.cat([logits, hard_neg_logits], dim=1)

        # Labels: diagonal (each query's positive is at index i)
        labels = torch.arange(query_emb.size(0), device=query_emb.device)

        return F.cross_entropy(logits, labels)
