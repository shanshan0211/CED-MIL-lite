from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(slots=True)
class DSMILOutput:
    logits: torch.Tensor
    instance_logits: torch.Tensor
    attention: torch.Tensor
    aux_loss: torch.Tensor | None = None


class DSMIL(nn.Module):
    """A lightweight DSMIL-style dual-stream MIL baseline.

    The model combines:
    1. an instance classifier over all patches, and
    2. a bag branch that attends to patches using class-specific critical
       instances as anchors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.instance_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bag_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> DSMILOutput:
        h = self.proj(patch_features)
        if patch_mask is None:
            patch_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)

        instance_logits = self.instance_classifier(h)
        masked_instance_logits = instance_logits.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        critical_idx = masked_instance_logits.argmax(dim=1)

        gather_idx = critical_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        critical_tokens = torch.gather(h, dim=1, index=gather_idx)

        q_all = self.query(h)
        q_critical = self.query(critical_tokens)
        attn_scores = torch.einsum("bnd,bcd->bnc", q_all, q_critical) / math.sqrt(h.size(-1))
        attn_scores = attn_scores.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(attn_scores, dim=1)

        v_all = self.value(h)
        bag_reps = torch.einsum("bnc,bnd->bcd", attn, v_all)
        bag_logits = torch.cat(
            [head(bag_reps[:, idx, :]) for idx, head in enumerate(self.bag_classifiers)],
            dim=1,
        )
        max_instance_logits = masked_instance_logits.max(dim=1).values
        logits = 0.5 * (bag_logits + max_instance_logits)
        return DSMILOutput(logits=logits, instance_logits=instance_logits, attention=attn)
