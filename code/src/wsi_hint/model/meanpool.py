from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class MeanPoolOutput:
    logits: torch.Tensor


class MeanPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, patch_features: torch.Tensor, patch_mask: torch.Tensor | None = None, **kwargs) -> MeanPoolOutput:
        h = self.proj(patch_features)
        if patch_mask is None:
            pooled = h.mean(dim=1)
        else:
            mask = patch_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled)
        return MeanPoolOutput(logits=logits)
