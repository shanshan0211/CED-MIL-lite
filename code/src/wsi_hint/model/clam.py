from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class CLAMOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    aux_loss: torch.Tensor | None = None


class CLAMMIL(nn.Module):
    """A lightweight CLAM-style multi-branch attention baseline.

    This keeps the key inductive bias of CLAM: class-specific attention
    branches that pool different evidence for each class. We intentionally
    omit the original instance clustering loss to keep the baseline compatible
    with the current training loop.
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
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, num_classes)
        self.classifiers = nn.ModuleList(
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
    ) -> CLAMOutput:
        h = self.proj(patch_features)
        if patch_mask is None:
            patch_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)

        gated = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        scores = self.attn_w(gated)
        scores = scores.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)

        class_reps = torch.einsum("bnc,bnd->bcd", attn, h)
        logits = torch.cat(
            [head(class_reps[:, idx, :]) for idx, head in enumerate(self.classifiers)],
            dim=1,
        )
        return CLAMOutput(logits=logits, attention=attn)
