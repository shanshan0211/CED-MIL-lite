"""Patch-level Support-Query Cross-Attention classifier for wsi_hint."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class PatchSQOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    aux_loss: torch.Tensor | None = None
    projection: torch.Tensor | None = None


class PatchSQPool(nn.Module):
    """Patch-level support-query cross-attention pooling.

    Given query patch embeddings and support patch embeddings,
    computes cross-attention weights and returns a single refined
    slide-level representation.
    """

    def __init__(self, dim: int, temperature: float | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature or math.sqrt(dim)

    def forward(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_patches:   [Nq, D]
        support_patches: [Ns, D]
        returns:         [D]  aggregated representation
        """
        logits = torch.matmul(query_patches, support_patches.transpose(0, 1)) / self.temperature
        attn = F.softmax(logits, dim=-1)          # [Nq, Ns]
        refined = torch.matmul(attn, support_patches)  # [Nq, D]
        return refined.mean(dim=0)


class PatchSQClassifier(nn.Module):
    """Slide classifier using patch-level SQ pooling.

    In standard (non-episodic) training mode, falls back to gated
    attention pooling.  When ``support_patches`` is provided at forward
    time, activates the SQ cross-attention path instead.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
        max_support_patches: int = 1024,
        max_query_patches: int = 1024,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.max_support_patches = max_support_patches
        self.max_query_patches = max_query_patches

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.sq_pool = PatchSQPool(hidden_dim, temperature=math.sqrt(hidden_dim))

    def _attention_pool(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb, _ = self._attention_pool_with_weights(h, mask)
        return emb

    def _attention_pool_with_weights(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h)))
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        a = F.softmax(a, dim=-2)
        return (a * h).sum(dim=-2), a.squeeze(-1)

    @staticmethod
    def _deterministic_subsample(t: torch.Tensor, n: int) -> torch.Tensor:
        if t.shape[0] <= n:
            return t
        idx = torch.linspace(0, t.shape[0] - 1, n, device=t.device).long()
        return t[idx]

    def forward(
        self,
        features: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        support_patches: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        features: [B, N, D] batch of slides
        patch_mask: [B, N] valid-patch mask (named to match wsi_hint training framework)
        support_patches: [S, D] raw-dim support patches (optional, for SQ path)
        """
        mask = patch_mask
        if features.ndim == 2:
            features = features.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)

        h = self.proj(features)  # [B, N, hidden]
        B = h.shape[0]

        if support_patches is not None:
            if support_patches.shape[-1] == self.input_dim:
                s_proj = self.proj(support_patches)
            else:
                s_proj = support_patches
            s_proj = self._deterministic_subsample(s_proj, self.max_support_patches)

            out = []
            for b in range(B):
                valid = h[b][mask[b]] if mask is not None else h[b]
                valid = self._deterministic_subsample(valid, self.max_query_patches)
                z = self.sq_pool(valid, s_proj)
                out.append(z)
            slide_emb = torch.stack(out, dim=0)
            attn_weights = torch.zeros(B, 1, device=h.device)
        else:
            slide_emb, attn_weights = self._attention_pool_with_weights(h, mask)

        logits = self.classifier(slide_emb)
        return PatchSQOutput(logits=logits, attention=attn_weights)
