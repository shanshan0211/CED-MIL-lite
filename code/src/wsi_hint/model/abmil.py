from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ced_head import CEDEvidenceHead


@dataclass(slots=True)
class ABMILOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    aux_loss: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    role_gates: torch.Tensor | None = None
    role_attentions: list[torch.Tensor] | None = None
    evidence_prototypes: torch.Tensor | None = None
    residual_prototype: torch.Tensor | None = None
    counterfactual_logits: torch.Tensor | None = None
    shared_only_logits: torch.Tensor | None = None
    residual_only_logits: torch.Tensor | None = None


class ABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        instance_dropout: float = 0.0,
        use_ced_head: bool = False,
        ced_attn_dim: int = 128,
        ced_lambda_sep: float = 0.05,
        ced_lambda_align: float = 0.05,
        ced_lambda_cf: float = 0.05,
        ced_lambda_residual: float = 0.02,
        ced_lambda_balance: float = 0.01,
        ced_sep_margin: float = 1.0,
        ced_cf_margin: float = 0.2,
        ced_use_cf: bool = True,
    ) -> None:
        super().__init__()
        self.instance_dropout = float(instance_dropout)
        self.use_ced_head = bool(use_ced_head)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        self.ced_head = (
            CEDEvidenceHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                attn_dim=ced_attn_dim,
                dropout=dropout,
                lambda_sep=ced_lambda_sep,
                lambda_align=ced_lambda_align,
                lambda_cf=ced_lambda_cf,
                lambda_residual=ced_lambda_residual,
                lambda_bal=ced_lambda_balance,
                sep_margin=ced_sep_margin,
                cf_margin=ced_cf_margin,
                use_cf=ced_use_cf,
            )
            if self.use_ced_head else None
        )

    def forward(self, patch_features: torch.Tensor, patch_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs) -> ABMILOutput:
        h = self.proj(patch_features)
        scores = self.attention(h).squeeze(-1)
        if patch_mask is None:
            patch_mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
        if self.training and self.instance_dropout > 0:
            valid_before = patch_mask
            keep = torch.rand(scores.shape, device=scores.device) >= self.instance_dropout
            patch_mask = valid_before & keep
            empty = ~patch_mask.any(dim=1)
            if empty.any():
                patch_mask = patch_mask.clone()
                first_valid = valid_before.int().argmax(dim=1)
                patch_mask[empty, first_valid[empty]] = True
        scores = scores.masked_fill(~patch_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        if self.ced_head is not None:
            ced_out = self.ced_head(h, token_mask=patch_mask, labels=labels)
            return ABMILOutput(
                logits=ced_out.logits,
                attention=weights,
                aux_loss=ced_out.aux_loss,
                role_gates=ced_out.role_gates,
                role_attentions=ced_out.role_attentions,
                evidence_prototypes=ced_out.evidence_prototypes,
                residual_prototype=ced_out.residual_prototype,
                counterfactual_logits=ced_out.counterfactual_logits,
                shared_only_logits=ced_out.shared_only_logits,
                residual_only_logits=ced_out.residual_only_logits,
            )
        pooled = torch.einsum("bn,bnd->bd", weights, h)
        logits = self.classifier(pooled)
        return ABMILOutput(logits=logits, attention=weights)
