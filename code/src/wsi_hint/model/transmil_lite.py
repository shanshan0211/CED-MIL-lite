from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ced_head import CEDEvidenceHead


@dataclass(slots=True)
class TransMILLiteOutput:
    logits: torch.Tensor
    cls_token: torch.Tensor
    aux_loss: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    role_gates: torch.Tensor | None = None
    role_attentions: list[torch.Tensor] | None = None
    evidence_prototypes: torch.Tensor | None = None
    residual_prototype: torch.Tensor | None = None
    counterfactual_logits: torch.Tensor | None = None
    shared_only_logits: torch.Tensor | None = None
    residual_only_logits: torch.Tensor | None = None


class TransMILLite(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
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
        self.use_ced_head = bool(use_ced_head)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> TransMILLiteOutput:
        h = self.patch_proj(patch_features)
        batch_size = h.size(0)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, h], dim=1)
        if patch_mask is None:
            key_padding_mask = None
        else:
            cls_mask = patch_mask.new_ones((batch_size, 1))
            key_padding_mask = ~(torch.cat([cls_mask, patch_mask], dim=1))
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        cls_token = encoded[:, 0, :]
        token_states = encoded[:, 1:, :]
        token_mask = patch_mask if patch_mask is not None else torch.ones(
            token_states.shape[:2], dtype=torch.bool, device=token_states.device,
        )
        if self.ced_head is not None:
            ced_out = self.ced_head(token_states, token_mask=token_mask, labels=labels)
            return TransMILLiteOutput(
                logits=ced_out.logits,
                cls_token=cls_token,
                aux_loss=ced_out.aux_loss,
                role_gates=ced_out.role_gates,
                role_attentions=ced_out.role_attentions,
                evidence_prototypes=ced_out.evidence_prototypes,
                residual_prototype=ced_out.residual_prototype,
                counterfactual_logits=ced_out.counterfactual_logits,
                shared_only_logits=ced_out.shared_only_logits,
                residual_only_logits=ced_out.residual_only_logits,
            )
        logits = self.classifier(cls_token)
        return TransMILLiteOutput(logits=logits, cls_token=cls_token)
