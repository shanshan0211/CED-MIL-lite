from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(slots=True)
class CEDEvidenceOutput:
    logits: torch.Tensor
    role_gates: torch.Tensor
    role_attentions: list[torch.Tensor]
    evidence_prototypes: torch.Tensor
    residual_prototype: torch.Tensor
    counterfactual_logits: torch.Tensor | None
    shared_only_logits: torch.Tensor | None
    residual_only_logits: torch.Tensor | None
    aux_loss: torch.Tensor


class _RoleAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.attn(tokens).squeeze(-1)
        score = score.masked_fill(~mask, float("-inf"))
        return F.softmax(score, dim=-1)


class CEDEvidenceHead(nn.Module):
    """Plugin-style evidence decomposition head.

    This head consumes token-level hidden states from an arbitrary MIL backbone.
    It can therefore be attached to ABMIL-like or Transformer-like encoders
    without changing the upstream feature extractor substantially.
    """

    NUM_ROLES = 3
    ROLE_CLS0 = 0
    ROLE_CLS1 = 1
    ROLE_SHARED = 2

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        attn_dim: int = 128,
        dropout: float = 0.15,
        lambda_sep: float = 0.05,
        lambda_align: float = 0.05,
        lambda_cf: float = 0.05,
        lambda_residual: float = 0.02,
        lambda_bal: float = 0.01,
        sep_margin: float = 1.0,
        cf_margin: float = 0.2,
        use_cf: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("CEDEvidenceHead currently supports binary classification only.")
        self.num_classes = num_classes
        self.lambda_sep = lambda_sep
        self.lambda_align = lambda_align
        self.lambda_cf = lambda_cf
        self.lambda_residual = lambda_residual
        self.lambda_bal = lambda_bal
        self.sep_margin = sep_margin
        self.cf_margin = cf_margin
        self.use_cf = use_cf

        self.role_gate = nn.Linear(hidden_dim, self.NUM_ROLES)
        self.role_attns = nn.ModuleList([
            _RoleAttention(hidden_dim, attn_dim) for _ in range(self.NUM_ROLES)
        ])
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _zero_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def _compose_slide_feature(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z_shared: torch.Tensor,
        z_residual: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([z0, z1, z_shared, z0 - z1, z_shared - z_residual], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CEDEvidenceOutput:
        batch_size, token_count, _ = tokens.shape
        if token_mask is None:
            token_mask = torch.ones(batch_size, token_count, dtype=torch.bool, device=tokens.device)

        gate_logits = self.role_gate(tokens)
        gate_probs = F.softmax(gate_logits, dim=-1)

        prototypes: list[torch.Tensor] = []
        role_attn_list: list[torch.Tensor] = []
        for role_index in range(self.NUM_ROLES):
            alpha = self.role_attns[role_index](tokens, token_mask)
            role_attn_list.append(alpha)
            role_weight = gate_probs[:, :, role_index] * alpha
            norm = role_weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            proto = torch.einsum("bn,bnd->bd", role_weight, tokens) / norm
            prototypes.append(proto)

        z0, z1, z_shared = prototypes
        proto_stack = torch.stack(prototypes, dim=1)

        global_context = self._masked_mean(tokens, token_mask)
        z_residual = self.residual_proj(torch.cat([global_context, z0, z1, z_shared], dim=-1))
        logits = self.classifier(self._compose_slide_feature(z0, z1, z_shared, z_residual))

        counterfactual_logits = None
        shared_only_logits = None
        residual_only_logits = None
        if self.use_cf:
            logits_wo_0 = self.classifier(self._compose_slide_feature(self._zero_like(z0), z1, z_shared, z_residual))
            logits_wo_1 = self.classifier(self._compose_slide_feature(z0, self._zero_like(z1), z_shared, z_residual))
            chosen_labels = logits.argmax(dim=-1) if labels is None else labels
            use_drop0 = (chosen_labels == 0).unsqueeze(-1)
            counterfactual_logits = torch.where(use_drop0, logits_wo_0, logits_wo_1)

            shared_only_logits = self.classifier(
                self._compose_slide_feature(self._zero_like(z0), self._zero_like(z1), z_shared, self._zero_like(z_residual))
            )
            residual_only_logits = self.classifier(
                self._compose_slide_feature(self._zero_like(z0), self._zero_like(z1), self._zero_like(z_shared), z_residual)
            )

        aux_loss = self._compute_aux_losses(
            z0=z0,
            z1=z1,
            z_shared=z_shared,
            gate_probs=gate_probs,
            role_attentions=role_attn_list,
            logits=logits,
            counterfactual_logits=counterfactual_logits,
            shared_only_logits=shared_only_logits,
            residual_only_logits=residual_only_logits,
            labels=labels,
            mask=token_mask,
        )

        return CEDEvidenceOutput(
            logits=logits,
            role_gates=gate_probs,
            role_attentions=role_attn_list,
            evidence_prototypes=proto_stack,
            residual_prototype=z_residual,
            counterfactual_logits=counterfactual_logits,
            shared_only_logits=shared_only_logits,
            residual_only_logits=residual_only_logits,
            aux_loss=aux_loss,
        )

    def _compute_aux_losses(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z_shared: torch.Tensor,
        gate_probs: torch.Tensor,
        role_attentions: list[torch.Tensor],
        logits: torch.Tensor,
        counterfactual_logits: torch.Tensor | None,
        shared_only_logits: torch.Tensor | None,
        residual_only_logits: torch.Tensor | None,
        labels: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        device = z0.device
        loss = torch.tensor(0.0, device=device)

        if self.lambda_sep > 0:
            dist_01 = F.pairwise_distance(z0, z1, p=2)
            loss = loss + self.lambda_sep * F.relu(self.sep_margin - dist_01).mean()

        if self.lambda_align > 0 and labels is not None:
            norm_0 = z0.norm(dim=-1)
            norm_1 = z1.norm(dim=-1)
            margin_align = torch.where(labels == 0, norm_0 - norm_1, norm_1 - norm_0)
            loss = loss + self.lambda_align * F.relu(0.1 - margin_align).mean()

        if self.use_cf and self.lambda_cf > 0 and labels is not None and counterfactual_logits is not None and shared_only_logits is not None:
            true_full = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            true_drop = counterfactual_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            l_cf = F.relu(self.cf_margin - (true_full - true_drop)).mean()
            shared_probs = F.softmax(shared_only_logits, dim=-1)
            shared_entropy = -(shared_probs * torch.log(shared_probs.clamp(min=1e-8))).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(self.num_classes), device=device))
            loss = loss + self.lambda_cf * (l_cf + 0.5 * (max_entropy - shared_entropy).mean())

        if self.use_cf and self.lambda_residual > 0 and residual_only_logits is not None:
            residual_probs = F.softmax(residual_only_logits, dim=-1)
            residual_entropy = -(residual_probs * torch.log(residual_probs.clamp(min=1e-8))).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(self.num_classes), device=device))
            loss = loss + self.lambda_residual * (max_entropy - residual_entropy).mean()

        if self.lambda_bal > 0:
            mask_f = mask.float().unsqueeze(-1)
            avg_gate = (gate_probs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            uniform = torch.ones_like(avg_gate) / self.NUM_ROLES
            l_gate_bal = F.kl_div(torch.log(avg_gate.clamp(min=1e-8)), uniform, reduction="batchmean")
            pair_sims = []
            for i in range(len(role_attentions)):
                for j in range(i + 1, len(role_attentions)):
                    pair_sims.append(F.cosine_similarity(role_attentions[i], role_attentions[j], dim=-1).mean())
            l_role_div = torch.stack(pair_sims).mean() if pair_sims else torch.tensor(0.0, device=device)
            loss = loss + self.lambda_bal * (l_gate_bal + l_role_div)

        return loss
