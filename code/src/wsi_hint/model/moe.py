from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(slots=True)
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor


class TopKRouter(nn.Module):
    """Token-choice top-k router with load-balancing auxiliary loss."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        topk: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (dispatch_weights, expert_indices, aux_loss).

        dispatch_weights: (B*N, topk) - normalized routing weights
        expert_indices:   (B*N, topk) - which expert for each token
        aux_loss:         scalar - load balancing loss
        """
        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))

        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(flat) * self.jitter_noise
            logits = self.gate(flat + noise)
        else:
            logits = self.gate(flat)

        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Load balancing loss (Switch Transformer style)
        tokens_per_expert = F.one_hot(topk_indices.reshape(-1), self.num_experts).float().mean(dim=0)
        avg_prob = probs.mean(dim=0)
        aux_loss = (tokens_per_expert * avg_prob).sum() * self.num_experts

        return topk_weights, topk_indices, aux_loss


class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEFFN(nn.Module):
    """Mixture-of-Experts Feed-Forward with SwiGLU activation and load balancing.

    Uses grouped matmul simulation for efficiency with moderate expert counts.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        topk: int = 2,
        intermediate_scale: int = 4,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        intermediate_dim = hidden_dim * intermediate_scale
        self.router = TopKRouter(hidden_dim, num_experts, topk, capacity_factor, jitter_noise)
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, intermediate_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> MoEOutput:
        orig_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        dispatch_weights, expert_indices, aux_loss = self.router(x)

        output = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = flat[mask]
            expert_out = self.experts[expert_idx](expert_input)
            weight_mask = (expert_indices[mask] == expert_idx).float()
            weight = (dispatch_weights[mask] * weight_mask).sum(dim=-1, keepdim=True)
            output[mask] = output[mask] + expert_out * weight

        return MoEOutput(hidden_states=output.view(orig_shape), aux_loss=aux_loss)


class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE FFN replacing standard FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_experts: int = 8,
        moe_topk: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe_ffn = MoEFFN(hidden_dim, num_experts=num_experts, topk=moe_topk, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = x + attended
        normed = self.norm2(x)
        moe_out = self.moe_ffn(normed)
        x = x + moe_out.hidden_states
        return x, moe_out.aux_loss
