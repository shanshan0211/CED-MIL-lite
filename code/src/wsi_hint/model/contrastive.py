from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP that projects pooled slide embeddings to a lower-dim
    L2-normalized space for contrastive learning."""

    def __init__(self, input_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls same-class embeddings together and pushes different-class
    embeddings apart in the projection space. Particularly effective
    for long-tail distributions where rare classes benefit from explicit
    clustering pressure.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = projections.device
        batch_size = projections.size(0)
        if batch_size <= 1:
            return projections.new_tensor(0.0)

        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        mask_pos = mask_pos * diag_mask.float()

        if mask_pos.sum() == 0:
            return projections.new_tensor(0.0)

        sim = torch.matmul(projections, projections.T) / self.temperature
        sim = sim.masked_fill(~diag_mask, float("-inf"))

        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        exp_sim = exp_sim * diag_mask.float()
        denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)

        log_prob = torch.log(exp_sim / denom + 1e-8)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / mask_pos.sum(dim=1).clamp(min=1.0)

        return -mean_log_prob.mean()


class RDropLoss(nn.Module):
    """R-Drop Regularization (Wu et al., 2021).

    Computes symmetric KL divergence between logits from two forward
    passes with different dropout masks, forcing prediction consistency.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        kl_pq = F.kl_div(p, q.detach().exp(), reduction="batchmean")
        kl_qp = F.kl_div(q, p.detach().exp(), reduction="batchmean")
        return self.alpha * (kl_pq + kl_qp) * 0.5
