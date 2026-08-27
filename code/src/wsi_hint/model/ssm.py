from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model inspired by Mamba (Gu & Dao, 2023).

    Pure PyTorch implementation without custom CUDA kernels. Uses a sequential
    scan which is efficient for moderate sequence lengths (< 1024).

    The key insight: A, B, C, delta are all input-dependent (selective),
    allowing the model to adaptively filter or retain information.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt_init_std = (1.0 / d_model) ** 0.5
        nn.init.uniform_(self.x_proj.weight, -dt_init_std, dt_init_std)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential selective scan.

        x: (B, L, D_inner)
        delta: (B, L, D_inner)
        A: (D_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = x.new_zeros(batch, d_inner, d_state)
        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        params = self.x_proj(x_conv)
        B = params[..., :self.d_state]
        C = params[..., self.d_state:2 * self.d_state]
        delta = F.softplus(params[..., -1:].expand(-1, -1, self.d_inner))

        A = -torch.exp(self.A_log)

        if mask is not None:
            x_conv = x_conv * mask.unsqueeze(-1).float()

        y = self._selective_scan(x_conv, delta, A, B, C)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.ssm(self.norm(x), mask=mask)


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba: runs SSM forward and backward, fuses results.

    Captures both left-to-right and right-to-left dependencies in the
    region sequence.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm_fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_bwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm(x)
        fwd = self.ssm_fwd(normed, mask=mask)
        bwd = self.ssm_bwd(normed.flip(1), mask=mask.flip(1) if mask is not None else None).flip(1)
        gate = self.gate(torch.cat([fwd, bwd], dim=-1))
        fused = gate * fwd + (1.0 - gate) * bwd
        return x + fused
