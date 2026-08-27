from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding from (x, y) patch coordinates.

    Normalizes coordinates to [0, 1] using observed min/max per batch,
    then generates sin/cos embeddings at multiple frequency bands.
    """

    def __init__(self, hidden_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError(f"hidden_dim must be divisible by 4, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.dim_per_axis = hidden_dim // 2
        self.temperature = temperature
        freq_bands = torch.arange(0, self.dim_per_axis, 2, dtype=torch.float32)
        inv_freq = 1.0 / (temperature ** (freq_bands / self.dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, N, 2) with [x, y] in pixel space."""
        batch_size, seq_len, _ = coords.shape
        coords_f = coords.float()
        coord_min = coords_f.amin(dim=1, keepdim=True)
        coord_max = coords_f.amax(dim=1, keepdim=True)
        coord_range = (coord_max - coord_min).clamp(min=1.0)
        normed = (coords_f - coord_min) / coord_range * 2 * math.pi

        x = normed[..., 0:1]
        y = normed[..., 1:2]
        inv = self.inv_freq.to(coords.device)

        x_enc = torch.cat([torch.sin(x * inv), torch.cos(x * inv)], dim=-1)
        y_enc = torch.cat([torch.sin(y * inv), torch.cos(y * inv)], dim=-1)

        return torch.cat([x_enc, y_enc], dim=-1)


class LearnablePositionalEncoding(nn.Module):
    """Lightweight learned positional encoding that projects raw (x, y)
    coordinates through a small MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_f = coords.float()
        coord_min = coords_f.amin(dim=1, keepdim=True)
        coord_max = coords_f.amax(dim=1, keepdim=True)
        normed = (coords_f - coord_min) / (coord_max - coord_min).clamp(min=1.0)
        return self.proj(normed)
