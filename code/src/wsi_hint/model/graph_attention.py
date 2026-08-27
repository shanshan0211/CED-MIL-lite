from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpatialRegionGraphAttention(nn.Module):
    """k-NN graph attention on region tokens using region center coordinates.

    Builds a k-nearest-neighbor graph in coordinate space, then performs
    multi-head attention restricted to spatial neighbors. This captures
    tissue microenvironment topology -- e.g. tumor-stroma interface patterns
    -- that standard global attention treats as orderless.

    Complexity: O(R^2) for distance computation where R = num_regions (<= 256),
    then O(R * k * d) for the sparse attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)

    def _build_knn_mask(
        self,
        region_coords: torch.Tensor,
        region_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Build k-NN adjacency mask from region center coordinates.

        Returns: (B, R, R) boolean mask where True means "is a neighbor".
        """
        dists = torch.cdist(region_coords.float(), region_coords.float())
        dists = dists.masked_fill(~region_mask.unsqueeze(1), float("inf"))
        dists = dists.masked_fill(~region_mask.unsqueeze(2), float("inf"))
        effective_k = min(k, region_coords.size(1))
        _, knn_idx = torch.topk(dists, k=effective_k, dim=-1, largest=False)
        adj = torch.zeros_like(dists, dtype=torch.bool)
        adj.scatter_(2, knn_idx, True)
        adj = adj & region_mask.unsqueeze(1) & region_mask.unsqueeze(2)
        return adj

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_coords: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, R, D = region_tokens.shape
        adj = self._build_knn_mask(region_coords, region_mask, self.k_neighbors)

        normed = self.norm(region_tokens)
        q = self.q_proj(normed).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        graph_mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn = attn.masked_fill(~graph_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, R, D)
        out = self.out_proj(out)

        region_tokens = region_tokens + out
        region_tokens = region_tokens + self.ffn(region_tokens)
        return region_tokens
