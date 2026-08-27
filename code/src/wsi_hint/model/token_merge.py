from __future__ import annotations

import torch
from torch import nn


class TokenMerging(nn.Module):
    """Token Merging (ToMe) adapted for MIL (Bolya et al., 2023).

    Merges the most similar patch tokens to reduce sequence length,
    lowering computation in downstream attention while preserving
    information. Uses bipartite soft matching: splits tokens into
    two sets, finds best matches via cosine similarity, merges top-r
    pairs by averaging.

    Unlike image ViT-ToMe which uses spatial alternation, we use
    random bipartite partitioning suited for unordered patch bags.
    """

    def __init__(self, merge_ratio: float = 0.25) -> None:
        super().__init__()
        self.merge_ratio = merge_ratio

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.training and self.merge_ratio <= 0:
            return tokens, mask, coords
        B, N, D = tokens.shape
        r = int(N * self.merge_ratio * 0.5)
        if r < 1 or N < 4:
            return tokens, mask, coords

        mid = N // 2
        a_tokens = tokens[:, :mid]
        b_tokens = tokens[:, mid:]

        a_norm = nn.functional.normalize(a_tokens, dim=-1)
        b_norm = nn.functional.normalize(b_tokens, dim=-1)
        sim = torch.bmm(a_norm, b_norm.transpose(1, 2))

        if mask is not None:
            a_mask = mask[:, :mid]
            b_mask = mask[:, mid:]
            valid_mask = a_mask.unsqueeze(2) & b_mask.unsqueeze(1)
            sim = sim.masked_fill(~valid_mask, float("-inf"))

        max_sim, b_idx = sim.max(dim=2)
        _, a_sorted = max_sim.sort(dim=1, descending=True)
        a_merge = a_sorted[:, :r]
        a_keep = a_sorted[:, r:]

        kept_a = torch.gather(a_tokens, 1, a_keep.unsqueeze(-1).expand(-1, -1, D))

        merge_a = torch.gather(a_tokens, 1, a_merge.unsqueeze(-1).expand(-1, -1, D))
        merge_b_idx = torch.gather(b_idx, 1, a_merge)
        merge_b = torch.gather(b_tokens, 1, merge_b_idx.unsqueeze(-1).expand(-1, -1, D))
        merged = (merge_a + merge_b) * 0.5

        b_merged_set = merge_b_idx
        b_all_idx = torch.arange(b_tokens.size(1), device=tokens.device).unsqueeze(0).expand(B, -1)
        b_is_merged = torch.zeros(B, b_tokens.size(1), dtype=torch.bool, device=tokens.device)
        b_is_merged.scatter_(1, b_merged_set, True)
        b_keep_mask = ~b_is_merged
        max_b_keep = int(b_keep_mask.sum(dim=1).max().item())
        if max_b_keep == 0:
            out_tokens = torch.cat([kept_a, merged], dim=1)
        else:
            b_keep_idx = torch.zeros(B, max_b_keep, dtype=torch.long, device=tokens.device)
            for i in range(B):
                keep_indices = b_all_idx[i][b_keep_mask[i]]
                b_keep_idx[i, :keep_indices.size(0)] = keep_indices
            kept_b = torch.gather(b_tokens, 1, b_keep_idx.unsqueeze(-1).expand(-1, -1, D))
            out_tokens = torch.cat([kept_a, kept_b, merged], dim=1)

        out_mask = None
        if mask is not None:
            kept_a_mask = torch.gather(a_mask, 1, a_keep)
            merge_a_mask = torch.gather(a_mask, 1, a_merge)
            merge_b_mask = torch.gather(b_mask, 1, merge_b_idx)
            merged_mask = merge_a_mask | merge_b_mask
            if max_b_keep == 0:
                out_mask = torch.cat([kept_a_mask, merged_mask], dim=1)
            else:
                kept_b_mask = torch.gather(b_mask, 1, b_keep_idx)
                out_mask = torch.cat([kept_a_mask, kept_b_mask, merged_mask], dim=1)

        out_coords = None
        if coords is not None:
            a_coords = coords[:, :mid]
            b_coords = coords[:, mid:]
            kept_a_c = torch.gather(a_coords, 1, a_keep.unsqueeze(-1).expand(-1, -1, 2))
            merge_a_c = torch.gather(a_coords, 1, a_merge.unsqueeze(-1).expand(-1, -1, 2))
            merge_b_c = torch.gather(b_coords, 1, merge_b_idx.unsqueeze(-1).expand(-1, -1, 2))
            merged_c = (merge_a_c + merge_b_c) * 0.5
            if max_b_keep == 0:
                out_coords = torch.cat([kept_a_c, merged_c], dim=1)
            else:
                kept_b_c = torch.gather(b_coords, 1, b_keep_idx.unsqueeze(-1).expand(-1, -1, 2))
                out_coords = torch.cat([kept_a_c, kept_b_c, merged_c], dim=1)

        return out_tokens, out_mask, out_coords
