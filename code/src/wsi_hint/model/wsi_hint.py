from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .positional import SinusoidalPositionalEncoding2D
from .moe import MoEFFN, MoETransformerLayer
from .ssm import BidirectionalMambaBlock
from .graph_attention import SpatialRegionGraphAttention
from .contrastive import ProjectionHead
from .token_merge import TokenMerging


@dataclass(slots=True)
class WSIHintOutput:
    logits: torch.Tensor
    slide_tokens: torch.Tensor
    region_tokens: torch.Tensor
    attention_index: torch.Tensor
    aux_loss: torch.Tensor
    projection: torch.Tensor | None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _batched_gather_regions(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        expanded = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
    elif tensor.ndim == 4:
        expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, tensor.size(-2), tensor.size(-1),
        )
    else:
        raise ValueError(f"Unsupported tensor rank: {tensor.ndim}")
    source = tensor.unsqueeze(1).expand(-1, indices.size(1), -1, *tensor.shape[2:])
    return torch.gather(source, 2, expanded)


def _compute_region_coords(
    patch_coords: torch.Tensor | None,
    patch_mask: torch.Tensor,
    region_size: int,
    region_count: int,
    pad_tokens: int,
    topk_idx: torch.Tensor | None,
) -> torch.Tensor | None:
    """Compute region center coordinates by averaging patch coords per group."""
    if patch_coords is None:
        return None
    B = patch_coords.size(0)
    if pad_tokens > 0:
        patch_coords = torch.cat(
            [patch_coords, patch_coords.new_zeros(B, pad_tokens, 2)], dim=1,
        )
        patch_mask_padded = torch.cat(
            [patch_mask, patch_mask.new_zeros(B, pad_tokens)], dim=1,
        )
    else:
        patch_mask_padded = patch_mask

    coord_groups = patch_coords.view(B, region_count, region_size, 2)
    mask_groups = patch_mask_padded.view(B, region_count, region_size)
    valid = mask_groups.sum(dim=-1, keepdim=True).clamp(min=1)
    region_coords = (coord_groups * mask_groups.unsqueeze(-1)).sum(dim=2) / valid

    if topk_idx is not None:
        region_coords = torch.gather(
            region_coords, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2),
        )
    return region_coords


class HierarchicalIndexBuilder(nn.Module):
    def __init__(self, hidden_dim: int, region_size: int, max_regions: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.region_size = region_size
        self.max_regions = max_regions
        self.region_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        patch_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns (region_tokens, patch_groups, region_mask, region_scores, residual_token, region_coords)."""
        batch_size, token_count, hidden_dim = patch_tokens.shape
        if patch_mask is None:
            patch_mask = torch.ones(batch_size, token_count, dtype=torch.bool, device=patch_tokens.device)

        region_count = max(1, math.ceil(token_count / self.region_size))
        pad_tokens = region_count * self.region_size - token_count

        orig_patch_mask = patch_mask
        if pad_tokens > 0:
            patch_tokens = torch.cat(
                [patch_tokens, patch_tokens.new_zeros(batch_size, pad_tokens, hidden_dim)], dim=1,
            )
            patch_mask = torch.cat(
                [patch_mask, patch_mask.new_zeros(batch_size, pad_tokens)], dim=1,
            )

        patch_groups = patch_tokens.view(batch_size, region_count, self.region_size, hidden_dim)
        mask_groups = patch_mask.view(batch_size, region_count, self.region_size)
        valid_counts = mask_groups.sum(dim=-1, keepdim=True).clamp(min=1)
        region_tokens = (patch_groups * mask_groups.unsqueeze(-1)).sum(dim=2) / valid_counts

        region_scores = self.region_gate(region_tokens).squeeze(-1)
        region_valid = mask_groups.any(dim=-1)
        region_scores = region_scores.masked_fill(~region_valid, float("-inf"))

        residual_token: torch.Tensor | None = None
        topk_idx: torch.Tensor | None = None
        if region_count > self.max_regions:
            _, topk_idx = torch.topk(region_scores, k=self.max_regions, dim=1)

            selected_mask = torch.zeros_like(region_scores, dtype=torch.bool)
            selected_mask.scatter_(1, topk_idx, True)
            discarded_mask = ~selected_mask & region_valid
            disc_scores = region_scores.masked_fill(~discarded_mask, float("-inf"))
            disc_weights = torch.softmax(disc_scores, dim=-1)
            residual_token = torch.einsum("br,brd->bd", disc_weights, region_tokens)

            region_tokens = torch.gather(
                region_tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            patch_groups = torch.gather(
                patch_groups, 1,
                topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.region_size, hidden_dim),
            )
            mask_groups = torch.gather(
                mask_groups, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.region_size),
            )

        region_mask = mask_groups.any(dim=-1)
        region_coords = _compute_region_coords(
            patch_coords, orig_patch_mask, self.region_size, region_count, pad_tokens, topk_idx,
        )
        return region_tokens, patch_groups, region_mask, region_scores, residual_token, region_coords


class MultiScaleRegionBuilder(nn.Module):
    def __init__(self, hidden_dim: int, fine_region_size: int, coarse_scale: int, max_regions: int) -> None:
        super().__init__()
        self.fine = HierarchicalIndexBuilder(hidden_dim, fine_region_size, max_regions)
        coarse_region_size = fine_region_size * coarse_scale
        self.coarse = HierarchicalIndexBuilder(hidden_dim, coarse_region_size, max(1, max_regions // coarse_scale))
        self.coarse_to_fine = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None,
        patch_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        fine_regions, fine_patches, fine_mask, fine_scores, fine_residual, fine_coords = self.fine(
            patch_tokens, patch_mask, patch_coords,
        )
        coarse_regions, _, coarse_mask, _, coarse_residual, _ = self.coarse(
            patch_tokens, patch_mask, patch_coords,
        )

        enhanced, _ = self.coarse_to_fine(
            fine_regions, coarse_regions, coarse_regions,
            key_padding_mask=~coarse_mask, need_weights=False,
        )
        fine_regions = self.norm(fine_regions + enhanced)

        residual = fine_residual
        if coarse_residual is not None:
            residual = (residual + coarse_residual) * 0.5 if residual is not None else coarse_residual

        return fine_regions, fine_patches, fine_mask, fine_scores, residual, fine_coords


class IndexedSparseRetriever(nn.Module):
    def __init__(self, hidden_dim: int, topk: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.topk = topk
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        region_tokens: torch.Tensor,
        patch_groups: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(
            self.query_proj(region_tokens), self.key_proj(region_tokens).transpose(1, 2),
        ) * self.scale
        valid = region_mask.unsqueeze(1) & region_mask.unsqueeze(2)
        scores = scores.masked_fill(~valid, float("-inf"))
        keep = min(self.topk, region_tokens.size(1))
        indices = torch.topk(scores, k=keep, dim=-1).indices
        gathered_regions = _batched_gather_regions(region_tokens, indices)
        gathered_patches = _batched_gather_regions(patch_groups, indices)
        fused = gathered_regions.unsqueeze(-2) + gathered_patches
        return fused, indices


class NestedRegionAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, use_moe: bool = False, num_experts: int = 8) -> None:
        super().__init__()
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoEFFN(hidden_dim, num_experts=num_experts, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        self._aux_loss = torch.tensor(0.0)

    def forward(self, region_tokens: torch.Tensor, retrieved_patch_groups: torch.Tensor) -> torch.Tensor:
        batch_size, region_count, candidate_count, patch_count, hidden_dim = retrieved_patch_groups.shape
        flattened = retrieved_patch_groups.view(batch_size * region_count, candidate_count * patch_count, hidden_dim)
        refined_local, _ = self.local_attention(flattened, flattened, flattened, need_weights=False)
        queries = region_tokens.view(batch_size * region_count, 1, hidden_dim)
        attended, _ = self.query_attention(queries, refined_local, refined_local, need_weights=False)
        attended = attended.view(batch_size, region_count, hidden_dim)
        output = self.norm(region_tokens + attended)
        if self.use_moe:
            moe_out = self.ffn(output)
            self._aux_loss = moe_out.aux_loss
            return output + moe_out.hidden_states
        return output + self.ffn(output)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss


class SemanticLengthNormalizer(nn.Module):
    def __init__(self, hidden_dim: int, latent_tokens: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.randn(latent_tokens, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, region_tokens: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        latents = self.latent_tokens.unsqueeze(0).expand(region_tokens.size(0), -1, -1)
        attended, _ = self.cross_attention(
            latents, region_tokens, region_tokens,
            key_padding_mask=~region_mask, need_weights=False,
        )
        latents = self.norm(latents + attended)
        return latents + self.ffn(latents)


class LayerScale(nn.Module):
    """LayerScale (Touvron et al., CaiT 2021): learnable per-channel scaling
    of residual connections, initialized to a small value (1e-4).
    Stabilizes training of deep transformers."""

    def __init__(self, dim: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class QKNormAttentionLayer(nn.Module):
    """Transformer layer with QK-Norm (Dehghani et al., 2023) and LayerScale.

    QK-Norm: L2-normalizes Q and K before computing attention scores,
    then scales by a learnable temperature. Prevents attention entropy
    collapse in deep models.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, layer_scale_init: float = 1e-4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1) * (self.head_dim ** -0.5))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ls1 = LayerScale(hidden_dim, layer_scale_init)
        self.ls2 = LayerScale(hidden_dim, layer_scale_init)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if src_key_padding_mask is not None:
            attn = attn.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        x = x + self.ls1(out)
        x = x + self.ls2(self.ffn(self.norm2(x)))
        return x


class HybridGlobalEncoder(nn.Module):
    """Alternating Mamba-SSM and Transformer layers with optional MoE,
    stochastic depth, LayerScale, and QK-Norm."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_ssm: bool = True,
        use_moe: bool = False,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        num_experts: int = 8,
        moe_topk: int = 2,
        drop_path_rate: float = 0.0,
        use_qk_norm: bool = False,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_types: list[str] = []
        self.drop_probs: list[float] = []

        for i in range(num_layers):
            dp = drop_path_rate * i / max(num_layers - 1, 1) if drop_path_rate > 0 else 0.0
            self.drop_probs.append(dp)
            is_moe_layer = use_moe and (i >= num_layers - max(1, num_layers // 3))

            if use_ssm and i % 2 == 0:
                self.layers.append(BidirectionalMambaBlock(hidden_dim, ssm_d_state, ssm_d_conv, ssm_expand))
                self.layer_types.append("ssm")
            elif is_moe_layer:
                self.layers.append(MoETransformerLayer(hidden_dim, num_heads, num_experts, moe_topk, dropout))
                self.layer_types.append("moe_attn")
            elif use_qk_norm:
                self.layers.append(QKNormAttentionLayer(hidden_dim, num_heads, dropout, layer_scale_init))
                self.layer_types.append("qk_attn")
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads, dropout=dropout,
                    batch_first=True, dim_feedforward=hidden_dim * 4, activation="gelu",
                )
                self.layers.append(layer)
                self.layer_types.append("attn")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        total_aux_loss = x.new_tensor(0.0)
        key_padding_mask = ~mask if mask is not None else None

        for layer, ltype, dp in zip(self.layers, self.layer_types, self.drop_probs):
            if self.training and dp > 0 and torch.rand(1).item() < dp:
                continue

            if ltype == "ssm":
                x = layer(x, mask=mask)
            elif ltype == "moe_attn":
                x, aux = layer(x, src_key_padding_mask=key_padding_mask)
                total_aux_loss = total_aux_loss + aux
            elif ltype == "qk_attn":
                x = layer(x, src_key_padding_mask=key_padding_mask)
            else:
                x = layer(x, src_key_padding_mask=key_padding_mask)

        return x, total_aux_loss


class MultiHeadAttentionPooling(nn.Module):
    """Learnable multi-head attention pooling (Lee et al., 2019).

    Instead of simple mean pooling, uses a set of learnable query vectors
    to attend over slide tokens, producing a weighted combination that
    captures class-discriminative information.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_queries: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.norm(out)
        return out.mean(dim=1)


class FeatureAugmentation(nn.Module):
    def __init__(self, mask_ratio: float = 0.1, noise_std: float = 0.01) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training:
            return x
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask_ratio > 0 and patch_mask is not None:
            drop = torch.rand(x.shape[:2], device=x.device) < self.mask_ratio
            drop = drop & patch_mask
            x = x.masked_fill(drop.unsqueeze(-1), 0.0)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WSIHint(nn.Module):
    """WSI-HINT: Hierarchical Indexing and Nested Transformer for WSI classification.

    Beyond-SOTA features:
      - Spatial positional encoding from patch coordinates
      - Multi-scale region hierarchy
      - Spatial k-NN graph attention on region tokens
      - Register tokens (Darcet 2024)
      - Bidirectional Mamba-Transformer hybrid with stochastic depth
      - Mixture-of-Experts FFN
      - Residual branch for discarded regions
      - Supervised contrastive projection head
      - Feature-level augmentation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        region_size: int,
        max_regions: int,
        retrieval_topk: int,
        latent_tokens: int,
        global_layers: int,
        dropout: float,
        num_classes: int,
        # --- SOTA enhancements ---
        use_positional_encoding: bool = False,
        use_multi_scale: bool = False,
        coarse_scale_factor: int = 4,
        use_ssm: bool = False,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_topk: int = 2,
        use_residual_branch: bool = False,
        feature_augmentation: bool = False,
        aug_mask_ratio: float = 0.1,
        aug_noise_std: float = 0.01,
        moe_aux_weight: float = 0.01,
        # --- Beyond-SOTA enhancements ---
        use_graph_attention: bool = False,
        graph_k_neighbors: int = 8,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        use_contrastive_head: bool = False,
        contrastive_proj_dim: int = 128,
        use_attention_pooling: bool = False,
        attention_pool_queries: int = 4,
        # --- Beyond-SOTA v3 ---
        use_token_merging: bool = False,
        token_merge_ratio: float = 0.25,
        use_qk_norm: bool = False,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.use_residual_branch = use_residual_branch
        self.use_graph_attention = use_graph_attention
        self.num_register_tokens = num_register_tokens
        self.use_contrastive_head = use_contrastive_head
        self.use_attention_pooling = use_attention_pooling
        self.moe_aux_weight = moe_aux_weight
        self.hidden_dim = hidden_dim

        self.patch_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_positional_encoding:
            self.pos_enc = SinusoidalPositionalEncoding2D(hidden_dim)
            self.pos_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        if feature_augmentation:
            self.feature_aug = FeatureAugmentation(aug_mask_ratio, aug_noise_std)
        else:
            self.feature_aug = None

        self.token_merger = TokenMerging(token_merge_ratio) if use_token_merging else None

        if use_multi_scale:
            self.index_builder = MultiScaleRegionBuilder(hidden_dim, region_size, coarse_scale_factor, max_regions)
        else:
            self.index_builder = HierarchicalIndexBuilder(hidden_dim, region_size, max_regions)

        if use_graph_attention:
            self.graph_attn = SpatialRegionGraphAttention(
                hidden_dim, num_heads=min(num_heads, 4), k_neighbors=graph_k_neighbors, dropout=dropout,
            )

        self.retriever = IndexedSparseRetriever(hidden_dim, retrieval_topk)
        self.nested_attention = NestedRegionAttention(hidden_dim, num_heads, dropout, use_moe=use_moe, num_experts=num_experts)
        self.normalizer = SemanticLengthNormalizer(hidden_dim, latent_tokens, num_heads, dropout)

        if use_residual_branch:
            self.residual_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_dim) * 0.02)

        self.global_encoder = HybridGlobalEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=global_layers,
            dropout=dropout,
            use_ssm=use_ssm,
            use_moe=use_moe,
            ssm_d_state=ssm_d_state,
            ssm_d_conv=ssm_d_conv,
            ssm_expand=ssm_expand,
            num_experts=num_experts,
            moe_topk=moe_topk,
            drop_path_rate=drop_path_rate,
            use_qk_norm=use_qk_norm,
            layer_scale_init=layer_scale_init,
        )

        if use_attention_pooling:
            self.attn_pool = MultiHeadAttentionPooling(
                hidden_dim, num_heads=num_heads, num_queries=attention_pool_queries, dropout=dropout,
            )

        if use_contrastive_head:
            self.contrastive_head = ProjectionHead(hidden_dim, contrastive_proj_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        **kwargs,
    ) -> WSIHintOutput:
        if self.feature_aug is not None:
            patch_features = self.feature_aug(patch_features, patch_mask)

        patch_tokens = self.patch_projection(patch_features)

        if self.use_positional_encoding and coords is not None:
            pe = self.pos_enc(coords)
            gate = self.pos_gate(patch_tokens)
            patch_tokens = patch_tokens + gate * pe

        if self.token_merger is not None:
            patch_tokens, patch_mask, coords = self.token_merger(patch_tokens, patch_mask, coords)

        region_tokens, patch_groups, region_mask, region_scores, residual_token, region_coords = (
            self.index_builder(patch_tokens, patch_mask, coords)
        )

        if self.use_graph_attention and region_coords is not None:
            region_tokens = self.graph_attn(region_tokens, region_coords, region_mask)

        retrieved_patch_groups, attention_index = self.retriever(region_tokens, patch_groups, region_mask)
        refined_regions = self.nested_attention(region_tokens, retrieved_patch_groups)

        slide_tokens = self.normalizer(refined_regions, region_mask)

        if self.use_residual_branch and residual_token is not None:
            res_projected = self.residual_proj(residual_token).unsqueeze(1)
            slide_tokens = torch.cat([slide_tokens, res_projected], dim=1)

        if self.num_register_tokens > 0:
            B = slide_tokens.size(0)
            regs = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
            slide_tokens = torch.cat([regs, slide_tokens], dim=1)

        latent_mask = torch.ones(slide_tokens.shape[:2], dtype=torch.bool, device=slide_tokens.device)
        slide_tokens, global_aux_loss = self.global_encoder(slide_tokens, mask=latent_mask)

        if self.num_register_tokens > 0:
            slide_tokens = slide_tokens[:, self.num_register_tokens:]

        if self.use_attention_pooling:
            pooled = self.attn_pool(slide_tokens, latent_mask[:, self.num_register_tokens:] if self.num_register_tokens > 0 else latent_mask)
        else:
            pooled = slide_tokens.mean(dim=1)
        logits = self.classifier(pooled)

        projection = self.contrastive_head(pooled) if self.use_contrastive_head else None

        total_aux = global_aux_loss + self.nested_attention.aux_loss * self.moe_aux_weight

        return WSIHintOutput(
            logits=logits,
            slide_tokens=slide_tokens,
            region_tokens=refined_regions,
            attention_index=attention_index,
            aux_loss=total_aux * self.moe_aux_weight,
            projection=projection,
        )
