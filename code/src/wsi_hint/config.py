from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    dataset_root: str = "data/raw_wsi"
    manifest_path: str = "artifacts/manifest.json"
    feature_dir: str = "artifacts/features"
    label_key: str = "project_code"
    patch_feature_dim: int = 1024
    max_patches: int = 16384
    patch_size: int = 256
    patch_stride: int = 256
    tissue_thumbnail_max_size: int = 2048
    tissue_saturation_threshold: float = 0.06
    tissue_value_threshold: float = 0.95
    load_coords: bool = True


@dataclass(slots=True)
class ModelConfig:
    input_dim: int = 1024
    hidden_dim: int = 512
    num_heads: int = 8
    region_size: int = 64
    max_regions: int = 256
    retrieval_topk: int = 8
    latent_tokens: int = 16
    global_layers: int = 4
    dropout: float = 0.1
    # Training-time: randomly drop instance patches (ABMIL); reduces shortcut reliance on few patches.
    instance_dropout: float = 0.0
    num_classes: int = 8
    # SOTA
    use_positional_encoding: bool = True
    use_multi_scale: bool = True
    coarse_scale_factor: int = 4
    use_ssm: bool = True
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    use_moe: bool = True
    num_experts: int = 8
    moe_topk: int = 2
    moe_aux_weight: float = 0.01
    use_residual_branch: bool = True
    feature_augmentation: bool = True
    aug_mask_ratio: float = 0.1
    aug_noise_std: float = 0.01
    # Beyond-SOTA
    use_graph_attention: bool = True
    graph_k_neighbors: int = 8
    num_register_tokens: int = 4
    drop_path_rate: float = 0.1
    use_contrastive_head: bool = True
    contrastive_proj_dim: int = 128
    # Beyond-SOTA v2
    use_attention_pooling: bool = True
    attention_pool_queries: int = 4
    # Beyond-SOTA v3
    use_token_merging: bool = True
    token_merge_ratio: float = 0.25
    use_qk_norm: bool = True
    layer_scale_init: float = 0.0001
    # CED-MIL-lite
    ced_attn_dim: int = 128
    ced_use_cf: bool = True
    ced_use_plugin_head: bool = False
    ced_lambda_sep: float = 0.05
    ced_lambda_align: float = 0.05
    ced_lambda_cf: float = 0.05
    ced_lambda_residual: float = 0.02
    ced_lambda_balance: float = 0.01
    ced_sep_margin: float = 1.0
    ced_cf_margin: float = 0.2


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    device: str = "cuda"
    # SOTA
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    gradient_clip_max_norm: float = 1.0
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    gradient_accumulation_steps: int = 4
    use_gradtail: bool = True
    gradtail_momentum: float = 0.9
    gradtail_min_weight: float = 0.5
    gradtail_max_weight: float = 3.0
    # Beyond-SOTA
    use_sam: bool = True
    sam_rho: float = 0.05
    use_rdrop: bool = True
    rdrop_alpha: float = 1.0
    use_supcon: bool = True
    supcon_weight: float = 0.1
    supcon_temperature: float = 0.07
    use_swa: bool = True
    swa_start_frac: float = 0.75
    tta_passes: int = 5
    # Beyond-SOTA v2
    use_asymmetric_loss: bool = False
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    # Beyond-SOTA v3
    use_self_distillation: bool = True
    distill_temperature: float = 4.0
    distill_alpha: float = 0.5
    use_gradient_centralization: bool = True


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass_config(instance: Any, overrides: dict[str, Any]) -> Any:
    values = asdict(instance)
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(instance)}
    for key, val in overrides.items():
        if key in field_types and isinstance(val, str):
            ft = field_types[key]
            if ft in ("float", float) or "float" in str(ft):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
            elif ft in ("int", int) or "int" in str(ft):
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    pass
        values[key] = val
    return type(instance)(**values)


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    dataset = _merge_dataclass_config(config.dataset, raw.get("dataset", {}))
    model = _merge_dataclass_config(config.model, raw.get("model", {}))
    training = _merge_dataclass_config(config.training, raw.get("training", {}))
    return AppConfig(dataset=dataset, model=model, training=training)
