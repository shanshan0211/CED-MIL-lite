from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .config import AppConfig, load_config
from .data import SlideFeatureDataset, pad_collate, pack_collate, write_manifest
from .feature_extraction import ExtractConfig, extract_from_manifest
from .gdc import fetch_case_metadata
from .model import ABMIL, CEDMIL, CLAMMIL, DSMIL, MeanPool, PatchSQClassifier, TransMILLite, WSIHint
from .slides import open_slide
from .training import (
    SAM,
    SWAModel,
    CosineWarmupScheduler,
    EMAModel,
    GradTailReweighter,
    TrainConfig,
    evaluate,
    evaluate_ensemble,
    evaluate_with_tta,
    model_soup,
    predict_probs,
    train_one_epoch,
)


def _inner_val_score(metrics, metric_name: str) -> float:
    """Score for early stopping inside each k-fold train fold."""
    if metric_name == "auc":
        return float(metrics.auc)
    if metric_name == "balanced_acc":
        return float(metrics.balanced_accuracy)
    return float(metrics.macro_f1)


def _set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _make_loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _build_model(config: AppConfig, model_name: str, num_classes: int) -> torch.nn.Module:
    mc = config.model
    match model_name:
        case "wsi-hint":
            return WSIHint(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_heads=mc.num_heads,
                region_size=mc.region_size,
                max_regions=mc.max_regions,
                retrieval_topk=mc.retrieval_topk,
                latent_tokens=mc.latent_tokens,
                global_layers=mc.global_layers,
                dropout=mc.dropout,
                num_classes=num_classes,
                use_positional_encoding=mc.use_positional_encoding,
                use_multi_scale=mc.use_multi_scale,
                coarse_scale_factor=mc.coarse_scale_factor,
                use_ssm=mc.use_ssm,
                ssm_d_state=mc.ssm_d_state,
                ssm_d_conv=mc.ssm_d_conv,
                ssm_expand=mc.ssm_expand,
                use_moe=mc.use_moe,
                num_experts=mc.num_experts,
                moe_topk=mc.moe_topk,
                use_residual_branch=mc.use_residual_branch,
                feature_augmentation=mc.feature_augmentation,
                aug_mask_ratio=mc.aug_mask_ratio,
                aug_noise_std=mc.aug_noise_std,
                moe_aux_weight=mc.moe_aux_weight,
                use_graph_attention=mc.use_graph_attention,
                graph_k_neighbors=mc.graph_k_neighbors,
                num_register_tokens=mc.num_register_tokens,
                drop_path_rate=mc.drop_path_rate,
                use_contrastive_head=mc.use_contrastive_head,
                contrastive_proj_dim=mc.contrastive_proj_dim,
                use_attention_pooling=mc.use_attention_pooling,
                attention_pool_queries=mc.attention_pool_queries,
                use_token_merging=mc.use_token_merging,
                token_merge_ratio=mc.token_merge_ratio,
                use_qk_norm=mc.use_qk_norm,
                layer_scale_init=mc.layer_scale_init,
            )
        case "abmil":
            return ABMIL(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
                instance_dropout=mc.instance_dropout,
                use_ced_head=mc.ced_use_plugin_head,
                ced_attn_dim=mc.ced_attn_dim,
                ced_lambda_sep=mc.ced_lambda_sep,
                ced_lambda_align=mc.ced_lambda_align,
                ced_lambda_cf=mc.ced_lambda_cf,
                ced_lambda_residual=mc.ced_lambda_residual,
                ced_lambda_balance=mc.ced_lambda_balance,
                ced_sep_margin=mc.ced_sep_margin,
                ced_cf_margin=mc.ced_cf_margin,
                ced_use_cf=mc.ced_use_cf,
            )
        case "clam":
            return CLAMMIL(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
            )
        case "dsmil":
            return DSMIL(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
            )
        case "meanpool":
            return MeanPool(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
            )
        case "transmil":
            return TransMILLite(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_heads=mc.num_heads,
                num_layers=max(1, mc.global_layers),
                num_classes=num_classes,
                dropout=mc.dropout,
                use_ced_head=mc.ced_use_plugin_head,
                ced_attn_dim=mc.ced_attn_dim,
                ced_lambda_sep=mc.ced_lambda_sep,
                ced_lambda_align=mc.ced_lambda_align,
                ced_lambda_cf=mc.ced_lambda_cf,
                ced_lambda_residual=mc.ced_lambda_residual,
                ced_lambda_balance=mc.ced_lambda_balance,
                ced_sep_margin=mc.ced_sep_margin,
                ced_cf_margin=mc.ced_cf_margin,
                ced_use_cf=mc.ced_use_cf,
            )
        case "ced-mil":
            return CEDMIL(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
                attn_dim=mc.ced_attn_dim,
                lambda_sep=mc.ced_lambda_sep,
                lambda_align=mc.ced_lambda_align,
                lambda_cf=mc.ced_lambda_cf,
                lambda_residual=mc.ced_lambda_residual,
                lambda_bal=mc.ced_lambda_balance,
                sep_margin=mc.ced_sep_margin,
                cf_margin=mc.ced_cf_margin,
                use_cf=mc.ced_use_cf,
            )
        case "patch-sq":
            return PatchSQClassifier(
                input_dim=mc.input_dim,
                hidden_dim=mc.hidden_dim,
                num_classes=num_classes,
                dropout=mc.dropout,
            )
        case _:
            raise ValueError(f"Unknown model: {model_name}")


def _make_train_config(config: AppConfig) -> TrainConfig:
    tc = config.training
    return TrainConfig(
        use_focal_loss=tc.use_focal_loss,
        focal_gamma=tc.focal_gamma,
        label_smoothing=tc.label_smoothing,
        gradient_clip_max_norm=tc.gradient_clip_max_norm,
        use_amp=tc.use_amp,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        use_gradtail=tc.use_gradtail,
        num_classes=config.model.num_classes,
        moe_aux_weight=config.model.moe_aux_weight,
        use_sam=tc.use_sam,
        sam_rho=tc.sam_rho,
        use_rdrop=tc.use_rdrop,
        rdrop_alpha=tc.rdrop_alpha,
        use_supcon=tc.use_supcon,
        supcon_weight=tc.supcon_weight,
        supcon_temperature=tc.supcon_temperature,
        use_asymmetric_loss=tc.use_asymmetric_loss,
        asl_gamma_pos=tc.asl_gamma_pos,
        asl_gamma_neg=tc.asl_gamma_neg,
        asl_clip=tc.asl_clip,
        use_mixup=tc.use_mixup,
        mixup_alpha=tc.mixup_alpha,
        use_self_distillation=tc.use_self_distillation,
        distill_temperature=tc.distill_temperature,
        distill_alpha=tc.distill_alpha,
        use_gradient_centralization=tc.use_gradient_centralization,
    )


def _split_indices(sample_count: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if sample_count <= 0:
        return [], []
    val_size = int(round(sample_count * val_fraction))
    val_size = max(1, min(val_size, sample_count - 1)) if sample_count > 1 else 0
    indices = list(range(sample_count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[val_size:], indices[:val_size]


def _group_kfold_indices(groups: list[str], folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    group_to_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)
    unique_groups = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    for idx, group in enumerate(unique_groups):
        fold_groups[idx % folds].append(group)
    splits: list[tuple[list[int], list[int]]] = []
    for fold_index in range(folds):
        test_groups = set(fold_groups[fold_index])
        test_indices: list[int] = []
        train_indices: list[int] = []
        for group, idxs in group_to_indices.items():
            if group in test_groups:
                test_indices.extend(idxs)
            else:
                train_indices.extend(idxs)
        splits.append((sorted(train_indices), sorted(test_indices)))
    return splits


def _group_train_val_split(
    indices: list[int],
    groups: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not indices:
        return [], []
    if not (0.0 < val_fraction < 1.0):
        return indices, []
    group_to_indices: dict[str, list[int]] = {}
    for index in indices:
        group_to_indices.setdefault(groups[index], []).append(index)
    unique_groups = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    val_group_count = int(round(len(unique_groups) * val_fraction))
    val_group_count = max(1, min(val_group_count, len(unique_groups) - 1)) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:val_group_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for group, idxs in group_to_indices.items():
        if group in val_groups:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)
    return sorted(train_indices), sorted(val_indices)


def _write_jsonl(path: str | Path | None, record: dict) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_collate_fn(config: AppConfig, use_pack: bool):
    if use_pack:
        pack_length = config.dataset.max_patches
        return lambda batch: pack_collate(batch, pack_length=pack_length)
    return pad_collate


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def command_dump_config(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def command_scan_dataset(args: argparse.Namespace) -> None:
    records = write_manifest(args.dataset_root, args.output)
    print(f"slides={len(records)} output={args.output}")


def command_make_binary_manifest(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    slides = payload.get("slides", [])
    if not isinstance(slides, list):
        raise TypeError("manifest.slides must be a list")

    pos_values = set(args.pos_values)
    neg_values = set(args.neg_values)

    if pos_values & neg_values:
        raise ValueError("pos-values and neg-values must be disjoint")

    kept: list[dict[str, Any]] = []
    for record in slides:
        if not isinstance(record, dict):
            continue
        v = record.get(args.label_key, None)
        if v in pos_values:
            r = dict(record)
            r[args.out_label_key] = args.pos_name
            kept.append(r)
        elif v in neg_values:
            r = dict(record)
            r[args.out_label_key] = args.neg_name
            kept.append(r)

    out_payload = dict(payload)
    out_payload["slides"] = kept
    out_payload["slide_count"] = len(kept)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"slides={len(kept)} output={args.output} out_label_key={args.out_label_key} pos={args.pos_name} neg={args.neg_name}")


def _parse_filter_specs(specs: list[str] | None) -> dict[str, set[str]]:
    parsed: dict[str, set[str]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Invalid filter '{spec}'. Expected KEY=VALUE.")
        key, value = spec.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid filter '{spec}'. Empty key is not allowed.")
        parsed.setdefault(key, set()).add(value)
    return parsed


def _record_matches_filters(record: dict[str, Any], include_filters: dict[str, set[str]], exclude_filters: dict[str, set[str]]) -> bool:
    for key, allowed_values in include_filters.items():
        if str(record.get(key, "")) not in allowed_values:
            return False
    for key, blocked_values in exclude_filters.items():
        if str(record.get(key, "")) in blocked_values:
            return False
    return True


def command_filter_manifest(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    slides = payload.get("slides", [])
    if not isinstance(slides, list):
        raise TypeError("manifest.slides must be a list")

    include_filters = _parse_filter_specs(args.where)
    exclude_filters = _parse_filter_specs(args.exclude)
    kept = [
        dict(record)
        for record in slides
        if isinstance(record, dict) and _record_matches_filters(record, include_filters, exclude_filters)
    ]

    out_payload = dict(payload)
    out_payload["slides"] = kept
    out_payload["slide_count"] = len(kept)
    out_payload["filtered_from"] = str(args.manifest)
    out_payload["filters"] = {
        "include": {key: sorted(values) for key, values in include_filters.items()},
        "exclude": {key: sorted(values) for key, values in exclude_filters.items()},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"slides={len(kept)} output={output}")

    if args.count_by:
        counts: dict[str, int] = {}
        for record in kept:
            key = str(record.get(args.count_by, ""))
            counts[key] = counts.get(key, 0) + 1
        print(f"count_by={args.count_by} values={json.dumps(counts, ensure_ascii=False, sort_keys=True)}")


def command_make_synthetic_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    for index, record in enumerate(payload["slides"]):
        token_count = min(config.dataset.max_patches, args.base_tokens + (index % args.token_stride) * args.token_step)
        features = torch.randn(token_count, config.model.input_dim, generator=generator)
        coords = torch.randint(0, 100000, (token_count, 2), generator=generator).float()
        torch.save({"features": features, "coords": coords, "meta": {"synthetic": True}}, output_dir / f"{record['slide_id']}.pt")
    print(f"generated={len(payload['slides'])} output_dir={output_dir}")


def command_extract_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    extract_config = ExtractConfig(
        patch_size=config.dataset.patch_size if args.patch_size is None else args.patch_size,
        patch_stride=config.dataset.patch_stride if args.patch_stride is None else args.patch_stride,
        tissue_thumbnail_max_size=config.dataset.tissue_thumbnail_max_size,
        tissue_saturation_threshold=config.dataset.tissue_saturation_threshold,
        tissue_value_threshold=config.dataset.tissue_value_threshold,
        max_patches=config.dataset.max_patches if args.max_patches is None else args.max_patches,
        seed=args.seed,
        batch_size=args.batch_size,
        encoder_name=args.encoder,
        encoder_path=args.encoder_path,
        encoder_input_size=args.encoder_input_size,
        normalize_mean=tuple(args.normalize_mean),
        normalize_std=tuple(args.normalize_std),
        output_dim=config.model.input_dim,
        device=args.device,
        dtype=args.dtype,
    )
    results = extract_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=extract_config,
        limit=args.limit,
    )
    if args.log_path:
        for record in results:
            _write_jsonl(args.log_path, record)
    print(f"slides={len(results)} output_dir={args.output_dir}")


def command_download_encoder(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Missing dependency: huggingface_hub") from exc
    token = None
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(repo_id=args.repo_id, local_dir=str(output_dir), token=token)
    print(f"repo_id={args.repo_id} local_dir={local_dir}")


def command_enrich_manifest(args: argparse.Namespace) -> None:
    input_path = Path(args.manifest)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    slides: list[dict] = payload.get("slides", [])
    patient_ids = [slide.get("patient_id", "") for slide in slides]
    mapping = fetch_case_metadata(patient_ids)
    updated = 0
    for slide in slides:
        pid = slide.get("patient_id", "")
        meta = mapping.get(pid)
        if meta is None:
            continue
        if slide.get("project_id") != meta.project_id:
            slide["project_id"] = meta.project_id
            updated += 1
        if slide.get("primary_diagnosis") != meta.primary_diagnosis:
            slide["primary_diagnosis"] = meta.primary_diagnosis
            updated += 1
    payload["slides"] = slides
    payload["gdc_enriched"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"slides={len(slides)} updated_fields={updated} output={output}")


def command_smoke_test(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model = _build_model(config, model_name="wsi-hint", num_classes=config.model.num_classes)
    features = torch.randn(args.batch_size, args.tokens, config.model.input_dim)
    mask = torch.ones(args.batch_size, args.tokens, dtype=torch.bool)
    coords = torch.randint(0, 100000, (args.batch_size, args.tokens, 2)).float()

    model.eval()
    with torch.no_grad():
        output = model(features, patch_mask=mask, coords=coords)
    print(
        f"logits={tuple(output.logits.shape)} "
        f"slide_tokens={tuple(output.slide_tokens.shape)} "
        f"region_tokens={tuple(output.region_tokens.shape)} "
        f"indices={tuple(output.attention_index.shape)} "
        f"aux_loss={output.aux_loss.item():.6f}"
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params:,} trainable_params={trainable_params:,}")

    features_desc = []
    if config.model.use_positional_encoding:
        features_desc.append("PositionalEncoding")
    if config.model.use_multi_scale:
        features_desc.append("MultiScale")
    if config.model.use_ssm:
        features_desc.append("Mamba-SSM")
    if config.model.use_moe:
        features_desc.append(f"MoE({config.model.num_experts}experts)")
    if config.model.use_residual_branch:
        features_desc.append("ResidualBranch")
    if config.model.feature_augmentation:
        features_desc.append("FeatureAug")
    if config.model.use_graph_attention:
        features_desc.append(f"GraphAttn(k={config.model.graph_k_neighbors})")
    if config.model.num_register_tokens > 0:
        features_desc.append(f"RegisterTokens({config.model.num_register_tokens})")
    if config.model.drop_path_rate > 0:
        features_desc.append(f"StochasticDepth({config.model.drop_path_rate})")
    if config.model.use_contrastive_head:
        features_desc.append("ContrastiveHead")
    if config.training.use_sam:
        features_desc.append("SAM")
    if config.training.use_rdrop:
        features_desc.append("R-Drop")
    if config.training.use_supcon:
        features_desc.append("SupCon")
    if config.training.use_swa:
        features_desc.append("SWA")
    if config.training.tta_passes > 1:
        features_desc.append(f"TTA({config.training.tta_passes})")
    if config.model.use_attention_pooling:
        features_desc.append(f"AttnPool(q={config.model.attention_pool_queries})")
    if config.training.use_asymmetric_loss:
        features_desc.append("AsymmetricLoss")
    if config.training.use_mixup:
        features_desc.append(f"MixUp(α={config.training.mixup_alpha})")
    if config.model.use_token_merging:
        features_desc.append(f"ToMe({config.model.token_merge_ratio})")
    if config.model.use_qk_norm:
        features_desc.append("QK-Norm+LayerScale")
    if config.training.use_self_distillation:
        features_desc.append(f"SelfDistill(T={config.training.distill_temperature})")
    if config.training.use_gradient_centralization:
        features_desc.append("GradCentral")
    print(f"active_features=[{', '.join(features_desc)}]")
    if output.projection is not None:
        print(f"contrastive_projection={tuple(output.projection.shape)}")


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tc = config.training
    _set_global_seed(args.seed)
    label_key = config.dataset.label_key if args.label_key is None else args.label_key
    dataset = SlideFeatureDataset(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    num_classes = len(dataset.label_to_index)
    model = _build_model(config, model_name=args.model, num_classes=num_classes)
    device = tc.device if args.device is None else args.device
    target_device = torch.device("cpu" if device == "cuda" and not torch.cuda.is_available() else device)
    model.to(target_device)

    train_indices, val_indices = _split_indices(len(dataset), args.val_fraction, args.seed)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices) if val_indices else None

    collate_fn = _get_collate_fn(config, args.use_pack)
    train_loader_generator = _make_loader_generator(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=tc.batch_size, shuffle=True,
        num_workers=tc.num_workers, collate_fn=collate_fn,
        generator=train_loader_generator, worker_init_fn=_seed_worker,
    )
    val_loader = (
        DataLoader(
            val_set, batch_size=tc.batch_size, shuffle=False,
            num_workers=tc.num_workers, collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
        )
        if val_set is not None
        else None
    )

    if tc.use_sam:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=tc.sam_rho, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)

    total_epochs = tc.epochs if args.epochs is None else args.epochs
    steps_per_epoch = max(1, len(train_loader) // max(1, tc.gradient_accumulation_steps))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = tc.warmup_epochs * steps_per_epoch
    base_opt = optimizer.base_optimizer if tc.use_sam else optimizer
    scheduler = CosineWarmupScheduler(base_opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=tc.min_lr)

    ema = EMAModel(model, decay=tc.ema_decay) if tc.use_ema else None
    swa = None
    swa_start_epoch = int(total_epochs * tc.swa_start_frac) if tc.use_swa else total_epochs + 1

    train_cfg = _make_train_config(config)
    train_cfg.num_classes = num_classes

    gradtail = GradTailReweighter(
        num_classes, momentum=tc.gradtail_momentum,
        min_weight=tc.gradtail_min_weight, max_weight=tc.gradtail_max_weight,
    ) if tc.use_gradtail else None

    best_val_loss = float("inf")
    for epoch in range(total_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, str(target_device),
            scheduler=scheduler, ema=ema, config=train_cfg, gradtail=gradtail,
        )

        if tc.use_swa and epoch >= swa_start_epoch:
            if swa is None:
                swa = SWAModel(ema.shadow if ema is not None else model)
            else:
                swa.update(ema.shadow if ema is not None else model)

        eval_model = ema.shadow if ema is not None else model
        eval_loader = val_loader if val_loader is not None else train_loader
        eval_metrics = evaluate(eval_model, eval_loader, str(target_device), config=train_cfg)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "train_balanced_acc": train_metrics.balanced_accuracy,
            "train_auc": train_metrics.auc,
            "eval_loss": eval_metrics.loss,
            "eval_acc": eval_metrics.accuracy,
            "eval_macro_f1": eval_metrics.macro_f1,
            "eval_balanced_acc": eval_metrics.balanced_accuracy,
            "eval_auc": eval_metrics.auc,
            "train_count": train_metrics.sample_count,
            "eval_count": eval_metrics.sample_count,
            "model": args.model,
            "label_key": label_key,
            "lr": optimizer.param_groups[0]["lr"],
        }
        _write_jsonl(args.log_path, record)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_metrics.loss:.4f} train_f1={train_metrics.macro_f1:.4f} train_auc={train_metrics.auc:.4f} "
            f"eval_loss={eval_metrics.loss:.4f} eval_f1={eval_metrics.macro_f1:.4f} eval_auc={eval_metrics.auc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if args.checkpoint and eval_metrics.loss <= best_val_loss:
            best_val_loss = eval_metrics.loss
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_state = {
                "model_state": (ema.state_dict() if ema is not None else model.state_dict()),
                "label_to_index": dataset.label_to_index,
                "epoch": epoch + 1,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "model": args.model,
                "label_key": label_key,
                "config": config.to_dict(),
                "eval_metrics": {
                    "loss": eval_metrics.loss,
                    "macro_f1": eval_metrics.macro_f1,
                    "auc": eval_metrics.auc,
                    "balanced_acc": eval_metrics.balanced_accuracy,
                },
            }
            torch.save(save_state, checkpoint_path)
            print(f"checkpoint={checkpoint_path} eval_f1={eval_metrics.macro_f1:.4f}")


@torch.no_grad()
def command_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    label_key = config.dataset.label_key if args.label_key is None else args.label_key
    dataset = SlideFeatureDataset(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = args.model or checkpoint.get("model", "wsi-hint")
    model = _build_model(config, model_name=model_name, num_classes=len(dataset.label_to_index))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = config.training.device if args.device is None else args.device
    target_device = torch.device("cpu" if device == "cuda" and not torch.cuda.is_available() else device)
    model.to(target_device)

    train_cfg = _make_train_config(config)
    train_cfg.num_classes = len(dataset.label_to_index)

    _, eval_indices = _split_indices(len(dataset), args.val_fraction, args.seed)
    eval_set = Subset(dataset, eval_indices) if eval_indices else dataset
    collate_fn = _get_collate_fn(config, False)
    eval_loader = DataLoader(
        eval_set, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, collate_fn=collate_fn,
    )
    metrics = evaluate(model, eval_loader, str(target_device), config=train_cfg)
    record = {
        "checkpoint": str(args.checkpoint),
        "loss": metrics.loss,
        "acc": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "balanced_acc": metrics.balanced_accuracy,
        "auc": metrics.auc,
        "count": metrics.sample_count,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "model": model_name,
        "label_key": label_key,
    }
    _write_jsonl(args.log_path, record)
    print(
        f"loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} f1={metrics.macro_f1:.4f} "
        f"balanced_acc={metrics.balanced_accuracy:.4f} auc={metrics.auc:.4f} count={metrics.sample_count}"
    )


def _dataset_groups(dataset: SlideFeatureDataset) -> list[str]:
    groups: list[str] = []
    for record in dataset.records:
        groups.append(str(record.get("patient_id") or record.get("case_id") or record.get("slide_id")))
    return groups


def _run_benchmark_heldout(config: AppConfig, args: argparse.Namespace) -> dict[str, Any]:
    tc = config.training
    _set_global_seed(args.seed)
    label_key = config.dataset.label_key if args.label_key is None else args.label_key

    train_dataset = SlideFeatureDataset(
        manifest_path=args.train_manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    test_dataset = SlideFeatureDataset(
        manifest_path=args.test_manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    if not train_dataset.records:
        raise ValueError("train_manifest contains no usable slides")
    if not test_dataset.records:
        raise ValueError("test_manifest contains no usable slides")
    if len(train_dataset.label_to_index) < 2:
        raise ValueError(
            f"train_manifest must contain at least 2 classes for label_key={label_key!r}; "
            f"got {sorted(train_dataset.label_to_index.keys())}"
        )
    if train_dataset.label_to_index != test_dataset.label_to_index:
        raise ValueError(
            "train/test label spaces do not match. "
            f"train={train_dataset.label_to_index} test={test_dataset.label_to_index}"
        )

    num_classes = len(train_dataset.label_to_index)
    model = _build_model(config, model_name=args.model, num_classes=num_classes)
    device = tc.device if args.device is None else args.device
    target_device = torch.device("cpu" if device == "cuda" and not torch.cuda.is_available() else device)
    model.to(target_device)

    all_train_indices = list(range(len(train_dataset)))
    train_groups = _dataset_groups(train_dataset)
    fit_indices, val_indices = _group_train_val_split(all_train_indices, train_groups, args.inner_val_fraction, args.seed)
    if not fit_indices:
        raise ValueError("No training samples remain after heldout validation split")

    fit_set = Subset(train_dataset, fit_indices)
    val_set = Subset(train_dataset, val_indices) if val_indices else None
    test_set = Subset(test_dataset, list(range(len(test_dataset))))

    collate_fn = _get_collate_fn(config, args.use_pack)
    train_loader_generator = _make_loader_generator(args.seed)
    train_loader = DataLoader(
        fit_set, batch_size=tc.batch_size, shuffle=True,
        num_workers=tc.num_workers, collate_fn=collate_fn,
        generator=train_loader_generator, worker_init_fn=_seed_worker,
    )
    val_loader = (
        DataLoader(
            val_set, batch_size=tc.batch_size, shuffle=False,
            num_workers=tc.num_workers, collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
        )
        if val_set is not None
        else None
    )
    test_loader = DataLoader(
        test_set, batch_size=tc.batch_size, shuffle=False,
        num_workers=tc.num_workers, collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
    )

    if tc.use_sam:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=tc.sam_rho, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // max(1, tc.gradient_accumulation_steps))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = tc.warmup_epochs * steps_per_epoch
    base_opt = optimizer.base_optimizer if tc.use_sam else optimizer
    scheduler = CosineWarmupScheduler(base_opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=tc.min_lr)
    ema = EMAModel(model, decay=tc.ema_decay) if tc.use_ema else None
    gradtail = GradTailReweighter(
        num_classes, momentum=tc.gradtail_momentum,
        min_weight=tc.gradtail_min_weight, max_weight=tc.gradtail_max_weight,
    ) if tc.use_gradtail else None

    train_cfg = _make_train_config(config)
    train_cfg.num_classes = num_classes

    best_epoch = args.epochs
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, str(target_device),
            scheduler=scheduler, ema=ema, config=train_cfg, gradtail=gradtail,
        )
        eval_model = ema.shadow if ema is not None else model
        eval_loader = val_loader if val_loader is not None else train_loader
        eval_metrics = evaluate(eval_model, eval_loader, str(target_device), config=train_cfg)
        score = _inner_val_score(eval_metrics, args.early_metric)
        improved = (score > best_score + args.min_delta) or (
            abs(score - best_score) <= args.min_delta and eval_metrics.loss < best_val_loss
        )
        if improved:
            best_score = score
            best_val_loss = eval_metrics.loss
            best_epoch = epoch
            state_src = ema.shadow if ema is not None else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "model": args.model,
            "label_key": label_key,
            "seed": args.seed,
            "train_loss": train_metrics.loss,
            "train_macro_f1": train_metrics.macro_f1,
            "train_balanced_acc": train_metrics.balanced_accuracy,
            "train_auc": train_metrics.auc,
            "eval_loss": eval_metrics.loss,
            "eval_macro_f1": eval_metrics.macro_f1,
            "eval_balanced_acc": eval_metrics.balanced_accuracy,
            "eval_auc": eval_metrics.auc,
            "fit_size": len(fit_indices),
            "val_size": len(val_indices),
            "test_size": len(test_dataset),
        }
        _write_jsonl(args.log_path, epoch_record)
        print(
            f"epoch={epoch} train_f1={train_metrics.macro_f1:.4f} train_auc={train_metrics.auc:.4f} "
            f"val_f1={eval_metrics.macro_f1:.4f} val_auc={eval_metrics.auc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    if tc.tta_passes > 1:
        test_metrics = evaluate_with_tta(model, test_loader, str(target_device), config=train_cfg, tta_passes=tc.tta_passes)
    else:
        test_metrics = evaluate(model, test_loader, str(target_device), config=train_cfg)

    if getattr(args, "save_preds", False) and args.log_path:
        probs, labels = predict_probs(
            model, test_loader, str(target_device),
            config=train_cfg, tta_passes=max(tc.tta_passes, 1),
        )
        probs_path = Path(args.log_path).with_suffix(".probs.pt")
        slide_ids = [test_dataset.records[i].get("slide_id", str(i)) for i in range(len(test_dataset.records))]
        torch.save(
            {
                "probs": probs,
                "labels": labels,
                "slide_ids": slide_ids,
                "split": "heldout_test",
                "model": args.model,
                "seed": args.seed,
            },
            probs_path,
        )
        print(f"saved probs -> {probs_path}")

    result = {
        "type": "heldout_test",
        "model": args.model,
        "label_key": label_key,
        "seed": args.seed,
        "early_metric": args.early_metric,
        "best_epoch": best_epoch,
        "fit_size": len(fit_indices),
        "val_size": len(val_indices),
        "test_size": len(test_dataset),
        "train_manifest": str(args.train_manifest),
        "test_manifest": str(args.test_manifest),
        "label_to_index": train_dataset.label_to_index,
        "loss": test_metrics.loss,
        "acc": test_metrics.accuracy,
        "macro_f1": test_metrics.macro_f1,
        "balanced_acc": test_metrics.balanced_accuracy,
        "auc": test_metrics.auc,
        "count": test_metrics.sample_count,
    }
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "label_to_index": train_dataset.label_to_index,
                "epoch": best_epoch,
                "seed": args.seed,
                "model": args.model,
                "label_key": label_key,
                "train_manifest": str(args.train_manifest),
                "test_manifest": str(args.test_manifest),
                "config": config.to_dict(),
                "eval_metrics": {
                    "loss": test_metrics.loss,
                    "macro_f1": test_metrics.macro_f1,
                    "auc": test_metrics.auc,
                    "balanced_acc": test_metrics.balanced_accuracy,
                },
            },
            checkpoint_path,
        )
        print(f"checkpoint={checkpoint_path}")
    _write_jsonl(args.log_path, result)
    print(
        f"heldout_test loss={test_metrics.loss:.4f} acc={test_metrics.accuracy:.4f} "
        f"f1={test_metrics.macro_f1:.4f} balanced_acc={test_metrics.balanced_accuracy:.4f} "
        f"auc={test_metrics.auc:.4f} count={test_metrics.sample_count}"
    )
    return result


def command_benchmark_heldout(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _run_benchmark_heldout(config, args)


@torch.no_grad()
def command_export_case_attention(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    config = load_config(args.config)
    if args.use_plugin_head:
        config.model.ced_use_plugin_head = True
    label_key = config.dataset.label_key if args.label_key is None else args.label_key
    dataset = SlideFeatureDataset(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    inv_label = {v: k for k, v in dataset.label_to_index.items()}

    model = _build_model(config, model_name=args.model, num_classes=len(dataset.label_to_index))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)

    device = config.training.device if args.device is None else args.device
    target_device = torch.device("cpu" if device == "cuda" and not torch.cuda.is_available() else device)
    model.to(target_device)
    model.eval()

    selected_ids = set(args.slide_ids or [])
    all_cases: list[dict] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if selected_ids and item["slide_id"] not in selected_ids:
            continue
        batch = pad_collate([item])
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if batch.coords is not None else None
        output = model(features, patch_mask=mask, coords=coords, labels=labels)
        probs = torch.softmax(output.logits, dim=-1)[0].detach().cpu()
        pred_idx = int(probs.argmax().item())
        true_idx = int(labels.item())
        confidence = float(probs[pred_idx].item())
        margin = float((probs.topk(k=min(2, probs.numel())).values[0] - probs.topk(k=min(2, probs.numel())).values[-1]).item())

        case_payload = {
            "slide_id": item["slide_id"],
            "metadata": item["metadata"],
            "true_label": inv_label[true_idx],
            "pred_label": inv_label[pred_idx],
            "correct": pred_idx == true_idx,
            "confidence": confidence,
            "margin": margin,
            "probs": {inv_label[i]: float(probs[i].item()) for i in range(len(inv_label))},
        }

        token_mask = batch.mask[0]
        token_count = int(token_mask.sum().item())
        coords_cpu = batch.coords[0, :token_count].cpu() if batch.coords is not None else None
        base_attention = getattr(output, "attention", None)
        if base_attention is not None:
            base_attention = base_attention[0, :token_count].detach().cpu()
            topk = min(args.topk, token_count)
            top_idx = torch.topk(base_attention, k=topk).indices.tolist()
            case_payload["base_top_patches"] = [
                {
                    "rank": rank + 1,
                    "patch_index": int(patch_index),
                    "weight": float(base_attention[patch_index].item()),
                    "coord": coords_cpu[patch_index].tolist() if coords_cpu is not None else None,
                }
                for rank, patch_index in enumerate(top_idx)
            ]

        role_attns = getattr(output, "role_attentions", None)
        role_gates = getattr(output, "role_gates", None)
        if role_attns is not None:
            role_names = ["class0", "class1", "shared"]
            role_payload: dict[str, list[dict]] = {}
            for role_name, attn in zip(role_names, role_attns, strict=True):
                weights = attn[0, :token_count].detach().cpu()
                topk = min(args.topk, token_count)
                top_idx = torch.topk(weights, k=topk).indices.tolist()
                role_payload[role_name] = [
                    {
                        "rank": rank + 1,
                        "patch_index": int(patch_index),
                        "weight": float(weights[patch_index].item()),
                        "coord": coords_cpu[patch_index].tolist() if coords_cpu is not None else None,
                    }
                    for rank, patch_index in enumerate(top_idx)
                ]
            case_payload["role_top_patches"] = role_payload
        if role_gates is not None:
            case_payload["mean_role_gates"] = role_gates[0, :token_count].mean(dim=0).detach().cpu().tolist()

        all_cases.append(case_payload)

    if not all_cases:
        raise ValueError("No cases selected for export.")

    if not selected_ids:
        correct_cases = sorted(
            [c for c in all_cases if c["correct"]],
            key=lambda x: (x["confidence"], x["margin"]),
            reverse=True,
        )
        error_cases = sorted(
            [c for c in all_cases if not c["correct"]],
            key=lambda x: (x["confidence"], x["margin"]),
            reverse=True,
        )
        selected_cases = correct_cases[:args.max_correct] + error_cases[:args.max_errors]
    else:
        selected_cases = all_cases

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases.json").write_text(json.dumps(selected_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_overlay(slide_path: str | None, slide_id: str, coords_cpu: torch.Tensor, maps: list[tuple[str, torch.Tensor]]) -> None:
        if not slide_path or not Path(slide_path).exists():
            return
        try:
            slide = open_slide(slide_path)
            width, height = slide.dimensions
            max_size = int(config.dataset.tissue_thumbnail_max_size)
            scale = min(max_size / max(width, 1), max_size / max(height, 1), 1.0)
            thumb_w = max(1, int(round(width * scale)))
            thumb_h = max(1, int(round(height * scale)))
            thumbnail = np.asarray(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
            try:
                slide.close()
            except Exception:
                pass
        except Exception as exc:
            print(f"warning: failed to render WSI thumbnail for {slide_id}: {exc}")
            return

        cols = 2
        rows = max(1, (len(maps) + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
        axes_list = np.atleast_1d(axes).reshape(-1)
        x = coords_cpu[:, 0].numpy() * scale
        y = coords_cpu[:, 1].numpy() * scale
        for ax, (title, weights) in zip(axes_list, maps, strict=False):
            ax.imshow(thumbnail)
            sc = ax.scatter(x, y, c=weights.numpy(), s=11, cmap="hot", alpha=0.55, linewidths=0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes_list[len(maps):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{slide_id}_overlay.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    selected_ids = {case["slide_id"] for case in selected_cases}
    for index in range(len(dataset)):
        item = dataset[index]
        if item["slide_id"] not in selected_ids:
            continue
        batch = pad_collate([item])
        features = batch.features.to(target_device)
        mask = batch.mask.to(target_device)
        labels = batch.labels.to(target_device)
        coords = batch.coords.to(target_device) if batch.coords is not None else None
        output = model(features, patch_mask=mask, coords=coords, labels=labels)

        token_count = int(batch.mask[0].sum().item())
        coords_cpu = batch.coords[0, :token_count].cpu() if batch.coords is not None else None
        if coords_cpu is None:
            continue

        maps: list[tuple[str, torch.Tensor]] = []
        base_attention = getattr(output, "attention", None)
        if base_attention is not None:
            maps.append(("base_attention", base_attention[0, :token_count].detach().cpu()))
        elif getattr(output, "role_gates", None) is not None:
            gate_max = output.role_gates[0, :token_count].max(dim=-1).values.detach().cpu()
            maps.append(("gate_max", gate_max))
        role_attns = getattr(output, "role_attentions", None)
        if role_attns is not None:
            role_names = ["role_class0", "role_class1", "role_shared"]
            for role_name, attn in zip(role_names, role_attns, strict=True):
                maps.append((role_name, attn[0, :token_count].detach().cpu()))

        cols = 2
        rows = max(1, (len(maps) + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
        axes_list = np.atleast_1d(axes).reshape(-1)
        x = coords_cpu[:, 0].numpy()
        y = -coords_cpu[:, 1].numpy()
        for ax, (title, weights) in zip(axes_list, maps, strict=False):
            sc = ax.scatter(x, y, c=weights.numpy(), s=9, cmap="hot")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes_list[len(maps):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{item['slide_id']}_attention.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        save_overlay(item["metadata"].get("svs_path"), item["slide_id"], coords_cpu, maps)

    print(f"exported_cases={len(selected_cases)} output_dir={output_dir}")


def command_benchmark_kfold(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _run_benchmark_kfold(config, args)


def _run_benchmark_kfold(config: AppConfig, args: argparse.Namespace) -> None:
    tc = config.training
    _set_global_seed(args.seed)
    label_key = config.dataset.label_key if args.label_key is None else args.label_key
    dataset = SlideFeatureDataset(
        manifest_path=args.manifest,
        feature_dir=args.feature_dir,
        label_key=label_key,
        load_coords=config.dataset.load_coords,
    )
    num_classes = len(dataset.label_to_index)
    if num_classes < 2:
        labels = ", ".join(repr(label) for label in dataset.label_to_index)
        raise ValueError(
            f"benchmark-kfold requires at least 2 classes, got {num_classes} for "
            f"label_key={label_key!r} with labels=[{labels}]. Check the manifest and label key."
        )
    device = tc.device if args.device is None else args.device
    target_device = torch.device("cpu" if device == "cuda" and not torch.cuda.is_available() else device)

    groups = [record.get("patient_id", record.get("slide_id", "")) for record in dataset.records]
    splits = _group_kfold_indices(groups, folds=args.folds, seed=args.seed)

    if args.overwrite:
        if args.log_path:
            Path(args.log_path).unlink(missing_ok=True)
        if args.summary_path:
            Path(args.summary_path).unlink(missing_ok=True)

    collate_fn = _get_collate_fn(config, args.use_pack)
    train_cfg = _make_train_config(config)
    train_cfg.num_classes = num_classes

    summary: list[dict] = []
    fold_state_dicts: list[dict[str, torch.Tensor]] = []
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        fold_seed = args.seed * 1000 + fold
        _set_global_seed(fold_seed)
        model = _build_model(config, model_name=args.model, num_classes=num_classes)
        model.to(target_device)

        if tc.use_sam:
            optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=tc.sam_rho, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)

        train_fold_indices, val_fold_indices = _group_train_val_split(
            train_indices, groups=groups, val_fraction=args.inner_val_fraction, seed=args.seed + fold,
        )

        train_set = Subset(dataset, train_fold_indices)
        val_set = Subset(dataset, val_fold_indices) if val_fold_indices else None
        test_set = Subset(dataset, test_indices)

        train_labels = [dataset[i]["label"] for i in train_fold_indices]
        from collections import Counter
        label_counts = Counter(train_labels)
        sample_weights = [1.0 / max(label_counts[l], 1) for l in train_labels]
        sampler_generator = _make_loader_generator(fold_seed)
        loader_generator = _make_loader_generator(fold_seed + 1)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_generator,
        )

        train_loader = DataLoader(
            train_set, batch_size=tc.batch_size, sampler=sampler,
            num_workers=tc.num_workers, collate_fn=collate_fn,
            generator=loader_generator, worker_init_fn=_seed_worker,
        )
        val_loader = (
            DataLoader(
                val_set, batch_size=tc.batch_size, shuffle=False,
                num_workers=tc.num_workers, collate_fn=collate_fn,
                worker_init_fn=_seed_worker,
            )
            if val_set is not None
            else None
        )
        test_loader = DataLoader(
            test_set, batch_size=tc.batch_size, shuffle=False,
            num_workers=tc.num_workers, collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
        )

        steps_per_epoch = max(1, len(train_loader) // max(1, tc.gradient_accumulation_steps))
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = tc.warmup_epochs * steps_per_epoch
        base_opt = optimizer.base_optimizer if tc.use_sam else optimizer
        scheduler = CosineWarmupScheduler(base_opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=tc.min_lr)
        ema = EMAModel(model, decay=tc.ema_decay) if tc.use_ema else None
        swa = None
        swa_start_epoch = int(args.epochs * tc.swa_start_frac) if tc.use_swa else args.epochs + 1
        gradtail = GradTailReweighter(
            num_classes, momentum=tc.gradtail_momentum,
            min_weight=tc.gradtail_min_weight, max_weight=tc.gradtail_max_weight,
        ) if tc.use_gradtail else None

        best_epoch = 0
        best_score = float("-inf")
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            train_one_epoch(
                model, train_loader, optimizer, str(target_device),
                scheduler=scheduler, ema=ema, config=train_cfg, gradtail=gradtail,
            )
            if tc.use_swa and epoch >= swa_start_epoch:
                if swa is None:
                    swa = SWAModel(ema.shadow if ema is not None else model)
                else:
                    swa.update(ema.shadow if ema is not None else model)
            if val_loader is None:
                continue
            eval_model = ema.shadow if ema is not None else model
            val_metrics = evaluate(eval_model, val_loader, str(target_device), config=train_cfg)
            score = _inner_val_score(val_metrics, args.early_metric)
            improved = (score > best_score + args.min_delta) or (
                abs(score - best_score) <= args.min_delta and val_metrics.loss < best_val_loss
            )
            if improved:
                best_score = score
                best_val_loss = val_metrics.loss
                best_epoch = epoch
                state_src = ema.shadow if ema is not None else model
                best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        if swa is not None:
            swa.apply_to(model)
        fold_state_dicts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if tc.tta_passes > 1:
            metrics = evaluate_with_tta(model, test_loader, str(target_device), config=train_cfg, tta_passes=tc.tta_passes)
        else:
            metrics = evaluate(model, test_loader, str(target_device), config=train_cfg)

        if getattr(args, "save_fold_preds", False) and args.log_path:
            fold_probs, fold_labels = predict_probs(
                model, test_loader, str(target_device),
                config=train_cfg, tta_passes=max(tc.tta_passes, 1),
            )
            fold_pred_path = Path(args.log_path).with_suffix(f".probs_fold{fold}.pt")
            test_slide_ids = [dataset.records[i].get("slide_id", str(i)) for i in test_indices]
            torch.save(
                {"probs": fold_probs, "labels": fold_labels, "slide_ids": test_slide_ids,
                 "fold": fold, "model": args.model, "seed": args.seed},
                fold_pred_path,
            )
        record = {
            "fold": fold,
            "folds": args.folds,
            "model": args.model,
            "label_key": label_key,
            "early_metric": args.early_metric,
            "loss": metrics.loss,
            "acc": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "balanced_acc": metrics.balanced_accuracy,
            "auc": metrics.auc,
            "count": metrics.sample_count,
            "train_size": len(train_fold_indices),
            "val_size": len(val_fold_indices),
            "test_size": len(test_indices),
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "inner_val_fraction": args.inner_val_fraction,
        }
        summary.append(record)
        _write_jsonl(args.log_path, record)
        print(
            f"fold={fold}/{args.folds} loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} "
            f"f1={metrics.macro_f1:.4f} balanced_acc={metrics.balanced_accuracy:.4f} "
            f"auc={metrics.auc:.4f} n={metrics.sample_count}"
        )

    if summary:
        avg_f1 = sum(r["macro_f1"] for r in summary) / len(summary)
        avg_auc = sum(r["auc"] for r in summary) / len(summary)
        avg_bacc = sum(r["balanced_acc"] for r in summary) / len(summary)
        print(f"\n=== K-Fold Summary ===")
        print(f"avg_macro_f1={avg_f1:.4f} avg_auc={avg_auc:.4f} avg_balanced_acc={avg_bacc:.4f}")

        if len(fold_state_dicts) >= 2:
            soup_state = model_soup(fold_state_dicts)
            soup_model = _build_model(config, model_name=args.model, num_classes=num_classes)
            soup_model.load_state_dict(soup_state, strict=True)
            soup_model.to(target_device)
            full_loader = DataLoader(
                dataset, batch_size=tc.batch_size, shuffle=False,
                num_workers=tc.num_workers, collate_fn=collate_fn,
                worker_init_fn=_seed_worker,
            )
            if tc.tta_passes > 1:
                soup_metrics = evaluate_with_tta(soup_model, full_loader, str(target_device), config=train_cfg, tta_passes=tc.tta_passes)
            else:
                soup_metrics = evaluate(soup_model, full_loader, str(target_device), config=train_cfg)
            print(
                f"model_soup: f1={soup_metrics.macro_f1:.4f} auc={soup_metrics.auc:.4f} "
                f"balanced_acc={soup_metrics.balanced_accuracy:.4f} n={soup_metrics.sample_count}"
            )
            summary.append({
                "type": "model_soup",
                "macro_f1": soup_metrics.macro_f1,
                "auc": soup_metrics.auc,
                "balanced_acc": soup_metrics.balanced_accuracy,
                "acc": soup_metrics.accuracy,
                "loss": soup_metrics.loss,
                "count": soup_metrics.sample_count,
            })

            probs, labels = predict_probs(
                soup_model, full_loader, str(target_device),
                config=train_cfg, tta_passes=max(tc.tta_passes, 1),
            )
            probs_path = Path(args.log_path).with_suffix(".probs.pt") if args.log_path else None
            if probs_path:
                torch.save({"probs": probs, "labels": labels}, probs_path)
                print(f"saved probs -> {probs_path}")

            soup_ckpt_path = Path(args.log_path).with_suffix(".soup.pt") if args.log_path else None
            if soup_ckpt_path:
                torch.save(soup_state, soup_ckpt_path)
                print(f"saved soup checkpoint -> {soup_ckpt_path}")

    if args.summary_path:
        Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_dropout_tag(p: float) -> str:
    s = f"{p:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def command_protocol_abmil_clean(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)

    manifest = args.manifest or base_config.dataset.manifest_path
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)
    baseline_instance_dropout = float(base_config.model.instance_dropout)

    def run_one(*, tag: str, early_metric: str, instance_dropout: float) -> None:
        config = copy.deepcopy(base_config)
        config.model.instance_dropout = float(instance_dropout)

        log_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{args.seed}.jsonl")
        summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{args.seed}_summary.json")

        call_args = argparse.Namespace(
            config=args.config,
            manifest=str(manifest),
            feature_dir=str(feature_dir),
            model="abmil",
            label_key=str(label_key),
            folds=args.folds,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            log_path=str(log_path),
            summary_path=str(summary_path),
            inner_val_fraction=args.inner_val_fraction,
            patience=args.patience,
            min_delta=args.min_delta,
            early_metric=early_metric,
            overwrite=True,
            use_pack=args.use_pack,
        )
        _run_benchmark_kfold(config, call_args)

    run_one(tag="aucstop", early_metric="auc", instance_dropout=baseline_instance_dropout)
    run_one(tag="f1stop", early_metric="macro_f1", instance_dropout=baseline_instance_dropout)
    run_one(tag=f"aucstop_idrop{_format_dropout_tag(0.0)}", early_metric="auc", instance_dropout=0.0)
    run_one(tag=f"aucstop_idrop{_format_dropout_tag(0.25)}", early_metric="auc", instance_dropout=0.25)


def _summarize_ablation_results(out_prefix: Path, seed: int, tags: list[str]) -> None:
    print("\n" + "=" * 40)
    print(f"=== Ablation Summary (Seed {seed}) ===")
    print("=" * 40)
    from statistics import mean
    for tag in tags:
        summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        folds = [x for x in data if isinstance(x, dict) and "fold" in x]
        soup = next((x for x in data if isinstance(x, dict) and x.get("type") == "model_soup"), None)
        
        print(f"--- [ {tag} ] ---")
        if folds:
            avg_auc = mean(x["auc"] for x in folds)
            avg_f1 = mean(x["macro_f1"] for x in folds)
            avg_bacc = mean(x["balanced_acc"] for x in folds)
            print(f"Fold Avg : AUC={avg_auc:.4f}  F1={avg_f1:.4f}  BAcc={avg_bacc:.4f}")
        if soup:
            print(f"Soup     : AUC={soup['auc']:.4f}  F1={soup['macro_f1']:.4f}  BAcc={soup['balanced_acc']:.4f}")
        print()


def _run_cedmil_ablation_seed(
    *,
    base_config: AppConfig,
    manifest: str,
    feature_dir: str,
    label_key: str,
    out_prefix: Path,
    folds: int,
    epochs: int,
    seed: int,
    device: str | None,
    inner_val_fraction: float,
    patience: int,
    min_delta: float,
    use_pack: bool,
    save_fold_preds: bool = False,
) -> list[str]:
    tags = ["abmil_baseline", "cedmil_nocf", "cedmil_withcf"]

    def run_one(tag: str, model_name: str, use_cf: bool) -> None:
        config = copy.deepcopy(base_config)
        # Keep the same strong protocol across all ablations.
        config.model.instance_dropout = 0.15
        if model_name == "ced-mil":
            config.model.ced_use_cf = use_cf

        log_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}.jsonl")
        summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")

        call_args = argparse.Namespace(
            config=None,
            manifest=str(manifest),
            feature_dir=str(feature_dir),
            model=model_name,
            label_key=str(label_key),
            folds=folds,
            epochs=epochs,
            seed=seed,
            device=device,
            log_path=str(log_path),
            summary_path=str(summary_path),
            inner_val_fraction=inner_val_fraction,
            patience=patience,
            min_delta=min_delta,
            early_metric="auc",
            overwrite=True,
            use_pack=use_pack,
            save_fold_preds=save_fold_preds,
        )
        _run_benchmark_kfold(config, call_args)

    print(f"\n>>> Running [1/3] ABMIL Strong Baseline (seed={seed})...")
    run_one(tag=tags[0], model_name="abmil", use_cf=False)

    print(f"\n>>> Running [2/3] CED-MIL-lite (w/o Counterfactual Ablation, seed={seed})...")
    run_one(tag=tags[1], model_name="ced-mil", use_cf=False)

    print(f"\n>>> Running [3/3] CED-MIL-lite (Full, seed={seed})...")
    run_one(tag=tags[2], model_name="ced-mil", use_cf=True)

    return tags


def command_protocol_cedmil_ablation(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)

    manifest = args.manifest or base_config.dataset.manifest_path
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)
    tags = _run_cedmil_ablation_seed(
        base_config=base_config,
        manifest=str(manifest),
        feature_dir=str(feature_dir),
        label_key=str(label_key),
        out_prefix=out_prefix,
        folds=args.folds,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        inner_val_fraction=args.inner_val_fraction,
        patience=args.patience,
        min_delta=args.min_delta,
        use_pack=args.use_pack,
    )
    _summarize_ablation_results(out_prefix, args.seed, tags)


def _read_seed_ablation_metrics(summary_path: Path) -> dict[str, float] | None:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    folds = [x for x in data if isinstance(x, dict) and "fold" in x]
    soup = next((x for x in data if isinstance(x, dict) and x.get("type") == "model_soup"), None)
    if not folds:
        return None
    from statistics import mean
    metrics = {
        "avg_auc": mean(x["auc"] for x in folds),
        "avg_f1": mean(x["macro_f1"] for x in folds),
        "avg_bacc": mean(x["balanced_acc"] for x in folds),
    }
    if soup is not None:
        metrics.update({
            "soup_auc": float(soup["auc"]),
            "soup_f1": float(soup["macro_f1"]),
            "soup_bacc": float(soup["balanced_acc"]),
        })
    return metrics


def _summarize_multiseed_ablation_results(out_prefix: Path, seeds: list[int], tags: list[str], summary_path: str | None = None) -> None:
    from statistics import mean, stdev

    def fmt(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{mean(values):.4f} +- {stdev(values):.4f}"

    payload: dict[str, dict[str, object]] = {"seeds": {"values": seeds}}
    print("\n" + "=" * 56)
    print(f"=== Multi-Seed Ablation Summary (seeds={seeds}) ===")
    print("=" * 56)

    for tag in tags:
        gathered: list[dict[str, float]] = []
        for seed in seeds:
            path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")
            metrics = _read_seed_ablation_metrics(path)
            if metrics is not None:
                gathered.append(metrics)
        if not gathered:
            continue

        avg_auc = [m["avg_auc"] for m in gathered]
        avg_f1 = [m["avg_f1"] for m in gathered]
        avg_bacc = [m["avg_bacc"] for m in gathered]
        soup_auc = [m["soup_auc"] for m in gathered if "soup_auc" in m]
        soup_f1 = [m["soup_f1"] for m in gathered if "soup_f1" in m]
        soup_bacc = [m["soup_bacc"] for m in gathered if "soup_bacc" in m]

        payload[tag] = {
            "seed_count": len(gathered),
            "avg_auc": {"values": avg_auc, "mean": mean(avg_auc), "std": stdev(avg_auc) if len(avg_auc) > 1 else 0.0},
            "avg_f1": {"values": avg_f1, "mean": mean(avg_f1), "std": stdev(avg_f1) if len(avg_f1) > 1 else 0.0},
            "avg_bacc": {"values": avg_bacc, "mean": mean(avg_bacc), "std": stdev(avg_bacc) if len(avg_bacc) > 1 else 0.0},
            "soup_auc": {"values": soup_auc, "mean": mean(soup_auc) if soup_auc else None, "std": stdev(soup_auc) if len(soup_auc) > 1 else 0.0 if soup_auc else None},
            "soup_f1": {"values": soup_f1, "mean": mean(soup_f1) if soup_f1 else None, "std": stdev(soup_f1) if len(soup_f1) > 1 else 0.0 if soup_f1 else None},
            "soup_bacc": {"values": soup_bacc, "mean": mean(soup_bacc) if soup_bacc else None, "std": stdev(soup_bacc) if len(soup_bacc) > 1 else 0.0 if soup_bacc else None},
        }

        print(f"--- [ {tag} ] ---")
        print(f"Fold Avg : AUC={fmt(avg_auc)}  F1={fmt(avg_f1)}  BAcc={fmt(avg_bacc)}")
        if soup_auc:
            print(f"Soup     : AUC={fmt(soup_auc)}  F1={fmt(soup_f1)}  BAcc={fmt(soup_bacc)}")
        print()

    if summary_path:
        Path(summary_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved multi-seed summary -> {summary_path}")


def _read_seed_heldout_metrics(summary_path: Path) -> dict[str, float] | None:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    required = ("auc", "macro_f1", "balanced_acc", "acc", "loss")
    if not all(key in data for key in required):
        return None
    return {
        "auc": float(data["auc"]),
        "f1": float(data["macro_f1"]),
        "bacc": float(data["balanced_acc"]),
        "acc": float(data["acc"]),
        "loss": float(data["loss"]),
    }


def _summarize_multiseed_heldout_results(out_prefix: Path, seeds: list[int], tags: list[str], summary_path: str | None = None) -> None:
    from statistics import mean, stdev

    def pack(values: list[float]) -> dict[str, object]:
        return {
            "values": values,
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
        }

    payload: dict[str, dict[str, object]] = {"seeds": {"values": seeds}}
    print("\n" + "=" * 56)
    print(f"=== Multi-Seed Heldout Summary (seeds={seeds}) ===")
    print("=" * 56)

    for tag in tags:
        gathered: list[dict[str, float]] = []
        for seed in seeds:
            path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")
            metrics = _read_seed_heldout_metrics(path)
            if metrics is not None:
                gathered.append(metrics)
        if not gathered:
            continue
        auc = [m["auc"] for m in gathered]
        f1 = [m["f1"] for m in gathered]
        bacc = [m["bacc"] for m in gathered]
        acc = [m["acc"] for m in gathered]
        loss = [m["loss"] for m in gathered]
        payload[tag] = {
            "seed_count": len(gathered),
            "auc": pack(auc),
            "f1": pack(f1),
            "bacc": pack(bacc),
            "acc": pack(acc),
            "loss": pack(loss),
        }
        auc_msg = f"{mean(auc):.4f} +- {stdev(auc):.4f}" if len(auc) > 1 else f"{auc[0]:.4f}"
        f1_msg = f"{mean(f1):.4f} +- {stdev(f1):.4f}" if len(f1) > 1 else f"{f1[0]:.4f}"
        bacc_msg = f"{mean(bacc):.4f} +- {stdev(bacc):.4f}" if len(bacc) > 1 else f"{bacc[0]:.4f}"
        print(f"--- [ {tag} ] ---")
        print(f"Heldout  : AUC={auc_msg}  F1={f1_msg}  BAcc={bacc_msg}")
        print()

    if summary_path:
        Path(summary_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved heldout multi-seed summary -> {summary_path}")


def command_summarize_multiseed_ablation(args: argparse.Namespace) -> None:
    out_prefix = Path(args.out_prefix)
    seeds = list(args.seeds)
    if args.preset == "plugin-ablation":
        tags = ["abmil_base", "abmil_plugin", "transmil_base", "transmil_plugin"]
    elif args.preset == "cedmil-ablation":
        tags = ["abmil_baseline", "cedmil_nocf", "cedmil_withcf"]
    else:
        tags = list(args.tags or [])
        if not tags:
            raise ValueError("preset=custom requires --tags")
    _summarize_multiseed_ablation_results(out_prefix, seeds, tags, args.summary_path)


def command_protocol_cedmil_multiseed(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    manifest = args.manifest or base_config.dataset.manifest_path
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)
    tags: list[str] | None = None

    for idx, seed in enumerate(args.seeds, start=1):
        print("\n" + "#" * 56)
        print(f"### Multi-seed run {idx}/{len(args.seeds)} | seed={seed}")
        print("#" * 56)
        tags = _run_cedmil_ablation_seed(
            base_config=base_config,
            manifest=str(manifest),
            feature_dir=str(feature_dir),
            label_key=str(label_key),
            out_prefix=out_prefix,
            folds=args.folds,
            epochs=args.epochs,
            seed=seed,
            device=args.device,
            inner_val_fraction=args.inner_val_fraction,
            patience=args.patience,
            min_delta=args.min_delta,
            use_pack=args.use_pack,
            save_fold_preds=bool(getattr(args, "save_fold_preds", False)),
        )
        _summarize_ablation_results(out_prefix, seed, tags)

    if tags is not None:
        _summarize_multiseed_ablation_results(out_prefix, args.seeds, tags, args.summary_path)


def command_protocol_baselines_multiseed(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    manifest = args.manifest or base_config.dataset.manifest_path
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)

    models = list(args.models)
    tags = [m.replace("-", "_") for m in models]

    for idx, seed in enumerate(args.seeds, start=1):
        print("\n" + "#" * 56)
        print(f"### Baselines multi-seed run {idx}/{len(args.seeds)} | seed={seed}")
        print("#" * 56)

        for model_name, tag in zip(models, tags, strict=True):
            config = copy.deepcopy(base_config)
            if model_name == "abmil":
                config.model.instance_dropout = float(args.instance_dropout)
            else:
                config.model.instance_dropout = 0.0
            if model_name == "ced-mil":
                config.model.ced_use_cf = bool(args.ced_use_cf)

            log_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}.jsonl")
            summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")

            call_args = argparse.Namespace(
                config=None,
                manifest=str(manifest),
                feature_dir=str(feature_dir),
                model=model_name,
                label_key=str(label_key),
                folds=args.folds,
                epochs=args.epochs,
                seed=seed,
                device=args.device,
                log_path=str(log_path),
                summary_path=str(summary_path),
                inner_val_fraction=args.inner_val_fraction,
                patience=args.patience,
                min_delta=args.min_delta,
                early_metric="auc",
                overwrite=True,
                use_pack=args.use_pack,
                save_fold_preds=bool(getattr(args, "save_fold_preds", False)),
            )
            print(f"\n>>> Running model={model_name} seed={seed}...")
            _run_benchmark_kfold(config, call_args)

    _summarize_multiseed_ablation_results(out_prefix, args.seeds, tags, args.summary_path)


def command_protocol_plugin_ablation_multiseed(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    manifest = args.manifest or base_config.dataset.manifest_path
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)

    specs = [
        ("abmil", "abmil_base", False),
        ("abmil", "abmil_plugin", True),
        ("transmil", "transmil_base", False),
        ("transmil", "transmil_plugin", True),
    ]

    for idx, seed in enumerate(args.seeds, start=1):
        print("\n" + "#" * 56)
        print(f"### Plugin ablation multi-seed run {idx}/{len(args.seeds)} | seed={seed}")
        print("#" * 56)

        for model_name, tag, use_plugin in specs:
            config = copy.deepcopy(base_config)
            config.model.ced_use_plugin_head = bool(use_plugin)
            config.model.instance_dropout = float(args.instance_dropout) if model_name == "abmil" else 0.0
            config.model.ced_use_cf = bool(args.ced_use_cf)

            log_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}.jsonl")
            summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")
            call_args = argparse.Namespace(
                config=None,
                manifest=str(manifest),
                feature_dir=str(feature_dir),
                model=model_name,
                label_key=str(label_key),
                folds=args.folds,
                epochs=args.epochs,
                seed=seed,
                device=args.device,
                log_path=str(log_path),
                summary_path=str(summary_path),
                inner_val_fraction=args.inner_val_fraction,
                patience=args.patience,
                min_delta=args.min_delta,
                early_metric="auc",
                overwrite=True,
                use_pack=args.use_pack,
                save_fold_preds=bool(getattr(args, "save_fold_preds", False)),
            )
            print(f"\n>>> Running model={model_name} variant={tag} seed={seed}...")
            _run_benchmark_kfold(config, call_args)

    tags = [tag for _, tag, _ in specs]
    _summarize_multiseed_ablation_results(out_prefix, args.seeds, tags, args.summary_path)


def command_protocol_plugin_ablation_heldout(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    feature_dir = args.feature_dir or base_config.dataset.feature_dir
    label_key = args.label_key or base_config.dataset.label_key
    out_prefix = Path(args.out_prefix)

    all_specs: list[tuple[str, str, bool]] = [
        ("abmil", "abmil_base", False),
        ("abmil", "abmil_plugin", True),
        ("transmil", "transmil_base", False),
        ("transmil", "transmil_plugin", True),
    ]
    want = getattr(args, "tags", None)
    if want:
        allowed = {t for t in want}
        specs = [s for s in all_specs if s[1] in allowed]
        unknown = allowed - {s[1] for s in specs}
        if unknown:
            raise ValueError(f"Unknown --tags entries (expected abmil_base, ...): {sorted(unknown)}")
        if not specs:
            raise ValueError("--tags filtered out all arms")
    else:
        specs = all_specs

    for idx, seed in enumerate(args.seeds, start=1):
        print("\n" + "#" * 56)
        print(f"### Plugin ablation heldout run {idx}/{len(args.seeds)} | seed={seed}")
        print("#" * 56)

        for model_name, tag, use_plugin in specs:
            config = copy.deepcopy(base_config)
            config.model.ced_use_plugin_head = bool(use_plugin)
            config.model.instance_dropout = float(args.instance_dropout) if model_name == "abmil" else 0.0
            config.model.ced_use_cf = bool(args.ced_use_cf)

            log_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}.jsonl")
            summary_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}_summary.json")
            checkpoint_path = out_prefix.with_name(f"{out_prefix.name}_{tag}_s{seed}.pt") if args.save_checkpoints else None
            call_args = argparse.Namespace(
                config=None,
                train_manifest=str(args.train_manifest),
                test_manifest=str(args.test_manifest),
                feature_dir=str(feature_dir),
                model=model_name,
                label_key=str(label_key),
                epochs=args.epochs,
                seed=seed,
                device=args.device,
                log_path=str(log_path),
                summary_path=str(summary_path),
                checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
                inner_val_fraction=args.inner_val_fraction,
                patience=args.patience,
                min_delta=args.min_delta,
                early_metric="auc",
                use_pack=args.use_pack,
                save_preds=bool(getattr(args, "save_preds", False)),
            )
            print(f"\n>>> Running heldout model={model_name} variant={tag} seed={seed}...")
            _run_benchmark_heldout(config, call_args)

    tags_summary = [tag for _, tag, _ in specs]
    _summarize_multiseed_heldout_results(out_prefix, args.seeds, tags_summary, args.summary_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wsi-hint")
    parser.add_argument("--config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_config = subparsers.add_parser("dump-config")
    dump_config.add_argument("--output", required=True)
    dump_config.set_defaults(func=command_dump_config)

    scan_dataset = subparsers.add_parser("scan-dataset")
    scan_dataset.add_argument("--dataset-root", required=True)
    scan_dataset.add_argument("--output", required=True)
    scan_dataset.set_defaults(func=command_scan_dataset)

    enrich_manifest = subparsers.add_parser("enrich-manifest")
    enrich_manifest.add_argument("--manifest", required=True)
    enrich_manifest.add_argument("--output", required=True)
    enrich_manifest.set_defaults(func=command_enrich_manifest)

    binary_manifest = subparsers.add_parser("make-binary-manifest")
    binary_manifest.add_argument("--manifest", required=True)
    binary_manifest.add_argument("--label-key", required=True)
    binary_manifest.add_argument("--pos-values", nargs="+", required=True)
    binary_manifest.add_argument("--neg-values", nargs="+", required=True)
    binary_manifest.add_argument("--pos-name", default="POS")
    binary_manifest.add_argument("--neg-name", default="NEG")
    binary_manifest.add_argument("--out-label-key", default="label")
    binary_manifest.add_argument("--output", required=True)
    binary_manifest.set_defaults(func=command_make_binary_manifest)

    filter_manifest = subparsers.add_parser("filter-manifest")
    filter_manifest.add_argument("--manifest", required=True)
    filter_manifest.add_argument("--output", required=True)
    filter_manifest.add_argument("--where", nargs="*", default=None, help="Include filters as KEY=VALUE.")
    filter_manifest.add_argument("--exclude", nargs="*", default=None, help="Exclude filters as KEY=VALUE.")
    filter_manifest.add_argument("--count-by", default=None, help="Optional field for quick count summary.")
    filter_manifest.set_defaults(func=command_filter_manifest)

    synthetic = subparsers.add_parser("make-synthetic-features")
    synthetic.add_argument("--manifest", required=True)
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--base-tokens", type=int, default=256)
    synthetic.add_argument("--token-stride", type=int, default=31)
    synthetic.add_argument("--token-step", type=int, default=32)
    synthetic.set_defaults(func=command_make_synthetic_features)

    extract_cmd = subparsers.add_parser("extract-features")
    extract_cmd.add_argument("--manifest", required=True)
    extract_cmd.add_argument("--output-dir", required=True)
    extract_cmd.add_argument("--encoder", default="phikon-v2")
    extract_cmd.add_argument("--encoder-path", default=None)
    extract_cmd.add_argument("--encoder-input-size", type=int, default=224)
    extract_cmd.add_argument("--normalize-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    extract_cmd.add_argument("--normalize-std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    extract_cmd.add_argument("--patch-size", type=int, default=None)
    extract_cmd.add_argument("--patch-stride", type=int, default=None)
    extract_cmd.add_argument("--max-patches", type=int, default=None)
    extract_cmd.add_argument("--batch-size", type=int, default=16)
    extract_cmd.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    extract_cmd.add_argument("--device", default="cpu")
    extract_cmd.add_argument("--seed", type=int, default=7)
    extract_cmd.add_argument("--limit", type=int, default=None)
    extract_cmd.add_argument("--log-path", default=None)
    extract_cmd.set_defaults(func=command_extract_features)

    download_cmd = subparsers.add_parser("download-encoder")
    download_cmd.add_argument("--repo-id", default="owkin/phikon-v2")
    download_cmd.add_argument("--output-dir", required=True)
    download_cmd.add_argument("--token-file", default=None)
    download_cmd.set_defaults(func=command_download_encoder)

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--tokens", type=int, default=2048)
    smoke.set_defaults(func=command_smoke_test)

    train = subparsers.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--feature-dir", required=True)
    train.add_argument("--model", choices=["wsi-hint", "abmil", "clam", "dsmil", "meanpool", "transmil", "ced-mil", "patch-sq"], default="wsi-hint")
    train.add_argument("--label-key", default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--checkpoint", default=None)
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-path", default=None)
    train.add_argument("--use-pack", action="store_true", help="Use pack-based collation")
    train.set_defaults(func=command_train)

    eval_cmd = subparsers.add_parser("eval")
    eval_cmd.add_argument("--manifest", required=True)
    eval_cmd.add_argument("--feature-dir", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--model", choices=["wsi-hint", "abmil", "clam", "dsmil", "meanpool", "transmil", "ced-mil", "patch-sq"], default=None)
    eval_cmd.add_argument("--label-key", default=None)
    eval_cmd.add_argument("--device", default=None)
    eval_cmd.add_argument("--val-fraction", type=float, default=0.1)
    eval_cmd.add_argument("--seed", type=int, default=7)
    eval_cmd.add_argument("--log-path", default=None)
    eval_cmd.set_defaults(func=command_eval)

    heldout = subparsers.add_parser("benchmark-heldout")
    heldout.add_argument("--train-manifest", required=True)
    heldout.add_argument("--test-manifest", required=True)
    heldout.add_argument("--feature-dir", required=True)
    heldout.add_argument("--model", choices=["wsi-hint", "abmil", "clam", "dsmil", "meanpool", "transmil", "ced-mil", "patch-sq"], default="wsi-hint")
    heldout.add_argument("--label-key", default=None)
    heldout.add_argument("--epochs", type=int, default=30)
    heldout.add_argument("--seed", type=int, default=7)
    heldout.add_argument("--device", default=None)
    heldout.add_argument("--log-path", default=None)
    heldout.add_argument("--summary-path", default=None)
    heldout.add_argument("--checkpoint", default=None)
    heldout.add_argument("--inner-val-fraction", type=float, default=0.1)
    heldout.add_argument("--patience", type=int, default=10)
    heldout.add_argument("--min-delta", type=float, default=0.0)
    heldout.add_argument(
        "--early-metric",
        choices=["macro_f1", "auc", "balanced_acc"],
        default="auc",
        help="Inner validation metric for checkpoint selection on the train split.",
    )
    heldout.add_argument("--use-pack", action="store_true", help="Use pack-based collation")
    heldout.add_argument("--save-preds", action="store_true", help="Save per-sample probs for the heldout test split as <log_path>.probs.pt")
    heldout.set_defaults(func=command_benchmark_heldout)

    export_case = subparsers.add_parser("export-case-attention")
    export_case.add_argument("--manifest", required=True)
    export_case.add_argument("--feature-dir", required=True)
    export_case.add_argument("--checkpoint", required=True)
    export_case.add_argument("--output-dir", required=True)
    export_case.add_argument("--model", choices=["abmil", "transmil"], required=True)
    export_case.add_argument("--label-key", default=None)
    export_case.add_argument("--device", default=None)
    export_case.add_argument("--use-plugin-head", action="store_true")
    export_case.add_argument("--slide-ids", nargs="+", default=None)
    export_case.add_argument("--topk", type=int, default=20)
    export_case.add_argument("--max-correct", type=int, default=2)
    export_case.add_argument("--max-errors", type=int, default=2)
    export_case.set_defaults(func=command_export_case_attention)

    kfold = subparsers.add_parser("benchmark-kfold")
    kfold.add_argument("--manifest", required=True)
    kfold.add_argument("--feature-dir", required=True)
    kfold.add_argument("--model", choices=["wsi-hint", "abmil", "clam", "dsmil", "meanpool", "transmil", "ced-mil", "patch-sq"], default="wsi-hint")
    kfold.add_argument("--label-key", default=None)
    kfold.add_argument("--folds", type=int, default=5)
    kfold.add_argument("--epochs", type=int, default=30)
    kfold.add_argument("--seed", type=int, default=7)
    kfold.add_argument("--device", default=None)
    kfold.add_argument("--log-path", default=None)
    kfold.add_argument("--summary-path", default=None)
    kfold.add_argument("--inner-val-fraction", type=float, default=0.1)
    kfold.add_argument("--patience", type=int, default=10)
    kfold.add_argument("--min-delta", type=float, default=0.0)
    kfold.add_argument(
        "--early-metric",
        choices=["macro_f1", "auc", "balanced_acc"],
        default="macro_f1",
        help="Inner validation metric for checkpoint selection (default: macro_f1). "
        "Use auc or balanced_acc when ranking / minority-class behavior matters more.",
    )
    kfold.add_argument("--overwrite", action="store_true")
    kfold.add_argument("--use-pack", action="store_true", help="Use pack-based collation")
    kfold.add_argument(
        "--save-fold-preds", action="store_true",
        help="Save per-sample (probs, labels, slide_ids) for the test split of every fold "
             "as <log_path>.probs_fold{N}.pt; required for DeLong's test and ROC/calibration plots.",
    )
    kfold.set_defaults(func=command_benchmark_kfold)

    protocol = subparsers.add_parser("protocol-abmil-clean")
    protocol.add_argument("--manifest", default=None)
    protocol.add_argument("--feature-dir", default=None)
    protocol.add_argument("--label-key", default=None)
    protocol.add_argument("--folds", type=int, default=5)
    protocol.add_argument("--epochs", type=int, default=50)
    protocol.add_argument("--seed", type=int, default=0)
    protocol.add_argument("--device", default="cuda")
    protocol.add_argument("--inner-val-fraction", type=float, default=0.15)
    protocol.add_argument("--patience", type=int, default=15)
    protocol.add_argument("--min-delta", type=float, default=0.0)
    protocol.add_argument("--use-pack", action="store_true", help="Use pack-based collation")
    protocol.add_argument("--out-prefix", default="artifacts/clean_phikon_abmil")
    protocol.set_defaults(func=command_protocol_abmil_clean)

    cedmil_ablation = subparsers.add_parser("protocol-cedmil-ablation")
    cedmil_ablation.add_argument("--manifest", default=None)
    cedmil_ablation.add_argument("--feature-dir", default=None)
    cedmil_ablation.add_argument("--label-key", default=None)
    cedmil_ablation.add_argument("--folds", type=int, default=5)
    cedmil_ablation.add_argument("--epochs", type=int, default=50)
    cedmil_ablation.add_argument("--seed", type=int, default=0)
    cedmil_ablation.add_argument("--device", default=None)
    cedmil_ablation.add_argument("--inner-val-fraction", type=float, default=0.15)
    cedmil_ablation.add_argument("--patience", type=int, default=15)
    cedmil_ablation.add_argument("--min-delta", type=float, default=0.0)
    cedmil_ablation.add_argument("--use-pack", action="store_true")
    cedmil_ablation.add_argument("--out-prefix", default="artifacts/cedmil_ablation")
    cedmil_ablation.set_defaults(func=command_protocol_cedmil_ablation)

    cedmil_multiseed = subparsers.add_parser("protocol-cedmil-multiseed")
    cedmil_multiseed.add_argument("--manifest", default=None)
    cedmil_multiseed.add_argument("--feature-dir", default=None)
    cedmil_multiseed.add_argument("--label-key", default=None)
    cedmil_multiseed.add_argument("--folds", type=int, default=5)
    cedmil_multiseed.add_argument("--epochs", type=int, default=50)
    cedmil_multiseed.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 42, 123, 256])
    cedmil_multiseed.add_argument("--device", default=None)
    cedmil_multiseed.add_argument("--inner-val-fraction", type=float, default=0.15)
    cedmil_multiseed.add_argument("--patience", type=int, default=15)
    cedmil_multiseed.add_argument("--min-delta", type=float, default=0.0)
    cedmil_multiseed.add_argument("--use-pack", action="store_true")
    cedmil_multiseed.add_argument("--out-prefix", default="artifacts/cedmil_multiseed")
    cedmil_multiseed.add_argument("--summary-path", default="artifacts/cedmil_multiseed_summary.json")
    cedmil_multiseed.add_argument("--save-fold-preds", action="store_true")
    cedmil_multiseed.set_defaults(func=command_protocol_cedmil_multiseed)

    baselines_multiseed = subparsers.add_parser("protocol-baselines-multiseed")
    baselines_multiseed.add_argument("--manifest", default=None)
    baselines_multiseed.add_argument("--feature-dir", default=None)
    baselines_multiseed.add_argument("--label-key", default=None)
    baselines_multiseed.add_argument("--folds", type=int, default=5)
    baselines_multiseed.add_argument("--epochs", type=int, default=50)
    baselines_multiseed.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 42, 123, 256])
    baselines_multiseed.add_argument(
        "--models",
        nargs="+",
        default=["meanpool", "abmil", "clam", "dsmil", "transmil", "wsi-hint", "ced-mil"],
    )
    baselines_multiseed.add_argument("--device", default=None)
    baselines_multiseed.add_argument("--inner-val-fraction", type=float, default=0.15)
    baselines_multiseed.add_argument("--patience", type=int, default=15)
    baselines_multiseed.add_argument("--min-delta", type=float, default=0.0)
    baselines_multiseed.add_argument("--use-pack", action="store_true")
    baselines_multiseed.add_argument("--out-prefix", default="artifacts/baselines_multiseed")
    baselines_multiseed.add_argument("--summary-path", default="artifacts/baselines_multiseed_summary.json")
    baselines_multiseed.add_argument("--instance-dropout", type=float, default=0.15)
    baselines_multiseed.add_argument("--ced-use-cf", action="store_true", default=True)
    baselines_multiseed.add_argument("--save-fold-preds", action="store_true")
    baselines_multiseed.set_defaults(func=command_protocol_baselines_multiseed)

    plugin_ablation = subparsers.add_parser("protocol-plugin-ablation-multiseed")
    plugin_ablation.add_argument("--manifest", default=None)
    plugin_ablation.add_argument("--feature-dir", default=None)
    plugin_ablation.add_argument("--label-key", default=None)
    plugin_ablation.add_argument("--folds", type=int, default=5)
    plugin_ablation.add_argument("--epochs", type=int, default=50)
    plugin_ablation.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 42])
    plugin_ablation.add_argument("--device", default=None)
    plugin_ablation.add_argument("--inner-val-fraction", type=float, default=0.15)
    plugin_ablation.add_argument("--patience", type=int, default=15)
    plugin_ablation.add_argument("--min-delta", type=float, default=0.0)
    plugin_ablation.add_argument("--use-pack", action="store_true")
    plugin_ablation.add_argument("--instance-dropout", type=float, default=0.15)
    plugin_ablation.add_argument("--ced-use-cf", action="store_true", default=True)
    plugin_ablation.add_argument("--out-prefix", default="artifacts/plugin_ablation_multiseed")
    plugin_ablation.add_argument("--summary-path", default="artifacts/plugin_ablation_multiseed_summary.json")
    plugin_ablation.add_argument("--save-fold-preds", action="store_true")
    plugin_ablation.set_defaults(func=command_protocol_plugin_ablation_multiseed)

    heldout_plugin_ablation = subparsers.add_parser("protocol-plugin-ablation-heldout")
    heldout_plugin_ablation.add_argument("--train-manifest", required=True)
    heldout_plugin_ablation.add_argument("--test-manifest", required=True)
    heldout_plugin_ablation.add_argument("--feature-dir", default=None)
    heldout_plugin_ablation.add_argument("--label-key", default=None)
    heldout_plugin_ablation.add_argument("--epochs", type=int, default=50)
    heldout_plugin_ablation.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 42])
    heldout_plugin_ablation.add_argument("--device", default=None)
    heldout_plugin_ablation.add_argument("--inner-val-fraction", type=float, default=0.15)
    heldout_plugin_ablation.add_argument("--patience", type=int, default=15)
    heldout_plugin_ablation.add_argument("--min-delta", type=float, default=0.0)
    heldout_plugin_ablation.add_argument("--use-pack", action="store_true")
    heldout_plugin_ablation.add_argument("--save-checkpoints", action="store_true")
    heldout_plugin_ablation.add_argument("--instance-dropout", type=float, default=0.15)
    heldout_plugin_ablation.add_argument("--ced-use-cf", action="store_true", default=True)
    heldout_plugin_ablation.add_argument("--out-prefix", default="artifacts/plugin_ablation_heldout")
    heldout_plugin_ablation.add_argument("--summary-path", default="artifacts/plugin_ablation_heldout_summary.json")
    heldout_plugin_ablation.add_argument(
        "--save-preds",
        action="store_true",
        help="Save held-out test probabilities to <out-prefix>_<tag>_s<seed>.probs.pt for DeLong / ROC.",
    )
    heldout_plugin_ablation.add_argument(
        "--tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Optional subset of arms: abmil_base abmil_plugin transmil_base transmil_plugin (default: all four).",
    )
    heldout_plugin_ablation.set_defaults(func=command_protocol_plugin_ablation_heldout)

    summarize_ablation = subparsers.add_parser("summarize-multiseed-ablation")
    summarize_ablation.add_argument("--out-prefix", required=True)
    summarize_ablation.add_argument("--seeds", type=int, nargs="+", required=True)
    summarize_ablation.add_argument("--summary-path", default=None)
    summarize_ablation.add_argument("--preset", choices=["plugin-ablation", "cedmil-ablation", "custom"], default="plugin-ablation")
    summarize_ablation.add_argument("--tags", nargs="+", default=None)
    summarize_ablation.set_defaults(func=command_summarize_multiseed_ablation)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
