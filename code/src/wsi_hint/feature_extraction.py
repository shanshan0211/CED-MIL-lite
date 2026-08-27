from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .encoders import build_encoder
from .slides import SlideTilingConfig, generate_patch_coordinates, open_slide, read_patch_rgb


@dataclass(slots=True)
class ExtractConfig:
    patch_size: int
    patch_stride: int
    tissue_thumbnail_max_size: int
    tissue_saturation_threshold: float
    tissue_value_threshold: float
    max_patches: int
    seed: int
    batch_size: int
    encoder_name: str
    encoder_path: str | None
    encoder_input_size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    output_dim: int
    device: str
    dtype: str


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def _normalize(images: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    mean_t = images.new_tensor(mean).view(1, 3, 1, 1)
    std_t = images.new_tensor(std).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "float32":
            return torch.float32
        case "float16":
            return torch.float16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


@torch.no_grad()
def extract_slide_features(
    slide_path: str | Path,
    output_path: str | Path,
    config: ExtractConfig,
    encoder: torch.nn.Module | None = None,
) -> dict:
    slide_path = Path(slide_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiling = SlideTilingConfig(
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        tissue_thumbnail_max_size=config.tissue_thumbnail_max_size,
        tissue_saturation_threshold=config.tissue_saturation_threshold,
        tissue_value_threshold=config.tissue_value_threshold,
        max_patches=config.max_patches,
        seed=config.seed,
    )
    coords = generate_patch_coordinates(slide_path, tiling)
    target_device = _resolve_device(config.device)
    out_dtype = _resolve_dtype(config.dtype)
    if encoder is None:
        encoder = build_encoder(config.encoder_name, output_dim=config.output_dim, encoder_path=config.encoder_path).to(target_device)
        encoder.eval()
    use_amp = target_device.type == "cuda"
    features: list[torch.Tensor] = []
    coord_array: list[tuple[int, int]] = []
    batch: list[torch.Tensor] = []
    slide = open_slide(slide_path)
    try:
        for coord in coords:
            patch = read_patch_rgb(slide, coord, config.patch_size)
            if config.encoder_input_size != config.patch_size:
                patch = patch.resize((config.encoder_input_size, config.encoder_input_size), resample=Image.BILINEAR)
            batch.append(_pil_to_tensor(patch))
            coord_array.append((coord.x, coord.y))
            if len(batch) >= config.batch_size:
                images = torch.stack(batch, dim=0).to(target_device)
                images = _normalize(images, config.normalize_mean, config.normalize_std)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out = encoder(images)
                else:
                    out = encoder(images)
                features.append(out.features.float().detach().to("cpu"))
                batch = []
        if batch:
            images = torch.stack(batch, dim=0).to(target_device)
            images = _normalize(images, config.normalize_mean, config.normalize_std)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = encoder(images)
            else:
                out = encoder(images)
            features.append(out.features.float().detach().to("cpu"))
    finally:
        slide.close()
    feature_tensor = torch.cat(features, dim=0).to(dtype=out_dtype)
    payload = {
        "slide_path": str(slide_path),
        "patch_size": config.patch_size,
        "patch_stride": config.patch_stride,
        "max_patches": config.max_patches,
        "feature_dim": int(feature_tensor.shape[1]),
        "patch_count": int(feature_tensor.shape[0]),
    }
    torch.save({"features": feature_tensor, "coords": coord_array, "meta": payload}, output_path)
    return payload


def extract_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: ExtractConfig,
    limit: int | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    import time as _time
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    slides = payload["slides"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_device = _resolve_device(config.device)
    encoder = build_encoder(config.encoder_name, output_dim=config.output_dim, encoder_path=config.encoder_path).to(target_device)
    encoder.eval()
    print(f"Encoder loaded: {config.encoder_name} on {target_device} (output_dim={config.output_dim})")

    results: list[dict] = []
    total = min(len(slides), limit) if limit is not None else len(slides)
    t0 = _time.time()
    for index, record in enumerate(slides):
        if limit is not None and index >= limit:
            break
        slide_id = record["slide_id"]
        slide_path = record["svs_path"]
        out_path = output_dir / f"{slide_id}.pt"
        if skip_existing and out_path.exists():
            loaded = torch.load(out_path, map_location="cpu", weights_only=False)
            pc = loaded["features"].shape[0] if isinstance(loaded, dict) else loaded.shape[0]
            fd = loaded["features"].shape[1] if isinstance(loaded, dict) else loaded.shape[1]
            info = {"slide_path": slide_path, "patch_count": int(pc), "feature_dim": int(fd)}
            results.append({"slide_id": slide_id, **info, "output": str(out_path)})
            print(f"[{index+1}/{total}] SKIP (exists) slide={slide_id} patches={pc}")
            continue
        info = extract_slide_features(slide_path, out_path, config, encoder=encoder)
        result = {"slide_id": slide_id, **info, "output": str(out_path)}
        results.append(result)
        elapsed = _time.time() - t0
        eta = elapsed / (index + 1) * (total - index - 1) if index > 0 else 0
        print(f"[{index+1}/{total}] slide={slide_id} patches={result['patch_count']} dim={result['feature_dim']} eta={eta:.0f}s")
    print(f"Done: {len(results)} slides in {_time.time() - t0:.1f}s")
    return results
