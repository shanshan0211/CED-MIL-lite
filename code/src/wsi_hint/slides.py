from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    s = np.where(cmax == 0, 0.0, delta / (cmax + 1e-8))
    v = cmax
    h = np.zeros_like(v)
    mask = delta > 1e-8
    idx = (cmax == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = (cmax == g) & mask
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = (cmax == b) & mask
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _open_slide(path: str | Path):
    from openslide import OpenSlide

    return OpenSlide(str(path))


@dataclass(slots=True)
class SlideTilingConfig:
    patch_size: int
    patch_stride: int
    tissue_thumbnail_max_size: int
    tissue_saturation_threshold: float
    tissue_value_threshold: float
    max_patches: int
    seed: int


@dataclass(slots=True)
class PatchCoordinate:
    x: int
    y: int


def build_tissue_mask(
    slide_path: str | Path,
    thumbnail_max_size: int,
    saturation_threshold: float,
    value_threshold: float,
) -> tuple[np.ndarray, float]:
    slide = _open_slide(slide_path)
    width, height = slide.dimensions
    scale = min(thumbnail_max_size / max(width, 1), thumbnail_max_size / max(height, 1), 1.0)
    thumb_w = max(1, int(round(width * scale)))
    thumb_h = max(1, int(round(height * scale)))
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    rgb = np.asarray(thumbnail)
    hsv = _rgb_to_hsv(rgb)
    s = hsv[..., 1]
    v = hsv[..., 2]
    tissue = (s >= float(saturation_threshold)) & (v <= float(value_threshold))
    return tissue, scale


def generate_patch_coordinates(
    slide_path: str | Path,
    config: SlideTilingConfig,
) -> list[PatchCoordinate]:
    slide = _open_slide(slide_path)
    width, height = slide.dimensions
    tissue_mask, scale = build_tissue_mask(
        slide_path,
        thumbnail_max_size=config.tissue_thumbnail_max_size,
        saturation_threshold=config.tissue_saturation_threshold,
        value_threshold=config.tissue_value_threshold,
    )
    stride = config.patch_stride
    patch_size = config.patch_size
    grid_x = list(range(0, max(width - patch_size + 1, 1), stride))
    grid_y = list(range(0, max(height - patch_size + 1, 1), stride))
    coords: list[PatchCoordinate] = []
    for y in grid_y:
        ty = min(int(round(y * scale)), tissue_mask.shape[0] - 1)
        for x in grid_x:
            tx = min(int(round(x * scale)), tissue_mask.shape[1] - 1)
            if bool(tissue_mask[ty, tx]):
                coords.append(PatchCoordinate(x=x, y=y))
    if len(coords) <= config.max_patches:
        return coords
    rng = np.random.default_rng(config.seed)
    selected = rng.choice(len(coords), size=config.max_patches, replace=False)
    return [coords[int(i)] for i in selected]


def open_slide(slide_path: str | Path):
    return _open_slide(slide_path)


def read_patch_rgb(slide, coord: PatchCoordinate, patch_size: int) -> Image.Image:
    patch = slide.read_region((coord.x, coord.y), 0, (patch_size, patch_size)).convert("RGB")
    return patch
