from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SlideFeatureBatch:
    features: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    slide_ids: list[str]
    metadata: list[dict[str, Any]]
    coords: torch.Tensor | None


class SlideFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        feature_dir: str | Path,
        label_key: str = "project_code",
        feature_suffix: str = ".pt",
        filter_missing: bool = True,
        load_coords: bool = True,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.feature_dir = Path(feature_dir)
        self.load_coords = load_coords
        all_records: list[dict[str, Any]] = payload["slides"]
        if filter_missing:
            self.records = [
                record
                for record in all_records
                if (self.feature_dir / f"{record['slide_id']}{feature_suffix}").exists()
            ]
        else:
            self.records = all_records
        self.label_key = label_key
        self.feature_suffix = feature_suffix
        labels = [record.get(label_key, "UNK") for record in self.records]
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.records)

    def _feature_path(self, slide_id: str) -> Path:
        return self.feature_dir / f"{slide_id}{self.feature_suffix}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        feature_path = self._feature_path(record["slide_id"])
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature tensor: {feature_path}")
        loaded = torch.load(feature_path, map_location="cpu", weights_only=False)

        coords = None
        if isinstance(loaded, dict) and "features" in loaded:
            features = loaded["features"]
            if self.load_coords and "coords" in loaded:
                raw_coords = loaded["coords"]
                if isinstance(raw_coords, torch.Tensor):
                    coords = raw_coords.float()
                elif isinstance(raw_coords, list):
                    coords = torch.tensor(raw_coords, dtype=torch.float32)
        else:
            features = loaded

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {feature_path}, got {type(features)!r}")
        if features.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {feature_path}, got shape {tuple(features.shape)}")

        label_name = record.get(self.label_key, "UNK")
        item: dict[str, Any] = {
            "slide_id": record["slide_id"],
            "features": features.float(),
            "label": self.label_to_index[label_name],
            "metadata": record,
        }
        if coords is not None and coords.ndim == 2 and coords.shape[0] == features.shape[0]:
            item["coords"] = coords
        return item


def pad_collate(batch: list[dict[str, Any]]) -> SlideFeatureBatch:
    """Standard padding collation: pads to max length in batch."""
    max_tokens = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_tokens, feature_dim, dtype=batch[0]["features"].dtype)
    mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    has_coords = all("coords" in item for item in batch)
    coords = torch.zeros(len(batch), max_tokens, 2, dtype=torch.float32) if has_coords else None

    for batch_index, item in enumerate(batch):
        token_count = item["features"].shape[0]
        features[batch_index, :token_count] = item["features"]
        mask[batch_index, :token_count] = True
        if has_coords and coords is not None:
            coords[batch_index, :token_count] = item["coords"]

    return SlideFeatureBatch(
        features=features, mask=mask, labels=labels,
        slide_ids=slide_ids, metadata=metadata, coords=coords,
    )


def pack_collate(batch: list[dict[str, Any]], pack_length: int = 4096) -> SlideFeatureBatch:
    """Pack-based collation (inspired by Pack-based MIL, 2025).

    Packs multiple slides into fixed-length sequences to maximize GPU
    utilization and avoid information loss from truncation. When a slide
    exceeds pack_length, it is split across packs. Smaller slides fill
    remaining space in existing packs.
    """
    feature_dim = batch[0]["features"].shape[1]

    packs: list[list[torch.Tensor]] = [[]]
    pack_masks: list[list[bool]] = [[]]
    pack_labels: list[int] = [batch[0]["label"]]
    pack_slide_ids: list[str] = [batch[0]["slide_id"]]
    pack_metadata: list[dict[str, Any]] = [batch[0]["metadata"]]
    pack_coords_list: list[list[torch.Tensor]] = [[]]
    current_lengths: list[int] = [0]

    has_coords = all("coords" in item for item in batch)
    sorted_batch = sorted(batch, key=lambda x: x["features"].shape[0], reverse=True)

    for item in sorted_batch:
        feats = item["features"]
        n = feats.shape[0]
        item_coords = item.get("coords")

        placed = False
        for pack_idx in range(len(packs)):
            remaining = pack_length - current_lengths[pack_idx]
            if remaining >= min(n, 1):
                take = min(n, remaining)
                packs[pack_idx].append(feats[:take])
                current_lengths[pack_idx] += take
                if has_coords and item_coords is not None:
                    pack_coords_list[pack_idx].append(item_coords[:take])
                placed = True
                break

        if not placed:
            packs.append([feats[:pack_length]])
            pack_masks.append([])
            pack_labels.append(item["label"])
            pack_slide_ids.append(item["slide_id"])
            pack_metadata.append(item["metadata"])
            pack_coords_list.append([])
            current_lengths.append(min(n, pack_length))
            if has_coords and item_coords is not None:
                pack_coords_list[-1].append(item_coords[:pack_length])

    num_packs = len(packs)
    features_out = torch.zeros(num_packs, pack_length, feature_dim)
    mask_out = torch.zeros(num_packs, pack_length, dtype=torch.bool)
    labels_out = torch.tensor(pack_labels[:num_packs], dtype=torch.long)
    coords_out = torch.zeros(num_packs, pack_length, 2) if has_coords else None

    for i, pack in enumerate(packs):
        if not pack:
            continue
        cat = torch.cat(pack, dim=0)[:pack_length]
        length = cat.shape[0]
        features_out[i, :length] = cat
        mask_out[i, :length] = True
        if has_coords and coords_out is not None and pack_coords_list[i]:
            cat_coords = torch.cat(pack_coords_list[i], dim=0)[:pack_length]
            coords_out[i, :length] = cat_coords

    return SlideFeatureBatch(
        features=features_out,
        mask=mask_out,
        labels=labels_out,
        slide_ids=pack_slide_ids[:num_packs],
        metadata=pack_metadata[:num_packs],
        coords=coords_out,
    )
