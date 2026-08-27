from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(slots=True)
class EncoderOutput:
    features: torch.Tensor


class SimpleCNNEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        feats = self.stem(images)
        pooled = self.pool(feats).flatten(1)
        return EncoderOutput(features=self.proj(pooled))


class TorchScriptEncoder(nn.Module):
    def __init__(self, script_path: str | Path) -> None:
        super().__init__()
        self.script_path = str(script_path)
        self.module = torch.jit.load(self.script_path, map_location="cpu")

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        output = self.module(images)
        if isinstance(output, torch.Tensor):
            features = output
        elif isinstance(output, dict) and "features" in output:
            features = output["features"]
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            features = output[0]
        else:
            raise TypeError(f"Unsupported TorchScript output type: {type(output)!r}")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor features, got {type(features)!r}")
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={tuple(features.shape)}")
        return EncoderOutput(features=features)


class HFViTCLSEncoder(nn.Module):
    def __init__(self, model_name_or_path: str | Path) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise RuntimeError("Missing dependency: transformers") from exc
        self.model_name_or_path = str(model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        outputs = self.model(pixel_values=images)
        if not hasattr(outputs, "last_hidden_state"):
            raise TypeError("Expected outputs.last_hidden_state from transformers model")
        hidden = outputs.last_hidden_state
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError(f"Expected last_hidden_state as 3D tensor, got {type(hidden)!r} shape={getattr(hidden, 'shape', None)}")
        features = hidden[:, 0, :]
        return EncoderOutput(features=features)


class TimmHfHubEncoder(nn.Module):
    def __init__(self, repo_id: str, output_dim: int) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("Missing dependency: timm") from exc
        self.repo_id = repo_id
        self.output_dim = output_dim
        self.model = timm.create_model(
            f"hf-hub:{repo_id}",
            pretrained=True,
            num_classes=0,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        output = self.model(images)
        if isinstance(output, torch.Tensor):
            features = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            features = output[0]
        else:
            raise TypeError(f"Unsupported timm output type: {type(output)!r}")
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={tuple(features.shape)}")
        if features.size(1) != self.output_dim:
            raise ValueError(f"Expected feature dim {self.output_dim}, got {features.size(1)}")
        return EncoderOutput(features=features)


class TimmGenericEncoder(nn.Module):
    """Uses any timm model by name (e.g. resnet50, vit_base_patch16_224)."""

    def __init__(self, model_name: str, output_dim: int | None = None) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("Missing dependency: timm") from exc
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            raw_dim = self.model(dummy).shape[1]
        self.raw_dim = raw_dim
        self.proj: nn.Module | None = None
        if output_dim is not None and output_dim != raw_dim:
            self.proj = nn.Linear(raw_dim, output_dim)
            self.out_dim = output_dim
        else:
            self.out_dim = raw_dim

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        features = self.model(images)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={tuple(features.shape)}")
        if self.proj is not None:
            features = self.proj(features)
        return EncoderOutput(features=features)


class TorchVisionEncoder(nn.Module):
    """Uses torchvision pretrained models (downloads from PyTorch CDN)."""

    def __init__(self, model_name: str = "resnet50", output_dim: int | None = None) -> None:
        super().__init__()
        import torchvision.models as tv_models
        weights_map = {
            "resnet50": ("resnet50", tv_models.ResNet50_Weights.DEFAULT, 2048),
            "resnet18": ("resnet18", tv_models.ResNet18_Weights.DEFAULT, 512),
            "resnet101": ("resnet101", tv_models.ResNet101_Weights.DEFAULT, 2048),
            "convnext_tiny": ("convnext_tiny", tv_models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
            "convnext_small": ("convnext_small", tv_models.ConvNeXt_Small_Weights.DEFAULT, 768),
            "efficientnet_v2_s": ("efficientnet_v2_s", tv_models.EfficientNet_V2_S_Weights.DEFAULT, 1280),
        }
        if model_name not in weights_map:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(weights_map)}")
        fn_name, weights, raw_dim = weights_map[model_name]
        backbone = getattr(tv_models, fn_name)(weights=weights)
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, "head"):
            backbone.head = nn.Identity()
        self.backbone = backbone
        self.raw_dim = raw_dim
        self.proj: nn.Module | None = None
        if output_dim is not None and output_dim != raw_dim:
            self.proj = nn.Linear(raw_dim, output_dim)
            self.out_dim = output_dim
        else:
            self.out_dim = raw_dim

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        features = self.backbone(images)
        if features.ndim != 2:
            features = features.flatten(1)
        if self.proj is not None:
            features = self.proj(features)
        return EncoderOutput(features=features)


class LocalUNIEncoder(nn.Module):
    """Loads UNI (MahmoodLab) from a local directory containing pytorch_model.bin."""

    def __init__(self, local_dir: str | Path) -> None:
        super().__init__()
        import timm
        import json
        local_dir = Path(local_dir)
        self.model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1.0,
            num_classes=0,
            dynamic_img_size=True,
            global_pool="token",
        )
        weights_path = local_dir / "pytorch_model.bin"
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        features = self.model(images)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={tuple(features.shape)}")
        return EncoderOutput(features=features)


_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "encoders"


def build_encoder(name: str, output_dim: int, *, encoder_path: str | Path | None = None) -> nn.Module:
    match name:
        case "uni":
            local_dir = _ASSETS_DIR / "MahmoodLab_UNI"
            if local_dir.exists() and (local_dir / "pytorch_model.bin").exists():
                return LocalUNIEncoder(local_dir=local_dir)
            return TimmHfHubEncoder(repo_id="MahmoodLab/UNI", output_dim=output_dim)
        case "phikon-v2":
            local_dir = _ASSETS_DIR / "owkin_phikon_v2"
            if local_dir.exists() and (local_dir / "model.safetensors").exists():
                return HFViTCLSEncoder(model_name_or_path=str(local_dir))
            if encoder_path is None:
                raise ValueError("encoder_path is required for encoder=phikon-v2 (local dir or HF id)")
            return HFViTCLSEncoder(model_name_or_path=encoder_path)
        case "torchscript":
            if encoder_path is None:
                raise ValueError("encoder_path is required for encoder=torchscript")
            return TorchScriptEncoder(script_path=encoder_path)
        case "simple-cnn":
            return SimpleCNNEncoder(output_dim=output_dim)
        case _ if name.startswith("timm:"):
            timm_model_name = name[5:]
            return TimmGenericEncoder(model_name=timm_model_name, output_dim=output_dim)
        case _ if name.startswith("tv:"):
            tv_model_name = name[3:]
            return TorchVisionEncoder(model_name=tv_model_name, output_dim=output_dim)
        case _:
            raise ValueError(f"Unknown encoder: {name}. Use 'timm:<name>' or 'tv:<name>'")
