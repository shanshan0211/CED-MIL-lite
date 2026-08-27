"""Test loading local encoder models."""
import torch
import timm
import json

# --- UNI ---
print("=" * 50)
print("Testing UNI (MahmoodLab)")
local_path = "assets/encoders/MahmoodLab_UNI"
with open(f"{local_path}/config.json") as f:
    config = json.load(f)
print(f"Architecture: {config['architecture']}")

model = timm.create_model(
    "vit_large_patch16_224",
    init_values=1.0,
    num_classes=0,
    dynamic_img_size=True,
    global_pool="token",
)
state = torch.load(f"{local_path}/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))
    print(f"Output shape: {out.shape}")
print("UNI loaded successfully!\n")

# --- Virchow2 ---
print("=" * 50)
print("Testing Virchow2 (Paige AI)")
local_path = "assets/encoders/paige_ai_Virchow2"
with open(f"{local_path}/config.json") as f:
    config = json.load(f)
print(f"Architecture: {config['architecture']}")
args = config.get("model_args", {})

model = timm.create_model(
    "vit_huge_patch14_224",
    img_size=args.get("img_size", 224),
    init_values=args.get("init_values", 1e-5),
    num_classes=0,
    reg_tokens=args.get("reg_tokens", 4),
    mlp_ratio=args.get("mlp_ratio", 5.3375),
    global_pool="",
    dynamic_img_size=True,
)
from safetensors.torch import load_file
state = load_file(f"{local_path}/model.safetensors")
model.load_state_dict(state, strict=True)
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))
    print(f"Output shape: {out.shape}")
    if out.ndim == 3:
        cls_token = out[:, 0, :]
        print(f"CLS token dim: {cls_token.shape[-1]}")
print("Virchow2 loaded successfully!")
