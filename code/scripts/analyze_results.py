"""Analyze benchmark results and model sizes."""
import json
import sys
sys.path.insert(0, "src")

from wsi_hint.config import load_config
from wsi_hint.cli import _build_model


def main():
    for model_name in ["abmil", "wsi_hint", "transmil", "meanpool"]:
        path = f"artifacts/real_summary_{model_name}.json"
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except FileNotFoundError:
            print(f"  {model_name}: no summary file")
            continue
        folds = [d for d in data if "fold" in d]
        soup = [d for d in data if d.get("type") == "model_soup"]

        print(f"\n=== {model_name.upper()} Per-Fold ===")
        for f in folds:
            fold = f["fold"]
            f1 = f["macro_f1"]
            auc = f["auc"]
            bacc = f["balanced_acc"]
            ep = f["best_epoch"]
            tr = f["train_size"]
            te = f["test_size"]
            print(f"  Fold {fold}: F1={f1:.4f} AUC={auc:.4f} BAcc={bacc:.4f} | train={tr} test={te} best_ep={ep}")
        if soup:
            s = soup[0]
            print(f"  Soup:    F1={s['macro_f1']:.4f} AUC={s['auc']:.4f} BAcc={s['balanced_acc']:.4f}")

    print("\n=== Model Parameter Counts ===")
    config = load_config("configs/real_data.yaml")
    for name in ["wsi-hint", "abmil", "transmil", "meanpool"]:
        m = _build_model(config, name, 2)
        p = sum(pp.numel() for pp in m.parameters())
        t = sum(pp.numel() for pp in m.parameters() if pp.requires_grad)
        print(f"  {name:12s}: {p:>12,} total  {t:>12,} trainable")


if __name__ == "__main__":
    main()
