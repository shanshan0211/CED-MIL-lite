"""Cross-seed and cross-model ensemble evaluation.

Loads per-sample probability files (.probs.pt) from multiple model soup runs,
averages probabilities, and computes ensemble metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


def load_probs(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["probs"], data["labels"]


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = probs.argmax(dim=-1)
    nc = probs.shape[1]
    labels_np = labels.numpy()
    preds_np = preds.numpy()

    macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(labels_np, preds_np)

    if nc == 2:
        auc = roc_auc_score(labels_np, probs[:, 1].numpy())
    else:
        try:
            auc = roc_auc_score(
                labels_np, probs.numpy(), multi_class="ovr", average="macro"
            )
        except ValueError:
            auc = 0.0

    acc = (preds == labels).float().mean().item()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "n": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probs_files", nargs="+", help=".probs.pt files")
    args = parser.parse_args()

    all_probs = []
    ref_labels = None

    print(f"\n{'='*60}")
    print(f"Ensemble Evaluation: {len(args.probs_files)} models")
    print(f"{'='*60}\n")

    for pf in args.probs_files:
        probs, labels = load_probs(pf)
        if ref_labels is None:
            ref_labels = labels
        else:
            assert torch.equal(ref_labels, labels), f"Label mismatch in {pf}"

        metrics = compute_metrics(probs, labels)
        print(
            f"  {Path(pf).stem:40s}  "
            f"AUC={metrics['auc']:.4f}  F1={metrics['macro_f1']:.4f}  "
            f"BalAcc={metrics['balanced_accuracy']:.4f}"
        )
        all_probs.append(probs)

    ens_probs = torch.stack(all_probs).mean(dim=0)
    ens_metrics = compute_metrics(ens_probs, ref_labels)

    print(f"\n{'─'*60}")
    print(
        f"  {'ENSEMBLE (prob avg)':40s}  "
        f"AUC={ens_metrics['auc']:.4f}  F1={ens_metrics['macro_f1']:.4f}  "
        f"BalAcc={ens_metrics['balanced_accuracy']:.4f}"
    )
    print(f"{'─'*60}")

    ens_max_probs = torch.stack(all_probs).max(dim=0).values
    ens_max_probs = ens_max_probs / ens_max_probs.sum(dim=-1, keepdim=True)
    max_metrics = compute_metrics(ens_max_probs, ref_labels)
    print(
        f"  {'ENSEMBLE (max conf)':40s}  "
        f"AUC={max_metrics['auc']:.4f}  F1={max_metrics['macro_f1']:.4f}  "
        f"BalAcc={max_metrics['balanced_accuracy']:.4f}"
    )

    weights = torch.softmax(
        torch.tensor([compute_metrics(p, ref_labels)["auc"] for p in all_probs]) * 5,
        dim=0,
    )
    weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
    w_metrics = compute_metrics(weighted_probs, ref_labels)
    print(
        f"  {'ENSEMBLE (AUC-weighted)':40s}  "
        f"AUC={w_metrics['auc']:.4f}  F1={w_metrics['macro_f1']:.4f}  "
        f"BalAcc={w_metrics['balanced_accuracy']:.4f}"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
