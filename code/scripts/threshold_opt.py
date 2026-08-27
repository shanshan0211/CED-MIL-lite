"""Threshold optimization for ensemble probabilities."""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from collections import Counter

files = [
    "artifacts/ens_abmil_s123.probs.pt",
    "artifacts/ens_abmil_s0.probs.pt",
    "artifacts/ens_abmil_s7.probs.pt",
]

all_probs = []
for f in files:
    data = torch.load(f, map_location="cpu", weights_only=True)
    all_probs.append(data["probs"])
    labels = data["labels"]

aucs = [roc_auc_score(labels.numpy(), p[:, 1].numpy()) for p in all_probs]
weights = torch.softmax(torch.tensor(aucs) * 5, dim=0)
ens_probs = sum(w * p for w, p in zip(weights, all_probs))

pos_probs = ens_probs[:, 1].numpy()
labels_np = labels.numpy()

auc = roc_auc_score(labels_np, pos_probs)
print(f"Ensemble AUC = {auc:.4f}")
print(f"Label distribution: {Counter(labels_np.tolist())}")

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.01):
    preds = (pos_probs >= th).astype(int)
    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    ba = balanced_accuracy_score(labels_np, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

preds_opt = (pos_probs >= best_th).astype(int)
print(f"\nOptimal threshold (F1) = {best_th:.2f}")
print(f"  F1        = {f1_score(labels_np, preds_opt, average='macro', zero_division=0):.4f}")
print(f"  BalAcc    = {balanced_accuracy_score(labels_np, preds_opt):.4f}")
print(f"  Pred dist = {Counter(preds_opt.tolist())}")

best_ba, best_th_ba = 0, 0.5
for th in np.arange(0.05, 0.95, 0.01):
    preds = (pos_probs >= th).astype(int)
    ba = balanced_accuracy_score(labels_np, preds)
    if ba > best_ba:
        best_ba = ba
        best_th_ba = th

preds_ba = (pos_probs >= best_th_ba).astype(int)
print(f"\nOptimal threshold (BalAcc) = {best_th_ba:.2f}")
print(f"  F1        = {f1_score(labels_np, preds_ba, average='macro', zero_division=0):.4f}")
print(f"  BalAcc    = {balanced_accuracy_score(labels_np, preds_ba):.4f}")
print(f"  Pred dist = {Counter(preds_ba.tolist())}")

print("\n--- All thresholds scan ---")
print(f"{'Threshold':>10s} {'F1':>8s} {'BalAcc':>8s}")
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = (pos_probs >= th).astype(int)
    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    ba = balanced_accuracy_score(labels_np, preds)
    print(f"{th:10.2f} {f1:8.4f} {ba:8.4f}")
