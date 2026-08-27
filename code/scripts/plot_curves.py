"""ROC and Calibration plotting for WSI MIL benchmarks.

Reads per-sample probabilities saved by `wsi-hint benchmark-kfold` (one file
per fold, written by the patched `cli.py` when ``--save-fold-preds`` is set,
or, as a fallback, the slide-level model-soup ``.probs.pt`` produced by the
default kfold runner).

Generates publication-quality figures:
  * ROC curves with per-fold thin lines and a thick mean curve, plus AUC ± std
    in the legend.
  * Reliability diagram (calibration curve) with confidence histogram and ECE.
  * Optional decision-curve analysis (DCA) for clinical contextualization.

CLI
---
python scripts/plot_curves.py \\
    --probs artifacts/run_abmil_base_s0.probs.pt   \\
    --probs artifacts/run_abmil_plugin_s0.probs.pt \\
    --names "ABMIL Baseline" "ABMIL + CED Plugin" \\
    --output artifacts/figures/roc_abmil.png

For per-fold curves, pass multiple ``--probs-fold`` groups separated by ``--``:
python scripts/plot_curves.py \\
    --probs-fold artifacts/run_a_s0.probs_fold1.pt artifacts/run_a_s0.probs_fold2.pt -- \\
    --probs-fold artifacts/run_b_s0.probs_fold1.pt artifacts/run_b_s0.probs_fold2.pt    \\
    --names "ABMIL Baseline" "ABMIL + CED Plugin" \\
    --output artifacts/figures/roc_per_fold.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_probs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (positive-class probability, label) from a single .probs.pt file."""
    import torch

    d = torch.load(path, map_location="cpu", weights_only=True)
    probs = d["probs"]
    labels = d["labels"]
    if hasattr(probs, "numpy"):
        probs = probs.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        probs = probs[:, 1]
    return probs.reshape(-1), labels


def load_probs_group(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Load and concatenate multiple per-fold .probs files; also return per-fold pairs."""
    per_fold = [load_probs(Path(p)) for p in paths]
    cat_p = np.concatenate([s for s, _ in per_fold]) if per_fold else np.empty(0)
    cat_l = np.concatenate([l for _, l in per_fold]) if per_fold else np.empty(0, dtype=int)
    return cat_p, cat_l, per_fold


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Expected Calibration Error (Naeini et al., 2015)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == (y_prob[mask] >= 0.5)).mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(
    arms: list[tuple[str, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]],
    output: Path,
    title: str = "ROC Curves",
) -> dict:
    """Plot ROC for one or more arms.

    `arms` is a list of (name, y_score_concat, y_true_concat, [per-fold (s,l)]).
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    info: dict[str, dict] = {}

    base_grid = np.linspace(0.0, 1.0, 200)
    for i, (name, y_score, y_true, per_fold) in enumerate(arms):
        color = PALETTE[i % len(PALETTE)]

        fold_aucs: list[float] = []
        interp_tprs: list[np.ndarray] = []
        for s, l in per_fold:
            if l.size == 0 or len(np.unique(l)) < 2:
                continue
            fpr, tpr, _ = roc_curve(l, s)
            ax.plot(fpr, tpr, color=color, lw=0.8, alpha=0.25)
            fold_aucs.append(float(sk_auc(fpr, tpr)))
            interp_tprs.append(np.interp(base_grid, fpr, tpr))

        fpr_all, tpr_all, _ = roc_curve(y_true, y_score)
        auc_all = float(sk_auc(fpr_all, tpr_all))

        if interp_tprs:
            mean_tpr = np.mean(interp_tprs, axis=0)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
            std = float(np.std(fold_aucs, ddof=0)) if len(fold_aucs) > 1 else 0.0
            label = f"{name}  (AUC = {auc_all:.3f}, fold = {np.mean(fold_aucs):.3f} ± {std:.3f})"
            ax.plot(base_grid, mean_tpr, color=color, lw=2.0, label=label)
        else:
            ax.plot(fpr_all, tpr_all, color=color, lw=2.0, label=f"{name}  (AUC = {auc_all:.3f})")

        info[name] = {
            "auc_concat": auc_all,
            "fold_aucs": fold_aucs,
            "n": int(y_true.size),
        }

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(output)
    plt.close(fig)
    return info


def plot_calibration(
    arms: list[tuple[str, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]],
    output: Path,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.05})
    ax_top, ax_bot = axes
    info: dict[str, dict] = {}

    ax_top.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="Perfect calibration")

    for i, (name, y_score, y_true, _per_fold) in enumerate(arms):
        color = PALETTE[i % len(PALETTE)]
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
        except ValueError:
            frac_pos, mean_pred = np.array([]), np.array([])
        ece = expected_calibration_error(y_true, y_score, n_bins=n_bins)
        brier = brier_score(y_true, y_score)
        info[name] = {"ece": ece, "brier": brier, "n": int(y_true.size)}

        ax_top.plot(mean_pred, frac_pos, "o-", color=color, lw=1.6, markersize=5,
                    label=f"{name}  (ECE = {ece:.3f}, Brier = {brier:.3f})")
        ax_bot.hist(y_score, bins=np.linspace(0, 1, n_bins + 1), alpha=0.45, color=color)

    ax_top.set_ylabel("Empirical positive rate")
    ax_top.set_title(title)
    ax_top.set_xlim(-0.01, 1.01)
    ax_top.set_ylim(-0.01, 1.01)
    ax_top.legend(loc="upper left", fontsize=8.5)

    ax_bot.set_xlabel("Predicted probability")
    ax_bot.set_ylabel("Count")
    ax_bot.set_xlim(-0.01, 1.01)
    fig.savefig(output)
    plt.close(fig)
    return info


# ---------------------------------------------------------------------------
# Demo data (only used when --demo is passed)
# ---------------------------------------------------------------------------

def _make_demo_arms() -> list[tuple[str, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]:
    rng = np.random.default_rng(0)
    n_pos, n_neg = 40, 55
    arms = []
    for name, sep, miscal in [
        ("ABMIL Baseline", 1.4, 0.10),
        ("ABMIL + CED Plugin", 1.7, 0.04),
        ("TransMIL Baseline", 1.45, 0.07),
        ("TransMIL + CED Plugin", 1.75, 0.03),
    ]:
        per_fold = []
        scores_all = []
        labels_all = []
        for _ in range(5):
            s_pos = rng.normal(loc=0.5 + sep * 0.18, scale=0.18, size=n_pos // 5)
            s_neg = rng.normal(loc=0.5 - sep * 0.18, scale=0.18, size=n_neg // 5)
            scores = np.clip(np.concatenate([s_pos, s_neg]) + miscal * rng.normal(size=n_pos // 5 + n_neg // 5), 0.001, 0.999)
            labels = np.concatenate([np.ones(n_pos // 5, dtype=int), np.zeros(n_neg // 5, dtype=int)])
            per_fold.append((scores, labels))
            scores_all.append(scores)
            labels_all.append(labels)
        arms.append((name, np.concatenate(scores_all), np.concatenate(labels_all), per_fold))
    return arms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _split_groups(values: list[str], delim: str = "--") -> list[list[str]]:
    groups: list[list[str]] = [[]]
    for v in values:
        if v == delim:
            groups.append([])
        else:
            groups[-1].append(v)
    return [g for g in groups if g]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs-fold", nargs="+", action="append", default=[],
                        help="Per-fold probs files for one arm. Repeat for multiple arms.")
    parser.add_argument("--probs", nargs="+", default=[],
                        help="One probs file per arm (alternative to --probs-fold).")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=False, default="artifacts/figures/roc_calibration.png")
    parser.add_argument("--title-roc", default="ROC Curves (TCGA-COAD/READ, 5-fold)")
    parser.add_argument("--title-cal", default="Reliability Diagram (TCGA-COAD/READ, 5-fold)")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--demo", action="store_true",
                        help="Generate template figures from synthetic data (no probs required).")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        arms = _make_demo_arms()
    else:
        arms_raw: list[list[Path]] = []
        if args.probs_fold:
            arms_raw = [[Path(p) for p in g] for g in args.probs_fold]
        elif args.probs:
            arms_raw = [[Path(p)] for p in args.probs]
        else:
            parser.error("Pass --probs, --probs-fold, or --demo.")

        names = args.names or [f"Arm {i+1}" for i in range(len(arms_raw))]
        if len(names) != len(arms_raw):
            parser.error(f"--names ({len(names)}) must match number of arms ({len(arms_raw)}).")

        arms = []
        for name, paths in zip(names, arms_raw):
            cat_p, cat_l, per_fold = load_probs_group(paths)
            arms.append((name, cat_p, cat_l, per_fold))

    roc_path = out_path.with_name(out_path.stem + "_roc.png")
    cal_path = out_path.with_name(out_path.stem + "_calibration.png")
    info_roc = plot_roc(arms, roc_path, title=args.title_roc)
    info_cal = plot_calibration(arms, cal_path, title=args.title_cal, n_bins=args.n_bins)

    summary_path = out_path.with_suffix(".json")
    payload = {
        "roc": info_roc,
        "calibration": info_cal,
        "roc_figure": str(roc_path),
        "calibration_figure": str(cal_path),
        "demo": bool(args.demo),
    }
    summary_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    print(f"saved ROC -> {roc_path}")
    print(f"saved Calibration -> {cal_path}")
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
