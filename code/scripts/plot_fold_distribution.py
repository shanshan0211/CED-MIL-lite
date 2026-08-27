"""Per-fold metric distribution plots (companion to plot_curves.py).

Whereas `plot_curves.py` requires per-sample probabilities (only available
after running with ``--save-fold-preds``), this script renders the variance
of fold-level metrics that are *already* logged in every ``*.jsonl`` file
produced by ``wsi-hint benchmark-kfold``. It therefore turns the existing
3-seed × 4-arm × 5-fold artifacts into a publication-quality figure
immediately, without re-training.

Outputs
-------
A two-panel figure (one panel per metric, default AUC + macro-F1) where each
condition is rendered as a half-violin + box + scatter overlay, with a
horizontal bar marking the bootstrap-mean and 95% CI.

Usage
-----
python scripts/plot_fold_distribution.py \
    --condition-glob "artifacts/canonical_plugin_ablation_phikon_{tag}_s{seed}.jsonl" \
    --tags abmil_base abmil_plugin transmil_base transmil_plugin \
    --labels "ABMIL Baseline" "ABMIL + CED Plugin" "TransMIL Baseline" "TransMIL + CED Plugin" \
    --seeds 0 7 42 \
    --metrics auc macro_f1 \
    --output artifacts/figures/fold_distribution_main.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from jsonl_fold_metrics import fold_metric_values
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.45,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = ["#5e8db8", "#c75a4f", "#6aa364", "#9472a8", "#d9933b"]
METRIC_LABELS = {
    "auc": "AUC",
    "macro_f1": "Macro-F1",
    "balanced_acc": "Balanced Accuracy",
    "acc": "Accuracy",
}


def _load_metric(jsonl_paths: Sequence[Path], metric: str) -> list[float]:
    out: list[float] = []
    for p in jsonl_paths:
        out.extend(fold_metric_values(Path(p), metric))
    return out


def _bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return float(arr.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def _half_violin(ax: plt.Axes, values: list[float], position: float, color: str,
                 width: float = 0.45) -> None:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or arr.std() < 1e-9:
        return
    parts = ax.violinplot(
        [arr], positions=[position - 0.04], widths=width, showmeans=False,
        showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        path = body.get_paths()[0]
        verts = path.vertices
        verts[:, 0] = np.clip(verts[:, 0], -np.inf, position - 0.04)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.32)


def plot_distribution(
    condition_glob: str,
    tags: list[str],
    labels: list[str],
    seeds: list[int],
    metrics: list[str],
    output: Path,
    title: str | None = None,
) -> dict:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 5.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    summary: dict[str, dict] = {}

    for ax, metric in zip(axes, metrics):
        positions = list(range(len(tags)))
        for i, tag in enumerate(tags):
            files = []
            for seed in seeds:
                p = Path(condition_glob.format(tag=tag, seed=seed))
                if p.exists():
                    files.append(p)
            values = _load_metric(files, metric)
            if not values:
                continue
            color = PALETTE[i % len(PALETTE)]
            mean, lo, hi = _bootstrap_ci(values)
            summary.setdefault(metric, {})[tag] = {
                "mean": mean, "ci_low": lo, "ci_high": hi,
                "n": len(values), "values": values,
            }
            _half_violin(ax, values, position=i, color=color)
            ax.boxplot(
                [values], positions=[i + 0.04], widths=0.20,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
                medianprops=dict(color="white", lw=1.4),
                whiskerprops=dict(color=color, lw=1.0),
                capprops=dict(color=color, lw=1.0),
                flierprops=dict(marker="x", markersize=3, markeredgecolor=color, alpha=0.6),
            )
            jitter = (np.random.default_rng(7 * (i + 1)).uniform(-0.06, 0.06, size=len(values)))
            ax.scatter(np.full_like(values, i + 0.04, dtype=float) + jitter, values,
                       color=color, edgecolor="white", lw=0.4, s=24, alpha=0.85, zorder=3)
            if not np.isnan(lo):
                ax.errorbar([i + 0.34], [mean], yerr=[[mean - lo], [hi - mean]],
                            fmt="o", color="black", ecolor="black",
                            elinewidth=1.1, capsize=3, markersize=4, zorder=4)
                ax.text(i + 0.42, mean,
                        f"{mean:.3f}\n[{lo:.3f}, {hi:.3f}]",
                        fontsize=8, va="center", ha="left", color="black",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_xlim(-0.6, len(tags) + 0.4)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

    payload = {
        "metrics": metrics,
        "tags": tags,
        "labels": labels,
        "seeds": seeds,
        "title": title,
        "results": {
            m: {tag: {k: v for k, v in d.items() if k != "values"}
                for tag, d in summary[m].items()}
            for m in summary
        },
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition-glob", required=True,
                   help="Pattern with {tag} and {seed}.")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--metrics", nargs="+", default=["auc", "macro_f1"],
                   choices=list(METRIC_LABELS))
    p.add_argument("--title", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if len(args.tags) != len(args.labels):
        raise SystemExit("--tags and --labels must have the same length.")

    plot_distribution(
        condition_glob=args.condition_glob,
        tags=args.tags,
        labels=args.labels,
        seeds=args.seeds,
        metrics=args.metrics,
        output=Path(args.output),
        title=args.title,
    )
    print(f"saved -> {args.output}")
    print(f"saved -> {Path(args.output).with_suffix('.json')}")


if __name__ == "__main__":
    main()
