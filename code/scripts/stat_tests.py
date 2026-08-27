"""Statistical tests for paired AUC/F1/BAcc comparisons in WSI MIL benchmarks.

Implements three families of tests, all with consistent CLI:
1. **DeLong's test** (Sun & Xu 2014, fast O(n log n) implementation) for
   comparing two correlated AUCs. Operates on per-sample (y_true, y_score)
   loaded from per-fold .probs.pt files written by `wsi-hint benchmark-kfold`.
2. **Bootstrap 95% CI** for any per-fold metric (AUC / F1 / BAcc), with
   stratified resampling at the fold level. Operates directly on the
   `*.jsonl` fold logs, so it works on **all existing artifacts** without
   re-running training.
3. **Paired permutation test** on fold-level deltas (mirrors what is already
   reported in the paper, kept here for one-stop reproducibility).

CLI examples
------------
# Bootstrap CI on existing fold logs (no new training needed):
python scripts/stat_tests.py bootstrap-ci \\
    --jsonl artifacts/canonical_plugin_ablation_mucnos_abmil_plugin_s0.jsonl \\
            artifacts/canonical_plugin_ablation_mucnos_abmil_plugin_s7.jsonl \\
            artifacts/canonical_plugin_ablation_mucnos_abmil_plugin_s42.jsonl \\
    --metric auc --n-boot 5000 --seed 0

# DeLong's test on per-fold saved probs (after running with --save-fold-preds):
python scripts/stat_tests.py delong \\
    --probs-a artifacts/run_abmil_base_s0.probs_fold1.pt \\
              artifacts/run_abmil_base_s0.probs_fold2.pt \\
    --probs-b artifacts/run_abmil_plugin_s0.probs_fold1.pt \\
              artifacts/run_abmil_plugin_s0.probs_fold2.pt

# Aggregate: bootstrap CI for every condition + DeLong wherever possible:
python scripts/stat_tests.py report \\
    --condition-glob "artifacts/canonical_plugin_ablation_*_{tag}_s{seed}.jsonl" \\
    --tags abmil_base abmil_plugin transmil_base transmil_plugin \\
    --seeds 0 7 42 \\
    --output artifacts/stat_tests_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from jsonl_fold_metrics import fold_metric_values


# ---------------------------------------------------------------------------
# Core: DeLong's test (Sun & Xu, 2014; fast variant)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks of `x` (used by DeLong for tied scores)."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j0 = i
        while j0 < n and z[j0] == z[i]:
            j0 += 1
        t[i:j0] = 0.5 * (i + j0 - 1) + 1
        i = j0
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def delong_var(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Per-sample DeLong influence functions for a single classifier.

    Parameters
    ----------
    scores : (n,) array of class-1 probabilities
    labels : (n,) array of {0,1} labels

    Returns
    -------
    auc : float
    v10 : (n_pos,) array of positive-sample influence values
    v01 : (n_neg,) array of negative-sample influence values
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        raise ValueError("Need both positive and negative samples for AUC.")

    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))

    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v10, v01


def delong_test(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray) -> dict:
    """Two-sided DeLong test for two correlated ROC AUCs on the same samples."""
    auc_a, v10_a, v01_a = delong_var(scores_a, labels)
    auc_b, v10_b, v01_b = delong_var(scores_b, labels)
    m = len(v10_a)
    n = len(v01_a)

    s10 = np.cov(np.vstack([v10_a, v10_b]), bias=False) / m
    s01 = np.cov(np.vstack([v01_a, v01_b]), bias=False) / n
    s = s10 + s01

    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = auc_a - auc_b
    if var <= 0:
        z = float("inf") if diff != 0 else 0.0
    else:
        z = diff / math.sqrt(var)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": diff,
        "var_delta": float(var),
        "z": float(z),
        "p_two_sided": float(p),
        "n_pos": int(m),
        "n_neg": int(n),
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Bootstrap CI (works on per-fold aggregate metrics OR per-sample probs)
# ---------------------------------------------------------------------------

def bootstrap_ci_from_values(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean of `values`.

    Useful when only fold-level metrics are available (no per-sample probs).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return {
            "mean": float(arr.mean()) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": n,
            "n_boot": n_boot,
        }
    boot_means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": n,
        "n_boot": n_boot,
    }


def bootstrap_paired_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean of paired differences (a - b)."""
    rng = np.random.default_rng(seed)
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError("Paired arrays must match in length.")
    n = len(arr_a)
    diff = arr_a - arr_b
    if n < 2:
        return {
            "mean_delta": float(diff.mean()) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_pairs": n,
        }
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    return {
        "mean_delta": float(diff.mean()),
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "n_pairs": n,
        "n_boot": n_boot,
    }


def bootstrap_ci_from_probs(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Stratified-bootstrap 95% CI for AUC computed per-sample."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both classes to compute AUC CI.")

    aucs = []
    for _ in range(n_boot):
        p = pos_idx[rng.integers(0, len(pos_idx), size=len(pos_idx))]
        q = neg_idx[rng.integers(0, len(neg_idx), size=len(neg_idx))]
        idx = np.concatenate([p, q])
        try:
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            continue
    aucs = np.asarray(aucs)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ci_low": float(np.quantile(aucs, alpha / 2)),
        "ci_high": float(np.quantile(aucs, 1 - alpha / 2)),
        "n_pos": int(len(pos_idx)),
        "n_neg": int(len(neg_idx)),
        "n_boot": int(len(aucs)),
    }


# ---------------------------------------------------------------------------
# Paired permutation (kept for one-stop reproducibility)
# ---------------------------------------------------------------------------

def paired_permutation(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 20000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b
    n = len(diff)
    if n == 0:
        return {"mean_delta": float("nan"), "p_two_sided": float("nan"), "n_pairs": 0}
    obs = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    return {
        "mean_delta": float(obs),
        "p_two_sided": p,
        "n_pairs": int(n),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Helpers for loading existing artifacts
# ---------------------------------------------------------------------------

def load_jsonl_metric(paths: Iterable[Path], metric: str) -> list[float]:
    """One value per (file, fold); ignores non-fold lines (e.g. soup records)."""
    out: list[float] = []
    for path in paths:
        out.extend(fold_metric_values(Path(path), metric))
    return out


def load_paired_jsonl(
    paths_a: Iterable[Path],
    paths_b: Iterable[Path],
    metric: str,
) -> tuple[list[float], list[float]]:
    """Load aligned per-fold metric values from two sets of jsonl logs.

    Pairs are aligned by file order, then by fold number.
    """
    paths_a = list(paths_a)
    paths_b = list(paths_b)
    if len(paths_a) != len(paths_b):
        raise ValueError("Need the same number of jsonl files in each arm.")
    a_vals: list[float] = []
    b_vals: list[float] = []
    for pa, pb in zip(paths_a, paths_b):
        rows_a = {int(json.loads(l)["fold"]): float(json.loads(l)[metric])
                  for l in Path(pa).read_text(encoding="utf-8").splitlines() if l.strip()}
        rows_b = {int(json.loads(l)["fold"]): float(json.loads(l)[metric])
                  for l in Path(pb).read_text(encoding="utf-8").splitlines() if l.strip()}
        common = sorted(set(rows_a) & set(rows_b))
        for f in common:
            a_vals.append(rows_a[f])
            b_vals.append(rows_b[f])
    return a_vals, b_vals


def load_probs_pt(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-sample (positive-class probs, labels) from .probs*.pt files."""
    import torch

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for p in paths:
        d = torch.load(Path(p), map_location="cpu", weights_only=True)
        probs = d["probs"]
        lab = d["labels"]
        if probs.ndim == 2 and probs.shape[1] >= 2:
            scores.append(probs[:, 1].numpy().astype(float))
        else:
            scores.append(np.asarray(probs).astype(float).reshape(-1))
        labels.append(np.asarray(lab).astype(int).reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_delong(args: argparse.Namespace) -> None:
    sa, la = load_probs_pt([Path(p) for p in args.probs_a])
    sb, lb = load_probs_pt([Path(p) for p in args.probs_b])
    if not np.array_equal(la, lb):
        raise ValueError(
            "Labels in arm A and arm B do not match sample-by-sample. "
            "DeLong's test requires the two classifiers to score the same instances."
        )
    res = delong_test(sa, sb, la)
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_bootstrap_ci(args: argparse.Namespace) -> None:
    values = load_jsonl_metric([Path(p) for p in args.jsonl], args.metric)
    res = bootstrap_ci_from_values(values, n_boot=args.n_boot, seed=args.seed)
    res["metric"] = args.metric
    res["files"] = args.jsonl
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_paired(args: argparse.Namespace) -> None:
    a, b = load_paired_jsonl(
        [Path(p) for p in args.jsonl_a],
        [Path(p) for p in args.jsonl_b],
        args.metric,
    )
    res = {
        "metric": args.metric,
        "n_pairs": len(a),
        "permutation": paired_permutation(a, b, n_perm=args.n_perm, seed=args.seed),
        "bootstrap_paired_delta": bootstrap_paired_delta(
            a, b, n_boot=args.n_boot, seed=args.seed,
        ),
    }
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> None:
    """One-shot: bootstrap CI + paired permutation/bootstrap for every condition.

    Pattern: ``--condition-glob 'artifacts/run_{tag}_s{seed}.jsonl'`` with
    ``--tags`` and ``--seeds`` will fan out to every (tag, seed) jsonl file.
    """
    pattern = args.condition_glob
    payload: dict[str, dict] = {"metric": args.metric, "tags": args.tags, "seeds": args.seeds}

    per_tag_files: dict[str, list[Path]] = {}
    for tag in args.tags:
        files = []
        for seed in args.seeds:
            p = Path(pattern.format(tag=tag, seed=seed))
            if p.exists():
                files.append(p)
        per_tag_files[tag] = files

    payload["bootstrap_ci"] = {}
    for tag, files in per_tag_files.items():
        if not files:
            continue
        values = load_jsonl_metric(files, args.metric)
        payload["bootstrap_ci"][tag] = bootstrap_ci_from_values(
            values, n_boot=args.n_boot, seed=args.seed,
        )

    payload["paired"] = {}
    if len(args.tags) >= 2 and args.pairs:
        for pair in args.pairs:
            tag_a, tag_b = pair.split("vs")
            tag_a, tag_b = tag_a.strip(), tag_b.strip()
            files_a = per_tag_files.get(tag_a, [])
            files_b = per_tag_files.get(tag_b, [])
            if not files_a or not files_b:
                continue
            a, b = load_paired_jsonl(files_a, files_b, args.metric)
            payload["paired"][pair] = {
                "n_pairs": len(a),
                "permutation": paired_permutation(a, b, n_perm=args.n_perm, seed=args.seed),
                "bootstrap_paired_delta": bootstrap_paired_delta(
                    a, b, n_boot=args.n_boot, seed=args.seed,
                ),
            }

    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("delong", help="DeLong's test on two arms of saved per-sample probs.")
    p.add_argument("--probs-a", nargs="+", required=True)
    p.add_argument("--probs-b", nargs="+", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_delong)

    p = sub.add_parser("bootstrap-ci", help="Bootstrap CI for the mean of a per-fold metric.")
    p.add_argument("--jsonl", nargs="+", required=True)
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1", "balanced_acc", "acc"])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_bootstrap_ci)

    p = sub.add_parser("paired", help="Paired permutation + bootstrap on per-fold metric deltas.")
    p.add_argument("--jsonl-a", nargs="+", required=True)
    p.add_argument("--jsonl-b", nargs="+", required=True)
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1", "balanced_acc", "acc"])
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_paired)

    p = sub.add_parser("report", help="Aggregate bootstrap CI + paired tests across conditions.")
    p.add_argument("--condition-glob", required=True,
                   help="Pattern with {tag} and {seed}, e.g. 'artifacts/run_{tag}_s{seed}.jsonl'")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="List like 'abmil_base vs abmil_plugin'")
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1", "balanced_acc", "acc"])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
