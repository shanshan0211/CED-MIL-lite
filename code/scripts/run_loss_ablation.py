r"""Loss-term Leave-One-Out (LOO) ablation runner (Item 3 of the quick-wins).

CED-MIL-lite combines several auxiliary losses on top of the main classifier
loss \(L_{cls}\):

  * \(L_{cf}\)        — counterfactual evidence ablation         (cf weight)
  * \(L_{align}\)     — class-prototype / shared-evidence align    (align)
  * \(L_{sep}\)       — role-separation margin                     (sep)
  * \(L_{residual}\)  — residual-only auxiliary supervision        (residual)
  * \(L_{balance}\)   — role-share balancing regularization        (balance)

This script generates one CED-MIL run per LOO variant by zeroing the
corresponding loss coefficient in `configs/ced_mil_phikon_canonical.yaml`
(via temporary YAML overrides), then launches the canonical 5-fold benchmark
with prediction saving so DeLong's test can be applied afterwards.

Usage
-----
# Run all 5 LOO variants on COAD/READ for seeds 0,7,42:
python scripts/run_loss_ablation.py --seeds 0 7 42

# Specific subset (e.g. only L_cf and L_sep):
python scripts/run_loss_ablation.py --terms cf sep --seeds 0

# Dry-run to inspect the generated commands:
python scripts/run_loss_ablation.py --seeds 0 --dry-run

# After all runs finish, summarise:
python scripts/run_loss_ablation.py --summarize-only \
    --out-prefix artifacts/loss_loo_ablation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import yaml

REPO = Path(__file__).resolve().parent.parent

LOSS_TERMS = {
    "full":     {},  # baseline = full CED-MIL
    "no_cf":       {"ced_lambda_cf": 0.0},
    "no_align":    {"ced_lambda_align": 0.0},
    "no_sep":      {"ced_lambda_sep": 0.0},
    "no_residual": {"ced_lambda_residual": 0.0},
    "no_balance":  {"ced_lambda_balance": 0.0},
}


def _patch_yaml(base_yaml: Path, model_overrides: dict, out_yaml: Path) -> None:
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    model = cfg.setdefault("model", {})
    for k, v in model_overrides.items():
        model[k] = v
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_one_variant(
    *,
    base_config: Path,
    overrides: dict,
    manifest: str,
    feature_dir: str,
    label_key: str,
    out_prefix: str,
    tag: str,
    seed: int,
    folds: int,
    epochs: int,
    device: str | None,
    inner_val_fraction: float,
    patience: int,
    save_fold_preds: bool,
    dry_run: bool,
) -> int:
    log_path = f"{out_prefix}_{tag}_s{seed}.jsonl"
    summary_path = f"{out_prefix}_{tag}_s{seed}_summary.json"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _patch_yaml(base_config, overrides, tmp_path)

        cmd = [
            sys.executable, "-m", "wsi_hint.cli",
            "--config", str(tmp_path),
            "benchmark-kfold",
            "--manifest", manifest,
            "--feature-dir", feature_dir,
            "--model", "ced-mil",
            "--label-key", label_key,
            "--folds", str(folds),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--early-metric", "auc",
            "--inner-val-fraction", str(inner_val_fraction),
            "--patience", str(patience),
            "--log-path", log_path,
            "--summary-path", summary_path,
            "--overwrite",
        ]
        if device:
            cmd.extend(["--device", device])
        if save_fold_preds:
            cmd.append("--save-fold-preds")

        print("\n" + "-" * 78)
        print(f">>> tag={tag}  seed={seed}  overrides={overrides or 'baseline'}")
        print("Command:", " ".join(shlex.quote(c) for c in cmd))
        print("-" * 78)
        if dry_run:
            return 0

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(REPO / "src"))
        return subprocess.call(cmd, cwd=str(REPO), env=env)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _summarize(out_prefix: Path, terms: Sequence[str], seeds: Sequence[int]) -> dict:
    """Aggregate per-fold AUC / F1 / BAcc across seeds for each LOO variant."""
    payload: dict[str, dict] = {"terms": list(terms), "seeds": list(seeds), "results": {}}
    for term in terms:
        per_seed_means = {"auc": [], "macro_f1": [], "balanced_acc": []}
        per_fold_values = {"auc": [], "macro_f1": [], "balanced_acc": []}
        for seed in seeds:
            jsonl = out_prefix.with_name(f"{out_prefix.name}_{term}_s{seed}.jsonl")
            if not jsonl.exists():
                continue
            fold_metrics = {"auc": [], "macro_f1": [], "balanced_acc": []}
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "fold" not in row:
                    continue
                for k in fold_metrics:
                    if k in row:
                        fold_metrics[k].append(float(row[k]))
            for k, vals in fold_metrics.items():
                if vals:
                    per_seed_means[k].append(mean(vals))
                    per_fold_values[k].extend(vals)
        agg = {}
        for k in per_seed_means:
            arr = per_seed_means[k]
            if arr:
                agg[k] = {
                    "seed_mean": mean(arr),
                    "seed_std": stdev(arr) if len(arr) > 1 else 0.0,
                    "fold_mean": mean(per_fold_values[k]),
                    "fold_std": stdev(per_fold_values[k]) if len(per_fold_values[k]) > 1 else 0.0,
                    "n_seeds": len(arr),
                    "n_folds": len(per_fold_values[k]),
                }
        payload["results"][term] = agg

    if "full" in payload["results"]:
        full = payload["results"]["full"]
        payload["delta_vs_full"] = {}
        for term in terms:
            if term == "full" or term not in payload["results"]:
                continue
            d = {}
            for k in ["auc", "macro_f1", "balanced_acc"]:
                if k in full and k in payload["results"][term]:
                    d[k] = payload["results"][term][k]["seed_mean"] - full[k]["seed_mean"]
            payload["delta_vs_full"][term] = d

    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/ced_mil_phikon_canonical.yaml")
    p.add_argument("--manifest", default="artifacts/manifest_enriched.json")
    p.add_argument("--feature-dir", default="artifacts/phikon_features")
    p.add_argument("--label-key", default="project_id")
    p.add_argument("--out-prefix", default="artifacts/loss_loo_ablation")
    p.add_argument("--terms", nargs="+", default=list(LOSS_TERMS),
                   choices=list(LOSS_TERMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--inner-val-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--save-fold-preds", action="store_true", default=True)
    p.add_argument("--summary-path", default=None)
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip training; only aggregate existing artifacts.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for seed in args.seeds:
            print("\n" + "#" * 78)
            print(f"### Loss LOO ablation: seed={seed}")
            print("#" * 78)
            for term in args.terms:
                rc = _run_one_variant(
                    base_config=Path(args.config),
                    overrides=LOSS_TERMS[term],
                    manifest=args.manifest,
                    feature_dir=args.feature_dir,
                    label_key=args.label_key,
                    out_prefix=str(out_prefix),
                    tag=term,
                    seed=seed,
                    folds=args.folds,
                    epochs=args.epochs,
                    device=args.device,
                    inner_val_fraction=args.inner_val_fraction,
                    patience=args.patience,
                    save_fold_preds=args.save_fold_preds,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    print(f"!!! tag={term} seed={seed} exited with rc={rc}; continuing.")

    payload = _summarize(out_prefix, args.terms, args.seeds)
    summary_path = Path(args.summary_path or f"{args.out_prefix}_summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {summary_path}")

    print("\n=== Loss LOO Summary (seed-averaged 5-fold mean) ===")
    rows = []
    for term in args.terms:
        if term not in payload["results"]:
            continue
        m = payload["results"][term]
        rows.append((
            term,
            m.get("auc", {}).get("seed_mean"),
            m.get("auc", {}).get("seed_std"),
            m.get("macro_f1", {}).get("seed_mean"),
            m.get("balanced_acc", {}).get("seed_mean"),
        ))
    if rows:
        print(f"{'tag':14s} {'AUC':>8s} {'±std':>7s} {'F1':>8s} {'BAcc':>8s}")
        for r in rows:
            tag = r[0]
            auc = "n/a" if r[1] is None else f"{r[1]:.4f}"
            std = "n/a" if r[2] is None else f"{r[2]:.4f}"
            f1 = "n/a" if r[3] is None else f"{r[3]:.4f}"
            bacc = "n/a" if r[4] is None else f"{r[4]:.4f}"
            print(f"{tag:14s} {auc:>8s} {std:>7s} {f1:>8s} {bacc:>8s}")


if __name__ == "__main__":
    main()
