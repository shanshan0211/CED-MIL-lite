"""Multi-seed runner that goes from 3 seeds → 8 seeds for the canonical
plug-in ablation experiments (Item 1 of the journal-revision quick-wins).

Wraps the existing `wsi-hint protocol-plugin-ablation-multiseed` CLI so that
it can be launched from a single, easy-to-edit script. By default it uses the
*new* seed set [0, 7, 42, 123, 256, 314, 1024, 2024] and saves per-fold
predictions for downstream DeLong / ROC / calibration analysis.

Usage
-----
# Full canonical 8-seed run on the COAD/READ task (Phikon features):
python scripts/run_multi_seed.py canonical-coad-read

# 8-seed run on the auxiliary MUC-vs-NOS task:
python scripts/run_multi_seed.py canonical-muc-nos

# Custom seed list:
python scripts/run_multi_seed.py canonical-coad-read --seeds 0 7 42 123 256

# Re-run only the *new* seeds and merge with existing 3-seed artifacts:
python scripts/run_multi_seed.py canonical-coad-read --seeds 123 256 314 1024 2024 \
    --out-prefix artifacts/canonical_plugin_ablation_extra
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [0, 7, 42, 123, 256, 314, 1024, 2024]

PRESETS: dict[str, dict[str, str]] = {
    "canonical-coad-read": {
        "config": "configs/ced_mil_phikon_canonical.yaml",
        "manifest": "artifacts/manifest_enriched.json",
        "feature_dir": "artifacts/phikon_features",
        "label_key": "project_id",
        "out_prefix": "artifacts/canonical_plugin_ablation_multiseed",
        "summary_path": "artifacts/canonical_plugin_ablation_multiseed_summary.json",
    },
    "canonical-muc-nos": {
        "config": "configs/ced_mil_phikon_canonical.yaml",
        "manifest": "artifacts/manifest_diagnosis_muc_vs_nos_coad.json",
        "feature_dir": "artifacts/phikon_features",
        "label_key": "diagnosis_bin",
        "out_prefix": "artifacts/canonical_plugin_ablation_mucnos_multiseed",
        "summary_path": "artifacts/canonical_plugin_ablation_mucnos_multiseed_summary.json",
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preset", choices=list(PRESETS), help="Which canonical experiment to run.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--inner-val-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--instance-dropout", type=float, default=0.15)
    p.add_argument("--ced-use-cf", action="store_true", default=True)
    p.add_argument("--out-prefix", default=None,
                   help="Override the preset's --out-prefix (useful for incremental runs).")
    p.add_argument("--summary-path", default=None)
    p.add_argument("--save-fold-preds", action="store_true", default=True,
                   help="Write per-sample probs for every fold (needed for DeLong/ROC).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the underlying command but do not execute it.")
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    out_prefix = args.out_prefix or cfg["out_prefix"]
    summary_path = args.summary_path or cfg["summary_path"]

    cmd = [
        sys.executable, "-m", "wsi_hint.cli",
        "protocol-plugin-ablation-multiseed",
        "--config", cfg["config"],
        "--manifest", cfg["manifest"],
        "--feature-dir", cfg["feature_dir"],
        "--label-key", cfg["label_key"],
        "--folds", str(args.folds),
        "--epochs", str(args.epochs),
        "--seeds", *[str(s) for s in args.seeds],
        "--inner-val-fraction", str(args.inner_val_fraction),
        "--patience", str(args.patience),
        "--instance-dropout", str(args.instance_dropout),
        "--out-prefix", out_prefix,
        "--summary-path", summary_path,
    ]
    if args.ced_use_cf:
        cmd.append("--ced-use-cf")
    if args.save_fold_preds:
        cmd.append("--save-fold-preds")
    if args.device:
        cmd.extend(["--device", args.device])

    print("=" * 78)
    print(f"Running preset: {args.preset}")
    print(f"Seeds         : {args.seeds}")
    print(f"Folds         : {args.folds}    Epochs: {args.epochs}")
    print(f"Out prefix    : {out_prefix}")
    print("Command       :", " ".join(shlex.quote(c) for c in cmd))
    print("=" * 78)

    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO / "src"))
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
