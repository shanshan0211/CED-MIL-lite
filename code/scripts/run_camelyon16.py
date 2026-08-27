"""Camelyon-16 binary classification benchmark runner (Item 4 of the quick-wins).

The Camelyon-16 corpus (Bejnordi et al., 2017, *JAMA*) is the de-facto external
sanity-check for slide-level MIL. Because the raw WSIs are large (~700 GB) and
must be obtained from the Grand Challenge portal, this script does *not* itself
download the slides — it instead provides a complete, reproducible pipeline
that turns a directory of `.tif` slides into a CED-MIL-lite benchmark report:

    1. ``prep``      — build a manifest JSON from the official train/test split,
                       inferring the binary label `tumor` vs `normal`.
    2. ``features``  — run the existing ``wsi-hint extract-features`` command
                       on every slide using the configured encoder
                       (Phikon-v2 by default, also supports UNI / ResNet50).
    3. ``benchmark`` — launch the canonical 4-arm plug-in ablation
                       (ABMIL/TransMIL × Baseline/Plugin) over multiple seeds,
                       saving per-fold predictions for downstream DeLong/ROC.
    4. ``report``    — aggregate metrics + run statistical tests.

Usage
-----
# 0. Get the slides (manual one-time step):
#    https://camelyon17.grand-challenge.org/Data/   (Camelyon-16 subset)
#    Place tumor + normal under <root>/training and <root>/testing
#    Place reference labels file under <root>/reference.csv

# 1. Build a manifest:
python scripts/run_camelyon16.py prep \
    --slides-root data/camelyon16 \
    --output      artifacts/manifest_camelyon16.json

# 2. Extract Phikon features (will take a while on a single GPU):
python scripts/run_camelyon16.py features \
    --manifest    artifacts/manifest_camelyon16.json \
    --feature-dir artifacts/phikon_features_camelyon16 \
    --encoder     phikon

# 3. Run the 4-arm × 8-seed × 5-fold benchmark with prediction saving:
python scripts/run_camelyon16.py benchmark \
    --manifest    artifacts/manifest_camelyon16.json \
    --feature-dir data/features/camelyon16 \
    --seeds 0 7 42 123 256 314 1024 2024

# 3b. Recommended Camelyon-16 reporting: train on official training slides,
#     evaluate on the official held-out test split (same protocol as the paper's seed=0 numbers):
python scripts/run_camelyon16.py benchmark-heldout

# 4. Run statistical tests + ROC/calibration figures:
python scripts/run_camelyon16.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent

DEFAULT_OUT_PREFIX = "artifacts/camelyon16_plugin_ablation_multiseed"
DEFAULT_SUMMARY_PATH = "artifacts/camelyon16_plugin_ablation_multiseed_summary.json"
# Pre-extracted Phikon-v2 tensors (771-d) used by this workspace; override with --feature-dir.
DEFAULT_FEATURE_DIR = "data/features/camelyon16"
DEFAULT_MANIFEST = "artifacts/manifest_camelyon16.json"
DEFAULT_TRAIN_MANIFEST = "artifacts/manifest_camelyon16_train.json"
DEFAULT_TEST_MANIFEST = "artifacts/manifest_camelyon16_test.json"
DEFAULT_CONFIG = "configs/camelyon16_phikon771.yaml"
DEFAULT_HELDOUT_PREFIX = "artifacts/camelyon16_plugin_ablation_heldout_multiseed"
DEFAULT_HELDOUT_SUMMARY = "artifacts/camelyon16_plugin_ablation_heldout_multiseed_summary.json"


# ---------------------------------------------------------------------------
# 1. Manifest preparation
# ---------------------------------------------------------------------------

def cmd_prep(args: argparse.Namespace) -> None:
    """Build a Camelyon-16 manifest JSON in the wsi-hint format.

    Slide naming convention assumed: ``tumor_NNN.tif`` / ``normal_NNN.tif``
    under ``<root>/training`` and ``<root>/testing``. If a reference CSV is
    supplied, it overrides the file-name heuristic for the test set.
    """
    root = Path(args.slides_root)
    train_dir = root / "training"
    test_dir = root / "testing"
    if not train_dir.exists() or not test_dir.exists():
        sys.exit(
            f"Expected '{train_dir}' and '{test_dir}'. Did you place the official\n"
            f"Camelyon-16 split under {root}? (https://camelyon17.grand-challenge.org/Data/)"
        )

    test_labels: dict[str, str] = {}
    if args.reference_csv:
        with open(args.reference_csv, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                slide_id = row[0].strip()
                label = row[1].strip().lower() if len(row) > 1 else ""
                if "tumor" in label or "macro" in label or "micro" in label or "itc" in label:
                    test_labels[slide_id] = "tumor"
                elif "normal" in label or label == "negative":
                    test_labels[slide_id] = "normal"

    records: list[dict] = []
    for split, slide_dir in [("train", train_dir), ("test", test_dir)]:
        for path in sorted(slide_dir.rglob("*.tif")):
            slide_id = path.stem
            if split == "test" and test_labels:
                label = test_labels.get(slide_id, "normal")
            else:
                label = "tumor" if slide_id.lower().startswith("tumor") else "normal"
            records.append({
                "slide_id": slide_id,
                "patient_id": slide_id,
                "svs_path": str(path),
                "split": split,
                "label": label,
                "diagnosis_bin": 1 if label == "tumor" else 0,
                "project_id": "TCGA-CAMELYON16",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    n_train = sum(1 for r in records if r["split"] == "train")
    n_test = sum(1 for r in records if r["split"] == "test")
    n_pos = sum(1 for r in records if r["diagnosis_bin"] == 1)
    print(f"saved manifest -> {out}")
    print(f"  total slides: {len(records)} (train={n_train}, test={n_test}, tumor={n_pos})")


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool, env: dict | None = None) -> int:
    print("\n+", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(REPO), env=env or os.environ)


def cmd_features(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "wsi_hint.cli",
        "--config", args.config,
        "extract-features",
        "--manifest", args.manifest,
        "--output-dir", args.feature_dir,
        "--encoder", args.encoder,
        "--patch-size", str(args.patch_size),
        "--patch-stride", str(args.patch_stride),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO / "src"))
    rc = _run(cmd, args.dry_run, env=env)
    sys.exit(rc)


# ---------------------------------------------------------------------------
# 3. Benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "wsi_hint.cli",
        "--config", args.config,
        "protocol-plugin-ablation-multiseed",
        "--manifest", args.manifest,
        "--feature-dir", args.feature_dir,
        "--label-key", "diagnosis_bin",
        "--folds", str(args.folds),
        "--epochs", str(args.epochs),
        "--seeds", *[str(s) for s in args.seeds],
        "--inner-val-fraction", str(args.inner_val_fraction),
        "--patience", str(args.patience),
        "--instance-dropout", str(args.instance_dropout),
        "--out-prefix", args.out_prefix,
        "--summary-path", args.summary_path,
        "--ced-use-cf",
    ]
    if args.save_fold_preds:
        cmd.append("--save-fold-preds")
    if args.device:
        cmd.extend(["--device", args.device])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO / "src"))
    rc = _run(cmd, args.dry_run, env=env)
    sys.exit(rc)


def cmd_benchmark_heldout(args: argparse.Namespace) -> None:
    """Official-style Camelyon-16 evaluation: train on training manifest, test on held-out test manifest."""
    cmd = [
        sys.executable, "-m", "wsi_hint.cli",
        "--config", args.config,
        "protocol-plugin-ablation-heldout",
        "--train-manifest", args.train_manifest,
        "--test-manifest", args.test_manifest,
        "--feature-dir", args.feature_dir,
        "--label-key", args.label_key,
        "--epochs", str(args.epochs),
        "--seeds", *[str(s) for s in args.seeds],
        "--inner-val-fraction", str(args.inner_val_fraction),
        "--patience", str(args.patience),
        "--instance-dropout", str(args.instance_dropout),
        "--out-prefix", args.out_prefix,
        "--summary-path", args.summary_path,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.ced_use_cf:
        cmd.append("--ced-use-cf")
    if args.save_preds:
        cmd.append("--save-preds")
    if getattr(args, "filter_tags", None):
        cmd.extend(["--tags", *args.filter_tags])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO / "src"))
    rc = _run(cmd, args.dry_run, env=env)
    sys.exit(rc)


# ---------------------------------------------------------------------------
# 4. Report — aggregate stats + figures
# ---------------------------------------------------------------------------

TAGS = ["abmil_base", "abmil_plugin", "transmil_base", "transmil_plugin"]
PAIRS = [
    "abmil_base vs abmil_plugin",
    "transmil_base vs transmil_plugin",
]


def cmd_report(args: argparse.Namespace) -> None:
    pattern = f"{args.out_prefix}_{{tag}}_s{{seed}}.jsonl"
    seeds = list(args.seeds)

    stats_out = Path("artifacts/camelyon16_stat_tests.json")
    cmd_stats = [
        sys.executable, str(REPO / "scripts" / "stat_tests.py"),
        "report",
        "--condition-glob", pattern,
        "--tags", *TAGS,
        "--seeds", *[str(s) for s in seeds],
        "--pairs", *PAIRS,
        "--metric", args.metric,
        "--n-boot", str(args.n_boot),
        "--n-perm", str(args.n_perm),
        "--output", str(stats_out),
    ]
    rc = _run(cmd_stats, args.dry_run)
    if rc != 0:
        sys.exit(rc)

    if not args.skip_figures:
        for tag in TAGS:
            probs_glob: list[str] = []
            for seed in seeds:
                for fold in range(1, args.folds + 1):
                    p = Path(f"{args.out_prefix}_{tag}_s{seed}.probs_fold{fold}.pt")
                    if p.exists():
                        probs_glob.append(str(p))
            if probs_glob:
                fig = f"artifacts/figures/camelyon16_{tag}_curves.png"
                cmd_fig = [
                    sys.executable, str(REPO / "scripts" / "plot_curves.py"),
                    "--probs-fold", *probs_glob,
                    "--names", tag,
                    "--output", fig,
                ]
                _run(cmd_fig, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="Build a Camelyon-16 manifest from official .tif split.")
    p.add_argument("--slides-root", required=True)
    p.add_argument("--reference-csv", default=None)
    p.add_argument("--output", default=DEFAULT_MANIFEST)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("features", help="Pre-extract patch features.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    p.add_argument("--encoder", default="phikon", choices=["phikon", "uni", "resnet50"])
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patch-stride", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("benchmark", help="Run the 4-arm plug-in ablation × seeds × folds.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42, 123, 256, 314, 1024, 2024])
    p.add_argument("--inner-val-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--instance-dropout", type=float, default=0.15)
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    p.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    p.add_argument("--save-fold-preds", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser(
        "benchmark-heldout",
        help="Train/val on training slides, evaluate on official test split (recommended for Camelyon-16 reporting).",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    p.add_argument("--test-manifest", default=DEFAULT_TEST_MANIFEST)
    p.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    p.add_argument("--label-key", default="label")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42, 123, 256, 314, 1024, 2024])
    p.add_argument("--inner-val-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--instance-dropout", type=float, default=0.15)
    p.add_argument("--out-prefix", default=DEFAULT_HELDOUT_PREFIX)
    p.add_argument("--summary-path", default=DEFAULT_HELDOUT_SUMMARY)
    p.add_argument("--save-preds", action="store_true", help="Write held-out .probs.pt per arm for DeLong / ROC.")
    p.add_argument(
        "--filter-tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Optional arms only, e.g. abmil_base abmil_plugin (passed through as wsi-hint --tags).",
    )
    p.add_argument("--ced-use-cf", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_benchmark_heldout)

    p = sub.add_parser("report", help="Aggregate stats + figures.")
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42, 123, 256, 314, 1024, 2024])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1", "balanced_acc"])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
