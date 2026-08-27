"""Aggregate Table 2 style metrics: mean±std over all fold rows in jsonl (8 seeds x 5 folds = 40)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from jsonl_fold_metrics import fold_metric_values

SEEDS = [0, 7, 42, 123, 256, 314, 1024, 2024]
TAGS = ["abmil_base", "abmil_plugin", "transmil_base", "transmil_plugin"]
MULTISEED_SUMMARY = ROOT / "artifacts" / "canonical_plugin_ablation_phikon_multiseed_summary_8seed.json"


def main() -> None:
    soup_block: dict = {}
    if MULTISEED_SUMMARY.exists():
        soup_block = json.loads(MULTISEED_SUMMARY.read_text(encoding="utf-8"))

    for tag in TAGS:
        all_auc, all_f1, all_bacc = [], [], []
        for s in SEEDS:
            p = ROOT / "artifacts" / f"canonical_plugin_ablation_phikon_{tag}_s{s}.jsonl"
            if not p.exists():
                print(f"MISSING {p}")
                continue
            all_auc.extend(fold_metric_values(p, "auc"))
            all_f1.extend(fold_metric_values(p, "macro_f1"))
            all_bacc.extend(fold_metric_values(p, "balanced_acc"))

        soup_auc, soup_f1, soup_bacc = [], [], []
        arm = soup_block.get(tag) if soup_block else None
        if arm:
            soup_auc = arm.get("soup_auc", {}).get("values") or []
            soup_f1 = arm.get("soup_f1", {}).get("values") or []
            soup_bacc = arm.get("soup_bacc", {}).get("values") or []

        def fmt(xs: list[float]) -> str:
            if not xs:
                return "n/a"
            m = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return f"{m:.4f} ± {sd:.4f}"

        print(f"{tag:16s}  Avg {fmt(all_auc)}  F1 {fmt(all_f1)}  BAcc {fmt(all_bacc)}  |  Soup AUC {fmt(soup_auc)}  F1 {fmt(soup_f1)}  BAcc {fmt(soup_bacc)}")


if __name__ == "__main__":
    main()
