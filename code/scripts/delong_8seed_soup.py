"""DeLong tests on model-soup `.probs.pt` for the 8-seed canonical Phikon plugin ablation.

Each `wsi-hint benchmark-kfold` run with `log_path=...jsonl` writes soup predictions to the
sibling file `...probs.pt` (same stem, extension replaced). For every seed, this script loads
plugin vs base soup scores on identical labels and runs Sun–Xu DeLong (same implementation
as `scripts/stat_tests.py`).

Usage
-----
    cd wsi_hint_project
    python scripts/delong_8seed_soup.py
    python scripts/delong_8seed_soup.py --output artifacts/delong_8seed_soup.json

Requires `torch` + `numpy` (already project deps). Optional `--fisher-exploratory` uses Fisher's method across seeds; this assumes
independent p-values and is **not** justified when every seed scores the **same**
cohort (only for curiosity).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from stat_tests import delong_test, load_probs_pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [0, 7, 42, 123, 256, 314, 1024, 2024]
PAIRS: list[tuple[str, str, str]] = [
    ("abmil_plugin", "abmil_base", "ABMIL Plugin vs Base (soup)"),
    ("transmil_plugin", "transmil_base", "TransMIL Plugin vs Base (soup)"),
]


def _fisher_combine(p_values: list[float]) -> dict | None:
    """Fisher's method; returns None if scipy unavailable."""
    try:
        from scipy.stats import chi2  # type: ignore[import-untyped]
    except ImportError:
        return None
    ps = [max(float(p), 1e-300) for p in p_values]
    stat = float(-2.0 * sum(math.log(p) for p in ps))
    df = 2 * len(ps)
    p_comb = float(1.0 - chi2.cdf(stat, df))
    return {"statistic": stat, "df": df, "p_combined": p_comb}


def _summarize_block(per_seed: dict[str, dict]) -> dict:
    deltas: list[float] = []
    ps: list[float] = []
    for _sk, row in per_seed.items():
        if row.get("missing") or row.get("error"):
            continue
        deltas.append(float(row["delta"]))
        ps.append(float(row["p_two_sided"]))
    if not deltas:
        return {}
    arr_d = np.asarray(deltas, dtype=float)
    arr_p = np.asarray(ps, dtype=float)
    return {
        "n_ok": int(len(deltas)),
        "mean_delta_auc": float(arr_d.mean()),
        "n_p_lt_0.05": int((arr_p < 0.05).sum()),
        "n_p_lt_0.01": int((arr_p < 0.01).sum()),
    }


def _run_pair(
    artifacts: Path,
    tag_a: str,
    tag_b: str,
    seeds: list[int],
    *,
    fisher: bool,
) -> dict:
    per_seed: dict[str, dict] = {}
    p_list: list[float] = []
    for s in seeds:
        pa = artifacts / f"canonical_plugin_ablation_phikon_{tag_a}_s{s}.probs.pt"
        pb = artifacts / f"canonical_plugin_ablation_phikon_{tag_b}_s{s}.probs.pt"
        if not pa.exists() or not pb.exists():
            per_seed[str(s)] = {
                "missing": True,
                "path_a": str(pa),
                "path_b": str(pb),
                "exists_a": pa.exists(),
                "exists_b": pb.exists(),
            }
            continue
        sa, la = load_probs_pt([pa])
        sb, lb = load_probs_pt([pb])
        if sa.shape != sb.shape or la.shape != lb.shape:
            per_seed[str(s)] = {
                "error": "shape_mismatch",
                "n_a": int(sa.shape[0]),
                "n_b": int(sb.shape[0]),
            }
            continue
        if not np.array_equal(la, lb):
            per_seed[str(s)] = {"error": "label_mismatch"}
            continue
        res = delong_test(sa, sb, la)
        per_seed[str(s)] = {k: v for k, v in res.items() if k != "var_delta"}
        p_list.append(float(res["p_two_sided"]))

    out: dict = {"tag_a": tag_a, "tag_b": tag_b, "per_seed": per_seed}
    out["summary"] = _summarize_block(per_seed)
    if p_list:
        out["p_values"] = p_list
        if fisher:
            out["fisher_exploratory"] = _fisher_combine(p_list)
            out["fisher_note"] = (
                "Same patients across seeds; Fisher combination is not a valid meta-analysis here."
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--fisher-exploratory",
        action="store_true",
        help="Also emit Fisher combined p (not valid for repeated scoring of the same cohort).",
    )
    args = p.parse_args()

    payload: dict = {
        "description": "DeLong on soup .probs.pt per seed (plugin arm = scores_a in delong_test output)",
        "seeds": args.seeds,
        "pairs": [],
    }
    missing_any = False
    for tag_a, tag_b, label in PAIRS:
        block = _run_pair(
            args.artifacts, tag_a, tag_b, list(args.seeds), fisher=bool(args.fisher_exploratory)
        )
        block["label"] = label
        payload["pairs"].append(block)
        for s in args.seeds:
            entry = block["per_seed"].get(str(s), {})
            if entry.get("missing"):
                missing_any = True

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved -> {args.output}", file=sys.stderr)

    if missing_any:
        print(
            "\nSome `.probs.pt` files are missing. Re-run k-fold with the same `log_path` stem; "
            "soup predictions are written next to each `*.jsonl` when training completes.\n"
            "Example (one seed, one arm — adjust for your machine):\n"
            "  PYTHONPATH=src python -m wsi_hint.cli benchmark-kfold ... "
            "--log-path artifacts/canonical_plugin_ablation_phikon_abmil_base_s0.jsonl\n",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
