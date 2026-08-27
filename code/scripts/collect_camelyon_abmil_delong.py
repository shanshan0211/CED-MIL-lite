"""Aggregate per-seed DeLong tests for ABMIL baseline vs plug-in (Camelyon-16 held-out).

Expects paired .probs.pt files from protocol-plugin-ablation-heldout --save-preds:
  <prefix>_abmil_base_s<seed>.probs.pt
  <prefix>_abmil_plugin_s<seed>.probs.pt

Example:
  python scripts/collect_camelyon_abmil_delong.py \\
      --prefix-seed0 artifacts/camelyon16_heldout_preds_s0 \\
      --prefix-rest artifacts/camelyon16_abmil_probs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prefix-seed0",
        required=True,
        help="Path prefix for seed 0 export (no extension), e.g. artifacts/camelyon16_heldout_preds_s0",
    )
    ap.add_argument(
        "--prefix-rest",
        required=True,
        help="Path prefix for other seeds, e.g. artifacts/camelyon16_abmil_probs",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 7, 42, 123, 256, 314, 1024, 2024],
    )
    ap.add_argument("--output", default="artifacts/camelyon16_abmil_delong_per_seed.json")
    args = ap.parse_args()

    stat_py = repo / "scripts" / "stat_tests.py"
    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def probs_paths(seed: int) -> tuple[Path, Path]:
        root = args.prefix_seed0 if seed == 0 else args.prefix_rest
        base = repo / f"{root}_abmil_base_s{seed}.probs.pt"
        plug = repo / f"{root}_abmil_plugin_s{seed}.probs.pt"
        return base, plug

    rows: list[dict] = []
    for seed in args.seeds:
        base, plug = probs_paths(seed)
        if not base.is_file() or not plug.is_file():
            rows.append(
                {
                    "seed": seed,
                    "error": "missing_probs",
                    "expected_base": str(base.relative_to(repo)),
                    "expected_plugin": str(plug.relative_to(repo)),
                }
            )
            continue
        cmd = [
            sys.executable,
            str(stat_py),
            "delong",
            "--probs-a",
            str(base),
            "--probs-b",
            str(plug),
        ]
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append({"seed": seed, "error": proc.stderr or proc.stdout or "delong failed"})
            continue
        try:
            payload = json.loads(proc.stdout.strip())
        except json.JSONDecodeError as e:
            rows.append({"seed": seed, "error": f"bad_json:{e}", "stdout": proc.stdout[:500]})
            continue
        rows.append({"seed": seed, **payload})

    valid_p = [
        float(r["p_two_sided"])
        for r in rows
        if isinstance(r.get("p_two_sided"), (int, float))
    ]
    summary = {
        "seeds_requested": list(args.seeds),
        "per_seed": rows,
        "abmil_delong_p_two_sided": {
            "values": valid_p,
            "count_below_05": sum(1 for p in valid_p if p < 0.05),
            "n_valid": len(valid_p),
            "median": sorted(valid_p)[len(valid_p) // 2] if valid_p else None,
            "min": min(valid_p) if valid_p else None,
            "max": max(valid_p) if valid_p else None,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["abmil_delong_p_two_sided"], indent=2))
    try:
        print(f"saved -> {out_path.relative_to(repo)}")
    except ValueError:
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
