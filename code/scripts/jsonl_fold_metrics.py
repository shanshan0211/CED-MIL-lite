"""Helpers for reading per-fold metrics from benchmark *.jsonl logs.

Some logs append the same fold twice (e.g. resume); downstream stats should
use one value per fold (last line wins).
"""

from __future__ import annotations

import json
from pathlib import Path


def fold_metric_values(path: Path, metric: str) -> list[float]:
    """Return metric values for folds 1..K in order, one value per fold index."""
    by_fold: dict[int, float] = {}
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "fold" not in row or metric not in row:
            continue
        by_fold[int(row["fold"])] = float(row[metric])
    return [by_fold[k] for k in sorted(by_fold)]
