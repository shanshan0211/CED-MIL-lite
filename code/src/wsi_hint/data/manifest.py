from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SlideRecord:
    slide_id: str
    slide_name: str
    case_id: str
    patient_id: str
    sample_id: str
    project_code: str
    svs_path: str
    annotation_path: str | None
    annotation_count: int
    annotation_categories: list[str]


def _parse_tcga_filename(file_name: str) -> dict[str, str]:
    stem = Path(file_name).stem
    parts = stem.split(".")[0].split("-")
    project_code = parts[1] if len(parts) > 1 else "UNK"
    patient_id = "-".join(parts[:3]) if len(parts) >= 3 else stem
    sample_id = "-".join(parts[:4]) if len(parts) >= 4 else patient_id
    return {
        "case_id": stem,
        "patient_id": patient_id,
        "sample_id": sample_id,
        "project_code": project_code,
    }


def _read_annotation_summary(path: Path | None) -> tuple[int, list[str]]:
    if path is None or not path.exists():
        return 0, []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    categories = sorted({row.get("category", "").strip() for row in rows if row.get("category", "").strip()})
    return len(rows), categories


def build_manifest(dataset_root: str | Path) -> list[SlideRecord]:
    root = Path(dataset_root)
    records: list[SlideRecord] = []
    for svs_path in sorted(root.rglob("*.svs")):
        slide_folder = svs_path.parent
        annotation_path = slide_folder / "annotations.txt"
        parsed = _parse_tcga_filename(svs_path.name)
        annotation_count, annotation_categories = _read_annotation_summary(annotation_path if annotation_path.exists() else None)
        records.append(
            SlideRecord(
                slide_id=slide_folder.name,
                slide_name=svs_path.name,
                case_id=parsed["case_id"],
                patient_id=parsed["patient_id"],
                sample_id=parsed["sample_id"],
                project_code=parsed["project_code"],
                svs_path=str(svs_path),
                annotation_path=str(annotation_path) if annotation_path.exists() else None,
                annotation_count=annotation_count,
                annotation_categories=annotation_categories,
            )
        )
    return records


def write_manifest(dataset_root: str | Path, output_path: str | Path) -> list[SlideRecord]:
    records = build_manifest(dataset_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_root": str(Path(dataset_root)),
        "slide_count": len(records),
        "slides": [asdict(record) for record in records],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
