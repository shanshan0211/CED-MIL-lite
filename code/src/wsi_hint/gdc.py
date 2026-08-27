from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class CaseMetadata:
    submitter_id: str
    project_id: str | None
    primary_diagnosis: str | None


def fetch_case_metadata(submitter_ids: list[str]) -> dict[str, CaseMetadata]:
    unique = sorted({s for s in submitter_ids if s})
    if not unique:
        return {}

    fields = [
        "submitter_id",
        "project.project_id",
        "diagnoses.primary_diagnosis",
    ]
    payload: dict[str, Any] = {
        "filters": {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": unique,
            },
        },
        "fields": ",".join(fields),
        "format": "JSON",
        "size": len(unique),
    }
    response = requests.post("https://api.gdc.cancer.gov/cases", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    hits = data.get("data", {}).get("hits", [])
    result: dict[str, CaseMetadata] = {}
    for hit in hits:
        submitter_id = hit.get("submitter_id")
        if not submitter_id:
            continue
        project_id = (hit.get("project") or {}).get("project_id")
        diagnoses = hit.get("diagnoses") or []
        primary_diagnosis = diagnoses[0].get("primary_diagnosis") if diagnoses else None
        result[submitter_id] = CaseMetadata(
            submitter_id=submitter_id,
            project_id=project_id,
            primary_diagnosis=primary_diagnosis,
        )
    return result
