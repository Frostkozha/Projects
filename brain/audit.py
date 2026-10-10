"""Metadata-only Brain audit (Brain plan v0.3, section 13).

Records request_id, status/error, artifact versions, evidence IDs/digest, timings, counts and resource
metrics. Never raw questions, evidence, outputs, reasoning, keys, identity or hidden item keys.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

ALLOWED_FIELDS = {"ts", "event", "request_id", "status", "error_code", "model_version", "runtime_version",
                  "prompt_version", "sampling_profile_version", "evidence_digest", "passage_ids", "metrics",
                  "resources", "detail"}


class InMemoryBrainAudit:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(_clean(record))


class JsonlBrainAudit:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        line = json.dumps(_clean(record), sort_keys=True, ensure_ascii=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _clean(record: dict) -> dict:
    unknown = set(record) - ALLOWED_FIELDS
    if unknown:
        raise ValueError("audit record has non-metadata fields")
    return {"ts": time.time(), **record}


def result_record(result, passage_ids: Optional[tuple[str, ...]], event: str = "brain_result") -> dict:
    return {
        "event": event, "request_id": str(result.request_id), "status": result.status,
        "error_code": result.error_code.value if result.error_code else None,
        "model_version": result.model_version, "runtime_version": result.runtime_version,
        "prompt_version": result.prompt_version, "sampling_profile_version": result.sampling_profile_version,
        "evidence_digest": result.evidence_digest, "passage_ids": list(passage_ids or ()),
        "metrics": result.metrics.model_dump(mode="json"),
    }
