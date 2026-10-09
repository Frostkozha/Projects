"""Metadata-only retrieval audit (Retriever spec v0.2, section 15).

Records never contain query text, passage text, vectors, names, credentials or exception payloads.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict

from .register import utcnow


class AuditUnavailable(RuntimeError):
    pass


class RetrievalAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    operation: str  # search | item_fetch | evidence
    service_id: Optional[str] = None
    session_ref: Optional[str] = None
    course_id: Optional[str] = None
    kb_version: Optional[str] = None
    index_version: Optional[str] = None
    profile_version: Optional[str] = None
    revocation_epoch: Optional[int] = None
    libraries_searched: tuple[str, ...] = ()
    passage_ids: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    dense_count: Optional[int] = None
    lexical_count: Optional[int] = None
    candidate_count: Optional[int] = None
    embedding_reused: Optional[bool] = None
    stage_ms: dict[str, float] = {}
    status: str
    reason: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: str = ""


class RetrievalAuditSink(Protocol):
    def write(self, record: RetrievalAuditRecord) -> None: ...


class InMemoryRetrievalAudit:
    def __init__(self):
        self.records: list[RetrievalAuditRecord] = []
        self._lock = threading.Lock()

    def write(self, record: RetrievalAuditRecord) -> None:
        with self._lock:
            self.records.append(record.model_copy(update={"timestamp": utcnow().isoformat()}))


class JsonlRetrievalAudit:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: RetrievalAuditRecord) -> None:
        line = record.model_copy(update={"timestamp": utcnow().isoformat()}).model_dump_json()
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            raise AuditUnavailable("audit write failed") from None
