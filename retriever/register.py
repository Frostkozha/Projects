"""Source register, eligibility and revocations (Retriever spec v0.2, sections 5, 8, 9).

Eligibility is re-evaluated at request time from UTC dates; an index rebuild is never needed to
enforce expiry or review deadlines. Revocations live outside immutable snapshots and are never
rolled back with an index.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .schema import LIBRARY_IDS, MACHINE_ID_RE, SHA256_RE, LibraryId, SourceStatus

ELIGIBLE_STATUS = "live"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _aware(v: Optional[datetime]) -> Optional[datetime]:
    if v is not None and v.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC")
    return v


class RightsRecord(_M):
    rights_reference: str
    allows_local_indexing: bool
    allows_excerpt_display: bool
    allows_model_training: bool = False
    constraints: str = ""
    reviewer: str
    reviewed_at: datetime
    supporting_document: str
    expires_at: Optional[datetime] = None

    @field_validator("reviewed_at", "expires_at")
    @classmethod
    def _tz(cls, v):
        return _aware(v)


class ContentApproval(_M):
    approver: str
    approved_at: datetime

    @field_validator("approved_at")
    @classmethod
    def _tz(cls, v):
        return _aware(v)


class ParserRecord(_M):
    parser_id: Literal["markdown", "json", "pdf_text"]
    parser_version: str
    normalization_version: str
    parsed_output_approved: bool
    verified_page_labels: bool = False


class SourceRecord(_M):
    source_id: str
    source_version: str
    title: str
    owner: str
    edition: Optional[str] = None
    publication_year: Optional[int] = None
    source_tier: int
    library_id: LibraryId
    course_ids: tuple[str, ...]
    tenant_id: str
    topic_ids: tuple[str, ...]
    language: Literal["en"] = "en"
    file_ref: str
    file_sha256: str
    format: Literal["markdown", "json", "pdf"]
    acquisition_record: str
    rights: Optional[RightsRecord]
    content_approval: Optional[ContentApproval]
    status: SourceStatus
    review_due_at: datetime
    effective_from: datetime
    effective_to: Optional[datetime] = None
    superseded_version: Optional[str] = None
    parser: ParserRecord
    default_section: str = "Document"
    extraction_checks: tuple[str, ...] = ()  # reviewed verbatim strings that must survive extraction
    synthetic_fixture: bool = False

    @field_validator("source_id", "source_version", "tenant_id")
    @classmethod
    def _mid(cls, v):
        if not MACHINE_ID_RE.match(v):
            raise ValueError("invalid machine id")
        return v

    @field_validator("course_ids", "topic_ids", "extraction_checks", mode="before")
    @classmethod
    def _tuple(cls, v):
        return tuple(v) if isinstance(v, list) else v

    @field_validator("review_due_at", "effective_from", "effective_to")
    @classmethod
    def _tz(cls, v):
        return _aware(v)

    @model_validator(mode="after")
    def _checks(self):
        if not SHA256_RE.match(self.file_sha256):
            raise ValueError("file_sha256 must be lowercase sha256 hex")
        if not self.title.strip():
            raise ValueError("title required")
        return self

    def build_problems(self) -> list[str]:
        """Reasons this record cannot be indexed live (quarantine). Empty means indexable."""
        p = []
        if self.rights is None:
            p.append("rights_missing")
        else:
            if not self.rights.allows_local_indexing:
                p.append("rights_no_indexing")
            if not self.rights.allows_excerpt_display:
                p.append("rights_no_excerpt_display")
            if not MACHINE_ID_RE.match(self.rights.rights_reference):
                p.append("rights_reference_invalid")
        if self.content_approval is None:
            p.append("content_approval_missing")
        if self.status != ELIGIBLE_STATUS:
            p.append(f"status_{self.status}")
        if not self.parser.parsed_output_approved:
            p.append("parsed_output_not_approved")
        if not self.topic_ids:
            p.append("no_enabled_topics")
        if not self.course_ids:
            p.append("no_course")
        return p

    def eligible(self, now: datetime, course_id: str, tenant_id: str) -> bool:
        """Runtime eligibility: live, in date, rights valid, matching course/tenant."""
        if self.build_problems():
            return False
        if course_id not in self.course_ids or tenant_id != self.tenant_id:
            return False
        if now >= self.review_due_at:  # treated as under_review immediately
            return False
        if now < self.effective_from or (self.effective_to is not None and now >= self.effective_to):
            return False
        if self.rights.expires_at is not None and now >= self.rights.expires_at:
            return False
        return True


class SourceRegister:
    def __init__(self, records: list[SourceRecord]):
        keys = [(r.source_id, r.source_version) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate source_id/source_version in register")
        self.records = {(r.source_id, r.source_version): r for r in records}

    @classmethod
    def load(cls, path: str | Path) -> "SourceRegister":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([SourceRecord.model_validate(r) for r in data["sources"]])

    def get(self, source_id: str, source_version: str) -> Optional[SourceRecord]:
        return self.records.get((source_id, source_version))


# ----------------------------------------------------------------------------- revocations


class RevocationRegistry:
    """Trusted, versioned, file-backed revocation list. Epoch increases on every change."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._epoch = 0
        self._revoked: dict[str, Optional[set[str]]] = {}  # source_id -> versions (None = all)
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._epoch = int(data["epoch"])
        self._revoked = {}
        for item in data["revoked"]:
            v = item.get("source_version")
            if v is None:
                self._revoked[item["source_id"]] = None
            else:
                cur = self._revoked.setdefault(item["source_id"], set())
                if cur is not None:
                    cur.add(v)

    def _save(self) -> None:
        if self.path is None:
            return
        items = []
        for sid, vers in sorted(self._revoked.items()):
            if vers is None:
                items.append({"source_id": sid, "source_version": None})
            else:
                items.extend({"source_id": sid, "source_version": v} for v in sorted(vers))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".revocations-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"epoch": self._epoch, "revoked": items}, fh, indent=2)
        os.replace(tmp, self.path)

    def refresh(self) -> None:
        if self.path is not None and self.path.exists():
            with self._lock:
                self._load()

    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def revoke(self, source_id: str, source_version: Optional[str] = None) -> int:
        with self._lock:
            if source_version is None:
                self._revoked[source_id] = None
            else:
                cur = self._revoked.setdefault(source_id, set())
                if cur is not None:
                    cur.add(source_version)
            self._epoch += 1
            self._save()
            return self._epoch

    def is_revoked(self, source_id: str, source_version: str) -> bool:
        with self._lock:
            if source_id not in self._revoked:
                return False
            vers = self._revoked[source_id]
            return vers is None or source_version in vers


def known_library(lib: str) -> bool:
    return lib in LIBRARY_IDS
