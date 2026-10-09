"""Canonical retrieval contract (Retriever spec v0.2, sections 6-7, 14).

These types are the single source of truth for the gate/orchestrator boundary: the gate package
imports them instead of keeping its own dictionaries.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator

SCHEMA_VERSION = "retrieval-result-0.2"
RETRIEVAL_POLICY_VERSION = "retrieval-policy-0.2"
LIBRARY_IDS = ("lib1", "lib2", "lib3", "lib4", "lib5", "lib6")
MAX_PASSAGES = 5
MACHINE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

LibraryId = Literal["lib1", "lib2", "lib3", "lib4", "lib5", "lib6"]
SourceStatus = Literal["live", "archived", "blocked", "pending_review", "under_review"]


class Status(str, Enum):
    ok = "ok"
    no_evidence = "no_evidence"
    error = "error"


class Reason(str, Enum):
    QUALIFYING_PASSAGES = "QUALIFYING_PASSAGES"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    NO_MATCHES = "NO_MATCHES"
    NO_ELIGIBLE_PASSAGES = "NO_ELIGIBLE_PASSAGES"
    APPROVED_ITEM_EVIDENCE = "APPROVED_ITEM_EVIDENCE"
    FAILURE = "FAILURE"


NO_EVIDENCE_REASONS = {Reason.BELOW_THRESHOLD, Reason.NO_MATCHES, Reason.NO_ELIGIBLE_PASSAGES}


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    INVALID_SCOPE = "INVALID_SCOPE"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    FORBIDDEN = "FORBIDDEN"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    SOURCE_STATE_CHANGED = "SOURCE_STATE_CHANGED"
    INVALID_EMBEDDING_REF = "INVALID_EMBEDDING_REF"
    INDEX_UNAVAILABLE = "INDEX_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    SEARCH_FAILED = "SEARCH_FAILED"
    RERANK_FAILED = "RERANK_FAILED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    ITEM_EVIDENCE_UNAVAILABLE = "ITEM_EVIDENCE_UNAVAILABLE"
    CONFLICT_EVIDENCE_INCOMPLETE = "CONFLICT_EVIDENCE_INCOMPLETE"
    AUDIT_UNAVAILABLE = "AUDIT_UNAVAILABLE"


HTTP_STATUS = {
    ErrorCode.INVALID_REQUEST: 422, ErrorCode.INVALID_SCOPE: 422, ErrorCode.INVALID_EMBEDDING_REF: 422,
    ErrorCode.INPUT_TOO_LONG: 413, ErrorCode.UNAUTHORIZED: 401, ErrorCode.FORBIDDEN: 403,
    ErrorCode.SCOPE_MISMATCH: 409, ErrorCode.VERSION_MISMATCH: 409, ErrorCode.SOURCE_STATE_CHANGED: 409,
    ErrorCode.CAPACITY_EXCEEDED: 429,
}


def http_status_for(code: ErrorCode) -> int:
    return HTTP_STATUS.get(code, 503)


class RetrievalError(Exception):
    """Internal control flow for a typed error result. Carries a code only, never text."""

    def __init__(self, code: ErrorCode):
        super().__init__(code.value)
        self.code = code


def _machine_id(v: str) -> str:
    if not isinstance(v, str) or not MACHINE_ID_RE.match(v):
        raise ValueError("invalid machine id")
    return v


def _uuid(v: str) -> str:
    try:
        parsed = uuid.UUID(v)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("request_id must be a UUID") from exc
    if str(parsed) != v.lower():
        raise ValueError("request_id must be a canonical UUID")
    return str(parsed)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# ----------------------------------------------------------------------------- requests


class RetrievalRequest(_Strict):
    """Exactly the spec's fields. Library list validity is checked by the service (typed errors)."""

    request_id: StrictStr
    query_text: StrictStr
    allowed_libraries: tuple[StrictStr, ...]
    preferred_libraries: tuple[StrictStr, ...]
    course_id: StrictStr
    kb_version: StrictStr
    strategy: Literal["all_active"]

    @field_validator("allowed_libraries", "preferred_libraries", mode="before")
    @classmethod
    def _list_to_tuple(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("request_id")
    @classmethod
    def _rid(cls, v):
        return _uuid(v)

    @field_validator("course_id", "kb_version")
    @classmethod
    def _ids(cls, v):
        return _machine_id(v)

    @field_validator("query_text")
    @classmethod
    def _query(cls, v):
        if not v.strip():
            raise ValueError("query_text must be nonempty")
        return v


class ItemFetchRequest(_Strict):
    request_id: StrictStr
    item_id: StrictStr
    item_version: StrictStr
    course_id: StrictStr
    kb_version: StrictStr

    @field_validator("request_id")
    @classmethod
    def _rid(cls, v):
        return _uuid(v)

    @field_validator("item_id", "item_version", "course_id", "kb_version")
    @classmethod
    def _ids(cls, v):
        return _machine_id(v)


@dataclass(frozen=True)
class EmbeddingRef:
    """Internal reusable query vector. Never serialized, logged or exposed."""

    vector: np.ndarray
    encoder_id: str
    encoder_revision: Optional[str]
    tokenizer_revision: Optional[str]
    pooling: str
    dimension: int
    dtype: str
    normalization: str
    text_sha256: str
    preprocessing_fingerprint: str

    def __repr__(self) -> str:
        return "EmbeddingRef(<redacted>)"


@dataclass(frozen=True)
class RetrievalContext:
    """Trusted, server-resolved context. Never accepted from a request body."""

    service_id: str
    tenant_id: str
    course_id: str
    authorized_libraries: frozenset[str]
    registry_version: str
    gate_permitted: bool
    embedding_ref: Optional[EmbeddingRef] = None
    session_ref: Optional[str] = None


# ----------------------------------------------------------------------------- output


class Locator(_Strict):
    kind: Literal["pdf_page", "slide", "section", "caption", "table"]
    start: Optional[int]
    end: Optional[int]
    label: StrictStr
    anchor: Optional[StrictStr]

    @model_validator(mode="after")
    def _rules(self):
        if not self.label.strip():
            raise ValueError("locator label required")
        if self.kind in ("pdf_page", "slide"):
            if self.start is None or self.end is None or self.start < 1 or self.end < self.start:
                raise ValueError("pdf_page/slide need positive 1-based start <= end")
        else:
            if not self.anchor or not self.anchor.strip():
                raise ValueError("section/caption/table need a stable anchor")
            if (self.start is None) != (self.end is None):
                raise ValueError("page range must be complete or absent")
            if self.start is not None and (self.start < 1 or self.end < self.start):
                raise ValueError("invalid enclosing page range")
        return self


class EvidencePassage(_Strict):
    passage_id: StrictStr
    source_id: StrictStr
    source_version: StrictStr
    title: StrictStr
    library_id: LibraryId
    locator: Locator
    section_path: tuple[StrictStr, ...]
    edition: Optional[StrictStr]
    publication_year: Optional[int]
    text: StrictStr
    text_sha256: StrictStr
    review_status: SourceStatus
    rights_reference: StrictStr
    relevance_score: Optional[float]
    score_type: Optional[Literal["cross_encoder_logit"]]
    evidence_uri: StrictStr

    @field_validator("section_path", mode="before")
    @classmethod
    def _sp(cls, v):
        return tuple(v) if isinstance(v, list) else v

    @model_validator(mode="after")
    def _rules(self):
        for f in (self.passage_id, self.source_id, self.source_version, self.rights_reference):
            _machine_id(f)
        if not self.title.strip() or not self.text:
            raise ValueError("title and text required")
        if (self.relevance_score is None) != (self.score_type is None):
            raise ValueError("score and score_type go together")
        if self.relevance_score is not None and not math.isfinite(self.relevance_score):
            raise ValueError("scores must be finite")
        return self


class ConflictOut(_Strict):
    conflict_id: StrictStr
    topic_id: StrictStr
    passage_ids: tuple[StrictStr, ...]
    description: StrictStr
    policy_reference: StrictStr


class RetrievalResult(_Strict):
    schema_version: Literal["retrieval-result-0.2"] = SCHEMA_VERSION
    request_id: StrictStr
    status: Status
    reason: Reason
    error_code: Optional[ErrorCode]
    passages: tuple[EvidencePassage, ...]
    coverage: Literal["full", "partial", "unknown"]
    conflicts: tuple[ConflictOut, ...]
    libraries_searched: tuple[StrictStr, ...]
    kb_version: Optional[StrictStr]
    retrieval_policy_version: StrictStr = RETRIEVAL_POLICY_VERSION
    index_version: Optional[StrictStr]
    profile_version: Optional[StrictStr]
    revocation_epoch: Optional[int]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for k in ("status", "reason", "error_code"):
                v = data.get(k)
                if isinstance(v, str):
                    data[k] = {"status": Status, "reason": Reason, "error_code": ErrorCode}[k](v)
            for k in ("passages", "conflicts", "libraries_searched"):
                if isinstance(data.get(k), list):
                    data[k] = tuple(data[k])
        return data

    @model_validator(mode="after")
    def _invariants(self):
        ids = [p.passage_id for p in self.passages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate passage ids")
        if self.status == Status.ok:
            if not 1 <= len(self.passages) <= MAX_PASSAGES or self.error_code is not None:
                raise ValueError("ok needs 1-5 passages and no error code")
            if self.reason not in (Reason.QUALIFYING_PASSAGES, Reason.APPROVED_ITEM_EVIDENCE):
                raise ValueError("invalid ok reason")
            if self.reason == Reason.APPROVED_ITEM_EVIDENCE:
                if any(p.relevance_score is not None for p in self.passages) or self.libraries_searched:
                    raise ValueError("item fetch carries no ranking scores or searched libraries")
            elif self.coverage != "unknown":
                raise ValueError("free-text search coverage is unknown")
        else:
            if self.passages or self.conflicts or self.coverage != "unknown":
                raise ValueError("non-ok results carry no evidence")
            if self.status == Status.no_evidence:
                if self.error_code is not None or self.reason not in NO_EVIDENCE_REASONS:
                    raise ValueError("no_evidence needs a specific reason and no error code")
            elif self.error_code is None or self.reason != Reason.FAILURE:
                raise ValueError("error needs FAILURE and an error code")
        return self

    @classmethod
    def error(cls, request_id: str, code: ErrorCode, **versions) -> "RetrievalResult":
        return cls(request_id=request_id, status=Status.error, reason=Reason.FAILURE, error_code=code, passages=(),
                   coverage="unknown", conflicts=(), libraries_searched=(),
                   kb_version=versions.get("kb_version"), index_version=versions.get("index_version"),
                   profile_version=versions.get("profile_version"),
                   revocation_epoch=versions.get("revocation_epoch"))

    @classmethod
    def no_evidence(cls, request_id: str, reason: Reason, libraries_searched=(), **versions) -> "RetrievalResult":
        return cls(request_id=request_id, status=Status.no_evidence, reason=reason, error_code=None, passages=(),
                   coverage="unknown", conflicts=(), libraries_searched=tuple(libraries_searched),
                   kb_version=versions.get("kb_version"), index_version=versions.get("index_version"),
                   profile_version=versions.get("profile_version"),
                   revocation_epoch=versions.get("revocation_epoch"))
