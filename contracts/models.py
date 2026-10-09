"""Canonical Brain-draft and verification contracts (Verifier spec v0.2, section 4).

Shared by the Brain adapter, verifier and orchestrator. Strict scalar types, forbidden extra fields,
explicit enums and cross-field invariants. The old free-text ``draft_text`` contract does not exist here:
free text with response-wide citations cannot bind evidence to individual statements.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, field_validator, model_validator

ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SENTENCES = 16
MAX_SENTENCE_CHARS = 600
MAX_TOTAL_CHARS = 6000
MAX_CITES = 5
MAX_DEPENDENCIES = 15

DRAFT_SCHEMA = "brain-draft-0.2"
REQUEST_SCHEMA = "verify-request-0.2"
RESULT_SCHEMA = "verification-result-0.2"


class ContractError(ValueError):
    """Malformed transport (duplicate keys, NaN/Infinity, invalid JSON)."""


def _reject_constant(name):
    raise ContractError(f"non-finite JSON constant {name}")


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ContractError("duplicate JSON key")
    return dict(pairs)


def strict_json_loads(data: bytes | str):
    """Parse JSON rejecting duplicate keys and NaN/Infinity before any model validation."""
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ContractError("not utf-8") from None
    try:
        return json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except ContractError:
        raise
    except ValueError:
        raise ContractError("invalid JSON") from None


def check_id(v: str) -> str:
    if not isinstance(v, str) or not ID_RE.match(v):
        raise ValueError("invalid id")
    return v


def check_uuid(v: str) -> str:
    try:
        parsed = uuid.UUID(v)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("request_id must be a UUID") from None
    if str(parsed) != v.lower():
        raise ValueError("request_id must be a canonical UUID")
    return str(parsed)


def check_sha(v: str) -> str:
    if not isinstance(v, str) or not SHA256_RE.match(v):
        raise ValueError("hash must be 64 lowercase hex characters")
    return v


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _tuple(v):
    return tuple(v) if isinstance(v, list) else v


# ----------------------------------------------------------------------------- enums


class ResponseCode(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"


class OperationalError(str, Enum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROFILE_MISMATCH = "PROFILE_MISMATCH"
    REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
    EVIDENCE_INTEGRITY = "EVIDENCE_INTEGRITY"
    EVIDENCE_LIMIT = "EVIDENCE_LIMIT"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    QUEUE_FULL = "QUEUE_FULL"
    WORKER_FAILED = "WORKER_FAILED"
    AUDIT_UNAVAILABLE = "AUDIT_UNAVAILABLE"
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"


class ContentReason(str, Enum):
    INVALID_DRAFT = "INVALID_DRAFT"
    MISSING_CITATION = "MISSING_CITATION"
    UNKNOWN_CITATION = "UNKNOWN_CITATION"
    IRRELEVANT_CITATION = "IRRELEVANT_CITATION"
    SOURCE_INELIGIBLE = "SOURCE_INELIGIBLE"
    HARD_FACT_MISMATCH = "HARD_FACT_MISMATCH"
    HARD_FACT_UNRESOLVED = "HARD_FACT_UNRESOLVED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNCERTAIN_SUPPORT = "UNCERTAIN_SUPPORT"
    UNSAFE_TRIM = "UNSAFE_TRIM"
    NO_VISIBLE_SUPPORT = "NO_VISIBLE_SUPPORT"
    HIDDEN_ANSWER_FAILED = "HIDDEN_ANSWER_FAILED"
    CONFLICT_INCOMPLETE = "CONFLICT_INCOMPLETE"
    COVERAGE_PARTIAL = "COVERAGE_PARTIAL"
    OUTPUT_POLICY_UNCERTAIN = "OUTPUT_POLICY_UNCERTAIN"
    POLICY_VIOLATION = "POLICY_VIOLATION"


PolicyCategory = Literal["real_person_advice", "privacy_disclosure", "assessed_work", "instruction_override",
                         "internal_content"]
CheckState = Literal["pass", "fail", "not_run"]


# ----------------------------------------------------------------------------- draft


class DraftSentence(_Strict):
    sentence_id: StrictStr
    text: StrictStr
    kind_hint: Literal["factual", "question", "transition", "advice"]  # untrusted model suggestion
    visibility: Literal["student", "internal"]
    cites: tuple[StrictStr, ...]
    depends_on: tuple[StrictStr, ...]

    @field_validator("cites", "depends_on", mode="before")
    @classmethod
    def _t(cls, v):
        return _tuple(v)

    @field_validator("sentence_id")
    @classmethod
    def _sid(cls, v):
        return check_id(v)

    @field_validator("text")
    @classmethod
    def _text(cls, v):
        if not v.strip() or len(v) > MAX_SENTENCE_CHARS:
            raise ValueError("sentence text must be nonempty and at most 600 code points")
        return v

    @model_validator(mode="after")
    def _lists(self):
        for c in self.cites + self.depends_on:
            check_id(c)
        if len(set(self.cites)) != len(self.cites) or len(self.cites) > MAX_CITES:
            raise ValueError("cites must be unique and at most 5")
        if len(set(self.depends_on)) != len(self.depends_on) or len(self.depends_on) > MAX_DEPENDENCIES:
            raise ValueError("depends_on must be unique and at most 15")
        return self


class DraftAnswer(_Strict):
    schema_version: Literal["brain-draft-0.2"]
    status: Literal["draft", "no_evidence"]
    sentences: tuple[DraftSentence, ...]
    used_passage_ids: tuple[StrictStr, ...]

    @field_validator("sentences", "used_passage_ids", mode="before")
    @classmethod
    def _t(cls, v):
        return _tuple(v)

    @model_validator(mode="after")
    def _invariants(self):
        if self.status == "no_evidence":
            if self.sentences or self.used_passage_ids:
                raise ValueError("no_evidence requires empty sentences and used_passage_ids")
            return self
        if not 1 <= len(self.sentences) <= MAX_SENTENCES:
            raise ValueError("a draft has 1 to 16 sentences")
        if sum(len(s.text) for s in self.sentences) > MAX_TOTAL_CHARS:
            raise ValueError("draft exceeds 6,000 code points")
        ids = [s.sentence_id for s in self.sentences]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sentence ids")
        seen: set[str] = set()
        for s in self.sentences:
            for d in s.depends_on:
                if d == s.sentence_id or d not in seen:
                    raise ValueError("dependencies must refer to earlier sentences")
            seen.add(s.sentence_id)
        union = tuple(sorted({c for s in self.sentences for c in s.cites}))
        if tuple(self.used_passage_ids) != union:
            raise ValueError("used_passage_ids must equal the sorted union of cites")
        if not any(s.visibility == "student" for s in self.sentences):
            raise ValueError("a draft needs at least one student-visible sentence")
        return self


class VerifyRequest(_Strict):
    schema_version: Literal["verify-request-0.2"]
    request_id: StrictStr
    draft: DraftAnswer
    prompt_version: StrictStr

    @field_validator("request_id")
    @classmethod
    def _rid(cls, v):
        return check_uuid(v)

    @field_validator("prompt_version")
    @classmethod
    def _pv(cls, v):
        return check_id(v)


# ----------------------------------------------------------------------------- result


class NLIScore(_Strict):
    contradiction: float
    entailment: float
    neutral: float

    @model_validator(mode="after")
    def _finite(self):
        for v in (self.contradiction, self.entailment, self.neutral):
            if not math.isfinite(v) or not 0.0 <= v <= 1.0:
                raise ValueError("scores must be finite and in [0,1]")
        return self


class SentenceChecks(_Strict):
    format: CheckState
    citation: CheckState
    hard_fact: CheckState
    entailment: CheckState


class SentenceResult(_Strict):
    sentence_id: StrictStr
    decision: Literal["keep", "delete"]
    reason_codes: tuple[ContentReason, ...]
    factual_checked: StrictBool
    checks: SentenceChecks
    nli_scores: dict[str, NLIScore]

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _rc(cls, v):
        return tuple(ContentReason(x) if isinstance(x, str) else x for x in v)


class InvalidCitation(_Strict):
    sentence_id: StrictStr
    passage_id: Optional[StrictStr]
    reason: ContentReason


class VerificationResult(_Strict):
    schema_version: Literal["verification-result-0.2"] = RESULT_SCHEMA
    request_id: StrictStr
    verifier_version: StrictStr
    thresholds_version: StrictStr
    evidence_digest: StrictStr
    draft_digest: StrictStr
    status: Literal["approved", "rejected", "error"]
    disposition: Literal["pass", "trimmed", "fallback", "refuse", "escalate", "unavailable"]
    response_code: Optional[ResponseCode]
    reply_key: Optional[StrictStr]
    error_code: Optional[OperationalError]
    verified_text: StrictStr
    final_sentence_ids: tuple[StrictStr, ...]
    internal_sentence_ids: tuple[StrictStr, ...]
    citation_map: dict[str, tuple[StrictStr, ...]]
    sentence_results: tuple[SentenceResult, ...]
    internal_sentences: tuple[DraftSentence, ...]
    supported_claims: StrictInt
    unsupported_claims: StrictInt
    invalid_citations: tuple[InvalidCitation, ...]
    partial_support: StrictBool
    source_conflict: StrictBool
    output_policy_violations: tuple[PolicyCategory, ...]
    near_miss: StrictBool
    decision_digest: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for k in ("response_code", "error_code"):
                v = data.get(k)
                if isinstance(v, str):
                    data[k] = (ResponseCode if k == "response_code" else OperationalError)(v)
            for k in ("final_sentence_ids", "internal_sentence_ids", "sentence_results", "internal_sentences",
                      "invalid_citations", "output_policy_violations"):
                if isinstance(data.get(k), list):
                    data[k] = tuple(data[k])
            if isinstance(data.get("citation_map"), dict):
                data["citation_map"] = {k: tuple(v) for k, v in data["citation_map"].items()}
        return data

    @model_validator(mode="after")
    def _invariants(self):
        check_uuid(self.request_id)
        for h in (self.evidence_digest, self.draft_digest, self.decision_digest):
            check_sha(h)
        if self.supported_claims < 0 or self.unsupported_claims < 0:
            raise ValueError("claim counts are nonnegative")
        empty_payload = (not self.verified_text and not self.final_sentence_ids and not self.internal_sentence_ids
                         and not self.citation_map and not self.internal_sentences)
        if self.status == "approved":
            if self.disposition not in ("pass", "trimmed") or self.response_code not in (
                    ResponseCode.A1, ResponseCode.A2, ResponseCode.A4):
                raise ValueError("approved requires pass/trimmed and A1/A2/A4")
            if not self.verified_text or not self.final_sentence_ids:
                raise ValueError("approved requires verified text and visible ids")
            if self.reply_key is not None or self.error_code is not None or self.output_policy_violations:
                raise ValueError("approved carries no reply key, error code or policy violation")
            if not set(self.citation_map) <= set(self.final_sentence_ids):
                raise ValueError("citation_map keys must be kept student sentences")
            if [s.sentence_id for s in self.internal_sentences] != list(self.internal_sentence_ids):
                raise ValueError("internal payload must match internal_sentence_ids")
        elif self.status == "rejected":
            if self.disposition not in ("fallback", "refuse", "escalate") or self.response_code not in (
                    ResponseCode.A5, ResponseCode.A6, ResponseCode.A7):
                raise ValueError("rejected requires fallback/refuse/escalate and A5/A6/A7")
            if not self.reply_key or self.error_code is not None or not empty_payload:
                raise ValueError("rejected requires a fixed reply key, no error code and an empty payload")
        else:
            if self.disposition != "unavailable" or self.response_code is not None or self.error_code is None:
                raise ValueError("error requires unavailable, null response code and an error code")
            if self.reply_key != "service_unavailable" or not empty_payload:
                raise ValueError("error requires service_unavailable and an empty payload")
        return self

