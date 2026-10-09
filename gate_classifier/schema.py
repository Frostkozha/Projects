"""Strict request, context, prediction and decision types (spec sections 5, 7, 9)."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator, model_validator

SCHEMA_VERSION = "gate-decision-0.2"

MODES = ("answer", "tutor", "quiz")
TOPIC_LABELS = ("course_related", "outside_course", "nonmedical", "unclear")
RISK_LABELS = (
    "real_person_advice",
    "imminent_emergency",
    "self_harm_crisis",
    "assessed_work",
    "instruction_override",
    "private_data_request",
)
LIBRARY_IDS = ("lib1", "lib2", "lib3", "lib4", "lib5", "lib6")
MULTICLASS_SUM_TOLERANCE = 1e-5


class Mode(str, Enum):
    answer = "answer"
    tutor = "tutor"
    quiz = "quiz"


class Route(str, Enum):
    retrieve = "retrieve"
    reply = "reply"
    clarify = "clarify"
    escalate = "escalate"
    unavailable = "unavailable"


class Reason(str, Enum):
    COURSE_REQUEST = "COURSE_REQUEST"
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"
    REAL_PERSON_REQUEST = "REAL_PERSON_REQUEST"
    EMERGENCY = "EMERGENCY"
    SELF_HARM = "SELF_HARM"
    ASSESSED_WORK = "ASSESSED_WORK"
    INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"
    PRIVATE_DATA_REQUEST = "PRIVATE_DATA_REQUEST"
    OUTSIDE_COURSE = "OUTSIDE_COURSE"
    NONMEDICAL = "NONMEDICAL"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    NO_ACTIVE_SOURCES = "NO_ACTIVE_SOURCES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SAFETY_UNCERTAIN = "SAFETY_UNCERTAIN"
    GATE_FAILURE = "GATE_FAILURE"
    MAINTENANCE = "MAINTENANCE"
    EXAM_DISABLED = "EXAM_DISABLED"


class ResponseCode(str, Enum):
    A1 = "A1"  # supported answer
    A2 = "A2"  # partial support / declared source conflict
    A3 = "A3"  # clarification
    A4 = "A4"  # restricted general educational explanation (feature not approved in v0.2)
    A5 = "A5"  # no evidence / outside scope / unsupported
    A6 = "A6"  # refusal (real person, integrity, security)
    A7 = "A7"  # emergency / crisis


class ModeSource(str, Enum):
    ui = "ui"
    text_switch = "text_switch"
    session = "session"
    predicted = "predicted"
    default = "default"


class InferenceStatus(str, Enum):
    completed = "completed"
    skipped_rule = "skipped_rule"
    failed = "failed"
    not_run = "not_run"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    EMPTY_INPUT = "EMPTY_INPUT"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_CONFLICT = "SESSION_CONFLICT"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MAINTENANCE = "MAINTENANCE"
    EXAM_DISABLED = "EXAM_DISABLED"


ERROR_HTTP_STATUS = {
    ErrorCode.INVALID_REQUEST: 422,
    ErrorCode.EMPTY_INPUT: 422,
    ErrorCode.INPUT_TOO_LONG: 413,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.SESSION_EXPIRED: 409,
    ErrorCode.SESSION_CONFLICT: 409,
    ErrorCode.CAPACITY_EXCEEDED: 429,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.MAINTENANCE: 503,
    ErrorCode.EXAM_DISABLED: 503,
}


class GateError(Exception):
    """A request-level failure mapped to the Section 5.3 error envelope."""

    def __init__(self, code: ErrorCode, session_update: "SessionUpdate | None" = None):
        super().__init__(code.value)  # never carries request text
        self.code = code
        self.session_update = session_update

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# --------------------------------------------------------------------------- request


class GateRequest(_Strict):
    """Public body of POST /v1/gate. Unknown fields and type coercion are rejected."""

    text: StrictStr
    requested_mode: Optional[Mode] = None
    session_id: Optional[StrictStr] = None

    @field_validator("requested_mode", mode="before")
    @classmethod
    def _mode_from_str(cls, v):
        # Strict mode would otherwise refuse the JSON string for the enum.
        if v is None:
            return None
        if isinstance(v, str) and v in MODES:
            return Mode(v)
        if isinstance(v, Mode):
            return v
        raise ValueError("invalid requested_mode")

    @field_validator("session_id")
    @classmethod
    def _uuid(cls, v):
        if v is None:
            return v
        try:
            parsed = uuid.UUID(v)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("session_id must be a UUID string") from exc
        if str(parsed) != v.lower():
            raise ValueError("session_id must be a canonical UUID string")
        return str(parsed)


# --------------------------------------------------------------------------- sessions


SessionStateName = Literal["idle", "awaiting_response", "paused", "closed"]


class SessionState(BaseModel):
    """Server-side session (Section 9). Only the orchestrator commits changes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    owner_code: str
    tenant_id: str
    course_id: str
    mode: Mode
    topic_id: Optional[str] = None
    topic_text: Optional[str] = None
    pending_question: Optional[str] = None
    pending_item_id: Optional[str] = None
    item_version: Optional[str] = None
    state: SessionStateName = "idle"
    revision: int = 0
    expires_at: datetime
    policy_version: str
    kb_version: str
    answered_items: tuple[str, ...] = ()

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class SessionUpdate(_Strict):
    session_id: StrictStr
    expected_revision: int
    operation: Literal["pause", "close", "reset_pending"]


DeploymentState = Literal["active", "suspended", "exam_shutdown", "disallowed"]


class GateContext(BaseModel):
    """Trusted context supplied by the orchestrator after authentication/authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    course_id: str
    tenant_id: str
    owner_code: str
    authorized_active_libraries: tuple[str, ...]
    topic_registry_version: str
    kb_version: str
    policy_version: str
    deployment_state: DeploymentState = "active"
    session: Optional[SessionState] = None

    @field_validator("authorized_active_libraries")
    @classmethod
    def _known_libs(cls, v):
        for lib in v:
            if lib not in LIBRARY_IDS:
                raise ValueError(f"unknown library id {lib}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate library ids")
        return tuple(sorted(v))


# --------------------------------------------------------------------------- predictions


def _check_unit(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return value


def _check_multiclass(name: str, scores: dict, labels: tuple[str, ...]) -> dict:
    if not isinstance(scores, dict) or set(scores) != set(labels):
        raise ValueError(f"{name} must contain exactly {labels}")
    out = {k: _check_unit(f"{name}.{k}", scores[k]) for k in labels}
    if abs(sum(out.values()) - 1.0) > MULTICLASS_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1")
    return out


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: dict[str, float]
    topic_scope: dict[str, float]
    risks: dict[str, float]
    libraries: dict[str, Optional[float]]

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, data):
        if not isinstance(data, dict):
            raise ValueError("prediction must be an object")
        if set(data) != {"mode", "topic_scope", "risks", "libraries"}:
            raise ValueError("prediction keys mismatch")
        risks = data["risks"]
        if not isinstance(risks, dict) or set(risks) != set(RISK_LABELS):
            raise ValueError("risks must contain exactly the declared risks")
        libs = data["libraries"]
        if not isinstance(libs, dict) or set(libs) != set(LIBRARY_IDS):
            raise ValueError("libraries must contain exactly lib1..lib6")
        return {
            "mode": _check_multiclass("mode", data["mode"], MODES),
            "topic_scope": _check_multiclass("topic_scope", data["topic_scope"], TOPIC_LABELS),
            "risks": {k: _check_unit(f"risks.{k}", risks[k]) for k in RISK_LABELS},
            "libraries": {k: (None if libs[k] is None else _check_unit(f"libraries.{k}", libs[k])) for k in LIBRARY_IDS},
        }


# --------------------------------------------------------------------------- rule flags


class RuleFlags(BaseModel):
    """Internal, never student-visible. Matched categories only; never the matched text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    imminent_emergency: bool = False
    self_harm_crisis: bool = False
    real_person_advice: bool = False
    private_data_request: bool = False
    instruction_override: bool = False
    assessed_work: bool = False
    ambiguous_clinical: bool = False
    unsupported_modality: bool = False
    bare_reply: bool = False
    text_mode_switch: Optional[Mode] = None
    rules_version: str = ""

    @property
    def crisis_hard(self) -> bool:
        return self.imminent_emergency or self.self_harm_crisis


# --------------------------------------------------------------------------- decision


class RetrievalPlan(_Strict):
    allowed_libraries: tuple[str, ...]
    preferred_libraries: tuple[str, ...]
    query_text: StrictStr
    strategy: Literal["all_active"] = "all_active"


class DecisionFlags(_Strict):
    pii_detected: StrictBool
    pii_redacted: StrictBool
    context_used: StrictBool


class Versions(_Strict):
    bundle: Optional[StrictStr]
    encoder_revision: Optional[StrictStr]
    policy: Optional[StrictStr]
    config_sha256: Optional[StrictStr]
    library_registry: Optional[StrictStr]
    kb: Optional[StrictStr]


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["gate-decision-0.2"] = SCHEMA_VERSION
    request_id: str
    route: Route
    reason: Reason
    response_code: Optional[ResponseCode]
    reply_key: Optional[str]
    effective_mode: Optional[Mode]
    mode_source: Optional[ModeSource]
    retrieval_plan: Optional[RetrievalPlan]
    prediction: Optional[Prediction]
    flags: DecisionFlags
    inference_status: InferenceStatus
    versions: Versions

    @model_validator(mode="after")
    def _invariants(self):
        r = self.route
        if r == Route.retrieve:
            if self.retrieval_plan is None or self.effective_mode is None or self.mode_source is None:
                raise ValueError("retrieve requires a plan and a mode")
            if self.response_code is not None or self.reply_key is not None:
                raise ValueError("retrieve has no final code")
            if not self.retrieval_plan.allowed_libraries:
                raise ValueError("retrieve requires at least one allowed library")
        else:
            if self.retrieval_plan is not None:
                raise ValueError("blocked routes have no retrieval plan")
            if self.effective_mode is not None or self.mode_source is not None:
                raise ValueError("blocked routes have no mode")
            if self.reply_key is None:
                raise ValueError("blocked routes need a reply key")
        expected = {
            Route.clarify: {ResponseCode.A3},
            Route.escalate: {ResponseCode.A7},
            Route.reply: {ResponseCode.A5, ResponseCode.A6},
            Route.unavailable: {None},
            Route.retrieve: {None},
        }[r]
        if self.response_code not in expected:
            raise ValueError(f"route {r.value} cannot carry {self.response_code}")
        if self.inference_status in (InferenceStatus.failed, InferenceStatus.not_run):
            if r != Route.unavailable and r != Route.escalate:
                raise ValueError("failed/not_run inference only on unavailable (or hard-rule escalate) decisions")
        if self.inference_status != InferenceStatus.completed and self.prediction is not None:
            raise ValueError("prediction only accompanies completed inference")
        return self


# Canonical internal vector reference shared with the retriever (never serialized or logged).
from retriever.schema import EmbeddingRef  # noqa: E402


class GateResult:
    """Pairs the serializable decision with internal-only data."""

    __slots__ = ("decision", "redacted_text", "embedding_ref", "session_update", "rule_flags", "pii_categories")

    def __init__(self, decision, redacted_text, embedding_ref=None, session_update=None, rule_flags=None, pii_categories=()):
        self.decision: GateDecision = decision
        self.redacted_text: Optional[str] = redacted_text
        self.embedding_ref: Optional[EmbeddingRef] = embedding_ref
        self.session_update: Optional[SessionUpdate] = session_update
        self.rule_flags: Optional[RuleFlags] = rule_flags
        self.pii_categories: tuple[str, ...] = tuple(pii_categories)

    def __repr__(self) -> str:
        return f"GateResult(route={self.decision.route.value}, reason={self.decision.reason.value})"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_request_id() -> str:
    return str(uuid.uuid4())


class ErrorBody(_Strict):
    error_code: ErrorCode
    request_id: StrictStr

