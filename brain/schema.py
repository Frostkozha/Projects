"""Brain service wrappers (Brain plan v0.3, section 7).

``DraftAnswer`` is imported from the shared contracts package, never redefined. ``BrainResult`` carries
trusted versions, the evidence digest and observed metrics; unobserved values are null, never invented.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import Field, model_validator

from contracts.draft import SHA256, DraftAnswer, MachineID, StrictModel
from contracts.evidence import FrozenEvidence

from .request_schema import BrainRequest

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from uuid import UUID

REQUEST_SCHEMA = "brain-request-0.2"
RESULT_SCHEMA = "brain-result-0.2"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    QUEUE_FULL = "QUEUE_FULL"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CANCELLED = "CANCELLED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RUNTIME_INCOMPATIBLE = "RUNTIME_INCOMPATIBLE"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"
    MEMORY_LIMIT = "MEMORY_LIMIT"


# INVALID_OUTPUT is a content rejection (coordinator renders fixed A5); everything else is a service failure.
CONTENT_ERRORS = frozenset({ErrorCode.INVALID_OUTPUT})


class BrainError(Exception):
    """Typed failure inside the Brain; the message is a stable code, never request or model text."""

    def __init__(self, code: ErrorCode | str):
        self.code = ErrorCode(code)
        super().__init__(self.code.value)


NonNegFloat = Annotated[float, Field(ge=0)]
NonNegInt = Annotated[int, Field(ge=0)]


class BrainMetrics(StrictModel):
    queue_ms: Optional[NonNegFloat]
    prompt_ms: Optional[NonNegFloat]
    generation_ms: Optional[NonNegFloat]
    total_ms: Optional[NonNegFloat]
    prompt_tokens: Optional[NonNegInt]
    completion_tokens: Optional[NonNegInt]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "unknown"]]

    @model_validator(mode="after")
    def _finite(self) -> Self:
        for v in (self.queue_ms, self.prompt_ms, self.generation_ms, self.total_ms):
            if v is not None and not math.isfinite(v):
                raise ValueError("nonfinite_timing")
        return self

    @classmethod
    def empty(cls) -> "BrainMetrics":
        return cls(queue_ms=None, prompt_ms=None, generation_ms=None, total_ms=None, prompt_tokens=None,
                   completion_tokens=None, finish_reason=None)


class BrainResult(StrictModel):
    schema_version: Literal["brain-result-0.2"]
    request_id: UUID
    status: Literal["ok", "no_evidence", "error"]
    draft: Optional[DraftAnswer]
    error_code: Optional[ErrorCode]
    model_version: Optional[MachineID]
    runtime_version: Optional[MachineID]
    prompt_version: Optional[MachineID]
    sampling_profile_version: Optional[MachineID]
    evidence_digest: Optional[SHA256]
    metrics: BrainMetrics

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if self.status == "ok":
            if self.draft is None or self.draft.status != "draft" or self.error_code is not None:
                raise ValueError("ok requires a draft with status draft and no error code")
            if self.metrics.finish_reason != "stop":
                raise ValueError("ok requires finish_reason stop")
        elif self.status == "no_evidence":
            if self.draft is None or self.draft.status != "no_evidence" or self.error_code is not None:
                raise ValueError("no_evidence requires the empty no_evidence draft and no error code")
        else:
            if self.draft is not None or self.error_code is None:
                raise ValueError("error requires a null draft and a declared error code")
        return self


class PresentationHint(StrictModel):
    level: Literal["basic", "standard"] = "standard"
    max_visible_sentences: Annotated[int, Field(ge=1, le=4)] = 4


@dataclass(frozen=True)
class BrainContext:
    """Trusted coordinator context. Never derived from student input; never shown to the model whole.

    ``deadline`` is an absolute ``time.monotonic()`` value for the whole interaction; the Brain caps it at its
    own 30-second admission bound. ``approved_item_question`` is used only by ``tutor_clue``; the hidden item
    key is never part of this context.
    """

    evidence: FrozenEvidence
    expected_evidence_digest: str
    deadline: float
    presentation: PresentationHint = field(default_factory=PresentationHint)
    approved_item_question: Optional[str] = None
    # Current registry check run immediately before invocation; False aborts (no re-retrieval).
    source_check: Optional[Callable[[], Awaitable[bool]]] = None


def error_result(request_id, code: ErrorCode | str, *, identity: dict, evidence_digest: Optional[str],
                 metrics: Optional[BrainMetrics] = None) -> BrainResult:
    return BrainResult(schema_version=RESULT_SCHEMA, request_id=request_id, status="error", draft=None,
                       error_code=ErrorCode(code), evidence_digest=evidence_digest,
                       metrics=metrics or BrainMetrics.empty(), **identity)


__all__ = ["BrainContext", "BrainError", "BrainMetrics", "BrainRequest", "BrainResult", "CONTENT_ERRORS",
           "ErrorCode", "PresentationHint", "REQUEST_SCHEMA", "RESULT_SCHEMA", "error_result"]
