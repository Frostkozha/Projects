"""Public response types and the fixed error envelope (Integration plan v0.3, section 5).

No risk scores, diagnostic text, hidden answers, raw drafts or internal version hashes appear here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field

from contracts.draft import MachineID, StrictModel

ERROR_STATUS = {
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403, "NOTICE_REQUIRED": 403,
    "CONTENT_UNAVAILABLE": 404,
    "SESSION_CONFLICT": 409, "SESSION_EXPIRED": 409, "VERSION_MISMATCH": 409, "IDEMPOTENCY_CONFLICT": 409,
    "REQUEST_IN_PROGRESS": 409, "REQUEST_EXPIRED": 409, "REQUEST_CANCELLED": 409,
    "INPUT_TOO_LONG": 413,
    "INVALID_REQUEST": 422, "EMPTY_INPUT": 422,
    "CAPACITY_EXCEEDED": 429, "RATE_LIMITED": 429,
    "SERVICE_UNAVAILABLE": 503, "MAINTENANCE": 503, "EXAM_DISABLED": 503,
}


class ApiError(Exception):
    def __init__(self, code: str, request_id: Optional[str] = None):
        if code not in ERROR_STATUS:
            raise ValueError("undeclared error code")
        super().__init__(code)
        self.code = code
        self.request_id = request_id

    @property
    def status(self) -> int:
        return ERROR_STATUS[self.code]


class LocatorOut(StrictModel):
    kind: str
    start: Optional[int]
    end: Optional[int]
    label: str
    anchor: Optional[str]


class Citation(StrictModel):
    citation_id: Annotated[int, Field(ge=1, le=5)]
    passage_id: MachineID
    title: str
    source_version: MachineID
    locator: LocatorOut
    url: str


class ActivityOption(StrictModel):
    option_id: Literal["A", "B", "C", "D", "E"]
    text: str


class Activity(StrictModel):
    item_id: MachineID
    item_version: MachineID
    mode: Literal["tutor", "quiz"]
    phase: Literal["question", "feedback"]
    question: str
    options: list[ActivityOption]


class SessionView(StrictModel):
    session_id: str
    revision: Annotated[int, Field(ge=0)]
    state: Literal["idle", "awaiting_response", "paused", "closed"]
    mode: Literal["answer", "tutor", "quiz"]
    expires_at: str


class StudentReply(StrictModel):
    schema_version: Literal["student-reply-0.2"]
    request_id: str
    response_code: Literal["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    content_type: Literal["generated", "fixed", "approved_item"]
    body: str
    notices: list[str]
    citations: list[Citation]
    activity: Optional[Activity]
    session: Optional[SessionView]


class ErrorEnvelope(StrictModel):
    error_code: str
    request_id: Optional[str]


class EvidenceView(StrictModel):
    passage_id: MachineID
    title: str
    source_version: MachineID
    locator: LocatorOut
    text: str


class NoticeView(StrictModel):
    notice_version: MachineID
    text: str
    accepted: bool
