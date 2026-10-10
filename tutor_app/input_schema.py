"""Exact public request bodies (Integration plan v0.3, section 5 and appendix A).

Nullable fields are required keys. ``text`` is not truncated or normalized here: the gate owns
normalization, the empty-text check and the code-point/token ceilings.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from uuid import UUID

from pydantic import Field, model_validator

from contracts.draft import MachineID, StrictModel

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class InteractionInput(StrictModel):
    text: str
    requested_mode: Literal["answer", "tutor", "quiz"] | None
    session_id: UUID | None
    expected_session_revision: Annotated[int, Field(ge=0)] | None
    notice_version: MachineID

    @model_validator(mode="after")
    def session_pair(self) -> Self:
        if (self.session_id is None) != (self.expected_session_revision is None):
            raise ValueError("session_revision_pair_required")
        return self


class CloseSessionInput(StrictModel):
    expected_session_revision: Annotated[int, Field(ge=0)]


class NoticeAcceptInput(StrictModel):
    notice_version: MachineID
    accepted: Literal[True]


class ReportInput(StrictModel):
    category: Literal["answer_error", "source_issue", "other"]
    message: Optional[Annotated[str, Field(max_length=2000)]]


class DevLoginInput(StrictModel):
    """Development-only synthetic identity selection (loopback profiles; never student_release)."""

    user: Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[a-z0-9-]+$")]
