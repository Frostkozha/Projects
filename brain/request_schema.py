"""BrainRequest wire schema (Brain plan v0.3, section 7 and appendix A)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from contracts.draft import MachineID, StrictModel

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python 3.11 has Self
    from typing_extensions import Self

from uuid import UUID


class BrainRequest(StrictModel):
    schema_version: Literal["brain-request-0.2"]
    request_id: UUID
    task: Literal["answer_draft", "tutor_clue"]
    question_redacted: Annotated[str, Field(min_length=1, max_length=4000)]
    passage_ids: Annotated[list[MachineID], Field(min_length=1, max_length=5)]
    prompt_version: MachineID
    sampling_profile_version: MachineID

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.question_redacted.strip():
            raise ValueError("empty_question")
        if len(set(self.passage_ids)) != len(self.passage_ids):
            raise ValueError("duplicate_passage_ids")
        return self
