"""Draft contract entry point (Brain plan v0.3, shared primitives).

``DraftAnswer`` and ``DraftSentence`` are defined once, in ``contracts.models`` (Verifier spec v0.2),
and re-exported here so the Brain, verifier and coordinator share one type. ``MachineID`` and
``StrictModel`` are the shared building blocks for new wire wrappers.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import DRAFT_SCHEMA, DraftAnswer, DraftSentence

MachineID = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


__all__ = ["DRAFT_SCHEMA", "DraftAnswer", "DraftSentence", "MachineID", "SHA256", "StrictModel"]
