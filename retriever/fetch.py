"""Approved Quiz-item evidence fetching (Retriever spec v0.2, section 11).

Passage IDs come only from the faculty-approved item mapping in the pinned snapshot, never from a
student. A bare answer such as ``B`` is never semantically searched.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .schema import MAX_PASSAGES


class EvidenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passage_ids: tuple[str, ...]
    coverage: Literal["full", "partial"] = "full"

    @field_validator("passage_ids", mode="before")
    @classmethod
    def _t(cls, v):
        return tuple(v) if isinstance(v, list) else v


class ItemMapping(BaseModel):
    """Faculty-authored item -> alternative sufficient evidence sets (one complete set suffices)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str
    item_version: str
    approved: bool
    evidence_sets: tuple[EvidenceSet, ...]

    @model_validator(mode="after")
    def _activation_rules(self):
        if not self.evidence_sets:
            raise ValueError("item needs at least one evidence set")
        for s in self.evidence_sets:
            if not s.passage_ids or len(set(s.passage_ids)) != len(s.passage_ids):
                raise ValueError("evidence set must be non-empty and duplicate-free")
            if len(s.passage_ids) > MAX_PASSAGES:
                raise ValueError("evidence set exceeds the five-passage limit; item cannot be activated")
        return self
