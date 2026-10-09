"""Threshold, overlap removal and conflict preservation (Retriever spec v0.2, sections 10.3, 10.4, 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict

from .chunk import span_overlap
from .schema import ErrorCode, RetrievalError


class ConflictSide(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    source_version: str
    passage_ids: tuple[str, ...]


class ConflictRecord(BaseModel):
    """Faculty-registered conflict between approved sources. Never inferred from score differences."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    conflict_id: str
    topic_id: str
    sides: tuple[ConflictSide, ...]
    faculty_review_status: str
    description: str
    description_approved: bool
    policy_reference: str

    @property
    def passage_ids(self) -> tuple[str, ...]:
        return tuple(pid for side in self.sides for pid in side.passage_ids)


@dataclass(frozen=True)
class Candidate:
    passage_id: str
    logit: float
    rrf: float
    source_key: tuple[str, str]
    section_path: tuple[str, ...]
    text_sha256: str
    spans: tuple[tuple[str, int, int], ...]


@dataclass
class Selection:
    passages: list[Candidate]
    conflicts: list[ConflictRecord]
    suppressed: list[str] = field(default_factory=list)  # internal diagnostics only

    @property
    def empty(self) -> bool:
        return not self.passages


def rank_key(c: Candidate):
    return (-c.logit, -c.rrf, c.passage_id)


def select(candidates: Sequence[Candidate], threshold: float, max_passages: int, overlap_limit: float,
           conflicts: Sequence[ConflictRecord] = ()) -> Selection:
    if threshold is None:
        raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE)  # never a threshold-zero fallback
    qualifying = sorted((c for c in candidates if c.logit >= threshold), key=rank_key)
    kept: list[Candidate] = []
    suppressed: list[str] = []
    for c in qualifying:
        redundant = False
        for k in kept:
            if k.source_key != c.source_key:
                continue  # cross-source duplicates are preserved (provenance / conflicts)
            if k.text_sha256 == c.text_sha256:
                redundant = True
            elif k.section_path == c.section_path and span_overlap(k.spans, c.spans) >= overlap_limit:
                redundant = True
            if redundant:
                break
        if redundant:
            suppressed.append(c.passage_id)
        else:
            kept.append(c)
    if not kept:
        return Selection([], [], suppressed)

    by_id = {c.passage_id: c for c in kept}
    selected_ids = [c.passage_id for c in kept[:max_passages]]
    relevant: list[ConflictRecord] = []
    changed = True
    while changed:
        changed = False
        for record in conflicts:
            if record in relevant or not any(pid in selected_ids for pid in record.passage_ids):
                continue
            if record.faculty_review_status != "approved" or not record.description_approved:
                raise RetrievalError(ErrorCode.CONFLICT_EVIDENCE_INCOMPLETE)
            if len(record.sides) < 2 or any(not any(p in by_id for p in side.passage_ids) for side in record.sides):
                raise RetrievalError(ErrorCode.CONFLICT_EVIDENCE_INCOMPLETE)
            relevant.append(record)
            changed = True
        required: list[str] = []
        for record in relevant:
            for side in record.sides:
                present = [p for p in side.passage_ids if p in by_id]
                best = min(present, key=lambda p: rank_key(by_id[p]))
                if best not in required:
                    required.append(best)
        if len(required) > max_passages:
            raise RetrievalError(ErrorCode.CONFLICT_EVIDENCE_INCOMPLETE)
        if required:
            others = [c.passage_id for c in kept if c.passage_id not in required]
            new = sorted(required + others[: max_passages - len(required)], key=lambda p: rank_key(by_id[p]))
            if new != selected_ids:
                selected_ids = new
                changed = True
    return Selection([by_id[p] for p in selected_ids], relevant, suppressed)


def coverage_for_search() -> str:
    """Free-text search cannot infer full support from a passing threshold."""
    return "unknown"

