"""Approved Tutor and Quiz records and the topic alias registry (Integration plan v0.3, section 7).

Items are imported whole or rejected. Hidden fields (``expected_answer``, ``correct_option_id``) stay in
trusted state and never enter public payloads, reports or logs. Topic resolution is exact: a whole
normalized alias or ``Quiz me on {alias}`` / ``Tutor me on {alias}`` (NFKC, casefold, collapsed spaces).
No fuzzy or model-written expansion.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import Field, model_validator

from contracts.draft import SHA256, MachineID, StrictModel
from contracts.json_codec import digest, load_object

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

OptionID = Literal["A", "B", "C", "D", "E"]


class Option(StrictModel):
    option_id: OptionID
    text: Annotated[str, Field(min_length=1, max_length=600)]


class ItemRecord(StrictModel):
    item_id: MachineID
    item_version: MachineID
    tenant_id: MachineID
    course_id: MachineID
    topic_id: MachineID
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    options: Annotated[list[Option], Field(min_length=2, max_length=5)]
    correct_option_id: OptionID
    reviewed_explanation: Annotated[str, Field(min_length=1, max_length=2000)]
    evidence_passage_ids: Annotated[list[MachineID], Field(min_length=1, max_length=5)]
    approval_reference: MachineID
    status: Literal["live", "archived", "blocked", "pending_review"]
    review_due_at: str

    @model_validator(mode="after")
    def _options(self) -> Self:
        ids = [o.option_id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate option ids")
        if self.correct_option_id not in ids:
            raise ValueError("correct option missing")
        if len(set(self.evidence_passage_ids)) != len(self.evidence_passage_ids):
            raise ValueError("duplicate evidence ids")
        return self


class TutorItem(StrictModel):
    item_id: MachineID
    item_version: MachineID
    tenant_id: MachineID
    course_id: MachineID
    topic_id: MachineID
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    expected_answer: Annotated[str, Field(min_length=1, max_length=2000)]
    reviewed_explanation: Annotated[str, Field(min_length=1, max_length=2000)]
    evidence_passage_ids: Annotated[list[MachineID], Field(min_length=1, max_length=5)]
    approval_reference: MachineID
    status: Literal["live", "archived", "blocked", "pending_review"]
    review_due_at: str


class Topic(StrictModel):
    topic_id: MachineID
    title: Annotated[str, Field(min_length=1, max_length=200)]
    aliases: Annotated[list[Annotated[str, Field(min_length=1, max_length=100)]], Field(min_length=1)]


class TopicRegistry(StrictModel):
    schema_version: Literal["topic-registry-0.2"]
    registry_version: MachineID
    course_id: MachineID
    topics: list[Topic]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        seen: dict[str, str] = {}
        for t in self.topics:
            for a in t.aliases:
                key = normalize(a)
                if key in seen and seen[key] != t.topic_id:
                    raise ValueError("alias maps to two topics")
                seen[key] = t.topic_id
        return self


class ItemBank(StrictModel):
    schema_version: Literal["item-bank-0.2"]
    bank_version: MachineID
    tenant_id: MachineID
    course_id: MachineID
    review_status: Literal["approved", "development_unreviewed", "synthetic_fixture"]
    quiz_items: list[ItemRecord]
    tutor_items: list[TutorItem]

    @model_validator(mode="after")
    def _scope(self) -> Self:
        keys = [(i.item_id, i.item_version) for i in [*self.quiz_items, *self.tutor_items]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate item id/version")
        for i in [*self.quiz_items, *self.tutor_items]:
            if i.tenant_id != self.tenant_id or i.course_id != self.course_id:
                raise ValueError("item outside bank scope")
        return self

    @property
    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


def item_hash(item: ItemRecord | TutorItem) -> SHA256:
    return digest(item.model_dump(mode="json"))


def load_bank(path: Path) -> ItemBank:
    return ItemBank.model_validate_json(_canon(path), strict=True)


def load_topics(path: Path) -> TopicRegistry:
    return TopicRegistry.model_validate_json(_canon(path), strict=True)


def _canon(path: Path) -> bytes:
    import json  # noqa: PLC0415

    return json.dumps(load_object(path.read_bytes(), max_bytes=4 * 1024 * 1024)).encode("utf-8")


# ----------------------------------------------------------------------------- exact matching

_SPACES = re.compile(r"\s+")
_COMMAND = re.compile(r"^(quiz|tutor) me on (.+?)[.!?]?$")
CONTINUE = {"continue", "next", "next question"}
SHOW_EXPLANATION = {"show the approved explanation", "show explanation", "show the explanation"}


def normalize(text: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def resolve_topic(text: str, registry: TopicRegistry) -> tuple[Optional[str], Optional[str]]:
    """(command_mode or None, topic_id or None) from an exact alias or exact command form."""
    t = normalize(text)
    mode = None
    m = _COMMAND.match(t)
    if m:
        mode, t = m.group(1), m.group(2).strip()
    for topic in registry.topics:
        if any(normalize(a) == t for a in topic.aliases):
            return mode, topic.topic_id
    return mode, None


def option_answer(text: str) -> Optional[str]:
    """Exact trimmed whole-message option ID, case-insensitive (A-E)."""
    t = text.strip()
    return t.upper() if len(t) == 1 and t.upper() in "ABCDE" else None


def control(text: str) -> Optional[str]:
    t = normalize(text).rstrip(".!")
    if t in CONTINUE:
        return "continue"
    if t in SHOW_EXPLANATION:
        return "show_explanation"
    return None


def due_order(items: list, review: dict[tuple[str, str], str], now: str, *, exclude: Optional[str] = None) -> list:
    """Live items ordered by due time then item_id; unseen items are due immediately."""
    out = []
    for it in items:
        if it.status != "live" or it.item_id == exclude or it.review_due_at <= now[:10]:
            continue
        due = review.get((it.item_id, it.item_version), "")
        if due <= now:
            out.append((due, it.item_id, it))
    return [x[2] for x in sorted(out, key=lambda x: (x[0], x[1]))]


def is_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None
