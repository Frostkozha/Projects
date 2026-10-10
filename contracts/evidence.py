"""Hash-bound frozen evidence (Brain plan v0.3, shared primitives).

``freeze_validated_bundle`` binds bytes; it does not establish source approval or user authority. Its
input must first pass the canonical retriever ``EvidencePassage`` validators (see ``bundle_value``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from contracts.json_codec import canonical_json, load_object

BUNDLE_FIELDS = {
    "request_id", "tenant_id", "course_id", "kb_version", "index_version",
    "profile_version", "retrieval_policy_version", "revocation_epoch",
    "registry_version", "passages",
}


@dataclass(frozen=True)
class FrozenEvidence:
    canonical_bytes: bytes
    evidence_digest: str
    passage_ids: tuple[str, ...]

    def fresh_value(self) -> dict:
        return load_object(self.canonical_bytes, max_bytes=len(self.canonical_bytes))


def freeze_validated_bundle(value: dict) -> FrozenEvidence:
    if not isinstance(value, dict) or set(value) != BUNDLE_FIELDS:
        raise ValueError("CONTEXT_MISMATCH")
    passages = value["passages"]
    if not isinstance(passages, list) or not 1 <= len(passages) <= 5:
        raise ValueError("INVALID_EVIDENCE")
    ids: list[str] = []
    for passage in passages:
        if not isinstance(passage, dict):
            raise ValueError("INVALID_EVIDENCE")
        pid, text, expected = passage.get("passage_id"), passage.get("text"), passage.get("text_sha256")
        if (not isinstance(pid, str) or not isinstance(text, str)
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("INVALID_EVIDENCE")
        if hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest() != expected:
            raise ValueError("INVALID_EVIDENCE")
        ids.append(pid)
    if len(ids) != len(set(ids)):
        raise ValueError("INVALID_EVIDENCE")
    raw = canonical_json(value)
    return FrozenEvidence(raw, hashlib.sha256(raw).hexdigest(), tuple(ids))


def bundle_value(*, request_id: str, tenant_id: str, course_id: str, retrieval, shown_passage_ids,
                 registry_version: str) -> dict:
    """Canonical bundle dict from a validated ``retriever.schema.RetrievalResult``.

    Every passage field is preserved (locator, rights, versions); only the shown passages, in shown order.
    """
    by_id = {p.passage_id: p for p in retrieval.passages}
    if any(pid not in by_id for pid in shown_passage_ids):
        raise ValueError("INVALID_EVIDENCE")
    return {
        "request_id": request_id, "tenant_id": tenant_id, "course_id": course_id,
        "kb_version": retrieval.kb_version, "index_version": retrieval.index_version,
        "profile_version": retrieval.profile_version,
        "retrieval_policy_version": retrieval.retrieval_policy_version,
        "revocation_epoch": retrieval.revocation_epoch, "registry_version": registry_version,
        "passages": [by_id[pid].model_dump(mode="json") for pid in shown_passage_ids],
    }
