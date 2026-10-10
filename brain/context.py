"""Trusted context construction from validated retriever passages (Brain plan v0.3, section 8).

Performs no retrieval and cannot add a passage from disk or model memory: it freezes exactly the
validated passages the coordinator selected, in order, and binds them to the request.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Optional

from contracts.evidence import FrozenEvidence, freeze_validated_bundle
from retriever.schema import EvidencePassage

from .schema import BrainContext, PresentationHint


def freeze_passages(*, request_id: str, tenant_id: str, course_id: str, passages: list[EvidencePassage],
                    kb_version: Optional[str], index_version: Optional[str], profile_version: Optional[str],
                    retrieval_policy_version: str, revocation_epoch: Optional[int],
                    registry_version: str) -> FrozenEvidence:
    for p in passages:
        if not isinstance(p, EvidencePassage):
            raise ValueError("INVALID_EVIDENCE")
        EvidencePassage.model_validate(p.model_dump())  # re-run the canonical passage validators
    return freeze_validated_bundle({
        "request_id": request_id, "tenant_id": tenant_id, "course_id": course_id, "kb_version": kb_version,
        "index_version": index_version, "profile_version": profile_version,
        "retrieval_policy_version": retrieval_policy_version, "revocation_epoch": revocation_epoch,
        "registry_version": registry_version, "passages": [p.model_dump(mode="json") for p in passages],
    })


def synthetic_passage(passage_id: str, text: str) -> EvidencePassage:
    """Clearly synthetic evaluation/probe passage; never part of the approved knowledge base."""
    return EvidencePassage(
        passage_id=passage_id, source_id="synthetic-src", source_version="synthetic-v1",
        title="Synthetic evaluation passage", library_id="lib1",
        locator={"kind": "section", "start": None, "end": None, "label": "Synthetic", "anchor": "synthetic"},
        section_path=("Synthetic",), edition=None, publication_year=None, text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), review_status="live",
        rights_reference="synthetic-rights", relevance_score=None, score_type=None,
        evidence_uri=f"/v1/evidence/{passage_id}?kb_version=synthetic")


def synthetic_context(passages: list[tuple[str, str]], *, request_id: Optional[str] = None,
                      deadline_seconds: float = 60.0, presentation: Optional[PresentationHint] = None,
                      approved_item_question: Optional[str] = None) -> tuple[str, BrainContext]:
    rid = request_id or str(uuid.uuid4())
    frozen = freeze_passages(request_id=rid, tenant_id="synthetic-tenant", course_id="synthetic-course",
                             passages=[synthetic_passage(pid, text) for pid, text in passages],
                             kb_version="synthetic-kb", index_version="synthetic-index",
                             profile_version="synthetic-profile", retrieval_policy_version="retrieval-policy-0.2",
                             revocation_epoch=0, registry_version="synthetic-registry")
    ctx = BrainContext(evidence=frozen, expected_evidence_digest=frozen.evidence_digest,
                       deadline=time.monotonic() + deadline_seconds,
                       presentation=presentation or PresentationHint(),
                       approved_item_question=approved_item_question)
    return rid, ctx
