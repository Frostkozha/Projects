"""Layer 1 support: trusted evidence bundle and live source eligibility (Verifier spec v0.2, sections 4.2, 5.2).

The bundle is built server-side by the orchestrator from the pinned RetrievalResult and holds exactly the
passages shown to the Brain. The caller can never submit evidence text. Hashes, scope and live
eligibility are checked at verification and again at delivery; a registry failure is an error, never
a permissive cached allow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Protocol

from contracts.models import OperationalError
from retriever.schema import EvidencePassage, RetrievalResult, Status


class VerifierFault(Exception):
    """Operational failure for the whole request (status error, service_unavailable)."""

    def __init__(self, code: OperationalError):
        super().__init__(code.value)
        self.code = code


class RegistryUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceBundle:
    """Exact passages shown to the Brain, bound to one request, tenant, course and pinned snapshot."""

    request_id: str
    tenant_id: str
    course_id: str
    retrieval: RetrievalResult
    shown_passage_ids: tuple[str, ...]

    @property
    def passages(self) -> tuple[EvidencePassage, ...]:
        by_id = {p.passage_id: p for p in self.retrieval.passages}
        return tuple(by_id[i] for i in self.shown_passage_ids if i in by_id)

    def binding(self) -> dict:
        """Canonical object for evidence_digest: exact text/hash, order and versions."""
        r = self.retrieval
        return {
            "request_id": self.request_id, "tenant_id": self.tenant_id, "course_id": self.course_id,
            "kb_version": r.kb_version, "index_version": r.index_version, "profile_version": r.profile_version,
            "revocation_epoch": r.revocation_epoch, "coverage": r.coverage,
            "passages": [{"passage_id": p.passage_id, "source_id": p.source_id, "source_version": p.source_version,
                          "library_id": p.library_id, "text": p.text, "text_sha256": p.text_sha256}
                         for p in self.passages],
            "conflicts": [{"conflict_id": c.conflict_id, "passage_ids": list(c.passage_ids)} for c in r.conflicts],
        }


class SourceRegistry(Protocol):
    """Trusted live registry: rights, approval, scope, review dates and revocation state."""

    def eligible(self, passages: tuple[EvidencePassage, ...], kb_version: str) -> dict[str, bool]: ...

    def epoch(self) -> int: ...


class RetrieverSourceRegistry:
    """Adapter over the retriever's live eligibility checks for one trusted retrieval context."""

    def __init__(self, service, context):
        self.service = service
        self.context = context

    def eligible(self, passages, kb_version):
        try:
            return self.service.passage_eligibility(tuple(p.passage_id for p in passages), kb_version, self.context)
        except Exception:
            raise RegistryUnavailable("registry lookup failed") from None

    def epoch(self) -> int:
        try:
            self.service.revocations.refresh()
            return self.service.revocations.epoch()
        except Exception:
            raise RegistryUnavailable("registry epoch unavailable") from None


def resolve_evidence(bundle: EvidenceBundle, request_id: str, tenant_id: str, course_id: str,
                     registry: SourceRegistry) -> tuple[dict[str, EvidencePassage], set[str]]:
    """Return (supplied passages by id, ineligible ids). Raises VerifierFault on any trust failure."""
    if bundle.request_id != request_id or bundle.tenant_id != tenant_id or bundle.course_id != course_id:
        raise VerifierFault(OperationalError.CONTEXT_MISMATCH)
    r = bundle.retrieval
    if r.status != Status.ok or not 1 <= len(r.passages) <= 5:
        raise VerifierFault(OperationalError.CONTEXT_MISMATCH)
    ids = list(bundle.shown_passage_ids)
    known = {p.passage_id for p in r.passages}
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= known:
        raise VerifierFault(OperationalError.EVIDENCE_INTEGRITY)
    supplied = {p.passage_id: p for p in bundle.passages}
    for p in supplied.values():
        if hashlib.sha256(p.text.encode("utf-8")).hexdigest() != p.text_sha256:
            raise VerifierFault(OperationalError.EVIDENCE_INTEGRITY)
    try:
        status = registry.eligible(tuple(supplied.values()), r.kb_version)
    except RegistryUnavailable:
        raise VerifierFault(OperationalError.REGISTRY_UNAVAILABLE) from None
    if set(status) != set(supplied):
        raise VerifierFault(OperationalError.REGISTRY_UNAVAILABLE)
    ineligible = {pid for pid, ok in status.items() if not ok or supplied[pid].review_status != "live"}
    return supplied, ineligible


def current_epoch(registry: SourceRegistry) -> int:
    try:
        return registry.epoch()
    except RegistryUnavailable:
        raise VerifierFault(OperationalError.REGISTRY_UNAVAILABLE) from None


def opt(v) -> Optional[str]:
    return None if v is None else str(v)
