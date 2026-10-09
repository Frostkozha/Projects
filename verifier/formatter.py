"""Deterministic formatter and delivery authorization (Verifier spec v0.2, section 16.3).

The formatter consumes only approved visible sentence IDs and their supporting citation subset, joins
the exact original sentence text in original order, appends code-owned notices and renders citations
from authoritative passage metadata (never model-written locators or links). Text is HTML-escaped.

Delivery authorization is the linearization point: under one lock it validates the result binding,
rechecks live source eligibility at the current revocation epoch and commits a DeliveryRecord holding
the payload digest. The UI accepts only a payload with a matching record. A revocation after
authorization cannot recall text already authorized; later requests still deny the source.
"""

from __future__ import annotations

import html
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from contracts.models import DraftAnswer, ResponseCode, VerificationResult

from . import digests
from .evidence import EvidenceBundle, RegistryUnavailable, SourceRegistry


class DeliveryRefused(Exception):
    """Delivery suppressed. ``code`` is a stable reason; ``unavailable`` marks operational faults."""

    def __init__(self, code: str, unavailable: bool = False):
        super().__init__(code)
        self.code = code
        self.unavailable = unavailable


@dataclass(frozen=True)
class CompiledPayload:
    request_id: str
    response_code: str
    text: str
    sentences: tuple[tuple[str, str, tuple[int, ...]], ...]  # (sentence_id, escaped text, citation numbers)
    citations: tuple[dict, ...]
    notices: tuple[str, ...]
    decision_digest: str
    digest: str

    def body(self) -> dict:
        return {"request_id": self.request_id, "response_code": self.response_code, "text": self.text,
                "sentences": [list(s[:2]) + [list(s[2])] for s in self.sentences],
                "citations": [dict(c) for c in self.citations], "notices": list(self.notices),
                "decision_digest": self.decision_digest}


def _payload_digest(body: dict) -> str:
    return digests.sha256_hex(body)


def check_binding(result: VerificationResult, draft: DraftAnswer, bundle: EvidenceBundle) -> None:
    if not isinstance(result, VerificationResult) or result.status != "approved":
        raise DeliveryRefused("not_approved")
    if result.request_id != bundle.request_id:
        raise DeliveryRefused("request_mismatch")
    if digests.draft_digest(draft) != result.draft_digest:
        raise DeliveryRefused("draft_digest_mismatch")
    if digests.evidence_digest(bundle) != result.evidence_digest:
        raise DeliveryRefused("evidence_digest_mismatch")
    if not digests.result_digest_valid(result):
        raise DeliveryRefused("decision_digest_mismatch")


def compile_payload(result: VerificationResult, draft: DraftAnswer, bundle: EvidenceBundle,
                    notices: dict[str, str]) -> CompiledPayload:
    """Build the exact student payload. Raises DeliveryRefused on any binding or consistency failure."""
    check_binding(result, draft, bundle)
    by_id = {s.sentence_id: s for s in draft.sentences}
    order = [s.sentence_id for s in draft.sentences]
    final = list(result.final_sentence_ids)
    if any(sid not in by_id or by_id[sid].visibility != "student" for sid in final):
        raise DeliveryRefused("hidden_or_unknown_sentence")
    if sorted(final, key=order.index) != final:
        raise DeliveryRefused("sentence_order_changed")
    if " ".join(by_id[sid].text for sid in final) != result.verified_text:
        raise DeliveryRefused("verified_text_mismatch")  # formatter never paraphrases
    passages = {p.passage_id: p for p in bundle.passages}
    numbers: dict[str, int] = {}
    citations: list[dict] = []
    sentences = []
    for sid in final:
        cited = result.citation_map.get(sid, ())
        if any(pid not in passages or pid not in by_id[sid].cites for pid in cited):
            raise DeliveryRefused("citation_not_supplied")
        nums = []
        for pid in cited:
            if pid not in numbers:
                p = passages[pid]
                numbers[pid] = len(numbers) + 1
                citations.append({"number": numbers[pid], "passage_id": p.passage_id, "source_id": p.source_id,
                                  "source_version": p.source_version, "title": html.escape(p.title),
                                  "locator": html.escape(p.locator.label), "evidence_uri": p.evidence_uri})
            nums.append(numbers[pid])
        sentences.append((sid, html.escape(by_id[sid].text), tuple(nums)))
    chosen: list[str] = []
    if result.disposition == "trimmed":
        chosen.append(notices["partial"])
    if result.source_conflict:
        chosen.append(notices["conflict"])
    if result.partial_support and result.disposition != "trimmed" and not result.source_conflict:
        chosen.append(notices["coverage_unknown"])
    if result.response_code == ResponseCode.A4:
        chosen.append(notices["a4_disclaimer"])
    chosen.append(notices["ai_notice"])
    body_text = " ".join(f"{t}" + "".join(f" [{n}]" for n in nums) for _, t, nums in sentences)
    text = "\n\n".join([body_text] + [html.escape(n) for n in chosen])
    payload = CompiledPayload(request_id=result.request_id, response_code=result.response_code.value, text=text,
                              sentences=tuple(sentences), citations=tuple(citations), notices=tuple(chosen),
                              decision_digest=result.decision_digest, digest="")
    return CompiledPayload(**{**payload.__dict__, "digest": _payload_digest(payload.body())})


@dataclass(frozen=True)
class DeliveryRecord:
    request_id: str
    payload_digest: str
    decision_digest: str
    revocation_epoch: int
    authorized_at: float


class DeliveryAuthorizer:
    """Atomic eligibility recheck + authorization record. ``record_sink(record)`` must persist durably."""

    def __init__(self, notices: dict[str, str], record_sink: Optional[Callable[[DeliveryRecord], None]] = None):
        self.notices = notices
        self._lock = threading.Lock()
        self._records: dict[str, DeliveryRecord] = {}
        self._sink = record_sink

    def authorize(self, result: VerificationResult, draft: DraftAnswer, bundle: EvidenceBundle,
                  registry: SourceRegistry, session_check: Optional[Callable[[], bool]] = None
                  ) -> tuple[CompiledPayload, DeliveryRecord]:
        payload = compile_payload(result, draft, bundle, self.notices)
        cited = tuple(p for p in bundle.passages if any(p.passage_id in v for v in result.citation_map.values()))
        with self._lock:
            try:
                epoch = registry.epoch()
                status = registry.eligible(cited, bundle.retrieval.kb_version)
            except RegistryUnavailable:
                raise DeliveryRefused("registry_unavailable", unavailable=True) from None
            if set(status) != {p.passage_id for p in cited}:
                raise DeliveryRefused("registry_unavailable", unavailable=True)
            if not all(status.values()) or any(p.review_status != "live" for p in cited):
                raise DeliveryRefused("source_ineligible")  # never repaired with stale evidence: A5
            if session_check is not None and not session_check():
                raise DeliveryRefused("stale_session")
            record = DeliveryRecord(result.request_id, payload.digest, result.decision_digest, epoch, time.time())
            if self._sink is not None:
                try:
                    self._sink(record)
                except Exception:  # noqa: BLE001
                    raise DeliveryRefused("audit_unavailable", unavailable=True) from None
            self._records[result.request_id] = record
        return payload, record

    def accept_for_display(self, payload: CompiledPayload) -> bool:
        """UI boundary: only an authorized compiled payload, untampered, is displayable."""
        rec = self._records.get(payload.request_id)
        if rec is None or rec.payload_digest != payload.digest:
            return False
        recomputed = _payload_digest(payload.body())
        return recomputed == rec.payload_digest
