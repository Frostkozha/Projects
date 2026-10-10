"""Deterministic formatter (Integration plan v0.3, sections 5, 6; Verifier spec 16.3).

Generated bodies are the exact verifier-approved sentences, in draft order, each followed by citation
markers for its verifier-approved supporting passages only. Titles, versions, locators and links come from
the authoritative passage records; links point only to this application's evidence endpoint. Bodies are
plain text: the UI renders them with ``textContent``. Nothing is paraphrased.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from contracts.models import DraftAnswer, VerificationResult
from verifier.evidence import EvidenceBundle
from verifier.formatter import DeliveryRefused, check_binding

from .public import Citation, LocatorOut


def evidence_url(passage_id: str, kb_version: str) -> str:
    return f"/v1/evidence/{quote(passage_id, safe='')}?kb_version={quote(kb_version, safe='')}"


def citation(n: int, p, kb_version: str) -> Citation:
    loc = p.locator
    return Citation(citation_id=n, passage_id=p.passage_id, title=p.title, source_version=p.source_version,
                    locator=LocatorOut(kind=loc.kind, start=loc.start, end=loc.end, label=loc.label, anchor=loc.anchor),
                    url=evidence_url(p.passage_id, kb_version))


def compile_generated(result: VerificationResult, draft: DraftAnswer, bundle: EvidenceBundle,
                      notices: dict[str, str]) -> tuple[str, list[str], list[Citation], list]:
    """Return (body, notices, citations, cited passages). Raises DeliveryRefused on any binding mismatch."""
    check_binding(result, draft, bundle)
    by_id = {s.sentence_id: s for s in draft.sentences}
    order = [s.sentence_id for s in draft.sentences]
    final = list(result.final_sentence_ids)
    if not final or any(sid not in by_id or by_id[sid].visibility != "student" for sid in final):
        raise DeliveryRefused("hidden_or_unknown_sentence")
    if sorted(final, key=order.index) != final:
        raise DeliveryRefused("sentence_order_changed")
    if " ".join(by_id[sid].text for sid in final) != result.verified_text:
        raise DeliveryRefused("verified_text_mismatch")
    passages = {p.passage_id: p for p in bundle.passages}
    numbers: dict[str, int] = {}
    cites: list[Citation] = []
    cited = []
    parts = []
    kb = bundle.retrieval.kb_version
    for sid in final:
        nums = []
        for pid in result.citation_map.get(sid, ()):
            if pid not in passages or pid not in by_id[sid].cites:
                raise DeliveryRefused("citation_not_supplied")
            if pid not in numbers:
                numbers[pid] = len(numbers) + 1
                cites.append(citation(numbers[pid], passages[pid], kb))
                cited.append(passages[pid])
            nums.append(numbers[pid])
        parts.append(by_id[sid].text + "".join(f" [{n}]" for n in nums))
    chosen = []
    if result.disposition == "trimmed":
        chosen.append(notices["partial"])
    if result.source_conflict:
        chosen.append(notices["conflict"])
    if result.partial_support and result.disposition != "trimmed" and not result.source_conflict:
        chosen.append(notices["coverage_unknown"])
    chosen.append(notices["ai_notice"])
    return " ".join(parts), chosen, cites, cited


def item_citations(passages: list, kb_version: str) -> list[Citation]:
    return [citation(i + 1, p, kb_version) for i, p in enumerate(passages[:5])]


QUIZ_INSTRUCTION = "Answer with the letter of one option (A to E)."
TUTOR_INSTRUCTION = ("Write your answer for yourself, then type \"show explanation\" to see the approved explanation "
                     "or \"continue\" for the next question.")


def quiz_feedback(correct: bool, correct_option: str, explanation: str, indicator: Optional[str]) -> str:
    head = "Correct." if correct else f"Not correct. The approved answer is {correct_option}."
    parts = [head, explanation]
    if indicator:
        parts.append(indicator)
    return "\n\n".join(parts)
