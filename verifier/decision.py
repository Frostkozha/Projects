"""Thresholds, sentence decisions, trimming, coverage and conflicts (Verifier spec v0.2, sections 5.4, 5.6).

Pair priority: contradiction >= Tc wins; else entailment >= T_hi supports; else entailment < T_lo is
unsupported; the remaining band (including equality at T_lo) is uncertain and rejected. Scores never
average across passages. A sentence needs at least one cited passage that alone supports it with
hard-fact checks passing, and no checked cited pair may be contradictory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from contracts.models import ContentReason, DraftSentence, ResponseCode

from .config import ThresholdProfile
from .hard_facts import Cues, HardFactResult

PairLabel = Literal["contradiction", "supported", "unsupported", "uncertain"]


def classify_pair(scores: dict[str, float], th: ThresholdProfile) -> PairLabel:
    if scores["contradiction"] >= th.Tc:
        return "contradiction"
    if scores["entailment"] >= th.T_hi:
        return "supported"
    if scores["entailment"] < th.T_lo:
        return "unsupported"
    return "uncertain"


@dataclass
class PairOutcome:
    passage_id: str
    hard_fact: HardFactResult
    scores: Optional[dict[str, float]] = None
    label: Optional[PairLabel] = None


@dataclass
class SentenceEval:
    sentence: DraftSentence
    index: int
    filler: bool
    cues: Cues
    reasons: list[ContentReason] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=lambda: {"format": "pass", "citation": "not_run",
                                                            "hard_fact": "not_run", "entailment": "not_run"})
    pairs: dict[str, PairOutcome] = field(default_factory=dict)
    support: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    trimmed: bool = False

    @property
    def sid(self) -> str:
        return self.sentence.sentence_id

    @property
    def visible(self) -> bool:
        return self.sentence.visibility == "student"

    @property
    def failed(self) -> bool:
        """Rejected checked sentence (allowlisted filler is never a checked sentence)."""
        return not self.filler and bool(self.reasons)


def apply_nli(ev: SentenceEval, th: ThresholdProfile) -> None:
    """Combine per-pair hard-fact and NLI outcomes into the sentence decision (no averaging)."""
    labels = {pid: classify_pair(p.scores, th) for pid, p in ev.pairs.items() if p.scores is not None}
    for pid, lab in labels.items():
        ev.pairs[pid].label = lab
    ev.checks["entailment"] = "fail"
    if any(lab == "contradiction" for lab in labels.values()):
        ev.reasons.append(ContentReason.CONTRADICTED)
        return
    if any(lab == "supported" and ev.pairs[pid].hard_fact.status == "fail" for pid, lab in labels.items()):
        # a top NLI score cannot rescue a deterministic mismatch (lexical false support)
        ev.checks["hard_fact"] = "fail"
        ev.reasons.append(ContentReason.HARD_FACT_MISMATCH)
        return
    support = [pid for pid in ev.sentence.cites
               if labels.get(pid) == "supported" and ev.pairs[pid].hard_fact.status in ("pass", "not_applicable")]
    if not support:
        unresolved_supported = any(labels.get(pid) == "supported" and p.hard_fact.status == "unresolved"
                                   for pid, p in ev.pairs.items())
        if unresolved_supported:
            ev.checks["hard_fact"] = "fail"
            ev.reasons.append(ContentReason.HARD_FACT_UNRESOLVED)
        elif any(lab == "uncertain" for lab in labels.values()):
            ev.reasons.append(ContentReason.UNCERTAIN_SUPPORT)
        else:
            ev.reasons.append(ContentReason.NOT_ENTAILED)
        return
    ev.checks["entailment"] = "pass"
    ev.support = support
    ev.dropped = [pid for pid in ev.sentence.cites if pid not in support]  # neutral extras: logged, not shown


# ----------------------------------------------------------------------------- answer level


def _starts_with(text: str, phrases) -> bool:
    low = text.lower().lstrip("\"'(")
    return any(re.match(rf"{re.escape(p)}\b", low) for p in phrases)


def trim_problem(evals: list[SentenceEval], bad: SentenceEval, dependent_starts) -> Optional[str]:
    """Deterministic conservative trimming validator. Returns a problem code, or None when safe."""
    if not bad.visible:
        return "hidden_sentence"
    visible = [e for e in evals if e.visible]
    if visible[-1] is not bad:
        return "not_last_visible"
    if any(bad.sid in e.sentence.depends_on for e in evals if e is not bad):
        return "has_dependents"
    c = bad.cues
    if c.high_risk or c.condition or c.exception or c.negation or c.number or c.unit:
        return "high_risk_cue"
    if c.question or bad.sentence.kind_hint == "question" or bad.sentence.text.rstrip().endswith("?"):
        return "question"
    if c.comparative or c.sequence or c.causal:
        return "comparison_sequence_or_causal"
    kept = [e for e in evals if e is not bad]
    for e in kept:
        if e.sentence.depends_on and not set(e.sentence.depends_on) <= {k.sid for k in kept}:
            return "kept_depends_on_deleted"
        if e.visible and not e.filler and _starts_with(e.sentence.text, dependent_starts):
            return "kept_not_self_contained"
    if _starts_with(bad.sentence.text, dependent_starts):
        return "deleted_continues_discourse"  # dependency scope uncertain: disable trimming
    return None


@dataclass
class AnswerDecision:
    status: Literal["approved", "rejected"]
    disposition: str
    response_code: ResponseCode
    reply_key: Optional[str] = None
    kept: list[SentenceEval] = field(default_factory=list)
    answer_reasons: list[ContentReason] = field(default_factory=list)
    partial_support: bool = False
    source_conflict: bool = False
    trimmed: bool = False


def _reject(*reasons: ContentReason) -> AnswerDecision:
    return AnswerDecision("rejected", "fallback", ResponseCode.A5, "no_source", answer_reasons=list(reasons))


def decide_answer(evals: list[SentenceEval], *, coverage: str, a4: bool, conflicts: list[tuple[str, ...]],
                  dependent_starts) -> AnswerDecision:
    """FA-07 counting, safe trimming, hidden-answer, visible-support, conflict and coverage rules."""
    failed = [e for e in evals if e.failed]
    if any(not e.visible for e in failed):
        for e in failed:
            if not e.visible:
                e.reasons.append(ContentReason.HIDDEN_ANSWER_FAILED)
        return _reject(ContentReason.HIDDEN_ANSWER_FAILED)
    if len(failed) >= 2:
        return _reject(*sorted({r for e in failed for r in e.reasons}, key=lambda r: r.value))
    trimmed = False
    if len(failed) == 1:
        bad = failed[0]
        if trim_problem(evals, bad, dependent_starts) is not None:
            bad.reasons.append(ContentReason.UNSAFE_TRIM)
            return _reject(ContentReason.UNSAFE_TRIM)
        bad.trimmed = trimmed = True
    kept = [e for e in evals if not e.failed]
    if not any(e.visible and not e.filler for e in kept):
        return _reject(ContentReason.NO_VISIBLE_SUPPORT)

    source_conflict = False
    if conflicts:
        supported_visible = {pid for e in kept if e.visible and not e.filler for pid in e.support}
        for sides in conflicts:
            if not set(sides) <= supported_visible:
                return _reject(ContentReason.CONFLICT_INCOMPLETE)
        source_conflict = True

    reasons: list[ContentReason] = []
    if coverage != "full":
        reasons.append(ContentReason.COVERAGE_PARTIAL)
    partial = coverage != "full" or trimmed or source_conflict
    if a4:
        code = ResponseCode.A4
    elif not partial:
        code = ResponseCode.A1
    else:
        code = ResponseCode.A2
    return AnswerDecision("approved", "trimmed" if trimmed else "pass", code, kept=kept, answer_reasons=reasons,
                          partial_support=partial, source_conflict=source_conflict, trimmed=trimmed)


POLICY_REPLY = {"real_person_advice": "real_person", "privacy_disclosure": "private_data",
                "assessed_work": "assessed_work", "instruction_override": "instruction_override",
                "internal_content": "private_data"}
