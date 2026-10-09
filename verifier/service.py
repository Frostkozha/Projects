"""Typed verify adapter and stage ordering (Verifier spec v0.2, sections 3, 5, 11, 12, 16.2).

verify(VerifyRequest, VerifyContext) -> VerificationResult. Trusted context is built server-side by the
orchestrator; nothing in it comes from a student JSON body. Stage order:

  readiness -> context -> no_evidence -> 0a format -> 0b output policy -> 1 evidence + citations
  -> token limits -> 2 hard facts -> 3 NLI (bounded worker) -> 5 answer level -> trim rechecks
  -> deadline -> audit commit -> result

Any mandatory infrastructure fault returns status error for the whole request and discards partial
approvals. Content failures follow the declared A5/A6/A7 routes. Logs hold codes, ids, scores and
versions only - never request, draft or evidence text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Protocol

from contracts.models import (
    ContentReason,
    DraftSentence,
    InvalidCitation,
    NLIScore,
    OperationalError,
    ResponseCode,
    SentenceChecks,
    SentenceResult,
    VerificationResult,
    VerifyRequest,
)

from . import digests
from .config import Rules, ThresholdProfile, VerifierProfile, load_profile, load_rules, load_thresholds
from .decision import POLICY_REPLY, AnswerDecision, PairOutcome, SentenceEval, apply_nli, decide_answer
from .evidence import EvidenceBundle, SourceRegistry, VerifierFault, current_epoch, resolve_evidence
from .format import Segmenter, validate_draft_format
from .hard_facts import RULES_VERSION, HardFactChecker
from .nli import NLIOutputInvalid, NLIUnavailable, to_scores
from .policy import OutputPolicyAdapter, PolicyUnavailable

log = logging.getLogger("verifier")

PILOT_MODES = ("answer", "tutor", "quiz")
FUTURE_MODES = {"virtual_patient": "virtual_patient", "open_ended": "open_ended", "image": "images"}
SMOKE_PAIR = ("Cells line the alveoli.", "Cells line the alveoli.")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


class AuditSink(Protocol):
    def write(self, record: dict) -> None: ...


class InMemoryVerifierAudit:
    """Development audit sink holding minimal metadata records (no text)."""

    def __init__(self, fail: bool = False):
        self.records: list[dict] = []
        self.fail = fail
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        if self.fail:
            raise OSError("audit store unavailable")
        with self._lock:
            self.records.append(dict(record))


class JsonlVerifierAudit:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


@dataclass(frozen=True)
class VerifyContext:
    """Trusted, server-owned verification context (never parsed from a student body)."""

    request_id: str
    route: str
    tenant_id: str
    course_id: str
    redacted_request: str
    mode: str
    evidence: EvidenceBundle
    registry: Optional[SourceRegistry]
    deadline: Optional[float] = None  # time.monotonic() bound propagated by the caller
    item_id: Optional[str] = None
    item_version: Optional[str] = None
    real_person_context: bool = False
    a4_authorized: bool = False  # gate-authorized educational-only route
    coverage: Literal["full", "partial", "unknown"] = "unknown"  # approved coverage adapter signal only
    versions: dict = field(default_factory=dict)  # policy/prompt/retrieval versions recorded in audit


@dataclass
class Verdict:
    result: VerificationResult
    answer_reasons: tuple[ContentReason, ...] = ()
    epoch: Optional[int] = None


class _Rejected(Exception):
    def __init__(self, decision: AnswerDecision):
        self.decision = decision


class Verifier:
    """Local three-class-NLI verifier. ``backend`` exposes count/count_pair; ``worker`` runs inference."""

    def __init__(self, *, profile: VerifierProfile, thresholds: ThresholdProfile, rules: Rules,
                 segmenter: Segmenter, worker, token_counter, backend_version: str, label_index: dict[str, int],
                 is_fixture_backend: bool, audit: Optional[AuditSink], policy: Optional[OutputPolicyAdapter] = None,
                 weights_sha256: Optional[str] = None, alarm=None):
        self.profile = profile
        self.thresholds = thresholds
        self.rules = rules
        self.segmenter = segmenter
        self.worker = worker
        self.tokens = token_counter
        self.backend_version = backend_version
        self.label_index = label_index
        self.is_fixture_backend = is_fixture_backend
        self.audit = audit
        self.policy = policy
        self.weights_sha256 = weights_sha256
        self.hard_facts = HardFactChecker(rules)
        self.alarm = alarm or (lambda code, request_id: log.error("alarm %s request=%s", code, request_id))
        self._smoke_ok = self._smoke_test()
        self.verifier_version = self._version()
        self.thresholds_version = f"{thresholds.profile_id}+{thresholds.version()}"

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_profile(cls, profile_path: str | Path, *, audit: Optional[AuditSink], fixture_backend=None,
                     policy: Optional[OutputPolicyAdapter] = None) -> "Verifier":
        """Build from a profile. Fixture mode requires an injected fixture backend (unit tests only)."""
        from .format import SpacyModelSegmenter, SpacySentencizer  # noqa: PLC0415
        from .worker import InProcessWorker, SubprocessWorker  # noqa: PLC0415

        profile = load_profile(profile_path)
        thresholds = load_thresholds(profile.path(profile.thresholds_path))
        rules = load_rules(profile.path(profile.rules_dir))
        seg = profile.segmenter
        segmenter = (SpacyModelSegmenter(str(profile.path(seg.model_path)), seg.model_version)
                     if seg.kind == "spacy_model" else SpacySentencizer())
        rt, nli = profile.runtime, profile.nli
        if profile.operating_mode == "fixture":
            if fixture_backend is None:
                raise NLIUnavailable("fixture mode needs an injected fixture backend")
            backend = fixture_backend
            worker = InProcessWorker(backend, rt.queue_capacity, nli.batch_size)
            return cls(profile=profile, thresholds=thresholds, rules=rules, segmenter=segmenter, worker=worker,
                       token_counter=backend, backend_version=backend.version, label_index=backend.label_index,
                       is_fixture_backend=True, audit=audit, policy=policy or OutputPolicyAdapter())
        if fixture_backend is not None:
            raise NLIUnavailable("a fixture backend cannot serve a real-model profile")
        spec = dict(local_path=str(profile.path(nli.local_path)), model_id=nli.model_id, revision=nli.revision,
                    expected_labels=nli.expected_labels, max_pair_tokens=nli.max_pair_tokens,
                    weights_sha256=nli.weights_sha256, torch_threads=nli.torch_threads)
        if rt.worker == "subprocess":
            from .nli import TokenizerCounter  # noqa: PLC0415

            worker = SubprocessWorker(spec, rt.queue_capacity, nli.batch_size)
            counter = TokenizerCounter(spec["local_path"])
            version, labels, sha = worker.version, worker.label_index, nli.weights_sha256
        else:
            from .nli import TransformersNLI  # noqa: PLC0415

            backend = TransformersNLI(**spec)
            worker = InProcessWorker(backend, rt.queue_capacity, nli.batch_size)
            counter, version, labels, sha = backend, backend.version, backend.label_index, backend.weights_sha256
        return cls(profile=profile, thresholds=thresholds, rules=rules, segmenter=segmenter, worker=worker,
                   token_counter=counter, backend_version=version, label_index=labels, is_fixture_backend=False,
                   audit=audit, policy=policy or OutputPolicyAdapter(), weights_sha256=sha)

    def _smoke_test(self) -> bool:
        try:
            out = self.worker.infer([SMOKE_PAIR], time.monotonic() + 60.0)
            to_scores(out, self.label_index, 1)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _version(self) -> str:
        parts = {
            "git_commit": self.profile.git_commit, "model": self.backend_version, "weights": self.weights_sha256,
            "parser": getattr(self.segmenter, "version", None), "rules": self.rules.digests,
            "hard_facts": RULES_VERSION, "thresholds": self.thresholds.model_dump(),
            "policy": getattr(self.policy, "version", None), "nli_limits": self.profile.nli.model_dump(),
        }
        prefix = "fixture-verifier-0.2" if self.is_fixture_backend else "verifier-0.2"
        return f"{prefix}+{digests.sha256_hex(parts)[:16]}"

    # ------------------------------------------------------------------ readiness

    def readiness(self) -> dict:
        """Process alive is not readiness. Fixture adapters never satisfy production readiness."""
        mode = self.profile.operating_mode
        th = self.thresholds
        checks = {
            "model_loaded": self.worker is not None and bool(getattr(self.worker, "healthy", True)),
            "label_mapping": set(self.label_index) == {"contradiction", "entailment", "neutral"},
            "smoke_test_finite": self._smoke_ok,
            "segmenter": self.segmenter is not None,
            "thresholds_compatible": th.model_digest is None or th.model_digest == self.weights_sha256,
            "policy_adapter": self.policy is not None,
            "audit_adapter": self.audit is not None,
            "backend_matches_mode": (mode == "fixture") == self.is_fixture_backend,
            "model_pinned": self.is_fixture_backend or bool(self.profile.nli.revision),
        }
        if mode == "production":
            checks.update({
                "thresholds_release_ready": th.release_ready,
                "rules_reviewed": self.rules.reviewed,
                "segmenter_production_grade": bool(getattr(self.segmenter, "production_grade", False)),
                "supervised_worker": self.profile.runtime.worker == "subprocess",
                "weights_hash_pinned": bool(self.profile.nli.weights_sha256),
            })
        ready = all(checks.values())
        return {"ready": ready, "checks": checks, "operating_mode": mode,
                "student_release_ready": ready and mode == "production",
                "verifier_version": self.verifier_version, "thresholds_version": self.thresholds_version}

    # ------------------------------------------------------------------ entry points

    def verify(self, request: VerifyRequest, ctx: VerifyContext) -> VerificationResult:
        return self.run(request, ctx).result

    def run(self, request: VerifyRequest, ctx: VerifyContext) -> Verdict:
        t0 = time.monotonic()
        deadline = t0 + self.profile.runtime.deadline_seconds
        if ctx.deadline is not None:
            deadline = min(deadline, ctx.deadline)
        ddig = digests.draft_digest(request.draft)
        try:
            edig = digests.evidence_digest(ctx.evidence)
        except Exception:  # noqa: BLE001
            edig = EMPTY_SHA
        state: dict = {"epoch": None}
        try:
            verdict = self._run(request, ctx, deadline, ddig, edig, state)
            if time.monotonic() > deadline:
                raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)
        except VerifierFault as fault:
            verdict = Verdict(self._error_result(request.request_id, ddig, edig, fault.code))
        return self._commit(verdict, ctx, t0, state)

    # ------------------------------------------------------------------ pipeline

    def _ready_or_fault(self) -> None:
        r = self.readiness()
        if r["ready"]:
            return
        c = r["checks"]
        model_keys = ("model_loaded", "label_mapping", "smoke_test_finite", "model_pinned")
        if not all(c[k] for k in model_keys):
            raise VerifierFault(OperationalError.MODEL_UNAVAILABLE)
        if not c["policy_adapter"]:
            raise VerifierFault(OperationalError.POLICY_UNAVAILABLE)
        if not c["audit_adapter"]:
            raise VerifierFault(OperationalError.AUDIT_UNAVAILABLE)
        raise VerifierFault(OperationalError.PROFILE_MISMATCH)

    def _check_context(self, request: VerifyRequest, ctx: VerifyContext) -> None:
        if ctx.request_id != request.request_id or ctx.route != "retrieve":
            raise VerifierFault(OperationalError.CONTEXT_MISMATCH)
        if ctx.mode not in PILOT_MODES:
            # future adapters (case, open-ended, image) are disabled: never a permissive substitute
            raise VerifierFault(OperationalError.CONTEXT_MISMATCH)
        if ctx.a4_authorized and not self.profile.features.get("a4_educational", False):
            raise VerifierFault(OperationalError.CONTEXT_MISMATCH)
        if ctx.coverage not in ("full", "partial", "unknown"):
            raise VerifierFault(OperationalError.CONTEXT_MISMATCH)

    @staticmethod
    def _tick(deadline: float) -> None:
        if time.monotonic() > deadline:
            raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)

    def _run(self, request: VerifyRequest, ctx: VerifyContext, deadline: float, ddig: str, edig: str,
             state: dict) -> Verdict:
        self._ready_or_fault()
        self._check_context(request, ctx)
        draft = request.draft
        rid = request.request_id
        if draft.status == "no_evidence":
            return self._rejected(rid, ddig, edig, AnswerDecision(
                "rejected", "fallback", ResponseCode.A5, "no_source",
                answer_reasons=[ContentReason.NO_VISIBLE_SUPPORT]), [])

        # ---- layer 0a: format and segmentation (whole draft)
        cues = {s.sentence_id: self.hard_facts.cues(s.text) for s in draft.sentences}
        allow = set(self.rules.allowlist)
        evals = [SentenceEval(sentence=s, index=i, filler=(s.text in allow and not s.cites), cues=cues[s.sentence_id])
                 for i, s in enumerate(draft.sentences)]
        problems = validate_draft_format(draft, self.segmenter)
        if any(problems.values()):
            for ev in evals:
                if problems[ev.sid]:
                    ev.checks["format"] = "fail"
                    ev.reasons.append(ContentReason.INVALID_DRAFT)
            return self._rejected(rid, ddig, edig, _a5(ContentReason.INVALID_DRAFT), evals, checked=False)
        self._tick(deadline)

        # ---- layer 0b: contextual output policy (request + trusted context + draft)
        if self.policy is None:
            raise VerifierFault(OperationalError.POLICY_UNAVAILABLE)
        texts = [s.text for s in draft.sentences]
        try:
            outcome = self.policy.evaluate(ctx.redacted_request, texts, ctx.real_person_context)
        except PolicyUnavailable:
            raise VerifierFault(OperationalError.POLICY_UNAVAILABLE) from None
        if outcome.crisis is not None:  # current request only; a generated fictional crisis cannot alert
            key = "emergency" if outcome.crisis == "emergency" else "self_harm"
            return self._rejected(rid, ddig, edig, AnswerDecision("rejected", "escalate", ResponseCode.A7, key),
                                  evals, checked=False)
        if outcome.violations:
            dec = AnswerDecision("rejected", "refuse", ResponseCode.A6, POLICY_REPLY[outcome.violations[0]],
                                 answer_reasons=[ContentReason.POLICY_VIOLATION])
            return self._rejected(rid, ddig, edig, dec, evals, checked=False,
                                  violations=tuple(dict.fromkeys(outcome.violations)))
        if outcome.uncertain:
            return self._rejected(rid, ddig, edig, _a5(ContentReason.OUTPUT_POLICY_UNCERTAIN), evals, checked=False)
        self._tick(deadline)

        # ---- layer 1: trusted evidence + citation integrity
        if ctx.registry is None:
            raise VerifierFault(OperationalError.REGISTRY_UNAVAILABLE)
        supplied, ineligible = resolve_evidence(ctx.evidence, rid, ctx.tenant_id, ctx.course_id, ctx.registry)
        state["epoch"] = current_epoch(ctx.registry)
        invalid: list[InvalidCitation] = []
        for ev in evals:
            if ev.filler:
                continue
            s = ev.sentence
            bad = False
            if not s.cites:
                ev.reasons.append(ContentReason.MISSING_CITATION)
                invalid.append(InvalidCitation(sentence_id=s.sentence_id, passage_id=None,
                                               reason=ContentReason.MISSING_CITATION))
                bad = True
            for pid in s.cites:
                if pid not in supplied:
                    reason = ContentReason.UNKNOWN_CITATION
                elif pid in ineligible:
                    reason = ContentReason.SOURCE_INELIGIBLE
                else:
                    continue
                if reason not in ev.reasons:
                    ev.reasons.append(reason)
                invalid.append(InvalidCitation(sentence_id=s.sentence_id, passage_id=pid, reason=reason))
                bad = True
            ev.checks["citation"] = "fail" if bad else "pass"
        self._tick(deadline)

        # ---- NLI token limits (verifier tokenizer, truncation disabled, no windowing)
        lim = self.profile.nli
        for p in supplied.values():
            if self.tokens.count(p.text) > lim.max_premise_tokens:
                raise VerifierFault(OperationalError.EVIDENCE_LIMIT)
        live = [ev for ev in evals if not ev.filler and not ev.reasons]
        for ev in live:
            if self.tokens.count(ev.sentence.text) > lim.max_hypothesis_tokens:
                ev.reasons.append(ContentReason.INVALID_DRAFT)
                ev.checks["format"] = "fail"
                return self._rejected(rid, ddig, edig, _a5(ContentReason.INVALID_DRAFT), evals, invalid=invalid)

        # ---- layer 2: hard facts (any mismatch rejects; unresolved pairs cannot support)
        for ev in live:
            for pid in ev.sentence.cites:
                ev.pairs[pid] = PairOutcome(pid, self.hard_facts.check(ev.sentence.text, supplied[pid].text))
            statuses = [p.hard_fact.status for p in ev.pairs.values()]
            if "fail" in statuses:
                ev.checks["hard_fact"] = "fail"
                ev.reasons.append(ContentReason.HARD_FACT_MISMATCH)
            elif all(st == "unresolved" for st in statuses):
                ev.checks["hard_fact"] = "fail"
                ev.reasons.append(ContentReason.HARD_FACT_UNRESOLVED)
            else:
                ev.checks["hard_fact"] = "pass"
        self._tick(deadline)

        # ---- layer 3: three-class NLI, premise = passage, hypothesis = sentence
        jobs = [(ev, pid) for ev in live if not ev.reasons for pid in ev.sentence.cites]
        if len(jobs) > self.profile.runtime.max_pairs:
            raise VerifierFault(OperationalError.EVIDENCE_LIMIT)
        pairs = [(supplied[pid].text, ev.sentence.text) for ev, pid in jobs]
        for premise, hypothesis in pairs:
            if self.tokens.count_pair(premise, hypothesis) > lim.max_pair_tokens:
                raise VerifierFault(OperationalError.EVIDENCE_LIMIT)
        if jobs:
            logits = self.worker.infer(pairs, deadline)
            try:
                scores = to_scores(logits, self.label_index, len(jobs))
            except NLIOutputInvalid:
                raise VerifierFault(OperationalError.WORKER_FAILED) from None
            for (ev, pid), sc in zip(jobs, scores):
                ev.pairs[pid].scores = sc
            for ev in {id(ev): ev for ev, _ in jobs}.values():
                apply_nli(ev, self.thresholds)
        self._tick(deadline)
        for ev in live:
            for pid in ev.dropped:
                invalid.append(InvalidCitation(sentence_id=ev.sid, passage_id=pid,
                                               reason=ContentReason.IRRELEVANT_CITATION))

        # ---- layer 5: answer level
        shown = set(supplied)
        conflicts = [tuple(c.passage_ids) for c in ctx.evidence.retrieval.conflicts if set(c.passage_ids) <= shown]
        decision = decide_answer(evals, coverage=ctx.coverage, a4=ctx.a4_authorized, conflicts=conflicts,
                                 dependent_starts=self.rules.cues["dependent_starts"])
        if decision.status == "rejected":
            return self._rejected(rid, ddig, edig, decision, evals, invalid=invalid)
        if decision.trimmed:
            # recheck output policy and live eligibility on the exact remainder (no rewording)
            kept_texts = [e.sentence.text for e in decision.kept]
            try:
                again = self.policy.evaluate(ctx.redacted_request, kept_texts, ctx.real_person_context)
            except PolicyUnavailable:
                raise VerifierFault(OperationalError.POLICY_UNAVAILABLE) from None
            if not again.clean:
                return self._rejected(rid, ddig, edig, _a5(ContentReason.OUTPUT_POLICY_UNCERTAIN), evals,
                                      invalid=invalid)
            _, still_bad = resolve_evidence(ctx.evidence, rid, ctx.tenant_id, ctx.course_id, ctx.registry)
            if still_bad & {pid for e in decision.kept for pid in e.support}:
                return self._rejected(rid, ddig, edig, _a5(ContentReason.SOURCE_INELIGIBLE), evals, invalid=invalid)
        return self._approved(rid, ddig, edig, decision, evals, invalid)

    # ------------------------------------------------------------------ result builders

    def _sentence_results(self, evals: list[SentenceEval], kept_ids: set[str], checked: bool) -> tuple:
        out = []
        for ev in evals:
            scores = {pid: NLIScore(**p.scores) for pid, p in ev.pairs.items() if p.scores is not None}
            out.append(SentenceResult(
                sentence_id=ev.sid, decision="keep" if ev.sid in kept_ids else "delete",
                reason_codes=tuple(dict.fromkeys(ev.reasons)), factual_checked=checked and not ev.filler,
                checks=SentenceChecks(**ev.checks), nli_scores=scores))
        return tuple(out)

    def _counts(self, evals, checked: bool) -> tuple[int, int, bool]:
        if not checked:
            return 0, 0, False
        assessed = [e for e in evals if not e.filler and e.checks["citation"] != "not_run"]
        failed = [e for e in assessed if e.failed]
        near = any(e.cues.high_risk for e in failed)
        return len(assessed) - len(failed), len(failed), near

    def _finish(self, fields: dict) -> VerificationResult:
        fields["decision_digest"] = digests.decision_digest(fields)
        return VerificationResult.model_validate(fields)

    def _base(self, rid, ddig, edig) -> dict:
        return dict(schema_version="verification-result-0.2", request_id=rid, verifier_version=self.verifier_version,
                    thresholds_version=self.thresholds_version, evidence_digest=edig, draft_digest=ddig)

    def _rejected(self, rid, ddig, edig, decision: AnswerDecision, evals, *, checked: bool = True,
                  invalid: Optional[list] = None, violations: tuple = ()) -> Verdict:
        supported, unsupported, near = self._counts(evals, checked)
        fields = self._base(rid, ddig, edig) | dict(
            status="rejected", disposition=decision.disposition, response_code=decision.response_code,
            reply_key=decision.reply_key, error_code=None, verified_text="", final_sentence_ids=(),
            internal_sentence_ids=(), citation_map={}, sentence_results=self._sentence_results(evals, set(), checked),
            internal_sentences=(), supported_claims=supported, unsupported_claims=unsupported,
            invalid_citations=tuple(invalid or ()), partial_support=False, source_conflict=False,
            output_policy_violations=violations, near_miss=near)
        return Verdict(self._finish(fields), tuple(decision.answer_reasons))

    def _approved(self, rid, ddig, edig, decision: AnswerDecision, evals, invalid) -> Verdict:
        kept = decision.kept
        visible = [e for e in kept if e.visible]
        internal: list[DraftSentence] = [e.sentence for e in kept if not e.visible]
        supported, unsupported, near = self._counts(evals, True)
        fields = self._base(rid, ddig, edig) | dict(
            status="approved", disposition=decision.disposition, response_code=decision.response_code,
            reply_key=None, error_code=None, verified_text=" ".join(e.sentence.text for e in visible),
            final_sentence_ids=tuple(e.sid for e in visible),
            internal_sentence_ids=tuple(s.sentence_id for s in internal),
            citation_map={e.sid: tuple(e.support) for e in visible if not e.filler},
            sentence_results=self._sentence_results(evals, {e.sid for e in kept}, True),
            internal_sentences=tuple(internal), supported_claims=supported, unsupported_claims=unsupported,
            invalid_citations=tuple(invalid), partial_support=decision.partial_support,
            source_conflict=decision.source_conflict, output_policy_violations=(), near_miss=near)
        return Verdict(self._finish(fields), tuple(decision.answer_reasons))

    def _error_result(self, rid, ddig, edig, code: OperationalError) -> VerificationResult:
        fields = self._base(rid, ddig, edig) | dict(
            status="error", disposition="unavailable", response_code=None, reply_key="service_unavailable",
            error_code=code, verified_text="", final_sentence_ids=(), internal_sentence_ids=(), citation_map={},
            sentence_results=(), internal_sentences=(), supported_claims=0, unsupported_claims=0,
            invalid_citations=(), partial_support=False, source_conflict=False, output_policy_violations=(),
            near_miss=False)
        return self._finish(fields)

    # ------------------------------------------------------------------ audit

    def _commit(self, verdict: Verdict, ctx: VerifyContext, t0: float, state: dict) -> Verdict:
        """Required decision metadata commits before generated delivery. Audit failure -> unavailable,
        except that a fixed A7 message stays deliverable (local alarm raised)."""
        r = verdict.result
        verdict.epoch = state.get("epoch")
        record = {
            "event": "verification", "request_id": r.request_id, "status": r.status, "disposition": r.disposition,
            "response_code": r.response_code.value if r.response_code else None,
            "error_code": r.error_code.value if r.error_code else None,
            "answer_reasons": [x.value for x in verdict.answer_reasons],
            "sentences": [{"id": s.sentence_id, "decision": s.decision, "reasons": [x.value for x in s.reason_codes],
                           "checks": s.checks.model_dump(),
                           "scores": {k: v.model_dump() for k, v in s.nli_scores.items()}}
                          for s in r.sentence_results],
            "supported_claims": r.supported_claims, "unsupported_claims": r.unsupported_claims,
            "invalid_citations": [[c.sentence_id, c.passage_id, c.reason.value] for c in r.invalid_citations],
            "near_miss": r.near_miss, "policy_violations": list(r.output_policy_violations),
            "verifier_version": r.verifier_version, "thresholds_version": r.thresholds_version,
            "draft_digest": r.draft_digest, "evidence_digest": r.evidence_digest,
            "decision_digest": r.decision_digest, "registry_epoch": verdict.epoch, "mode": ctx.mode,
            "versions": dict(ctx.versions), "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }
        try:
            if self.audit is None:
                raise OSError("no audit sink")
            self.audit.write(record)
        except Exception:  # noqa: BLE001
            self.alarm("verifier_audit_failed", r.request_id)
            if r.response_code == ResponseCode.A7:
                return verdict
            return Verdict(self._error_result(r.request_id, r.draft_digest, r.evidence_digest,
                                              OperationalError.AUDIT_UNAVAILABLE), epoch=verdict.epoch)
        return verdict


def _a5(reason: ContentReason) -> AnswerDecision:
    return AnswerDecision("rejected", "fallback", ResponseCode.A5, "no_source", answer_reasons=[reason])
