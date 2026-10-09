"""Compose gate operations and enforce the deadline (spec sections 5, 8, 11.3, 13)."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from pydantic import ValidationError

from .audit import AlertDispatcher, AlertEvent, AuditRecord, AuditSink, InMemoryAlertQueue, InMemoryAuditSink, OperationalAlarms
from .config import GateConfig, LibraryRegistry
from .decide import DecisionInputs, decide, propose_session_update
from .encoder import Encoder, EncoderError, FixtureTokenizer, Tokenizer, build_canonical_text
from .heads import Bundle, BundleError, FixtureScorer
from .normalize import normalize_text
from .policy_rules import RuleEngine
from .privacy import PrivacyError, Redactor
from .schema import (
    LIBRARY_IDS,
    DecisionFlags,
    EmbeddingRef,
    ErrorCode,
    GateContext,
    GateError,
    GateRequest,
    GateResult,
    InferenceStatus,
    Prediction,
    Route,
    SessionUpdate,
    Versions,
    utcnow,
)
from .session import SessionContextView, session_context


class DeadlineExceeded(RuntimeError):
    pass


class InferenceFailed(RuntimeError):
    pass


# ----------------------------------------------------------------------------- worker


class _Job:
    __slots__ = ("fn", "done", "result", "failed", "cancelled", "started")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result = None
        self.failed = False
        self.cancelled = False
        self.started: Optional[float] = None


class InferenceWorker:
    """One dedicated inference thread with a bounded queue.

    Timed-out jobs are cancelled before start or have their late result discarded. A job running
    longer than ``stuck_seconds`` causes the worker thread to be replaced (bounded restarts);
    after ``max_restarts`` the worker reports unhealthy so readiness turns false.
    Python threads cannot be killed; a production deployment should run inference in a separate
    process that can be terminated (see README, known limitations).
    """

    def __init__(self, capacity: int = 32, stuck_seconds: float = 10.0, max_restarts: int = 3):
        self._q: queue.Queue[_Job] = queue.Queue(maxsize=capacity)
        self._stuck_seconds = stuck_seconds
        self._max_restarts = max_restarts
        self.restarts = 0
        self._generation = 0
        self._current: Optional[_Job] = None
        self._lock = threading.Lock()
        self._spawn()

    @property
    def healthy(self) -> bool:
        return self.restarts <= self._max_restarts

    def _spawn(self) -> None:
        self._generation += 1
        t = threading.Thread(target=self._loop, args=(self._generation,), daemon=True, name=f"gate-infer-{self._generation}")
        t.start()

    def _loop(self, generation: int) -> None:
        while generation == self._generation:
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job.cancelled:
                continue
            with self._lock:
                self._current = job
                job.started = time.monotonic()
            try:
                job.result = job.fn()
            except Exception:  # error details may contain inputs; never propagate them
                job.failed = True
            finally:
                with self._lock:
                    self._current = None
                job.done.set()

    def _check_stuck(self) -> None:
        with self._lock:
            cur = self._current
            stuck = cur is not None and cur.started is not None and time.monotonic() - cur.started > self._stuck_seconds
            if stuck:
                cur.cancelled = True
                self._current = None
        if stuck:
            self.restarts += 1
            self._spawn()  # the old thread exits after its current job (generation changed)

    def run(self, fn: Callable[[], object], timeout: float):
        self._check_stuck()
        if not self.healthy:
            raise InferenceFailed("worker_unhealthy")
        job = _Job(fn)
        try:
            self._q.put_nowait(job)
        except queue.Full:
            raise GateError(ErrorCode.CAPACITY_EXCEEDED) from None
        if not job.done.wait(max(0.0, timeout)):
            job.cancelled = True  # late result is never read
            raise DeadlineExceeded("deadline")
        if job.failed or job.cancelled:
            raise InferenceFailed("inference_failed")
        return job.result


# ----------------------------------------------------------------------------- readiness


@dataclass
class Readiness:
    ready: bool
    status: str  # "ready" | "fixture_only" | "not_ready"
    operating_mode: str
    reasons: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- service


class GateService:
    def __init__(
        self,
        config: GateConfig,
        registry: LibraryRegistry,
        *,
        tokenizer: Optional[Tokenizer] = None,
        encoder: Optional[Encoder] = None,
        bundle: Optional[Bundle] = None,
        fixture_scorer: Optional[FixtureScorer] = None,
        redactor: Optional[Redactor] = None,
        rules: Optional[RuleEngine] = None,
        audit: Optional[AuditSink] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        alarms: Optional[OperationalAlarms] = None,
        worker: Optional[InferenceWorker] = None,
        load_errors: Optional[list[str]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.registry = registry
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.bundle = bundle
        self.fixture_scorer = fixture_scorer
        self.redactor = redactor or Redactor()
        self.rules = rules or RuleEngine()
        self.audit = audit if audit is not None else InMemoryAuditSink()
        self.alarms = alarms or OperationalAlarms()
        self.alerts = alert_dispatcher or AlertDispatcher(
            InMemoryAlertQueue(), self.alarms, self.audit, config.alerts.max_attempts, config.alerts.base_backoff_seconds
        )
        self.worker = worker or InferenceWorker(config.runtime.queue_capacity, config.runtime.stuck_worker_seconds,
                                                config.runtime.max_worker_restarts)
        self._load_errors = list(load_errors or [])
        self._monotonic = monotonic

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, config: GateConfig, registry: LibraryRegistry, **kw) -> "GateService":
        """Load the real encoder and bundle from local files. Failures make readiness false."""
        from .encoder import E5Encoder  # noqa: PLC0415
        from .heads import load_bundle  # noqa: PLC0415
        from .privacy import SpacyNameDetector  # noqa: PLC0415

        errors: list[str] = []
        encoder = bundle = tokenizer = None
        redactor = kw.pop("redactor", None)
        if config.operating_mode == "fixture":
            errors.append("fixture_mode_requires_fixture_scorer")
        else:
            enc = config.encoder
            try:
                encoder = E5Encoder(enc.local_path or "", enc.model_id, enc.revision or "", enc.tokenizer_revision or "",
                                    enc.prefix, enc.dimension, enc.max_tokens, enc.torch_threads)
                tokenizer = encoder.tokenizer
            except EncoderError as exc:
                errors.append(f"encoder:{exc}")
            if encoder is not None:
                try:
                    bundle = load_bundle(config.bundle_path or "", operating_mode=config.operating_mode,
                                         config_sha256=config.config_sha256(),
                                         preprocessing_fingerprint=encoder.preprocessing.fingerprint(),
                                         dimension=enc.dimension)
                except BundleError as exc:
                    errors.append(f"bundle:{exc.reason}")
            if redactor is None and config.privacy.name_detector == "spacy":
                try:
                    redactor = Redactor(SpacyNameDetector(config.privacy.spacy_model_path or "",
                                                          config.privacy.spacy_model_version))
                except PrivacyError as exc:
                    errors.append(f"privacy:{exc}")
        return cls(config, registry, tokenizer=tokenizer, encoder=encoder, bundle=bundle, redactor=redactor,
                   load_errors=errors, **kw)

    # ------------------------------------------------------------------ readiness

    def readiness(self) -> Readiness:
        cfg = self.config
        reasons = list(self._load_errors)
        if self.tokenizer is None:
            reasons.append("tokenizer_missing")
        if not self.worker.healthy:
            reasons.append("worker_unhealthy")
        if cfg.operating_mode == "fixture":
            if self.fixture_scorer is None and self.bundle is None:
                reasons.append("fixture_scorer_missing")
            reasons = [r for r in reasons if r != "fixture_mode_requires_fixture_scorer" or self.fixture_scorer is None]
            if reasons:
                return Readiness(False, "not_ready", cfg.operating_mode, reasons)
            return Readiness(True, "fixture_only", cfg.operating_mode, [])
        # development / production: real artifacts only, never fixture scores
        if self.fixture_scorer is not None:
            reasons.append("fixture_scorer_not_allowed")
        if self.encoder is None or getattr(self.encoder, "is_fixture", True):
            reasons.append("real_encoder_missing")
        if self.bundle is None:
            reasons.append("bundle_missing")
        elif self.bundle.is_fixture:
            reasons.append("fixture_bundle_not_allowed")
        elif self.bundle.manifest.get("config_sha256") != cfg.config_sha256():
            reasons.append("config_checksum_mismatch")
        if cfg.operating_mode == "production":
            reasons.extend(cfg.production_problems())
        reasons = sorted(set(reasons))
        return Readiness(not reasons, "ready" if not reasons else "not_ready", cfg.operating_mode, reasons)

    # ------------------------------------------------------------------ helpers

    def _versions(self, ctx: GateContext) -> Versions:
        bundle_v = self.bundle.version if self.bundle is not None else ("fixture" if self.fixture_scorer else None)
        enc_rev = None
        if self.encoder is not None:
            enc_rev = self.encoder.preprocessing.revision
        elif self.fixture_scorer is not None:
            enc_rev = "fixture"
        return Versions(bundle=bundle_v, encoder_revision=enc_rev, policy=self.config.policy_version,
                        config_sha256=self.config.config_sha256(), library_registry=self.registry.registry_version,
                        kb=ctx.kb_version)

    def _infer(self, model_text: str, canonical: str, enabled_libs: tuple[str, ...]):
        if self.fixture_scorer is not None and self.config.operating_mode == "fixture":
            raw = dict(self.fixture_scorer(canonical))
            vec = None
        else:
            if self.encoder is None or self.bundle is None:
                raise InferenceFailed("artifacts_missing")
            vec = self.encoder.encode([model_text])[0]
            if vec.shape != (self.config.encoder.dimension,) or not np.all(np.isfinite(vec)):
                raise InferenceFailed("bad_embedding")
            raw = self.bundle.heads.predict_mandatory(vec)
            try:
                raw["libraries"] = self.bundle.heads.predict_libraries(vec, enabled_libs)
            except Exception:  # optional heads: failure removes hints only
                raw["libraries"] = {lib: None for lib in LIBRARY_IDS}
        libs = dict(raw.get("libraries") or {})
        # disabled/unknown labels are not negative predictions: they are null
        raw["libraries"] = {lib: (libs.get(lib) if lib in enabled_libs else None) for lib in LIBRARY_IDS}
        return raw, vec

    def _count(self, text: str, special: bool) -> int:
        try:
            return int(self.tokenizer.count(text, special))  # type: ignore[union-attr]
        except Exception:
            raise PrivacyError("tokenizer_failed") from None  # treated as mandatory preprocessing failure

    @staticmethod
    def session_ref(ctx: GateContext) -> Optional[str]:
        if ctx.session is None:
            return None
        return hashlib.sha256(f"session:{ctx.session.session_id}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ main entry

    def evaluate(self, request: GateRequest, ctx: GateContext, request_id: str) -> GateResult:
        cfg = self.config
        start = self._monotonic()
        deadline = start + cfg.runtime.deadline_seconds
        versions = self._versions(ctx)
        no_flags = DecisionFlags(pii_detected=False, pii_redacted=False, context_used=False)
        empty_view = SessionContextView(False, None, None, None)

        def finish(inputs: DecisionInputs, redacted: Optional[str], categories=(), rules=None, vec=None,
                   canonical: Optional[str] = None) -> GateResult:
            decision = decide(inputs, cfg)
            if decision.route != Route.escalate and self._monotonic() > deadline:
                decision = decide(_with_failure(inputs, "inference"), cfg)
            decision = self._audit_decision(decision, inputs, ctx, categories, start, deadline)
            if decision.route == Route.escalate:
                category = "emergency" if decision.reply_key == "emergency" else "self_harm"
                self.alerts.dispatch(AlertEvent.new(request_id, self.session_ref(ctx), category))
            ref = None
            if vec is not None and canonical is not None and self.encoder is not None:
                ref = EmbeddingRef(vec, self.encoder.preprocessing.fingerprint(),
                                   hashlib.sha256((cfg.encoder.prefix + canonical).encode()).hexdigest())
            redacted_out = redacted if decision.route == Route.retrieve else None
            return GateResult(decision, redacted_out, ref, propose_session_update(decision, ctx), rules, categories)

        def base_inputs(**kw) -> DecisionInputs:
            values = dict(request_id=request_id, redacted_text=None, rules=None, prediction=None,
                          inference_status=InferenceStatus.not_run, context=ctx, session_view=empty_view,
                          requested_mode=request.requested_mode, flags=no_flags, versions=versions, failure=None)
            values.update(kw)
            return DecisionInputs(**values)

        # Row 1 and readiness come before any text processing.
        if ctx.deployment_state != "active":
            return finish(base_inputs(), None)
        if not self.readiness().ready:
            return finish(base_inputs(failure="preprocess"), None)

        # --- validation and canonical text (4xx errors propagate as GateError)
        text = normalize_text(request.text)
        if len(text) > cfg.limits.max_chars:
            raise GateError(ErrorCode.INPUT_TOO_LONG)
        try:
            pre_tokens = self._count(text, False)
            if pre_tokens > cfg.limits.max_current_tokens:
                raise GateError(ErrorCode.INPUT_TOO_LONG)
            rules = self.rules.evaluate(text)
            red = self.redactor.redact(text)
            red_tokens = self._count(red.text, False)
        except GateError:
            raise
        except Exception:
            return finish(base_inputs(failure="preprocess"), None)
        del text  # raw normalized text is not kept beyond risk detection and redaction
        if max(pre_tokens, red_tokens) > cfg.limits.max_current_tokens:
            raise GateError(ErrorCode.INPUT_TOO_LONG)

        view = session_context(ctx, utcnow())
        if view.valid_pending:
            ctx_tokens = self._count(" ".join(x for x in (view.topic, view.pending_question) if x), False)
            if ctx_tokens > cfg.limits.max_context_tokens:
                s = ctx.session
                raise GateError(ErrorCode.SESSION_CONFLICT,
                                SessionUpdate(session_id=s.session_id, expected_revision=s.revision,
                                              operation="reset_pending"))
        canonical = build_canonical_text(red.text, view.topic, view.pending_question)
        model_text = cfg.encoder.prefix + canonical
        if self._count(model_text, True) > cfg.limits.max_total_tokens:
            raise GateError(ErrorCode.INPUT_TOO_LONG)

        flags = DecisionFlags(pii_detected=red.detected, pii_redacted=red.detected, context_used=view.valid_pending)
        common = dict(redacted_text=red.text, rules=rules, session_view=view, flags=flags)

        # --- explicit crisis hard rule: inference may be skipped
        if rules.crisis_hard:
            return finish(base_inputs(inference_status=InferenceStatus.skipped_rule, **common), red.text,
                          red.categories, rules)

        enabled = self.registry.enabled_labels(ctx.course_id)
        remaining = deadline - self._monotonic()
        try:
            if remaining <= 0:
                raise DeadlineExceeded("deadline")
            raw, vec = self.worker.run(lambda: self._infer(model_text, canonical, enabled), remaining)
            prediction = Prediction.model_validate(raw)
        except GateError:
            raise  # 429 capacity
        except (DeadlineExceeded, InferenceFailed, ValidationError, ValueError, TypeError, KeyError):
            return finish(base_inputs(inference_status=InferenceStatus.failed, failure="inference", **common),
                          red.text, red.categories, rules)
        return finish(base_inputs(inference_status=InferenceStatus.completed, prediction=prediction, **common),
                      red.text, red.categories, rules, vec, canonical)

    def _audit_decision(self, decision, inputs, ctx, categories, start, deadline):
        plan = decision.retrieval_plan
        record = AuditRecord(
            event="gate_decision", request_id=decision.request_id, session_ref=self.session_ref(ctx),
            route=decision.route.value, reason=decision.reason.value,
            response_code=decision.response_code.value if decision.response_code else None,
            inference_status=decision.inference_status.value, versions=decision.versions.model_dump(),
            redaction_categories=tuple(categories), library_ids=plan.allowed_libraries if plan else (),
            latency_ms=round((self._monotonic() - start) * 1000, 3), service_status="ok",
        )
        try:
            self.audit.write(record)
        except Exception:
            self.alarms.raise_alarm("audit_write_failed", decision.request_id)
            if decision.route == Route.escalate:
                return decision  # the fixed support message is never withheld
            return decide(_with_failure(inputs, "inference"), self.config)
        if decision.route not in (Route.escalate, Route.unavailable) and self._monotonic() > deadline:
            return decide(_with_failure(inputs, "inference"), self.config)
        return decision


def _with_failure(inputs: DecisionInputs, stage: str) -> DecisionInputs:
    from dataclasses import replace  # noqa: PLC0415

    status = inputs.inference_status
    if status == InferenceStatus.completed:
        status = InferenceStatus.failed
    return replace(inputs, failure=stage, prediction=None, inference_status=status)


def default_fixture_tokenizer() -> FixtureTokenizer:
    return FixtureTokenizer()
