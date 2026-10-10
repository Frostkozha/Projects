"""Integration harness (spec section 11.3) with explicit development stubs.

This harness proves contracts and forbidden-call behavior. It is NOT the complete tutor:
the Fixture* adapters below are deterministic development stubs, not a retriever or a
generative model. Verification always runs the real ``verifier`` pipeline; fixture NLI scores
are injected only in tests and never satisfy production readiness.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from contracts.models import DraftAnswer, VerificationResult, VerifyRequest

from retriever.schema import ErrorCode as RErrorCode
from retriever.schema import Reason as RReason
from retriever.schema import Status as RStatus

from verifier import digests as vdigests
from verifier.config import load_rules
from verifier.evidence import EvidenceBundle, RegistryUnavailable, RetrieverSourceRegistry
from verifier.formatter import DeliveryAuthorizer, DeliveryRecord, DeliveryRefused
from verifier.service import VerifyContext

from .adapters import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION_VERSION,
    AdapterError,
    Brain,
    BrainRequest,
    ItemFetchRequest,
    Passage,
    PromptEvidence,
    RetrievalContext,
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    Verifier,
    build_prompt,
    validate_retrieval,
)
from .audit import AlertEvent, AuditRecord, RestrictedTextStore
from .config import LibraryRegistry
from .schema import (
    ERROR_HTTP_STATUS,
    ErrorCode,
    GateContext,
    GateError,
    GateRequest,
    Mode,
    Reason,
    ResponseCode,
    Route,
    new_request_id,
)
from .service import GateService
from .session import InMemorySessionStore

_UNAVAILABLE_CODES = {Reason.MAINTENANCE: ErrorCode.MAINTENANCE, Reason.EXAM_DISABLED: ErrorCode.EXAM_DISABLED}
MAX_PENDING_CHARS = 300
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Principal:
    """Authenticated pseudonymous identity supplied by the (out-of-scope) SSO layer."""

    owner_code: str
    tenant_id: str
    course_id: str
    authorized_libraries: Optional[frozenset[str]] = None


@dataclass
class FinalResponse:
    request_id: str
    http_status: int
    response_code: Optional[ResponseCode] = None
    text: Optional[str] = None
    citations: tuple[dict, ...] = ()
    error_code: Optional[ErrorCode] = None
    session_id: Optional[str] = None
    reason: Optional[str] = None

    def student_view(self) -> dict:
        """What a student client may see: no probabilities, heads, flags or versions."""
        if self.error_code is not None:
            return {"error_code": self.error_code.value, "request_id": self.request_id}
        return {"request_id": self.request_id, "response_code": self.response_code.value if self.response_code else None,
                "text": self.text, "citations": list(self.citations), "session_id": self.session_id}


@dataclass
class DeploymentControl:
    states: dict[str, str] = field(default_factory=dict)  # course_id -> state

    def state(self, course_id: str) -> str:
        return self.states.get(course_id, "active")


class Orchestrator:
    def __init__(self, gate: GateService, sessions: InMemorySessionStore, retriever: Retriever, brain: Brain,
                 verifier: Optional[Verifier], *, kb_version: str, topic_registry_version: str = "topics-dev-0.1",
                 deployment: Optional[DeploymentControl] = None, text_store: Optional[RestrictedTextStore] = None,
                 adapter_timeout_seconds: float = 10.0, service_id: str = "orchestrator"):
        self.gate = gate
        self.service_id = service_id
        self.config = gate.config
        self.registry: LibraryRegistry = gate.registry
        self.sessions = sessions
        self.retriever = retriever
        self.brain = brain
        self.verifier = verifier
        self.kb_version = kb_version
        self.topic_registry_version = topic_registry_version
        self.deployment = deployment or DeploymentControl()
        self.text_store = text_store or RestrictedTextStore(self.config.audit.restricted_text_store_enabled)
        self.adapter_timeout = adapter_timeout_seconds
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="adapter")
        self._lock = threading.Lock()
        rules = getattr(verifier, "rules", None) or load_rules(ROOT / "config/verifier")
        self.delivery = DeliveryAuthorizer(rules.notices, record_sink=self._record_delivery)

    # ------------------------------------------------------------------ context

    def build_context(self, owner_code: str, tenant_id: str, course_id: str, session_id: Optional[str],
                      authorized: Optional[frozenset[str]] = None) -> GateContext:
        course = self.registry.courses.get(course_id)
        if course is None or course.tenant_id != tenant_id:
            raise GateError(ErrorCode.FORBIDDEN)
        session = None
        if session_id is not None:
            session = self.sessions.lookup_owned(session_id, owner_code=owner_code, tenant_id=tenant_id,
                                                 course_id=course_id)
        return GateContext(
            course_id=course_id, tenant_id=tenant_id, owner_code=owner_code,
            authorized_active_libraries=self.registry.active_libraries(
                course_id, set(authorized) if authorized is not None else None),
            topic_registry_version=self.topic_registry_version, kb_version=self.kb_version,
            policy_version=self.config.policy_version, deployment_state=self.deployment.state(course_id),
            session=session,
        )

    # ------------------------------------------------------------------ helpers

    def _call(self, fn: Callable, *args, timeout: Optional[float] = None):
        fut = self._pool.submit(fn, *args)
        try:
            return fut.result(timeout=timeout or self.adapter_timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise AdapterError("adapter_timeout") from None
        except AdapterError:
            raise
        except Exception:  # adapter exceptions may contain prompt text: suppress them
            raise AdapterError("adapter_failed") from None

    def _error(self, request_id: str, code: ErrorCode) -> FinalResponse:
        return FinalResponse(request_id, ERROR_HTTP_STATUS[code], error_code=code)

    def _audit_final(self, request_id: str, code: Optional[str], status: str, detail: Optional[str] = None,
                     event: str = "final_response") -> bool:
        try:
            self.gate.audit.write(AuditRecord(event=event, request_id=request_id, response_code=code,
                                              service_status=status, detail_code=detail))
            return True
        except Exception:
            self.gate.alarms.raise_alarm("audit_write_failed", request_id)
            return False

    def _fixed(self, request_id: str, code: ResponseCode, reply_key: str, reason: str,
               session_id: Optional[str]) -> FinalResponse:
        self._audit_final(request_id, code.value, "ok", reply_key)
        return FinalResponse(request_id, 200, code, self.config.render_reply(reply_key), session_id=session_id,
                             reason=reason)

    def _apply_gate_update(self, update) -> None:
        if update is None:
            return
        extra = {}
        if update.operation == "reset_pending":
            extra = {"kb_version": self.kb_version, "policy_version": self.config.policy_version}
        try:
            self.sessions.apply_gate_update(update, **extra)
        except GateError:
            pass  # a concurrent commit already changed the session; nothing was advanced

    # ------------------------------------------------------------------ main flow

    def handle(self, principal: Optional[Principal], body) -> FinalResponse:
        request_id = new_request_id()
        if principal is None:
            return self._error(request_id, ErrorCode.UNAUTHORIZED)
        try:
            request = body if isinstance(body, GateRequest) else GateRequest.model_validate(body)
        except Exception:
            return self._error(request_id, ErrorCode.INVALID_REQUEST)
        try:
            ctx = self.build_context(principal.owner_code, principal.tenant_id, principal.course_id,
                                     request.session_id, principal.authorized_libraries)
            result = self.gate.evaluate(request, ctx, request_id)
        except GateError as exc:
            self._apply_gate_update(exc.session_update)
            return self._error(request_id, exc.code)

        decision = result.decision
        session_id = ctx.session.session_id if ctx.session else None
        if decision.route == Route.unavailable:
            self._audit_final(request_id, None, "unavailable", decision.reason.value)
            return self._error(request_id, _UNAVAILABLE_CODES.get(decision.reason, ErrorCode.SERVICE_UNAVAILABLE))
        if decision.route != Route.retrieve:
            self._apply_gate_update(result.session_update)
            return self._fixed(request_id, decision.response_code, decision.reply_key, decision.reason.value,
                               session_id)
        return self._generate(request_id, request, ctx, result)

    def _retrieval_context(self, ctx: GateContext, plan, embedding_ref) -> RetrievalContext:
        return RetrievalContext(
            service_id=self.service_id, tenant_id=ctx.tenant_id, course_id=ctx.course_id,
            authorized_libraries=frozenset(plan.allowed_libraries) if plan else frozenset(ctx.authorized_active_libraries),
            registry_version=self.registry.registry_version, gate_permitted=True, embedding_ref=embedding_ref,
            session_ref=self.gate.session_ref(ctx))

    def _fit_context(self, request_text, mode, evidence_passages, session_context, item_context, removable: bool):
        """Whole-passage removal (lowest-ranked first) until the full prompt fits the Brain budget."""
        brain = self.brain
        try:
            limit = int(brain.context_limit) - int(brain.reserved_output_tokens)
            count = brain.count_tokens
        except Exception:
            raise AdapterError("brain_budget_unavailable") from None
        passages = list(evidence_passages)
        while passages:
            evidence = PromptEvidence(passages=tuple(passages))
            prompt = build_prompt(request_text, mode, evidence, session_context, item_context)
            if count(prompt) <= limit:
                return evidence, prompt
            if not removable:
                raise AdapterError("context_budget_breaks_required_evidence")
            passages.pop()
        raise AdapterError("context_budget_removed_all_evidence")

    def _generate(self, request_id, request, ctx: GateContext, result) -> FinalResponse:
        decision = result.decision
        plan = decision.retrieval_plan
        mode = decision.effective_mode
        session = ctx.session
        session_id = session.session_id if session else None
        pending_item = bool(decision.flags.context_used and session is not None and session.pending_item_id
                            and session.item_version)
        rctx = self._retrieval_context(ctx, plan, result.embedding_ref)
        try:
            if pending_item:
                # a bare reply to an approved item fetches its approved evidence; never a search for "B"
                rreq = ItemFetchRequest(request_id=request_id, item_id=session.pending_item_id,
                                        item_version=session.item_version, course_id=ctx.course_id,
                                        kb_version=ctx.kb_version)
                fetch = self.retriever.fetch_item_evidence
            else:
                rreq = RetrievalRequest(request_id=request_id, query_text=result.redacted_text,
                                        allowed_libraries=plan.allowed_libraries,
                                        preferred_libraries=plan.preferred_libraries, course_id=ctx.course_id,
                                        kb_version=ctx.kb_version, strategy="all_active")
                fetch = self.retriever.retrieve
        except Exception:
            return self._unavailable(request_id, "retrieval_request_invalid")
        try:
            retrieval: RetrievalResult = self._call(fetch, rreq, rctx)
            if not isinstance(retrieval, RetrievalResult) or retrieval.request_id != request_id:
                raise AdapterError("retrieval_result_invalid")
            validate_retrieval(retrieval, plan.allowed_libraries, ctx.kb_version)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if retrieval.status == RStatus.no_evidence:
            return self._fixed(request_id, ResponseCode.A5, "no_source", "NO_EVIDENCE", session_id)

        item_context = None
        if pending_item:
            item_context = f"Pending item {session.pending_item_id} version {session.item_version}"
        session_context = session.topic_text if decision.flags.context_used and session else None
        # required evidence sets and conflict pairs may not be thinned to fit the prompt
        removable = retrieval.coverage == "unknown" and not retrieval.conflicts
        try:
            shown, prompt = self._fit_context(result.redacted_text, mode, retrieval.passages, session_context,
                                              item_context, removable)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        # revocation / eligibility re-check immediately before Brain invocation
        try:
            problem = self._call(self.retriever.revalidate, retrieval, rctx)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if problem is not None:
            return self._unavailable(request_id, "source_state_changed")
        brain_req = BrainRequest(question_text=result.redacted_text, mode=mode, session_context=session_context,
                                 evidence=shown, item_context=item_context,
                                 system_instruction_version=SYSTEM_INSTRUCTION_VERSION, prompt=prompt,
                                 request_id=request_id, tenant_id=ctx.tenant_id, course_id=ctx.course_id,
                                 retrieval=retrieval, registry_version=self.registry.registry_version)
        try:
            # the local Brain owns its own bounded deadline (30 s incl. queue); the adapter wait covers it
            brain_timeout = float(getattr(self.brain, "deadline_seconds", 0) or 0) + 5.0
            draft = self._call(self.brain.draft, brain_req, timeout=max(self.adapter_timeout, brain_timeout))
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if not isinstance(draft, DraftAnswer):
            # an invalid generated draft is a content rejection, never inferred onto old prose
            return self._fixed(request_id, ResponseCode.A5, "no_source", "INVALID_DRAFT", session_id)
        if draft.status == "no_evidence":
            return self._fixed(request_id, ResponseCode.A5, "no_source", "NO_EVIDENCE", session_id)
        if self.verifier is None:
            return self._unavailable(request_id, "verifier_missing")

        # trusted evidence bundle = exactly the passages shown to the Brain, bound to this request
        bundle = EvidenceBundle(request_id=request_id, tenant_id=ctx.tenant_id, course_id=ctx.course_id,
                                retrieval=retrieval, shown_passage_ids=shown.passage_ids)
        registry = self._source_registry(rctx)
        try:
            vreq = VerifyRequest(schema_version="verify-request-0.2", request_id=request_id, draft=draft,
                                 prompt_version=PROMPT_VERSION)
            vctx = VerifyContext(
                request_id=request_id, route="retrieve", tenant_id=ctx.tenant_id, course_id=ctx.course_id,
                redacted_request=result.redacted_text, mode=mode.value, evidence=bundle, registry=registry,
                deadline=time.monotonic() + self.adapter_timeout,
                item_id=session.pending_item_id if pending_item else None,
                item_version=session.item_version if pending_item else None,
                coverage=retrieval.coverage,  # approved item evidence sets carry faculty-defined coverage
                versions={"policy": ctx.policy_version, "kb": ctx.kb_version, "prompt": PROMPT_VERSION,
                          "index": retrieval.index_version, "profile": retrieval.profile_version})
        except Exception:
            return self._unavailable(request_id, "verify_request_invalid")
        try:
            verification = self._call(self.verifier.verify, vreq, vctx)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if (not isinstance(verification, VerificationResult) or verification.request_id != request_id
                or verification.draft_digest != vdigests.draft_digest(draft)
                or verification.evidence_digest != vdigests.evidence_digest(bundle)):
            return self._unavailable(request_id, "evidence_map_mismatch")  # draft never delivered
        if verification.status == "error":
            return self._unavailable(request_id, f"verifier_{verification.error_code.value.lower()}")
        if verification.status == "rejected":
            code = verification.response_code
            if code == ResponseCode.A7:
                category = verification.reply_key
                self.gate.alerts.dispatch(AlertEvent.new(request_id, self.gate.session_ref(ctx), category))
                return self._fixed(request_id, ResponseCode.A7, category, "CURRENT_REQUEST_CRISIS", session_id)
            if code == ResponseCode.A6:
                self._audit_final(request_id, "A6", "ok", "draft_policy_violation", event="incident")
                return self._fixed(request_id, ResponseCode.A6, verification.reply_key, "DRAFT_POLICY_VIOLATION",
                                   session_id)
            return self._fixed(request_id, ResponseCode.A5, verification.reply_key, "VERIFICATION_REJECTED",
                               session_id)

        # delivery authorization: binding + live eligibility + record, under one lock
        expected = (session.session_id, session.revision) if session else None

        def session_current() -> bool:
            if expected is None:
                return True
            current = self.sessions.get(expected[0])
            return current is not None and current.revision == expected[1]

        try:
            payload, _ = self.delivery.authorize(verification, draft, bundle, registry, session_current)
        except DeliveryRefused as exc:
            if exc.unavailable:
                return self._unavailable(request_id, f"delivery_{exc.code}")
            if exc.code == "stale_session":
                return self._error(request_id, ErrorCode.SESSION_CONFLICT)
            return self._fixed(request_id, ResponseCode.A5, "no_source", "DELIVERY_SUPPRESSED", session_id)

        # commit the session atomically after an authorized final response
        try:
            session_id = self._commit_session(ctx, mode, pending_item, shown, verification, rctx)
        except GateError as exc:
            return self._error(request_id, exc.code)
        code = ResponseCode(verification.response_code.value)
        if not self._audit_final(request_id, code.value, "ok"):
            return self._unavailable(request_id, "audit_failed", audit=False)
        self.text_store.maybe_store(request_id, result.redacted_text, verification.verified_text)
        return FinalResponse(request_id, 200, code, payload.text, payload.citations, session_id=session_id,
                             reason="VERIFIED")

    def _source_registry(self, rctx: RetrievalContext):
        factory = getattr(self.retriever, "source_registry", None)
        if callable(factory):
            return factory(rctx)
        if hasattr(self.retriever, "passage_eligibility"):
            return RetrieverSourceRegistry(self.retriever, rctx)
        return None  # the verifier fails closed with REGISTRY_UNAVAILABLE

    def _record_delivery(self, record: DeliveryRecord) -> None:
        self.gate.audit.write(AuditRecord(event="final_response", request_id=record.request_id,
                                          service_status="delivery_authorized",
                                          detail_code=f"payload:{record.payload_digest[:16]}"
                                                      f":epoch:{record.revocation_epoch}"))

    def _unavailable(self, request_id: str, detail: str, audit: bool = True) -> FinalResponse:
        if audit:
            self._audit_final(request_id, None, "unavailable", detail)
        return self._error(request_id, ErrorCode.SERVICE_UNAVAILABLE)

    def _commit_session(self, ctx: GateContext, mode: Mode, pending_item: bool, bundle: PromptEvidence,
                        verification: VerificationResult, rctx: RetrievalContext) -> Optional[str]:
        s = ctx.session
        new_pending: dict = {"pending_question": None, "pending_item_id": None, "item_version": None, "state": "idle"}
        item = None
        if mode == Mode.quiz and not pending_item:
            lookup = getattr(self.retriever, "approved_item_for", None)
            for p in bundle.passages:
                if p.library_id == "lib5" and lookup is not None:
                    item = lookup(rctx, ctx.kb_version, p.passage_id)
                    if item:
                        break
        if item:
            new_pending = {"pending_question": verification.verified_text[:MAX_PENDING_CHARS],
                           "pending_item_id": item[0], "item_version": item[1], "state": "awaiting_response"}
        elif mode == Mode.tutor and not pending_item:
            new_pending = {"pending_question": verification.verified_text[:MAX_PENDING_CHARS],
                           "pending_item_id": None, "item_version": None, "state": "awaiting_response"}
        if s is None:
            if mode == Mode.answer:
                return None
            created = self.sessions.create(owner_code=ctx.owner_code, tenant_id=ctx.tenant_id, course_id=ctx.course_id,
                                           mode=mode, policy_version=ctx.policy_version, kb_version=ctx.kb_version,
                                           **new_pending)
            return created.session_id
        changes = {"mode": mode, **new_pending}
        if pending_item:
            changes["answered_items"] = s.answered_items + (s.pending_item_id,)
        self.sessions.compare_and_swap(s.session_id, s.revision, **changes)
        return s.session_id


# ----------------------------------------------------------------------------- development stubs


class FixtureRetriever:
    """Development stub returning canonical RetrievalResult objects for configured passages.

    ``items`` maps item_id -> (item_version, passage_ids). Not the real retriever (see ``retriever``).
    """

    def __init__(self, passages: list[Passage], kb_version: str, status: str = "ok", raise_error: bool = False,
                 ignore_filter: bool = False, items: Optional[dict] = None, revalidate_result=None):
        self.passages = passages
        self.kb_version = kb_version
        self.status = status
        self.raise_error = raise_error
        self.ignore_filter = ignore_filter
        self.items = items or {}
        self.revalidate_result = revalidate_result
        self.calls: list = []
        self.item_calls: list = []
        self.registry = FixtureSourceRegistry()

    def _base(self, request_id, **kw):
        return dict(request_id=request_id, kb_version=self.kb_version, index_version="fixture-index",
                    profile_version="fixture-profile", revocation_epoch=0, **kw)

    def retrieve(self, request: RetrievalRequest, context: RetrievalContext) -> RetrievalResult:
        self.calls.append(request)
        if self.raise_error:
            raise RuntimeError(f"backend failure for {request.query_text}")  # suppressed by the harness
        if self.status == "error":
            return RetrievalResult.error(request.request_id, RErrorCode.SEARCH_FAILED, kb_version=self.kb_version)
        pool = self.passages if self.ignore_filter else [p for p in self.passages
                                                         if p.library_id in request.allowed_libraries]
        if self.status == "no_evidence" or not pool:
            return RetrievalResult.no_evidence(request.request_id, RReason.BELOW_THRESHOLD,
                                               request.allowed_libraries, kb_version=self.kb_version)
        return RetrievalResult(**self._base(request.request_id), status=RStatus.ok, reason=RReason.QUALIFYING_PASSAGES,
                               error_code=None, passages=tuple(pool[:5]), coverage="unknown", conflicts=(),
                               libraries_searched=tuple(sorted(request.allowed_libraries)))

    def fetch_item_evidence(self, request: ItemFetchRequest, context: RetrievalContext) -> RetrievalResult:
        self.item_calls.append(request)
        entry = self.items.get(request.item_id)
        if entry is None or entry[0] != request.item_version:
            return RetrievalResult.error(request.request_id, RErrorCode.ITEM_EVIDENCE_UNAVAILABLE,
                                         kb_version=self.kb_version)
        by_id = {p.passage_id: p for p in self.passages}
        chosen = tuple(by_id[pid].model_copy(update={"relevance_score": None, "score_type": None}) for pid in entry[1])
        return RetrievalResult(**self._base(request.request_id), status=RStatus.ok,
                               reason=RReason.APPROVED_ITEM_EVIDENCE, error_code=None, passages=chosen,
                               coverage="full", conflicts=(), libraries_searched=())

    def revalidate(self, result, context):
        return self.revalidate_result

    def source_registry(self, context):
        return self.registry

    def approved_item_for(self, context, kb_version, passage_id):
        for item_id, (version, pids) in sorted(self.items.items()):
            if passage_id in pids:
                return (item_id, version)
        return None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class FixtureSourceRegistry:
    """Development stub of the trusted live registry: every supplied live passage is eligible unless revoked."""

    def __init__(self, revoked: Optional[set] = None):
        self.revoked = revoked if revoked is not None else set()
        self._epoch = 0
        self.down = False

    def revoke(self, passage_id: str) -> None:
        self.revoked.add(passage_id)
        self._epoch += 1

    def eligible(self, passages, kb_version):
        if self.down:
            raise RegistryUnavailable("fixture registry down")
        return {p.passage_id: p.passage_id not in self.revoked for p in passages}

    def epoch(self) -> int:
        if self.down:
            raise RegistryUnavailable("fixture registry down")
        return self._epoch


class FixtureBrain:
    """Development stub: emits one sentence-level draft sentence copied from the first passage and citing it.

    ``script`` may return any ``contracts.models.DraftAnswer``. Not a generative model.
    """

    def __init__(self, script: Optional[Callable[[BrainRequest], DraftAnswer]] = None, context_limit: int = 4096,
                 reserved_output_tokens: int = 256):
        from .encoder import FixtureTokenizer  # noqa: PLC0415

        self.script = script
        self.context_limit = context_limit
        self.reserved_output_tokens = reserved_output_tokens
        self._tok = FixtureTokenizer()
        self.calls: list[BrainRequest] = []

    def count_tokens(self, text: str) -> int:
        return self._tok.count(text, True)

    def draft(self, request: BrainRequest) -> DraftAnswer:
        self.calls.append(request)
        if self.script is not None:
            return self.script(request)
        if not request.evidence.passages:
            return no_evidence_draft()
        p = request.evidence.passages[0]
        first = _SENTENCE_SPLIT.split(p.text.strip())[0]
        return sentence_draft((first, (p.passage_id,)))


def sentence_draft(*sentences: tuple, visibility: str = "student") -> DraftAnswer:
    """Build a brain-draft-0.2 draft from (text, cites) tuples (fixture helper)."""
    objs = [{"sentence_id": f"s{i + 1}", "text": text, "kind_hint": "factual", "visibility": visibility,
             "cites": list(cites), "depends_on": []} for i, (text, cites) in enumerate(sentences)]
    used = sorted({c for _, cites in sentences for c in cites})
    return DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": "draft", "sentences": objs,
                                       "used_passage_ids": used})


def no_evidence_draft() -> DraftAnswer:
    return DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": "no_evidence",
                                       "sentences": [], "used_passage_ids": []})


def fixture_verifier(score_fn: Optional[Callable] = None, audit=None):
    """The real verifier pipeline in fixture mode with injected NLI scores (tests/development only).

    Fixture scores never satisfy production readiness (Verifier spec 17).
    """
    from verifier.nli import FixtureNLI, substring_scores  # noqa: PLC0415
    from verifier.service import InMemoryVerifierAudit, Verifier  # noqa: PLC0415

    return Verifier.from_profile(ROOT / "config/verifier_development.yaml",
                                 audit=audit if audit is not None else InMemoryVerifierAudit(),
                                 fixture_backend=FixtureNLI(score_fn or substring_scores))
