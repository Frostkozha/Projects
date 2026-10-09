"""Integration harness (spec section 11.3) with explicit development stubs.

This harness proves contracts and forbidden-call behavior. It is NOT the complete tutor:
the Fixture* adapters below are deterministic development stubs, not a retriever, a
generative model or a claim verifier.
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from retriever.schema import ErrorCode as RErrorCode
from retriever.schema import Reason as RReason
from retriever.schema import Status as RStatus

from .adapters import (
    MODE_INSTRUCTIONS,
    SYSTEM_INSTRUCTION_VERSION,
    AdapterError,
    Brain,
    BrainRequest,
    DraftAnswer,
    EvidenceBundle,
    ItemFetchRequest,
    Passage,
    RetrievalContext,
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    VerificationResult,
    Verifier,
    build_prompt,
    check_citations,
    validate_retrieval,
)
from .audit import AlertEvent, AuditRecord, RestrictedTextStore
from .config import AI_NOTICE, LibraryRegistry
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


def final_code_from_verification(v: VerificationResult, invalid_citations: tuple[str, ...]) -> str:
    """Return one of: unavailable, A7_emergency, A7_self_harm, A6, A5, A2, A1 (spec 11.2)."""
    if v.status == "error":
        return "unavailable"
    if "imminent_emergency" in v.output_policy_violations:
        return "A7_emergency"
    if "self_harm_crisis" in v.output_policy_violations:
        return "A7_self_harm"
    if v.output_policy_violations:
        return "A6"
    if invalid_citations or v.invalid_citations or v.status == "rejected":
        return "A5"
    if len(v.unsupported_claims) > 1:
        return "A5"
    if len(v.unsupported_claims) == 1 and not v.removal_safe:
        return "A5"
    if v.partial_support or v.source_conflict or len(v.unsupported_claims) == 1:
        return "A2"
    if not v.verified_text.strip():
        return "A5"
    return "A1"


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

    def _call(self, fn: Callable, *args):
        fut = self._pool.submit(fn, *args)
        try:
            return fut.result(timeout=self.adapter_timeout)
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
            bundle = EvidenceBundle.build(passages)
            prompt = build_prompt(request_text, mode, bundle, session_context, item_context)
            if count(prompt) <= limit:
                return bundle, prompt
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
            bundle, prompt = self._fit_context(result.redacted_text, mode, retrieval.passages, session_context,
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
                                 evidence=bundle, item_context=item_context,
                                 system_instruction_version=SYSTEM_INSTRUCTION_VERSION, prompt=prompt)
        try:
            draft: DraftAnswer = self._call(self.brain.draft, brain_req)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if draft.status == "no_evidence":
            return self._fixed(request_id, ResponseCode.A5, "no_source", "NO_EVIDENCE", session_id)
        if self.verifier is None:
            return self._unavailable(request_id, "verifier_missing")
        invalid = check_citations(draft, bundle.passages)
        try:
            verification: VerificationResult = self._call(self.verifier.verify, draft, bundle,
                                                          result.redacted_text, mode, item_context)
        except AdapterError as exc:
            return self._unavailable(request_id, str(exc))
        if verification.evidence_digest != bundle.digest:
            return self._unavailable(request_id, "evidence_map_mismatch")  # draft never delivered
        outcome = final_code_from_verification(verification, invalid)
        if outcome == "unavailable":
            return self._unavailable(request_id, "verifier_error")
        if outcome.startswith("A7"):
            category = "emergency" if outcome == "A7_emergency" else "self_harm"
            self.gate.alerts.dispatch(AlertEvent.new(request_id, self.gate.session_ref(ctx), category))
            return self._fixed(request_id, ResponseCode.A7, category, "DRAFT_CRISIS", session_id)
        if outcome == "A6":
            self._audit_final(request_id, "A6", "ok", "draft_policy_violation", event="incident")
            return self._fixed(request_id, ResponseCode.A6, "real_person", "DRAFT_POLICY_VIOLATION", session_id)
        if outcome == "A5":
            return self._fixed(request_id, ResponseCode.A5, "no_source", "VERIFICATION_REJECTED", session_id)

        code = ResponseCode.A1 if outcome == "A1" else ResponseCode.A2
        numbers = {pid: n for n, pid in bundle.citation_map}
        by_id = {p.passage_id: p for p in bundle.passages}
        cited = [by_id[pid] for pid in draft.used_passage_ids if pid in by_id]
        citations = tuple({"number": numbers[p.passage_id], "passage_id": p.passage_id, "source_id": p.source_id,
                           "source_version": p.source_version, "title": p.title, "locator": p.locator.label,
                           "evidence_uri": p.evidence_uri} for p in cited)
        text = f"{verification.verified_text.strip()}\n\n{AI_NOTICE}"

        # commit the session atomically after an accepted final response
        try:
            session_id = self._commit_session(ctx, mode, pending_item, bundle, draft, rctx)
        except GateError as exc:
            return self._error(request_id, exc.code)
        if not self._audit_final(request_id, code.value, "ok"):
            return self._unavailable(request_id, "audit_failed", audit=False)
        self.text_store.maybe_store(request_id, result.redacted_text, verification.verified_text)
        return FinalResponse(request_id, 200, code, text, citations, session_id=session_id, reason="VERIFIED")

    def _unavailable(self, request_id: str, detail: str, audit: bool = True) -> FinalResponse:
        if audit:
            self._audit_final(request_id, None, "unavailable", detail)
        return self._error(request_id, ErrorCode.SERVICE_UNAVAILABLE)

    def _commit_session(self, ctx: GateContext, mode: Mode, pending_item: bool, bundle: EvidenceBundle,
                        draft: DraftAnswer, rctx: RetrievalContext) -> Optional[str]:
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
            new_pending = {"pending_question": draft.draft_text[:MAX_PENDING_CHARS], "pending_item_id": item[0],
                           "item_version": item[1], "state": "awaiting_response"}
        elif mode == Mode.tutor and not pending_item:
            new_pending = {"pending_question": draft.draft_text[:MAX_PENDING_CHARS], "pending_item_id": None,
                           "item_version": None, "state": "awaiting_response"}
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

    def approved_item_for(self, context, kb_version, passage_id):
        for item_id, (version, pids) in sorted(self.items.items()):
            if passage_id in pids:
                return (item_id, version)
        return None


class FixtureBrain:
    """Development stub: deterministic draft builder; ``script`` may override the draft."""

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
            return DraftAnswer(draft_text="", used_passage_ids=(), status="no_evidence")
        p = request.evidence.passages[0]
        lead = MODE_INSTRUCTIONS[request.mode].split(".")[0]
        return DraftAnswer(draft_text=f"{p.text} [{p.passage_id}] ({lead}.)", used_passage_ids=(p.passage_id,),
                           status="draft")


class FixtureVerifier:
    """Development stub: approves drafts whose citations are valid unless ``script`` says otherwise.

    It does NOT check medical claims against evidence; a real verifier must. It echoes the evidence
    digest it received so the orchestrator can prove Brain and verifier saw the same evidence map.
    """

    def __init__(self, script: Optional[Callable[..., VerificationResult]] = None):
        self.script = script
        self.calls = 0
        self.seen: list[EvidenceBundle] = []

    def verify(self, draft, evidence, request_text, mode, item_context) -> VerificationResult:
        self.calls += 1
        self.seen.append(evidence)
        if self.script is not None:
            v = self.script(draft, evidence.passages)
            return v if v.evidence_digest else v.model_copy(update={"evidence_digest": evidence.digest})
        return VerificationResult(status="approved", evidence_digest=evidence.digest, verified_text=draft.draft_text,
                                  supported_claims=draft.used_passage_ids, verifier_version="fixture-verifier")
