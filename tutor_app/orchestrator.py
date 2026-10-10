"""One request state machine (Integration plan v0.3, sections 3, 6, 7, 10; appendix E).

authenticated request -> notice -> idempotency -> owned session -> gate -> (fixed reply | approved item |
generated: retrieval -> frozen context -> live registry -> Brain -> verifier -> formatter) -> final
authorization transaction -> response.

Fixed gate replies stop before retrieval and never take the generated slot. Content outcomes (A3/A5/A6/A7)
are HTTP 200; operational faults are typed errors and never receive a fabricated A1-A7 code. No retries,
JSON repair, model switch, generative fallback, language filter or remote inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from brain.context import freeze_passages
from brain.request_schema import BrainRequest
from brain.schema import BrainContext, ErrorCode as BrainErrorCode, PresentationHint
from contracts.json_codec import canonical_json, digest, validate_wire
from contracts.models import VerificationResult, VerifyRequest
from gate_classifier.schema import ErrorCode as GateErrorCode
from gate_classifier.schema import GateContext, GateError, GateRequest, Mode, Reason, Route, SessionState
from retriever.schema import RetrievalRequest
from retriever.schema import Status as RStatus
from verifier import digests as vdigests
from verifier.evidence import EvidenceBundle
from verifier.formatter import DeliveryRefused
from verifier.service import VerifyContext

from . import items as itm
from .auth import Identity, owner_ref, support_ref
from .capacity import (CapacityRejected, CircuitBreaker, Deadline, GeneratedSlot, PendingLimiter, StageTimeout,
                       run_stage)
from .components import Components
from .config import TutorConfig
from .delivery import (AuthPlan, RequestStateError, SourceIneligible, Suspended, a7_outage_record, authorize,
                       session_row, view)
from .formatter import (QUIZ_INSTRUCTION, TUTOR_INSTRUCTION, compile_generated, item_citations, quiz_feedback)
from .input_schema import InteractionInput
from .public import Activity, ActivityOption, ApiError, StudentReply
from .review import practice_indicator
from .store import Store, parse_utc
from .transactions import SessionConflict, immediate_transaction

MAX_BODY = 32 * 1024
PROMPT_VERSION = "brain-prompt-0.2"
_GATE_ERRORS = {GateErrorCode.INPUT_TOO_LONG: "INPUT_TOO_LONG", GateErrorCode.EMPTY_INPUT: "EMPTY_INPUT",
                GateErrorCode.INVALID_REQUEST: "INVALID_REQUEST", GateErrorCode.FORBIDDEN: "FORBIDDEN",
                GateErrorCode.SESSION_EXPIRED: "SESSION_EXPIRED", GateErrorCode.SESSION_CONFLICT: "SESSION_CONFLICT",
                GateErrorCode.CAPACITY_EXCEEDED: "CAPACITY_EXCEEDED", GateErrorCode.UNAUTHORIZED: "UNAUTHORIZED",
                GateErrorCode.MAINTENANCE: "MAINTENANCE", GateErrorCode.EXAM_DISABLED: "EXAM_DISABLED",
                GateErrorCode.SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE"}
_DEPLOY_ERRORS = {"suspended": "MAINTENANCE", "exam_shutdown": "EXAM_DISABLED", "disallowed": "MAINTENANCE"}


class _Fixed(Exception):
    """Internal: route to a fixed reply (code, reply_key, reason)."""

    def __init__(self, code: str, reply_key: str, reason: str):
        super().__init__(reason)
        self.code, self.reply_key, self.reason = code, reply_key, reason


@dataclass
class Attempt:
    request_id: str
    identity: Identity
    body: InteractionInput
    deadline: Deadline
    cancel: asyncio.Event
    deployment_revision: int
    pinned: dict
    session: Optional[dict] = None
    stage_ms: dict = field(default_factory=dict)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise ApiError("REQUEST_CANCELLED", self.request_id)


class Coordinator:
    def __init__(self, cfg: TutorConfig, store: Store, components: Components, *, fingerprint_key: bytes,
                 bank: itm.ItemBank, topics: itm.TopicRegistry):
        self.cfg = cfg
        self.store = store
        self.c = components
        self._fp_key = fingerprint_key
        self.bank = bank
        self.topics = topics
        self.slot = GeneratedSlot(cfg.capacity.generated_waiting, cfg.capacity.generated_wait_seconds)
        self.pending = PendingLimiter(cfg.capacity.per_session_pending, cfg.capacity.per_identity_pending)
        self.input_sem = asyncio.Semaphore(cfg.capacity.input_capacity)
        b = cfg.breaker
        self.breakers = {n: CircuitBreaker(n, b.failures, b.window_seconds, b.cooldown_seconds)
                         for n in ("retriever", "brain", "verifier")}
        self.inflight: dict[tuple[str, str], Attempt] = {}  # (owner, idempotency key) -> attempt
        self.buffer: dict[str, tuple[float, dict, list]] = {}  # request_id -> (expiry, reply, cited sources)
        self.degraded: set[str] = set()
        self.notices = components.verifier.rules.notices if components.verifier else {}

    # ================================================================== entry point

    async def interact(self, identity: Identity, course_id: str, idem_key: str, raw: bytes) -> StudentReply:
        deadline = Deadline(self.cfg.deadlines.whole_seconds)
        try:
            uuid.UUID(idem_key)
        except (ValueError, TypeError, AttributeError):
            raise ApiError("INVALID_REQUEST") from None
        if course_id != self.cfg.course_id or course_id not in identity.course_ids:
            raise ApiError("FORBIDDEN")
        try:
            body = validate_wire(InteractionInput, raw, max_bytes=MAX_BODY)
        except Exception:  # noqa: BLE001 - never echo or log validation detail
            raise ApiError("INVALID_REQUEST") from None
        if self.input_sem.locked():
            raise ApiError("CAPACITY_EXCEEDED")
        async with self.input_sem:
            await self._require_notice(identity, body.notice_version)
            replay = await self._idempotency(identity, idem_key, raw)
            if replay is not None:
                return replay
            attempt = await self._admit(identity, idem_key, raw, body, deadline)
            key = (identity.owner_code, idem_key)
            self.inflight[key] = attempt
            try:
                async with self.pending.hold(identity.owner_code, str(body.session_id) if body.session_id else None):
                    reply = await self._run(attempt)
                return reply
            except CapacityRejected as exc:
                await self._fail(attempt, "failed", exc.code)
                raise ApiError(exc.code, attempt.request_id) from None
            except ApiError as exc:
                exc.request_id = exc.request_id or attempt.request_id
                await self._fail(attempt, "cancelled" if exc.code == "REQUEST_CANCELLED" else "failed", exc.code)
                raise
            except Exception:  # noqa: BLE001 - unexpected faults fail closed without detail
                await self._fail(attempt, "failed", "internal")
                raise ApiError("SERVICE_UNAVAILABLE", attempt.request_id) from None
            finally:
                self.inflight.pop(key, None)

    def cancel(self, identity: Identity, idem_key: str) -> bool:
        attempt = self.inflight.get((identity.owner_code, idem_key))
        if attempt is None:
            return False
        attempt.cancel.set()
        return True

    # ================================================================== admission

    async def _require_notice(self, identity: Identity, version: str) -> None:
        if version != self.cfg.notice.version:
            raise ApiError("NOTICE_REQUIRED")
        ok = await self.store.call(lambda c: c.execute(
            "SELECT 1 FROM notice_acceptance WHERE owner_code=? AND course_id=? AND notice_version=?",
            (identity.owner_code, self.cfg.course_id, version)).fetchone())
        if not ok:
            raise ApiError("NOTICE_REQUIRED")

    def _fingerprint(self, identity: Identity, raw: bytes) -> str:
        from contracts.json_codec import load_object  # noqa: PLC0415

        canon = canonical_json(load_object(raw, max_bytes=MAX_BODY))
        return hmac.new(self._fp_key, identity.owner_code.encode() + b"\x00" + canon, hashlib.sha256).hexdigest()

    async def _idempotency(self, identity: Identity, key: str, raw: bytes) -> Optional[StudentReply]:
        fp = self._fingerprint(identity, raw)
        row = await self.store.call(lambda c: c.execute(
            "SELECT request_id, body_fingerprint, state, content_type, fixed_payload FROM requests WHERE owner_code=? "
            "AND course_id=? AND idempotency_key=?", (identity.owner_code, self.cfg.course_id, key)).fetchone())
        if row is None:
            return None
        if not hmac.compare_digest(row["body_fingerprint"], fp):
            raise ApiError("IDEMPOTENCY_CONFLICT", row["request_id"])
        if row["state"] in ("admitted", "running"):
            raise ApiError("REQUEST_IN_PROGRESS", row["request_id"])
        if row["state"] != "authorized":
            raise ApiError("REQUEST_EXPIRED", row["request_id"])
        if row["fixed_payload"]:
            return StudentReply.model_validate_json(row["fixed_payload"])
        entry = self.buffer.get(row["request_id"])
        if entry is None or entry[0] < time.monotonic():
            self.buffer.pop(row["request_id"], None)
            raise ApiError("REQUEST_EXPIRED", row["request_id"])
        # a generated reply is repeated only after rechecking current delivery eligibility
        def check(conn):
            state, _ = Store.deployment(conn, self.cfg.course_id)
            ok = Store.sources_eligible(conn, entry[2], self.store.now())
            return state == "active" and all(ok.values())
        if not await self.store.call(check):
            raise ApiError("REQUEST_EXPIRED", row["request_id"])
        return StudentReply.model_validate(entry[1])

    async def _admit(self, identity, key, raw, body, deadline) -> Attempt:
        rid = str(uuid.uuid4())
        fp = self._fingerprint(identity, raw)

        def run(conn):
            state, revision = Store.deployment(conn, self.cfg.course_id)
            pinned = {"policy": self.c.gate_cfg.policy_version, "kb": self.c.kb_version,
                      "registry_epoch": Store.registry_epoch(conn), "deployment_revision": revision,
                      "item_bank": self.bank.bank_version, "topics": self.topics.registry_version,
                      "prompt": PROMPT_VERSION, "profile": self.cfg.profile}
            with immediate_transaction(conn):
                try:
                    conn.execute("INSERT INTO requests (request_id, owner_code, course_id, idempotency_key, "
                                 "body_fingerprint, session_id, expected_revision, pinned_versions, state, created_at, "
                                 "updated_at) VALUES (?,?,?,?,?,?,?,?, 'running', ?, ?)",
                                 (rid, identity.owner_code, self.cfg.course_id, key, fp,
                                  str(body.session_id) if body.session_id else None, body.expected_session_revision,
                                  json.dumps(pinned, sort_keys=True), self.store.now(), self.store.now()))
                except Exception:  # unique race: another attempt with this key won
                    raise ApiError("REQUEST_IN_PROGRESS") from None
            return state, revision, pinned
        state, revision, pinned = await self.store.call(run)
        return Attempt(rid, identity, body, deadline, asyncio.Event(), revision, pinned)

    async def _fail(self, attempt: Attempt, state: str, reason: str) -> None:
        def run(conn):
            with immediate_transaction(conn):
                conn.execute("UPDATE requests SET state=?, updated_at=? WHERE request_id=? AND state='running'",
                             (state, self.store.now(), attempt.request_id))
                try:
                    self.store.audit(conn, event="request_failed", request_id=attempt.request_id,
                                     owner_ref=owner_ref(attempt.identity.owner_code), reason=reason,
                                     stage_ms=attempt.stage_ms)
                except Exception:  # noqa: BLE001
                    self.degraded.add("audit")
        try:
            await self.store.call(run)
        except Exception:  # noqa: BLE001
            self.degraded.add("store")

    # ================================================================== main flow

    async def _run(self, a: Attempt) -> StudentReply:
        body = a.body
        if body.session_id is not None:
            a.session = await self._owned_session(a)
        deploy_state = await self.store.call(lambda c: Store.deployment(c, self.cfg.course_id)[0])
        gctx = self._gate_context(a, deploy_state)
        greq = GateRequest(text=body.text, requested_mode=body.requested_mode,
                           session_id=str(body.session_id) if body.session_id else None)
        a.check_cancel()
        t = time.monotonic()
        try:
            result = await run_stage("gate", self.cfg.deadlines.gate_seconds + 1.0, a.deadline,
                                     lambda until: self.c.thread(self.c.gate.evaluate, greq, gctx, a.request_id))
        except GateError as exc:
            if exc.session_update is not None and exc.session_update.operation == "reset_pending" and a.session:
                await self._reset_pending(a)
            raise ApiError(_GATE_ERRORS.get(exc.code, "SERVICE_UNAVAILABLE"), a.request_id) from None
        except StageTimeout:
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
        a.stage_ms["gate"] = round((time.monotonic() - t) * 1000, 1)
        d = result.decision
        if d.route == Route.unavailable:
            code = {Reason.MAINTENANCE: "MAINTENANCE", Reason.EXAM_DISABLED: "EXAM_DISABLED"}.get(d.reason,
                                                                                               "SERVICE_UNAVAILABLE")
            raise ApiError(code, a.request_id)
        if d.route == Route.escalate:
            return await self._a7(a, d.reply_key, d.reason.value)
        if d.route == Route.clarify:
            return await self._fixed(a, "A3", d.reply_key, d.reason.value)
        if d.route == Route.reply:
            op = "pause" if (d.response_code.value == "A6" and a.session) else None
            return await self._fixed(a, d.response_code.value, d.reply_key, d.reason.value, session_op=op)
        # ---- retrieve route: approved learning paths first, then the generated answer path
        try:
            item_reply = await self._learning_paths(a, d, result)
            if item_reply is not None:
                return item_reply
            return await self._generated(a, d, result)
        except _Fixed as f:
            return await self._fixed(a, f.code, f.reply_key, f.reason)

    async def _owned_session(self, a: Attempt) -> dict:
        sid = str(a.body.session_id)
        row = await self.store.call(lambda c: session_row(c, sid))
        ident = a.identity
        if row is None or row["owner_code"] != ident.owner_code or row["course_id"] != self.cfg.course_id \
                or row["tenant_id"] != ident.tenant_id:
            raise ApiError("FORBIDDEN", a.request_id)  # no disclosure of other sessions
        if row["state"] == "closed" or row["expires_at"] <= self.store.now():
            raise ApiError("SESSION_EXPIRED", a.request_id)
        if row["revision"] != a.body.expected_session_revision:
            raise ApiError("SESSION_CONFLICT", a.request_id)
        return row

    def _gate_context(self, a: Attempt, deploy_state: str) -> GateContext:
        s = a.session
        state = None
        if s is not None:
            pending_q = s["pending_question"] if s["state"] == "awaiting_response" else None
            state = SessionState(session_id=s["session_id"], owner_code=s["owner_code"], tenant_id=s["tenant_id"],
                                 course_id=s["course_id"], mode=Mode(s["mode"]), topic_id=s["topic_id"],
                                 topic_text=s["topic_text"], pending_question=pending_q,
                                 pending_item_id=s["pending_item_id"], item_version=s["item_version"],
                                 state=s["state"], revision=s["revision"], expires_at=parse_utc(s["expires_at"]),
                                 policy_version=s["policy_version"], kb_version=s["kb_version"])
        reg = self.c.library_registry
        return GateContext(course_id=self.cfg.course_id, tenant_id=self.cfg.tenant_id,
                           owner_code=a.identity.owner_code,
                           authorized_active_libraries=reg.active_libraries(self.cfg.course_id),
                           topic_registry_version=self.topics.registry_version, kb_version=self.c.kb_version or "none",
                           policy_version=self.c.gate_cfg.policy_version, deployment_state=deploy_state, session=state)

    async def _reset_pending(self, a: Attempt) -> None:
        s = a.session

        def run(conn):
            with immediate_transaction(conn):
                conn.execute("UPDATE sessions SET state='idle', pending_item_id=NULL, item_version=NULL, "
                             "pending_question=NULL, pending_kind=NULL, revision=revision+1 WHERE session_id=? AND "
                             "revision=? AND state='awaiting_response'", (s["session_id"], s["revision"]))
        try:
            await self.store.call(run)
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== fixed replies and A7

    def _reply(self, a: Attempt, code: str, ctype: str, body: str, *, notices=None, citations=None,
               activity=None) -> dict:
        return {"schema_version": "student-reply-0.2", "request_id": a.request_id, "response_code": code,
                "content_type": ctype, "body": body, "notices": list(notices or []),
                "citations": [c.model_dump(mode="json") for c in (citations or [])],
                "activity": activity.model_dump(mode="json") if activity else None, "session": None}

    def _plan(self, a: Attempt, payload: dict, *, route: str, reason: str, **kw) -> AuthPlan:
        sess = a.session
        return AuthPlan(request_id=a.request_id, owner_code=a.identity.owner_code, tenant_id=a.identity.tenant_id,
                        course_id=self.cfg.course_id, response_code=payload["response_code"],
                        content_type=payload["content_type"], payload_digest=digest(payload),
                        deployment_revision=a.deployment_revision, pinned_versions=a.pinned,
                        owner_ref=owner_ref(a.identity.owner_code), route=route, reason=reason,
                        session_id=sess["session_id"] if sess else None,
                        expected_revision=sess["revision"] if sess else None, stage_ms=a.stage_ms, **kw)

    async def _commit(self, a: Attempt, payload: dict, plan: AuthPlan) -> StudentReply:
        a.check_cancel()
        t = time.monotonic()
        try:
            sv = await asyncio.wait_for(asyncio.to_thread(authorize, self.store, plan,
                                                          ttl_minutes=self.cfg.session_ttl_minutes),
                                        timeout=max(0.1, min(self.cfg.deadlines.delivery_seconds + 3,
                                                             a.deadline.remaining())))
        except SessionConflict as exc:
            raise ApiError(str(exc) if str(exc) in ("SESSION_CONFLICT", "SESSION_EXPIRED") else "SESSION_CONFLICT",
                           a.request_id) from None
        except Suspended as exc:
            raise ApiError(_DEPLOY_ERRORS.get(exc.state, "MAINTENANCE"), a.request_id) from None
        except RequestStateError:
            raise ApiError("REQUEST_CANCELLED" if a.cancel.is_set() else "SERVICE_UNAVAILABLE", a.request_id) from None
        a.stage_ms["delivery"] = round((time.monotonic() - t) * 1000, 1)
        payload = dict(payload)
        payload["session"] = sv.model_dump(mode="json") if sv else None
        reply = StudentReply.model_validate(payload)
        if plan.fixed_payload is None and self.cfg.response_buffer_seconds:
            self.buffer[a.request_id] = (time.monotonic() + self.cfg.response_buffer_seconds,
                                         reply.model_dump(mode="json"), list(plan.cited_sources))
        return reply

    async def _fixed(self, a: Attempt, code: str, reply_key: str, reason: str, *, session_op=None,
                     incident=None) -> StudentReply:
        payload = self._reply(a, code, "fixed", self.c.gate_cfg.render_reply(reply_key))
        plan = self._plan(a, payload, route="fixed", reason=reason, session_op=session_op, incident=incident)
        # the deterministic fixed record is replayable for the same key (its session view is attached below)
        plan.fixed_payload = "pending"
        try:
            reply = await self._commit(a, payload, plan)
        except Exception as exc:
            if isinstance(exc, ApiError):
                raise
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
        await self._store_fixed(a.request_id, reply)
        return reply

    async def _store_fixed(self, request_id: str, reply: StudentReply) -> None:
        text = reply.model_dump_json()
        await self.store.call(lambda c: c.execute("UPDATE requests SET fixed_payload=? WHERE request_id=?",
                                                  (text, request_id)))

    async def _a7(self, a: Attempt, reply_key: str, reason: str) -> StudentReply:
        """Hard A7: fixed safety text always displays; never claims an alert was sent (I07, I09)."""
        category = "emergency" if reply_key == "emergency" else "self_harm"
        payload = self._reply(a, "A7", "fixed", self.c.gate_cfg.render_reply(reply_key))
        plan = self._plan(a, payload, route="escalate", reason=reason,
                          session_op="close" if a.session else None, welfare_category=category,
                          support_ref=support_ref(a.identity.owner_code, a.request_id))
        plan.fixed_payload = "pending"
        try:
            reply = await self._commit(a, payload, plan)
            await self._store_fixed(a.request_id, reply)
            return reply
        except Exception:  # noqa: BLE001 - outage exception: deliver the fixed text, mark degraded
            self.degraded.add("a7_audit_outbox")
            self.c.alarms.raise_alarm("a7_persistence_failed", a.request_id)
            a7_outage_record(self.store, plan)
            return StudentReply.model_validate(payload)

    # ================================================================== approved learning paths

    async def _learning_paths(self, a: Attempt, d, result) -> Optional[StudentReply]:
        s = a.session
        text = a.body.text
        pending = s is not None and s["state"] == "awaiting_response" and s["pending_item_id"]
        ctl = itm.control(text)
        if pending and s["pending_kind"] == "quiz":
            opt = itm.option_answer(text)
            if opt is not None:
                return await self._quiz_submit(a, s, opt)
            if ctl == "continue":
                return await self._present(a, "quiz", s["topic_id"], exclude=s["pending_item_id"])
        if pending and s["pending_kind"] == "tutor":
            if ctl == "show_explanation":
                return await self._tutor_explanation(a, s)
            if ctl == "continue":
                return await self._present(a, "tutor", s["topic_id"], exclude=s["pending_item_id"])
        if ctl == "continue" and s is not None and s["topic_id"] and s["mode"] in ("tutor", "quiz"):
            return await self._present(a, s["mode"], s["topic_id"])
        cmd_mode, topic = itm.resolve_topic(text, self.topics)
        requested = a.body.requested_mode if a.body.requested_mode in ("tutor", "quiz") else None
        mode = cmd_mode or requested
        if mode is None:
            return None  # Answer: grounded generated explanation
        short = itm.normalize(text).rstrip(".!?")
        if topic is None and s is not None and s["topic_id"] and short in ("quiz me", "tutor me", "quiz", "tutor",
                                                                           "start", "begin"):
            topic = s["topic_id"]
        if topic is None:
            if cmd_mode is not None or short in ("quiz me", "tutor me", "quiz", "tutor", "start", "begin"):
                raise _Fixed("A3", "clarify_topic", "NO_UNIQUE_TOPIC")  # no fuzzy topic expansion
            return None  # a substantive question in Tutor/Quiz mode gets a grounded explanation
        return await self._present(a, mode, topic)

    def _find(self, kind: str, item_id: str, version: str):
        pool = self.bank.quiz_items if kind == "quiz" else self.bank.tutor_items
        for it in pool:
            if it.item_id == item_id and it.item_version == version:
                return it
        return None

    async def _item_evidence(self, a: Attempt, item) -> list:
        """Exact approved item evidence: fetched by passage ID (never a search) and live-registry checked."""
        if item.status != "live" or item.review_due_at <= self.store.now()[:10]:
            raise _Fixed("A5", "no_source", "ITEM_NOT_ELIGIBLE")
        rctx = self.c.retrieval_context()
        passages = []
        for pid in item.evidence_passage_ids:
            try:
                p = await self.c.thread(self._get_evidence, pid, rctx)
            except Exception:  # noqa: BLE001 - unknown/revoked evidence: item unavailable, never substituted
                raise _Fixed("A5", "no_source", "ITEM_EVIDENCE_UNAVAILABLE") from None
            passages.append(p)
        try:
            ok = await self.c.thread(self.c.registry.eligible, tuple(passages), self.c.kb_version)
        except Exception:  # noqa: BLE001
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
        if not all(ok.values()):
            raise _Fixed("A5", "no_source", "ITEM_SOURCE_INELIGIBLE")
        return passages

    def _get_evidence(self, pid, rctx):
        r = self.c.retriever
        if hasattr(r, "get_evidence"):
            return r.get_evidence(pid, self.c.kb_version, rctx)
        by_id = {p.passage_id: p for p in r.passages}
        if pid not in by_id:
            raise KeyError("unknown")
        return by_id[pid]

    async def _present(self, a: Attempt, mode: str, topic_id: str, *, exclude: Optional[str] = None):
        pool = self.bank.quiz_items if mode == "quiz" else self.bank.tutor_items
        pool = [i for i in pool if i.topic_id == topic_id]
        blocked = await self.store.call(lambda c: c.execute("SELECT 1 FROM topic_blocks WHERE course_id=? AND "
                                                            "topic_id=?", (self.cfg.course_id, topic_id)).fetchone())
        if blocked:
            raise _Fixed("A5", "no_source", "TOPIC_BLOCKED")
        review = await self.store.call(lambda c: {(r["item_id"], r["item_version"]): r["due_at"] for r in c.execute(
            "SELECT item_id, item_version, due_at FROM item_review_state WHERE owner_code=? AND course_id=?",
            (a.identity.owner_code, self.cfg.course_id))})
        due = itm.due_order(pool, review if mode == "quiz" else {}, self.store.now(), exclude=exclude)
        if not due:
            if exclude is not None:
                payload = self._reply(a, "A5", "fixed", "You have completed the approved practice questions for "
                                                         "this topic.")
                plan = self._plan(a, payload, route="item", reason="practice_complete",
                                  session_op="reset_pending" if a.session else None)
                plan.fixed_payload = "pending"
                reply = await self._commit(a, payload, plan)
                await self._store_fixed(a.request_id, reply)
                return reply
            raise _Fixed("A5", "no_source", "NO_ELIGIBLE_PRACTICE_ITEM")
        item = due[0]
        await self._item_evidence(a, item)
        options = [ActivityOption(option_id=o.option_id, text=o.text) for o in getattr(item, "options", [])]
        activity = Activity(item_id=item.item_id, item_version=item.item_version, mode=mode, phase="question",
                            question=item.question, options=options)
        payload = self._reply(a, "A1", "approved_item", QUIZ_INSTRUCTION if mode == "quiz" else TUTOR_INSTRUCTION,
                              activity=activity)
        changes = {"mode": mode, "topic_id": topic_id, "topic_text": self._topic_title(topic_id),
                   "pending_item_id": item.item_id, "item_version": item.item_version,
                   "pending_question": item.question[:300], "pending_kind": mode, "state": "awaiting_response",
                   "policy_version": self.c.gate_cfg.policy_version, "kb_version": self.c.kb_version}
        plan = self._plan(a, payload, route="item", reason="APPROVED_ITEM_PRESENTED",
                          session_op="update" if a.session else "create", session_changes=changes,
                          evidence_ids=list(item.evidence_passage_ids))
        return await self._commit(a, payload, plan)

    def _topic_title(self, topic_id: str) -> Optional[str]:
        for t in self.topics.topics:
            if t.topic_id == topic_id:
                return t.title
        return None

    async def _quiz_submit(self, a: Attempt, s: dict, option: str) -> StudentReply:
        item = self._find("quiz", s["pending_item_id"], s["item_version"])
        if item is None or item.status != "live":
            await self._reset_pending(a)  # changed/expired item: invalidate, never use an old key (I31)
            raise ApiError("SESSION_CONFLICT", a.request_id)
        passages = await self._item_evidence(a, item)
        correct = option == item.correct_option_id
        indicator = None
        if self.cfg.features.topic_indicator:
            row = await self.store.call(lambda c: c.execute(
                "SELECT attempts, correct FROM topic_practice WHERE owner_code=? AND course_id=? AND topic_id=?",
                (a.identity.owner_code, self.cfg.course_id, item.topic_id)).fetchone())
            att, cor = (row["attempts"], row["correct"]) if row else (0, 0)
            att, cor = att + 1, cor + int(correct)
            indicator = (f"Practice indicator for this topic: {practice_indicator(correct=cor, attempts=att):.2f} "
                         f"from {att} attempt{'s' if att != 1 else ''}. This is a study indicator, not a grade.")
        options = [ActivityOption(option_id=o.option_id, text=o.text) for o in item.options]
        activity = Activity(item_id=item.item_id, item_version=item.item_version, mode="quiz", phase="feedback",
                            question=item.question, options=options)
        payload = self._reply(a, "A1", "approved_item",
                              quiz_feedback(correct, item.correct_option_id, item.reviewed_explanation, indicator),
                              citations=item_citations(passages, self.c.kb_version), activity=activity)
        changes = {"state": "idle", "pending_item_id": None, "item_version": None, "pending_question": None,
                   "pending_kind": None}
        plan = self._plan(a, payload, route="item", reason="QUIZ_FEEDBACK", session_op="update",
                          session_changes=changes,
                          cited_sources=[(p.source_id, p.source_version) for p in passages],
                          evidence_ids=[p.passage_id for p in passages],
                          learning={"item_id": item.item_id, "item_version": item.item_version,
                                    "topic_id": item.topic_id, "correct": correct})
        try:
            return await self._commit(a, payload, plan)
        except Exception as exc:
            if isinstance(exc, ApiError):
                raise
            raise ApiError("SESSION_CONFLICT", a.request_id) from None  # duplicate learning event

    async def _tutor_explanation(self, a: Attempt, s: dict) -> StudentReply:
        item = self._find("tutor", s["pending_item_id"], s["item_version"])
        if item is None or item.status != "live":
            await self._reset_pending(a)
            raise ApiError("SESSION_CONFLICT", a.request_id)
        passages = await self._item_evidence(a, item)
        activity = Activity(item_id=item.item_id, item_version=item.item_version, mode="tutor", phase="feedback",
                            question=item.question, options=[])
        payload = self._reply(a, "A1", "approved_item", item.reviewed_explanation,
                              citations=item_citations(passages, self.c.kb_version), activity=activity)
        plan = self._plan(a, payload, route="item", reason="TUTOR_EXPLANATION", session_op="update",
                          session_changes={"state": "awaiting_response"},
                          cited_sources=[(p.source_id, p.source_version) for p in passages],
                          evidence_ids=[p.passage_id for p in passages])
        return await self._commit(a, payload, plan)

    # ================================================================== generated answer path

    async def _generated(self, a: Attempt, d, result) -> StudentReply:
        async with self.slot.hold(a.deadline):
            a.check_cancel()
            rctx = self.c.retrieval_context(result.embedding_ref, d.retrieval_plan.allowed_libraries)
            plan_r = d.retrieval_plan
            rreq = RetrievalRequest(request_id=a.request_id, query_text=result.redacted_text,
                                    allowed_libraries=plan_r.allowed_libraries,
                                    preferred_libraries=plan_r.preferred_libraries, course_id=self.cfg.course_id,
                                    kb_version=self.c.kb_version, strategy="all_active")
            retrieval = await self._guarded("retriever", self.cfg.deadlines.retriever_seconds, a,
                                            lambda until: self.c.thread(self.c.retriever.retrieve, rreq, rctx))
            if retrieval.request_id != a.request_id or retrieval.status == RStatus.error:
                self.breakers["retriever"].failure()
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
            self.breakers["retriever"].success()
            if retrieval.status == RStatus.no_evidence:
                raise _Fixed("A5", "no_source", "NO_EVIDENCE")
            if any(p.library_id not in plan_r.allowed_libraries or p.review_status != "live"
                   for p in retrieval.passages):
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)  # scope violation never delivered (I11)
            passages = list(retrieval.passages)
            shown = tuple(p.passage_id for p in passages)
            # live registry immediately before Brain invocation (I21)
            problem = await self.c.thread(self.c.retriever.revalidate, retrieval, rctx)
            if problem is not None:
                raise _Fixed("A5", "no_source", "SOURCE_CHANGED")
            try:
                elig = await self.c.thread(self.c.registry.eligible, tuple(passages), retrieval.kb_version)
            except Exception:  # noqa: BLE001
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
            if not all(elig.values()):
                raise _Fixed("A5", "no_source", "SOURCE_INELIGIBLE")
            a.check_cancel()
            t = time.monotonic()
            try:
                frozen = freeze_passages(request_id=a.request_id, tenant_id=self.cfg.tenant_id,
                                         course_id=self.cfg.course_id, passages=passages,
                                         kb_version=retrieval.kb_version, index_version=retrieval.index_version,
                                         profile_version=retrieval.profile_version,
                                         retrieval_policy_version=retrieval.retrieval_policy_version,
                                         revocation_epoch=retrieval.revocation_epoch,
                                         registry_version=self.c.library_registry.registry_version)
            except ValueError:
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
            a.stage_ms["context"] = round((time.monotonic() - t) * 1000, 1)
            brain_cfg = getattr(self.c.brain, "cfg", None)
            sampling = brain_cfg.sampling.profile_version if brain_cfg else "grounded-dev-0.2"
            breq = BrainRequest(schema_version="brain-request-0.2", request_id=uuid.UUID(a.request_id),
                                task="answer_draft", question_redacted=result.redacted_text,
                                passage_ids=list(shown), prompt_version=PROMPT_VERSION,
                                sampling_profile_version=sampling)
            bctx = BrainContext(evidence=frozen, expected_evidence_digest=frozen.evidence_digest,
                                deadline=a.deadline.stage(self.cfg.deadlines.brain_seconds),
                                presentation=PresentationHint())
            brain_result = await self._guarded("brain", self.cfg.deadlines.brain_seconds + 5, a,
                                               lambda until: self.c.brain.generate(breq, bctx, cancel=a.cancel))
            if brain_result.status == "error":
                if brain_result.error_code == BrainErrorCode.CANCELLED:
                    raise ApiError("REQUEST_CANCELLED", a.request_id)
                if brain_result.error_code == BrainErrorCode.INVALID_OUTPUT:
                    self.breakers["brain"].success()
                    raise _Fixed("A5", "no_source", "INVALID_DRAFT")  # no repair, no regeneration
                self.breakers["brain"].failure()
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
            self.breakers["brain"].success()
            if (brain_result.evidence_digest != frozen.evidence_digest
                    or str(brain_result.request_id) != a.request_id):
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
            if brain_result.status == "no_evidence":
                raise _Fixed("A5", "no_source", "BRAIN_NO_EVIDENCE")
            draft = brain_result.draft
            bundle = EvidenceBundle(request_id=a.request_id, tenant_id=self.cfg.tenant_id,
                                    course_id=self.cfg.course_id, retrieval=retrieval, shown_passage_ids=shown)
            vreq = VerifyRequest(schema_version="verify-request-0.2", request_id=a.request_id, draft=draft,
                                 prompt_version=PROMPT_VERSION)
            vctx = VerifyContext(request_id=a.request_id, route="retrieve", tenant_id=self.cfg.tenant_id,
                                 course_id=self.cfg.course_id, redacted_request=result.redacted_text,
                                 mode=d.effective_mode.value, evidence=bundle, registry=self.c.registry,
                                 deadline=a.deadline.stage(self.cfg.deadlines.verifier_seconds),
                                 coverage=retrieval.coverage,
                                 versions={"policy": self.c.gate_cfg.policy_version, "kb": retrieval.kb_version,
                                           "prompt": PROMPT_VERSION, "index": retrieval.index_version,
                                           "profile": retrieval.profile_version})
            a.check_cancel()
            verification: VerificationResult = await self._guarded(
                "verifier", self.cfg.deadlines.verifier_seconds + 2, a,
                lambda until: self.c.thread(self.c.verifier.verify, vreq, vctx))
            if (verification.request_id != a.request_id
                    or verification.draft_digest != vdigests.draft_digest(draft)
                    or verification.evidence_digest != vdigests.evidence_digest(bundle)):
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
            if verification.status == "error":
                self.breakers["verifier"].failure()
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
            self.breakers["verifier"].success()
            if verification.status == "rejected":
                code = verification.response_code.value
                if code == "A7":
                    return await self._a7(a, verification.reply_key, "VERIFIER_A7")
                if code == "A6":
                    return await self._fixed(a, "A6", verification.reply_key, "DRAFT_POLICY_VIOLATION",
                                             incident={"category": "draft_policy", "near_miss": verification.near_miss})
                raise _Fixed("A5", verification.reply_key or "no_source", "VERIFICATION_REJECTED")
            try:
                body_text, notices, cites, cited = compile_generated(verification, draft, bundle, self.notices)
            except DeliveryRefused:
                raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
            code = verification.response_code.value
            if code == "A4":
                raise _Fixed("A5", "no_source", "A4_DISABLED")  # disabled educational route never activates
            payload = self._reply(a, code, "generated", body_text, notices=notices, citations=cites)
            changes = {"mode": a.session["mode"] if a.session else "answer",
                       "policy_version": self.c.gate_cfg.policy_version, "kb_version": self.c.kb_version}
            plan = self._plan(a, payload, route="generated", reason="VERIFIED",
                              session_op="update" if a.session else "create", session_changes=changes,
                              cited_sources=[(p.source_id, p.source_version) for p in cited],
                              evidence_ids=list(shown))
            try:
                return await self._commit(a, payload, plan)
            except ApiError:
                raise
            except SourceIneligible:
                pass
            # revocation between verification and authorization: controlled fixed A5 (I21)
            return await self._fixed(a, "A5", "no_source", "DELIVERY_SUPPRESSED")

    async def _guarded(self, name: str, ceiling: float, a: Attempt, fn):
        br = self.breakers[name]
        if not br.allow():
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id)
        t = time.monotonic()
        try:
            out = await run_stage(name, ceiling, a.deadline, fn)
        except StageTimeout:
            br.failure()
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
        except (ApiError, _Fixed):
            raise
        except Exception:  # noqa: BLE001 - adapter exceptions may contain text: suppressed
            br.failure()
            raise ApiError("SERVICE_UNAVAILABLE", a.request_id) from None
        a.stage_ms[name] = round((time.monotonic() - t) * 1000, 1)
        return out  # the caller records success only for non-fault outcomes (content A5 is not a fault)

    # ================================================================== sessions, notices, reports, evidence

    async def accept_notice(self, identity: Identity, version: str) -> None:
        if version != self.cfg.notice.version:
            raise ApiError("VERSION_MISMATCH")
        await self.store.call(lambda c: c.execute(
            "INSERT OR IGNORE INTO notice_acceptance VALUES (?,?,?,?)",
            (identity.owner_code, self.cfg.course_id, version, self.store.now())))

    async def notice_accepted(self, identity: Identity) -> bool:
        return bool(await self.store.call(lambda c: c.execute(
            "SELECT 1 FROM notice_acceptance WHERE owner_code=? AND course_id=? AND notice_version=?",
            (identity.owner_code, self.cfg.course_id, self.cfg.notice.version)).fetchone()))

    async def get_session(self, identity: Identity, session_id: str):
        row = await self.store.call(lambda c: session_row(c, session_id))
        if row is None or row["owner_code"] != identity.owner_code or row["course_id"] not in identity.course_ids:
            raise ApiError("FORBIDDEN")
        return view(row)

    async def close_session(self, identity: Identity, session_id: str, expected: int):
        from .transactions import close_session_cas  # noqa: PLC0415

        def run(conn):
            with immediate_transaction(conn):
                row = session_row(conn, session_id)
                if row is None or row["owner_code"] != identity.owner_code:
                    raise ApiError("FORBIDDEN")
                if row["state"] == "closed" or row["expires_at"] <= self.store.now():
                    raise ApiError("SESSION_EXPIRED")
                close_session_cas(conn, session_id=session_id, owner_code=identity.owner_code,
                                  tenant_id=identity.tenant_id, course_id=row["course_id"],
                                  expected_revision=expected, now_utc=self.store.now())
                self.store.audit(conn, event="session_closed", owner_ref=owner_ref(identity.owner_code))
                return view(session_row(conn, session_id))
        try:
            return await self.store.call(run)
        except SessionConflict:
            raise ApiError("SESSION_CONFLICT") from None

    async def report(self, identity: Identity, request_id: str, category: str, message: Optional[str]) -> None:
        from gate_classifier.privacy import Redactor  # noqa: PLC0415

        redacted = Redactor().redact(message).text if message else None

        def run(conn):
            r = conn.execute("SELECT owner_code FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if r is None or r["owner_code"] != identity.owner_code:
                raise ApiError("FORBIDDEN")
            n = conn.execute("SELECT COUNT(*) FROM response_reports WHERE owner_code=? AND created_at>?",
                             (identity.owner_code, self.store.now()[:10])).fetchone()[0]
            if n >= 20:
                raise ApiError("RATE_LIMITED")
            try:
                conn.execute("INSERT INTO response_reports VALUES (?,?,?,?,?,?)",
                             (str(uuid.uuid4()), request_id, identity.owner_code, category, redacted, self.store.now()))
            except Exception:  # noqa: BLE001
                raise ApiError("IDEMPOTENCY_CONFLICT") from None
        await self.store.call(run)

    async def evidence(self, identity: Identity, passage_id: str, kb_version: str):
        from .public import EvidenceView, LocatorOut  # noqa: PLC0415

        if kb_version != self.c.kb_version:
            raise ApiError("CONTENT_UNAVAILABLE")
        try:
            p = await self.c.thread(self._get_evidence, passage_id, self.c.retrieval_context())
            ok = await self.c.thread(self.c.registry.eligible, (p,), kb_version)
        except Exception:  # noqa: BLE001
            raise ApiError("CONTENT_UNAVAILABLE") from None
        if not ok.get(passage_id):
            raise ApiError("CONTENT_UNAVAILABLE")  # revoked: controlled unavailable-content response
        loc = p.locator
        return EvidenceView(passage_id=p.passage_id, title=p.title, source_version=p.source_version,
                            locator=LocatorOut(kind=loc.kind, start=loc.start, end=loc.end, label=loc.label,
                                               anchor=loc.anchor), text=p.text)
