"""Trusted session state, bounded context and mode resolution (spec section 9).

The gate only *proposes* pause/close/reset_pending. The orchestrator commits updates with
compare-and-swap on ``revision``; concurrent replies to one item cannot both advance it.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from .schema import (
    ErrorCode,
    GateContext,
    GateError,
    Mode,
    ModeSource,
    Prediction,
    RuleFlags,
    SessionState,
    SessionUpdate,
    utcnow,
)


@dataclass(frozen=True)
class SessionContextView:
    valid_pending: bool
    topic: Optional[str]
    pending_question: Optional[str]
    session_mode: Optional[Mode]


def session_context(ctx: GateContext, now: datetime) -> SessionContextView:
    """Return the context the gate may use. Version mismatches raise SESSION_CONFLICT with reset."""
    s = ctx.session
    if s is None:
        return SessionContextView(False, None, None, None)
    if s.owner_code != ctx.owner_code or s.tenant_id != ctx.tenant_id or s.course_id != ctx.course_id:
        raise GateError(ErrorCode.FORBIDDEN)
    if s.is_expired(now) or s.state == "closed":
        raise GateError(ErrorCode.SESSION_EXPIRED)
    if s.policy_version != ctx.policy_version or s.kb_version != ctx.kb_version:
        update = None
        if s.pending_item_id or s.pending_question:
            update = SessionUpdate(session_id=s.session_id, expected_revision=s.revision, operation="reset_pending")
        raise GateError(ErrorCode.SESSION_CONFLICT, session_update=update)
    valid = (
        s.state == "awaiting_response"
        and bool(s.pending_question)
        and (s.mode != Mode.quiz or bool(s.pending_item_id and s.item_version))
    )
    return SessionContextView(
        valid_pending=valid,
        topic=s.topic_text if valid else None,
        pending_question=s.pending_question if valid else None,
        session_mode=s.mode,
    )


def resolve_mode(requested: Optional[Mode], rules: RuleFlags, view: SessionContextView,
                 prediction: Optional[Prediction], mode_threshold: float) -> tuple[Mode, ModeSource]:
    """UI > explicit anchored text switch > owned session > confident prediction > default answer.

    Mode controls presentation only; it never alters authorization or risk thresholds.
    """
    if requested is not None:
        return requested, ModeSource.ui
    if rules.text_mode_switch is not None:
        return rules.text_mode_switch, ModeSource.text_switch
    if view.session_mode is not None:
        return view.session_mode, ModeSource.session
    if prediction is not None:
        best = max(prediction.mode, key=lambda k: (prediction.mode[k], k == "answer"))
        if prediction.mode[best] >= mode_threshold:
            return Mode(best), ModeSource.predicted
    return Mode.answer, ModeSource.default


# ----------------------------------------------------------------------------- store


class InMemorySessionStore:
    """Development store with atomic compare-and-swap. Stores no raw student history."""

    def __init__(self, ttl_minutes: int = 30, clock: Callable[[], datetime] = utcnow):
        self._lock = threading.Lock()
        self._data: dict[str, SessionState] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._clock = clock

    def create(self, *, owner_code: str, tenant_id: str, course_id: str, mode: Mode, policy_version: str,
               kb_version: str, **fields) -> SessionState:
        s = SessionState(
            session_id=str(uuid.uuid4()), owner_code=owner_code, tenant_id=tenant_id, course_id=course_id,
            mode=mode, policy_version=policy_version, kb_version=kb_version,
            expires_at=self._clock() + self._ttl, **fields,
        )
        with self._lock:
            self._data[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._data.get(session_id)

    def lookup_owned(self, session_id: str, *, owner_code: str, tenant_id: str, course_id: str) -> SessionState:
        s = self.get(session_id)
        if s is None:
            raise GateError(ErrorCode.SESSION_EXPIRED)
        if s.owner_code != owner_code or s.tenant_id != tenant_id or s.course_id != course_id:
            raise GateError(ErrorCode.FORBIDDEN)
        if s.is_expired(self._clock()) or s.state == "closed":
            raise GateError(ErrorCode.SESSION_EXPIRED)
        return s

    def compare_and_swap(self, session_id: str, expected_revision: int, **changes) -> SessionState:
        with self._lock:
            s = self._data.get(session_id)
            if s is None:
                raise GateError(ErrorCode.SESSION_EXPIRED)
            if s.revision != expected_revision:
                raise GateError(ErrorCode.SESSION_CONFLICT)
            new = s.model_copy(update={**changes, "revision": s.revision + 1,
                                       "expires_at": self._clock() + self._ttl})
            self._data[session_id] = new
            return new

    def apply_gate_update(self, update: SessionUpdate, **extra) -> SessionState:
        clear = {"pending_question": None, "pending_item_id": None, "item_version": None}
        if update.operation == "close":
            changes = {"state": "closed", **clear}
        elif update.operation == "pause":
            changes = {"state": "paused"}
        else:  # reset_pending
            changes = {"state": "idle", **clear}
        return self.compare_and_swap(update.session_id, update.expected_revision, **changes, **extra)
