"""Final binding and authorization transaction (Integration plan v0.3, section 8, appendix B).

One ``BEGIN IMMEDIATE`` transaction is the linearization point. It rechecks the request state, the
deployment state/revision, current source eligibility at the live registry epoch, and the session
compare-and-swap; then writes the minimal audit event, the delivery authorization digest, the session
transition and any learning event, and commits. Only the committed payload leaves the coordinator. A
revocation or suspension after commit cannot recall authorized bytes; later access is denied.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from .public import SessionView
from .review import next_review
from .store import Store, StoreError, parse_utc, utc_str
from .transactions import SessionConflict, immediate_transaction


class Suspended(Exception):
    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


class SourceIneligible(Exception):
    pass


class RequestStateError(Exception):
    pass


SESSION_COLUMNS = ("session_id", "owner_code", "tenant_id", "course_id", "mode", "topic_id", "topic_text",
                   "pending_item_id", "item_version", "pending_question", "pending_kind", "state", "revision",
                   "expires_at", "policy_version", "kb_version", "created_at")


@dataclass
class AuthPlan:
    request_id: str
    owner_code: str
    tenant_id: str
    course_id: str
    response_code: str
    content_type: str
    payload_digest: str
    deployment_revision: int
    pinned_versions: dict
    owner_ref: str
    route: str
    reason: str
    cited_sources: list[tuple[str, str]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    session_id: Optional[str] = None        # existing owned session
    expected_revision: Optional[int] = None
    session_op: Optional[str] = None        # None | create | update | pause | close
    session_changes: dict = field(default_factory=dict)
    learning: Optional[dict] = None         # item_id, item_version, topic_id, correct
    fixed_payload: Optional[str] = None
    stage_ms: dict = field(default_factory=dict)
    welfare_category: Optional[str] = None  # emergency | self_harm (A7)
    support_ref: Optional[str] = None
    incident: Optional[dict] = None         # category, near_miss (A6 draft policy)


def session_row(conn, session_id: str) -> Optional[dict]:
    r = conn.execute(f"SELECT {','.join(SESSION_COLUMNS)} FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return dict(r) if r else None


def view(row: dict) -> SessionView:
    return SessionView(session_id=row["session_id"], revision=row["revision"], state=row["state"], mode=row["mode"],
                       expires_at=row["expires_at"])


def authorize(store: Store, plan: AuthPlan, *, ttl_minutes: int = 30) -> Optional[SessionView]:
    """Run the final transaction on the store thread. Returns the committed SessionView (or None)."""

    def run(conn: sqlite3.Connection) -> Optional[SessionView]:
        now_dt = store.clock()
        now = utc_str(now_dt)
        expires = utc_str(now_dt + timedelta(minutes=ttl_minutes))
        with immediate_transaction(conn):
            req = conn.execute("SELECT state, owner_code FROM requests WHERE request_id=?",
                               (plan.request_id,)).fetchone()
            if req is None or req["state"] != "running" or req["owner_code"] != plan.owner_code:
                raise RequestStateError("request_not_running")
            state, revision = Store.deployment(conn, plan.course_id)
            if plan.response_code != "A7" and (state != "active" or revision != plan.deployment_revision):
                raise Suspended(state if state != "active" else "suspended")
            epoch = Store.registry_epoch(conn)
            if plan.cited_sources:
                ok = Store.sources_eligible(conn, plan.cited_sources, now)
                if not all(ok.values()):
                    raise SourceIneligible()
            result_view = _apply_session(conn, plan, now, expires)
            if plan.learning is not None:
                _learning(conn, plan, now_dt, now, result_view)
            if plan.welfare_category:
                conn.execute("INSERT OR IGNORE INTO welfare_outbox VALUES (?,?,?,?,?, 'pending', NULL, NULL)",
                             (str(uuid.uuid4()), plan.request_id, plan.welfare_category, plan.support_ref, now))
            if plan.incident:
                conn.execute("INSERT INTO incident_events VALUES (?,?,?,?,?, NULL)",
                             (str(uuid.uuid4()), plan.request_id, plan.incident["category"],
                              int(plan.incident.get("near_miss", 0)), now))
            pinned = json.dumps(plan.pinned_versions, sort_keys=True)
            conn.execute("INSERT INTO delivery_authorizations VALUES (?,?,?,?,?,?,?,?)",
                         (plan.request_id, plan.payload_digest, plan.response_code, plan.content_type, epoch,
                          revision, pinned, now))
            store.audit(conn, event="final_response", request_id=plan.request_id, owner_ref=plan.owner_ref,
                        route=plan.route, response_code=plan.response_code, reason=plan.reason,
                        component_status="authorized", versions=plan.pinned_versions, evidence_ids=plan.evidence_ids,
                        stage_ms=plan.stage_ms, authorization_digest=plan.payload_digest)
            conn.execute("UPDATE requests SET state='authorized', response_code=?, content_type=?, fixed_payload=?, "
                         "session_id=COALESCE(?, session_id), updated_at=? WHERE request_id=?",
                         (plan.response_code, plan.content_type, plan.fixed_payload,
                          result_view.session_id if result_view else None, now, plan.request_id))
        return result_view

    return store.run_sync(run)


def _apply_session(conn, plan: AuthPlan, now: str, expires: str) -> Optional[SessionView]:
    op = plan.session_op
    if plan.session_id is None:
        if op != "create":
            return None
        sid = str(uuid.uuid4())
        ch = plan.session_changes
        conn.execute(f"INSERT INTO sessions ({','.join(SESSION_COLUMNS)}) VALUES ({','.join('?' * len(SESSION_COLUMNS))})",
                     (sid, plan.owner_code, plan.tenant_id, plan.course_id, ch["mode"], ch.get("topic_id"),
                      ch.get("topic_text"), None, None, None, None, "idle", 0, expires, ch["policy_version"],
                      ch["kb_version"], now))
        plan.session_id, plan.expected_revision = sid, 0
        op = "update"
    row = session_row(conn, plan.session_id)
    if (row is None or row["owner_code"] != plan.owner_code or row["tenant_id"] != plan.tenant_id
            or row["course_id"] != plan.course_id):
        raise SessionConflict("SESSION_CONFLICT")
    if row["revision"] != plan.expected_revision:
        raise SessionConflict("SESSION_CONFLICT")
    if op is None:
        if row["state"] == "closed" or row["expires_at"] <= now:
            raise SessionConflict("SESSION_EXPIRED")
        return view(row)  # no transition (e.g. A3, A5): revision unchanged
    sets: dict = {}
    if op == "update":
        sets = dict(plan.session_changes)
        sets.pop("policy_version", None)
        sets.pop("kb_version", None)
        sets["expires_at"] = expires  # refreshed only by an authorized permitted activity
    elif op == "pause":
        sets = {"state": "paused", "pending_item_id": None, "item_version": None, "pending_question": None,
                "pending_kind": None}
    elif op == "close":
        sets = {"state": "closed", "pending_item_id": None, "item_version": None, "pending_question": None,
                "pending_kind": None}
    elif op == "reset_pending":
        sets = {"state": "idle", "pending_item_id": None, "item_version": None, "pending_question": None,
                "pending_kind": None}
    else:
        raise StoreError("unknown_session_op")
    allowed = set(SESSION_COLUMNS) - {"session_id", "owner_code", "tenant_id", "course_id", "revision", "created_at"}
    if set(sets) - allowed:
        raise StoreError("invalid_session_change")
    assign = ", ".join(f"{k}=?" for k in sets)
    n = conn.execute(f"UPDATE sessions SET {assign}, revision=revision+1 WHERE session_id=? AND revision=? "
                     "AND state!='closed' AND expires_at>?",
                     (*sets.values(), plan.session_id, plan.expected_revision, now)).rowcount
    if n != 1:
        raise SessionConflict("SESSION_CONFLICT")
    return view(session_row(conn, plan.session_id))


def _learning(conn, plan: AuthPlan, now_dt, now: str, sv: Optional[SessionView]) -> None:
    ev = plan.learning
    # unique (session_id, expected_revision, item_id, item_version) prevents duplicate advancement
    conn.execute("INSERT INTO learning_events VALUES (?,?,?,?,?,?,?,?)",
                 (plan.request_id, sv.session_id, sv.revision - 1, ev["item_id"], ev["item_version"],
                  ev["topic_id"], int(ev["correct"]), now))
    r = conn.execute("SELECT streak, attempts, correct FROM item_review_state WHERE owner_code=? AND course_id=? "
                     "AND item_id=? AND item_version=?", (plan.owner_code, plan.course_id, ev["item_id"],
                                                          ev["item_version"])).fetchone()
    prev, attempts, correct = (r["streak"], r["attempts"], r["correct"]) if r else (0, 0, 0)
    streak, due = next_review(now_dt, correct=bool(ev["correct"]), previous_streak=prev)
    conn.execute("INSERT INTO item_review_state VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(owner_code, course_id, item_id, "
                 "item_version) DO UPDATE SET streak=excluded.streak, due_at=excluded.due_at, "
                 "attempts=excluded.attempts, correct=excluded.correct",
                 (plan.owner_code, plan.course_id, ev["item_id"], ev["item_version"], streak, utc_str(due),
                  attempts + 1, correct + int(ev["correct"])))
    conn.execute("INSERT INTO topic_practice VALUES (?,?,?,1,?) ON CONFLICT(owner_code, course_id, topic_id) DO UPDATE "
                 "SET attempts=attempts+1, correct=correct+excluded.correct",
                 (plan.owner_code, plan.course_id, ev["topic_id"], int(ev["correct"])))


def a7_outage_record(store: Store, plan: AuthPlan) -> bool:
    """Best-effort durable outbox write when the normal A7 transaction failed. Never claims delivery."""
    def run(conn):
        with immediate_transaction(conn):
            conn.execute("INSERT OR IGNORE INTO welfare_outbox VALUES (?,?,?,?,?, 'pending', NULL, NULL)",
                         (str(uuid.uuid4()), plan.request_id, plan.welfare_category, plan.support_ref, store.now()))
    try:
        store.run_sync(run)
        return True
    except Exception:  # noqa: BLE001
        return False


def is_expired(row: dict, now: str) -> bool:
    return row["expires_at"] <= now or row["state"] == "closed"


def parse(row_time: str):
    return parse_utc(row_time)
