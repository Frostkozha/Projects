"""SQLite transaction primitives (Integration plan v0.3, appendix B)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def immediate_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    if conn.isolation_level is not None or conn.in_transaction:
        raise RuntimeError("dedicated_explicit_transaction_connection_required")
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


class SessionConflict(ValueError):
    pass


def close_session_cas(
    conn: sqlite3.Connection, *, session_id: str, owner_code: str,
    tenant_id: str, course_id: str, expected_revision: int, now_utc: str,
) -> None:
    if not conn.in_transaction:
        raise RuntimeError("authorization_transaction_required")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("invalid_revision")
    changed = conn.execute(
        """UPDATE sessions
           SET state='closed', revision=revision+1,
               pending_item_id=NULL, item_version=NULL,
               pending_question=NULL
           WHERE session_id=? AND owner_code=? AND tenant_id=? AND course_id=?
             AND revision=? AND expires_at>? AND state!='closed'""",
        (session_id, owner_code, tenant_id, course_id, expected_revision, now_utc),
    ).rowcount
    if changed != 1:
        raise SessionConflict("SESSION_CONFLICT")
