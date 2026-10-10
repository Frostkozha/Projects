"""SQLite application store, migrations and live registry (Integration plan v0.3, sections 4, 8, appendix B).

One dedicated connection lives on one dedicated database thread (``isolation_level=None``, WAL, foreign
keys, bounded busy timeout). Async code submits work with ``await store.call(fn)``; nothing blocks the
event loop and the connection is never shared between threads. Parameterized SQL only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .transactions import immediate_transaction

T = TypeVar("T")
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("aware datetime required")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path | str, *, clock: Callable[[], datetime] = utc_now):
        self.path = str(path)
        self.clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="app-db")
        self._thread_id: Optional[int] = None
        self._conn: Optional[sqlite3.Connection] = None
        self.fail_audit = False  # chaos switch for tests (I09/I10)
        self._pool.submit(self._open).result()

    # ------------------------------------------------------------------ connection

    def _open(self) -> None:
        self._thread_id = threading.get_ident()
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=1.0, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=1000")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        self._conn = conn

    def run_sync(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if threading.get_ident() == self._thread_id:
            raise StoreError("reentrant_store_call")
        return self._pool.submit(lambda: fn(self._conn)).result(timeout=10)

    async def call(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        loop = asyncio.get_running_loop()
        return await asyncio.wrap_future(self._pool.submit(lambda: fn(self._conn)), loop=loop)

    def close(self) -> None:
        def _close(conn):
            conn.close()
        try:
            self._pool.submit(_close, self._conn).result(timeout=5)
        finally:
            self._pool.shutdown(wait=True)

    def now(self) -> str:
        return utc_str(self.clock())

    # ------------------------------------------------------------------ migrations

    def migrate(self) -> list[str]:
        def run(conn: sqlite3.Connection) -> list[str]:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, sha256 TEXT NOT NULL,"
                         " applied_at TEXT NOT NULL)")
            applied = {r["version"]: r["sha256"] for r in conn.execute("SELECT version, sha256 FROM schema_migrations")}
            done = []
            for f in sorted(MIGRATIONS.glob("*.sql")):
                sql = f.read_text(encoding="utf-8")
                sha = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if f.stem in applied:
                    if applied[f.stem] != sha:
                        raise StoreError(f"migration {f.stem} changed after it was applied")
                    continue
                with immediate_transaction(conn):
                    for stmt in _statements(sql):
                        conn.execute(stmt)
                    conn.execute("INSERT INTO schema_migrations VALUES (?,?,?)", (f.stem, sha, self.now()))
                done.append(f.stem)
            return done
        return self.run_sync(run)

    def schema_versions(self) -> list[str]:
        return self.run_sync(lambda c: [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY 1")])

    def recover_startup(self) -> int:
        """Mark interrupted attempts failed. Model calls are never replayed (I27)."""
        def run(conn):
            with immediate_transaction(conn):
                return conn.execute("UPDATE requests SET state='failed', updated_at=? WHERE state IN "
                                    "('admitted','running')", (self.now(),)).rowcount
        return self.run_sync(run)

    # ------------------------------------------------------------------ live registry

    def load_registry(self, *, registry_version: str, sources: list[dict], course_id: str) -> None:
        """Project the reviewed source register into the live registry. Never weakens an existing block."""
        def run(conn):
            now = self.now()
            with immediate_transaction(conn):
                if conn.execute("SELECT 1 FROM registry_meta").fetchone() is None:
                    conn.execute("INSERT INTO registry_meta VALUES (1, ?, 0, ?)", (registry_version, now))
                for s in sources:
                    existing = conn.execute("SELECT status FROM source_live_state WHERE source_id=? AND "
                                            "source_version=?", (s["source_id"], s["source_version"])).fetchone()
                    if existing is None:
                        conn.execute("INSERT INTO source_live_state VALUES (?,?,?,?,?,?)",
                                     (s["source_id"], s["source_version"], s["status"], s.get("rights_expires_at"),
                                      s.get("review_due_at"), now))
                    # an existing row keeps its current (possibly revoked) status: a restore cannot re-allow it
                if conn.execute("SELECT 1 FROM deployment_state WHERE course_id=?", (course_id,)).fetchone() is None:
                    conn.execute("INSERT INTO deployment_state VALUES (?, 'active', 0, ?, 'bootstrap')", (course_id, now))
        self.run_sync(run)

    @staticmethod
    def registry_epoch(conn) -> int:
        row = conn.execute("SELECT epoch FROM registry_meta WHERE singleton=1").fetchone()
        if row is None:
            raise StoreError("registry_unavailable")
        return int(row["epoch"])

    @staticmethod
    def sources_eligible(conn, pairs: list[tuple[str, str]], now: str) -> dict[tuple[str, str], bool]:
        out = {}
        for sid, ver in pairs:
            r = conn.execute("SELECT status, rights_expires_at, review_due_at FROM source_live_state WHERE "
                             "source_id=? AND source_version=?", (sid, ver)).fetchone()
            out[(sid, ver)] = bool(r is not None and r["status"] == "live"
                                   and (r["rights_expires_at"] is None or r["rights_expires_at"] > now)
                                   and (r["review_due_at"] is None or r["review_due_at"] > now))
        return out

    def set_source_status(self, source_id: str, source_version: Optional[str], status: str) -> int:
        def run(conn):
            with immediate_transaction(conn):
                q = "UPDATE source_live_state SET status=?, updated_at=? WHERE source_id=?"
                args: list[Any] = [status, self.now(), source_id]
                if source_version is not None:
                    q += " AND source_version=?"
                    args.append(source_version)
                n = conn.execute(q, args).rowcount
                conn.execute("UPDATE registry_meta SET epoch=epoch+1, updated_at=? WHERE singleton=1", (self.now(),))
                return n
        return self.run_sync(run)

    @staticmethod
    def deployment(conn, course_id: str) -> tuple[str, int]:
        r = conn.execute("SELECT state, revision FROM deployment_state WHERE course_id=?", (course_id,)).fetchone()
        if r is None:
            raise StoreError("deployment_state_missing")
        return r["state"], int(r["revision"])

    def set_deployment(self, course_id: str, state: str, expected_revision: int, actor: str) -> int:
        def run(conn):
            with immediate_transaction(conn):
                n = conn.execute("UPDATE deployment_state SET state=?, revision=revision+1, updated_at=?, updated_by=? "
                                 "WHERE course_id=? AND revision=?",
                                 (state, self.now(), actor, course_id, expected_revision)).rowcount
                if n != 1:
                    raise StoreError("deployment_revision_conflict")
                self.audit(conn, event="deployment_change", reason=state, component_status=actor)
                return expected_revision + 1
        return self.run_sync(run)

    # ------------------------------------------------------------------ audit and outbox

    def audit(self, conn, **fields) -> None:
        """Metadata-only audit row inside the caller's transaction. Raises when the audit store is failing."""
        if self.fail_audit:
            raise StoreError("audit_unavailable")
        allowed = {"event", "request_id", "owner_ref", "route", "response_code", "reason", "component_status",
                   "versions", "evidence_ids", "stage_ms", "authorization_digest"}
        if set(fields) - allowed:
            raise StoreError("non_metadata_audit_field")
        row = {k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v) for k, v in fields.items()}
        cols = ["event_time"] + list(row)
        conn.execute(f"INSERT INTO audit_events ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [self.now(), *row.values()])


def _statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
