import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from contracts.json_codec import canonical_json, validate_wire
from tutor_app.input_schema import InteractionInput
from tutor_app.review import next_review, practice_indicator
from tutor_app.transactions import SessionConflict, close_session_cas, immediate_transaction


class IntegrationPrimitiveTests(unittest.TestCase):
    def test_json_uuid_and_boolean_revision(self):
        value = {"text": "Fixture question", "requested_mode": "answer",
            "session_id": str(UUID(int=1)), "expected_session_revision": 0, "notice_version": "n1"}
        self.assertIsInstance(validate_wire(InteractionInput, canonical_json(value), max_bytes=32768).session_id, UUID)
        with self.assertRaises(ValueError):
            validate_wire(InteractionInput, canonical_json(value | {"expected_session_revision": True}), max_bytes=32768)

    def test_review_schedule(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for previous, days in [(0, 1), (1, 3), (2, 7), (20, 7)]:
            streak, due = next_review(now, correct=True, previous_streak=previous)
            self.assertEqual(due, now + timedelta(days=days))
            self.assertLessEqual(streak, 3)
        self.assertEqual(next_review(now, correct=False, previous_streak=3), (0, now + timedelta(minutes=10)))
        self.assertEqual(practice_indicator(correct=2, attempts=3), 3/5)

    def test_rollback_and_stale_session(self):
        conn = sqlite3.connect(":memory:", isolation_level=None)
        self.addCleanup(conn.close)
        conn.execute("""CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, owner_code TEXT, tenant_id TEXT, course_id TEXT,
            revision INTEGER, state TEXT, expires_at TEXT, pending_item_id TEXT,
            item_version TEXT, pending_question TEXT)""")
        conn.execute("INSERT INTO sessions VALUES ('s','u','t','c',1,'awaiting_response','2099-01-01T00:00:00.000000Z','i','v','q')")
        args = dict(session_id="s", owner_code="u", tenant_id="t", course_id="c", expected_revision=1, now_utc="2026-01-01T00:00:00.000000Z")
        with self.assertRaises(RuntimeError):
            with immediate_transaction(conn):
                close_session_cas(conn, **args)
                raise RuntimeError("audit_write_failed")
        self.assertEqual(conn.execute("SELECT revision FROM sessions").fetchone(), (1,))
        with immediate_transaction(conn): close_session_cas(conn, **args)
        with self.assertRaises(SessionConflict):
            with immediate_transaction(conn): close_session_cas(conn, **args)
        self.assertEqual(conn.execute("SELECT revision,state,pending_question FROM sessions").fetchone(), (2, "closed", None))


if __name__ == "__main__":
    unittest.main()
