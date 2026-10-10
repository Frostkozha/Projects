"""Integration acceptance I21-I40 (Integration plan v0.3, section 13). Synthetic fixtures only."""

from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tutor_app import items as itm
from tutor_app.auth import HmacPseudonymResolver, IdentityError, TrustedProxyJwtIdentity
from tutor_app.capacity import CircuitBreaker
from tutor_app.config import TutorConfigError, load_tutor_config
from tutor_app.fixtures import FixtureBrain
from tutor_app.review import next_review
from tutor_app.store import parse_utc

from .conftest import COURSE, NOTICE, Q, ROOT, scores


def quiz(h, user="student-a"):
    status, out = h.ask("Quiz me on epithelium", user=user)
    assert status == 200 and out["activity"]["phase"] == "question", out
    return out


# ----------------------------------------------------------------------------- I21-I23 revocation, suspension

def test_I21_revocation_before_brain_means_no_generation(h):
    h.store.set_source_status("src-lib1", None, "revoked")
    h.store.set_source_status("src-lib2", None, "revoked")
    status, out = h.ask(Q)
    assert out["response_code"] == "A5" and h.calls["brain"] == 0


def test_I21_revocation_between_verify_and_authorization_is_fixed_A5(h):
    h.verifier.before_return = lambda r: h.store.set_source_status("src-lib1", None, "revoked")
    status, out = h.ask(Q)
    assert status == 200 and out["response_code"] == "A5" and out["content_type"] == "fixed"
    assert out["citations"] == []


def test_I22_revocation_after_authorization_denies_later_access(h):
    status, out = h.ask(Q)
    url = out["citations"][0]["url"]
    assert h.client.get(url).status_code == 200
    h.store.set_source_status("src-lib1", None, "revoked")
    r = h.client.get(url)
    assert (r.status_code, r.json()["error_code"]) == (404, "CONTENT_UNAVAILABLE")
    assert len(h.q("SELECT * FROM delivery_authorizations WHERE request_id=?", out["request_id"])) == 1


def test_I23_suspension_before_authorization_suppresses_delivery_and_learning(h):
    q = quiz(h)
    h.verifier.before_return = None
    orig = h.coord._item_evidence

    async def suspend_then(a, item):
        out = await orig(a, item)
        h.store.set_deployment(COURSE, "suspended", 0, "test")
        return out
    h.coord._item_evidence = suspend_then
    status, out = h.ask("B", session=q["session"])
    assert (status, out["error_code"]) == (503, "MAINTENANCE")
    assert h.q("SELECT * FROM learning_events") == []
    h.coord._item_evidence = orig
    status, out = h.ask(Q)  # gate sees the suspended deployment before any teaching call
    assert (status, out["error_code"]) == (503, "MAINTENANCE")
    h.store.set_deployment(COURSE, "exam_shutdown", 1, "test")
    assert h.ask(Q)[1]["error_code"] == "EXAM_DISABLED"


# ----------------------------------------------------------------------------- I24-I28 concurrency, idempotency

def test_I24_two_answers_at_one_pending_revision_one_event(h):
    q = quiz(h)
    s1, a = h.ask("B", session=q["session"])
    s2, b = h.ask("A", session=q["session"])  # same expected revision: compare-and-swap fails
    assert (s1, s2) == (200, 409) and b["error_code"] == "SESSION_CONFLICT"
    assert len(h.q("SELECT * FROM learning_events")) == 1


def test_I24_concurrent_submissions_through_the_coordinator(h):
    q = quiz(h)
    ident = h.tutor.identity.identify({}, {"tutor_dev_identity": h.cookies["student-a"]})

    async def both():
        def raw(opt):
            return json.dumps({"text": opt, "requested_mode": None, "session_id": q["session"]["session_id"],
                               "expected_session_revision": q["session"]["revision"],
                               "notice_version": NOTICE}).encode()
        tasks = [h.coord.interact(ident, COURSE, str(uuid.uuid4()), raw(o)) for o in ("A", "B")]
        return await asyncio.gather(*tasks, return_exceptions=True)
    results = asyncio.run(both())
    ok = [r for r in results if not isinstance(r, Exception)]
    assert len(ok) == 1 and len(h.q("SELECT * FROM learning_events")) == 1


def test_I25_same_key_while_running_and_completed_repeat(make):
    h = make(brain=FixtureBrain(delay=0.8))
    key = str(uuid.uuid4())
    out = {}

    def first():
        out["a"] = h.ask(Q, key=key)
    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.3)
    status, dup = h.ask(Q, key=key)
    t.join(10)
    assert (status, dup["error_code"]) == (409, "REQUEST_IN_PROGRESS")
    assert out["a"][0] == 200
    s, again = h.ask(Q, key=key)  # completed generated repeat: rechecked, same authorized payload
    assert s == 200 and again["request_id"] == out["a"][1]["request_id"] and h.calls["brain"] == 1
    h.store.set_source_status("src-lib1", None, "revoked")
    s, exp = h.ask(Q, key=key)
    assert (s, exp["error_code"]) == (409, "REQUEST_EXPIRED")


def test_I26_changed_body_or_owner_scope(h):
    key = str(uuid.uuid4())
    assert h.ask(Q, key=key)[0] == 200
    s, out = h.ask("Describe goblet cells.", key=key)
    assert (s, out["error_code"]) == (409, "IDEMPOTENCY_CONFLICT")
    h.login("student-b")
    s, other = h.ask(Q, key=key, user="student-b")  # same key, other owner: isolated
    assert s == 200 and h.calls["brain"] == 2


def test_I27_restart_marks_running_requests_failed_without_replay(make, tmp_path):
    h = make()
    rid = str(uuid.uuid4())
    key = str(uuid.uuid4())
    owner = h.tutor.identity.identify({}, {"tutor_dev_identity": h.cookies["student-a"]})
    fp = h.coord._fingerprint(owner, json.dumps({"text": Q, "requested_mode": None, "session_id": None,
                                                 "expected_session_revision": None,
                                                 "notice_version": NOTICE}).encode())
    h.store.run_sync(lambda c: c.execute(
        "INSERT INTO requests (request_id, owner_code, course_id, idempotency_key, body_fingerprint, pinned_versions, "
        "state, created_at, updated_at) VALUES (?,?,?,?,?, '{}', 'running', ?, ?)",
        (rid, owner.owner_code, COURSE, key, fp, h.store.now(), h.store.now())))
    assert h.store.recover_startup() == 1
    assert h.q("SELECT state FROM requests WHERE request_id=?", rid) == [{"state": "failed"}]
    s, out = h.ask(Q, key=key)
    assert (s, out["error_code"]) == (409, "REQUEST_EXPIRED") and h.calls["brain"] == 0


def test_I28_expired_private_buffer(make):
    h = make(cfg_update={"response_buffer_seconds": 0})
    key = str(uuid.uuid4())
    assert h.ask(Q, key=key)[0] == 200
    s, out = h.ask(Q, key=key)
    assert (s, out["error_code"]) == (409, "REQUEST_EXPIRED")
    assert all(r["fixed_payload"] is None for r in h.q("SELECT fixed_payload FROM requests WHERE "
                                                        "content_type='generated'"))
    k2 = str(uuid.uuid4())
    s1, f1 = h.ask("B", key=k2)          # fixed A3: deterministic record is replayable
    s2, f2 = h.ask("B", key=k2)
    assert s1 == s2 == 200 and f1 == f2


# ----------------------------------------------------------------------------- I29-I33 learning paths

def test_I29_bare_letter_uses_exact_item_fetch_not_search(h):
    q = quiz(h)
    searches = h.calls["retrieve"]
    s, out = h.ask("b", session=q["session"])
    assert s == 200 and out["activity"]["phase"] == "feedback" and out["body"].startswith("Correct.")
    assert h.calls["retrieve"] == searches and h.calls["brain"] == 0
    assert out["citations"][0]["passage_id"] == "fixture-p2"


def test_I30_invalid_option_and_unsafe_reply_never_count_as_wrong(make):
    h = make()
    q = quiz(h)
    s, out = h.ask("F", session=q["session"])
    assert out["response_code"] in ("A3", "A2", "A5")
    assert h.q("SELECT * FROM learning_events") == []
    sess = h.client.get(f"/v1/sessions/{q['session']['session_id']}").json()
    h.comps.gate.fixture_scorer = lambda c: scores(risks={"real_person_advice": 0.9})
    s, out = h.ask("my mother has this, what should she take", session=sess)
    assert out["response_code"] == "A6" and out["session"]["state"] == "paused"
    assert h.q("SELECT * FROM learning_events") == []


def test_I31_changed_item_or_versions_invalidate_pending(h):
    q = quiz(h)
    item_id = q["activity"]["item_id"]
    h.coord.bank = h.coord.bank.model_copy(update={"quiz_items": [i for i in h.coord.bank.quiz_items
                                                                  if i.item_id != item_id]})
    s, out = h.ask("B", session=q["session"])
    assert (s, out["error_code"]) == (409, "SESSION_CONFLICT")
    row = h.q("SELECT state, pending_item_id FROM sessions")[0]
    assert row == {"state": "idle", "pending_item_id": None}


def test_I31_kb_change_resets_pending(h):
    q = quiz(h)
    h.store.run_sync(lambda c: c.execute("UPDATE sessions SET kb_version='old-kb'"))
    s, out = h.ask("B", session=q["session"])
    assert (s, out["error_code"]) == (409, "SESSION_CONFLICT")
    assert h.q("SELECT pending_item_id FROM sessions")[0]["pending_item_id"] is None


def test_I32_fixed_tutor_path_has_no_brain_call(h):
    s, t = h.ask("Tutor me on epithelium")
    assert t["activity"]["mode"] == "tutor" and t["activity"]["options"] == []
    s, x = h.ask("show explanation", session=t["session"])
    assert x["body"] == "Simple squamous epithelium is a single layer of flat cells." and x["citations"]
    s, n = h.ask("continue", session=x["session"])
    assert n["activity"]["item_id"] != t["activity"]["item_id"]
    h.coord.bank = h.coord.bank.model_copy(update={"tutor_items": [i for i in h.coord.bank.tutor_items
                                                                   if i.item_id == n["activity"]["item_id"]]})
    s, done = h.ask("continue", session=n["session"])  # no other approved item: practice complete
    assert done["response_code"] == "A5" and done["session"]["state"] == "idle"
    assert h.calls["brain"] == 0 and h.q("SELECT * FROM learning_events") == []
    assert h.cfg.features.tutor_clue is False


def test_I33_scheduler_indicator_and_due_order(make):
    clock = {"now": datetime.now(timezone.utc).replace(microsecond=0)}
    h = make()
    h.store.clock = lambda: clock["now"]
    q = quiz(h)
    first = q["activity"]["item_id"]
    s, fb = h.ask("A" if first == "fq-1" else "B", session=q["session"])  # wrong answer
    assert "0.33 from 1 attempt" in fb["body"]
    row = h.q("SELECT streak, due_at, attempts, correct FROM item_review_state WHERE item_id=?", first)[0]
    assert row["streak"] == 0 and parse_utc(row["due_at"]) == clock["now"] + timedelta(minutes=10)
    q2 = quiz(h)
    assert q2["activity"]["item_id"] != first  # the missed item is not yet due; unseen items are
    for prev, days in [(0, 1), (1, 3), (2, 7), (3, 7)]:
        assert next_review(clock["now"], correct=True, previous_streak=prev)[1] == clock["now"] + timedelta(days=days)
    bank = h.coord.bank
    order = itm.due_order(bank.quiz_items, {}, "2026-10-10T00:00:00.000000Z")
    assert [i.item_id for i in order] == sorted(i.item_id for i in bank.quiz_items)


# ----------------------------------------------------------------------------- I34-I36 capacity, cancel, breakers

def test_I34_full_generation_queue_is_bounded(make):
    h = make(brain=FixtureBrain(delay=1.0), cfg_update={})
    h.coord.slot.max_waiting = 0
    h.login("student-b")
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("a", h.ask(Q)))
    t.start()
    time.sleep(0.3)
    s, out = h.ask(Q, user="student-b")
    t.join(10)
    assert (s, out["error_code"]) == (429, "CAPACITY_EXCEEDED") and res["a"][0] == 200
    s, fixed = h.ask("B", user="student-b")  # fixed routes never wait for the generated slot
    assert s == 200


def test_I35_cancel_produces_no_late_delivery(make):
    h = make(brain=FixtureBrain(delay=2.0))
    key = str(uuid.uuid4())
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("a", h.ask(Q, key=key)))
    t.start()
    time.sleep(0.4)
    r = h.client.post(f"/v1/courses/{COURSE}/interactions/cancel", headers={"Idempotency-Key": key})
    assert r.json() == {"cancelled": True}
    t.join(10)
    s, out = res["a"]
    assert (s, out["error_code"]) == (409, "REQUEST_CANCELLED")
    assert h.q("SELECT * FROM delivery_authorizations") == []
    assert h.q("SELECT state FROM requests") == [{"state": "cancelled"}]


def test_I36_breaker_counts_faults_only():
    t = {"now": 0.0}
    b = CircuitBreaker("brain", clock=lambda: t["now"])
    for _ in range(3):
        b.success()  # content outcomes (A5) are recorded as successes
    assert b.state == "closed"
    for _ in range(3):
        b.failure()
    assert b.state == "open" and not b.allow()
    t["now"] = 31
    assert b.allow() and not b.allow()  # exactly one half-open probe
    b.failure()
    assert b.state == "open"
    t["now"] = 62
    assert b.allow()
    b.success()
    assert b.state == "closed"


def test_I36_breaker_opens_after_three_brain_faults(make):
    from brain.schema import ErrorCode as BC

    h = make(brain=FixtureBrain(lambda r, c: BC.TRANSPORT_FAILURE))
    for _ in range(3):
        assert h.ask(Q)[0] == 503
    calls = h.calls["brain"]
    assert h.ask(Q)[0] == 503 and h.calls["brain"] == calls  # open: no call
    h2 = make(brain=FixtureBrain(lambda r, c: BC.INVALID_OUTPUT))
    for _ in range(4):
        assert h2.ask(Q)[1]["response_code"] == "A5"
    assert h2.coord.breakers["brain"].state == "closed"


# ----------------------------------------------------------------------------- I37-I40 readiness, identity, release

def test_I37_fixture_adapters_cannot_pass_real_model(tmp_path):
    import yaml

    data = yaml.safe_load((ROOT / "config/fixture.yaml").read_text())
    data["profile"] = "real_model"
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(TutorConfigError):
        load_tutor_config(p)
    from tutor_app.components import ComponentStatus
    from tutor_app.readiness import readiness

    class Comps:
        statuses = {"gate": ComponentStatus("gate", "fixture_scoring", True), "retriever": ComponentStatus(
            "retriever", "real", True), "verifier": ComponentStatus("verifier", "real", True),
            "brain": ComponentStatus("brain", "real", True)}
        brain = type("B", (), {"manifest": None, "readiness": lambda self: {"ready": True}})()

        def brain_ready(self):
            return True
    cfg = load_tutor_config(ROOT / "config/local-real.yaml")
    from tutor_app.store import Store

    store = Store(":memory:")
    store.migrate()
    store.load_registry(registry_version="r", sources=[], course_id=cfg.course_id)
    r = readiness(cfg, store, Comps(), identity_institutional=False)
    store.close()
    assert not r["ready"] and "gate_fixture_adapter_not_allowed" in r["reasons"]
    assert "brain_manifest_missing" in r["reasons"]


def test_I37_fixture_profile_is_never_labelled_ready(h):
    r = h.tutor.readiness()
    assert r["status"] == "fixture_only" and r["profile"] == "fixture"


@pytest.fixture(scope="module")
def keys():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    def pair():
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pub = k.public_key().public_bytes(serialization.Encoding.PEM,
                                          serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        return k, pub
    return pair(), pair()


def _jwt_adapter(pub):
    return TrustedProxyJwtIdentity(public_key=pub, algorithm="RS256", issuer="https://sso.example.edu",
                                   audience="histology-tutor", tenant_id="dev-tenant",
                                   course_roles={"histology-dev": ["student"]},
                                   resolver=HmacPseudonymResolver(b"k" * 40))


def _token(key, **over):
    import jwt

    now = int(time.time())
    claims = {"iss": "https://sso.example.edu", "aud": "histology-tutor", "sub": "s123", "iat": now, "nbf": now - 5,
              "exp": now + 300, "roles": ["student"], "tenant": "dev-tenant"}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256")


def test_I38_jwt_validation(keys):
    (k1, pub1), (k2, _) = keys
    a = _jwt_adapter(pub1)
    ident = a.identify({"authorization": "Bearer " + _token(k1)}, {})
    assert ident.course_ids == frozenset({"histology-dev"}) and not ident.synthetic and "s123" not in ident.owner_code
    bad = [_token(k2), _token(k1, iss="https://evil"), _token(k1, aud="other"), _token(k1, exp=int(time.time()) - 600),
           _token(k1, nbf=int(time.time()) + 600), _token(k1, exp=None)]
    for tok in bad:
        with pytest.raises(IdentityError) as e:
            a.identify({"authorization": "Bearer " + tok}, {})
        assert e.value.code == "UNAUTHORIZED"
    for tok in (_token(k1, roles=["visitor"]), _token(k1, tenant="other")):
        with pytest.raises(IdentityError) as e:
            a.identify({"authorization": "Bearer " + tok}, {})
        assert e.value.code == "FORBIDDEN"
    import jwt
    unsigned = jwt.encode({"sub": "x", "iss": "https://sso.example.edu", "aud": "histology-tutor"}, None,
                          algorithm="none")
    for headers in ({"authorization": "Bearer " + unsigned}, {"x-forwarded-user": "s123", "x-remote-user": "s123"}, {}):
        with pytest.raises(IdentityError):
            a.identify(headers, {"tutor_dev_identity": "anything"})


def test_I39_bundle_rollback_preserves_revocations(h, tmp_path):
    from tutor_app.bundle import activate

    h.store.set_source_status("src-lib1", None, "revoked")
    good = {"schema_version": "release-bundle-0.3", "bundle_id": "fixture-b1", "profile": "fixture",
            "shared_schemas": {"draft": "brain-draft-0.2"}, "application_commit": None,
            "dependency_lock_sha256": "a" * 64, "supported_migrations": ["0001_application", "0002_registry",
                                                                         "0003_audit_outbox"],
            "components": {}, "item_bank_digest": None, "evaluation_reference": None, "approval_reference": None}
    p1 = tmp_path / "b1.json"
    p1.write_text(json.dumps(good))
    rec = activate(h.cfg, h.store, p1)
    p2 = tmp_path / "b2.json"
    p2.write_text(json.dumps({**good, "bundle_id": "fixture-b2"}))
    rec2 = activate(h.cfg, h.store, p2)
    assert rec2["previous"]["bundle_id"] == "fixture-b1"
    p3 = tmp_path / "old.json"
    p3.write_text(json.dumps({**good, "bundle_id": "old", "supported_migrations": ["0001_application"]}))
    with pytest.raises(ValueError):
        activate(h.cfg, h.store, p3)
    assert h.q("SELECT status FROM source_live_state WHERE source_id='src-lib1'") == [{"status": "revoked"}]


def test_I40_fixture_load_run_records_latency(make):
    h = make()
    users = [f"load-{i}" for i in range(4)]
    for u in users:
        h.login(u)
    lat, statuses = [], []
    lock = threading.Lock()

    def worker(u):
        for _ in range(5):
            t = time.perf_counter()
            s, _ = h.ask(Q, user=u)
            with lock:
                lat.append((time.perf_counter() - t) * 1000)
                statuses.append(s)
    ts = [threading.Thread(target=worker, args=(u,)) for u in users]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    q = statistics.quantiles(lat, n=100)
    report = {"record_type": "fixture load run (synthetic adapters; not a real-model measurement)",
              "requests": len(lat), "status_counts": {str(k): statuses.count(k) for k in set(statuses)},
              "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(q[94], 1), "p99_ms": round(q[98], 1)}
    out = ROOT / "reports" / "tutor"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture-load.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert len(lat) == 20 and set(statuses) <= {200, 429}
