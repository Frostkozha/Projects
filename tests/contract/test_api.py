"""Internal HTTP API contract tests (spec sections 5.1, 5.3, 13)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gate_classifier.api import create_app
from tests.conftest import Harness, ScriptedScorer, scores

AUTH = {"Authorization": "Bearer service-secret", "x-gate-owner": "owner-a", "x-gate-tenant": "dev-tenant",
        "x-gate-course": "histology-dev"}


@pytest.fixture
def setup():
    h = Harness()
    app = create_app(h.service, lambda o, t, c, s: h.orch.build_context(o, t, c, s), ["service-secret"],
                     apply_session_update=h.sessions.apply_gate_update)
    return h, TestClient(app)


def envelope(r, status, code):
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error_code", "request_id"} and body["error_code"] == code


def test_retrieve_decision_returned(setup):
    h, c = setup
    r = c.post("/v1/gate", json={"text": "Explain simple squamous epithelium."}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "retrieve" and body["schema_version"] == "gate-decision-0.2"
    assert body["retrieval_plan"]["allowed_libraries"] == ["lib1", "lib2", "lib5", "lib6"]


def test_unauthorized(setup):
    _, c = setup
    envelope(c.post("/v1/gate", json={"text": "x"}), 401, "UNAUTHORIZED")
    envelope(c.post("/v1/gate", json={"text": "x"}, headers={**AUTH, "Authorization": "Bearer wrong"}), 401,
             "UNAUTHORIZED")


@pytest.mark.parametrize("payload", [b"not json", b"[1,2]", b'{"text": 5}', b'{"text": "a", "x": 1}',
                                     b'{"text": "a", "text": "b"}'])
def test_invalid_requests(setup, payload):
    _, c = setup
    envelope(c.post("/v1/gate", content=payload, headers={**AUTH, "content-type": "application/json"}), 422,
             "INVALID_REQUEST")


def test_empty_and_oversize(setup):
    _, c = setup
    envelope(c.post("/v1/gate", json={"text": "   "}, headers=AUTH), 422, "EMPTY_INPUT")
    envelope(c.post("/v1/gate", json={"text": "a" * 40000}, headers=AUTH), 413, "INPUT_TOO_LONG")
    envelope(c.post("/v1/gate", json={"text": "epithelium " * 500}, headers=AUTH), 413, "INPUT_TOO_LONG")


def test_forbidden_and_session_errors(setup):
    h, c = setup
    envelope(c.post("/v1/gate", json={"text": "x y"}, headers={**AUTH, "x-gate-course": "other"}), 403, "FORBIDDEN")
    envelope(c.post("/v1/gate", json={"text": "B", "session_id": "00000000-0000-4000-8000-000000000001"},
                    headers=AUTH), 409, "SESSION_EXPIRED")


def test_maintenance_and_exam_states(setup):
    h, c = setup
    h.orch.deployment.states["histology-dev"] = "suspended"
    envelope(c.post("/v1/gate", json={"text": "Explain epithelium."}, headers=AUTH), 503, "MAINTENANCE")
    h.orch.deployment.states["histology-dev"] = "exam_shutdown"
    envelope(c.post("/v1/gate", json={"text": "Explain epithelium."}, headers=AUTH), 503, "EXAM_DISABLED")


def test_error_body_never_echoes_text():
    h = Harness(ScriptedScorer(scores()), fixture_scorer=lambda t: (_ for _ in ()).throw(RuntimeError(t)))
    c = TestClient(create_app(h.service, lambda o, t, cc, s: h.orch.build_context(o, t, cc, s), ["service-secret"]))
    r = c.post("/v1/gate", json={"text": "Explain UNIQUE-MARKER-42 epithelium."}, headers=AUTH)
    envelope(r, 503, "SERVICE_UNAVAILABLE")
    assert "UNIQUE-MARKER-42" not in r.text


def test_ready_fixture_mode(setup):
    _, c = setup
    r = c.get("/health/ready")
    assert r.status_code == 200 and r.json() == {"status": "fixture_only"}
