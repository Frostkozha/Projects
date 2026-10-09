"""Internal HTTP wrapper: auth, body limit before parsing, strict schema, server-side evidence_ref."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.verifier.conftest import TEXTS, Kit, S, draft, request
from verifier.http import EvidenceRefError, create_app

TOKEN = "test-only-token"


@pytest.fixture
def client():
    kit = Kit()
    refs = {}

    def resolve(ref, req, service_id):
        if ref not in refs or service_id != "orchestrator":
            raise EvidenceRefError(ref)
        return kit.context(req, kit.bundle(req.request_id))

    app = create_app(kit.verifier, resolve, {"orchestrator": TOKEN})
    c = TestClient(app)
    c.refs, c.kit = refs, kit
    return c


def body(req, ref="ref-1") -> bytes:
    return json.dumps({"request": json.loads(req.model_dump_json()), "evidence_ref": ref}).encode()


def post(client, data, token=TOKEN):
    headers = {"content-type": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    return client.post("/v1/verify", content=data, headers=headers)


def test_auth_codes(client):
    req = request(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert post(client, body(req), token=None).status_code == 401
    assert post(client, body(req), token="wrong").status_code == 403


def test_body_limit_and_schema(client):
    assert post(client, b"{" + b" " * (300 * 1024) + b"}").status_code == 413
    assert post(client, b'{"request": {}, "request": {}}').status_code == 422
    req = request(draft(S("s1", TEXTS["p1"], ["p1"])))
    data = json.loads(body(req))
    data["evidence_text"] = "caller-supplied evidence is never accepted"
    assert post(client, json.dumps(data).encode()).status_code == 422


def test_domain_decision_200_and_unknown_ref_403(client):
    req = request(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert post(client, body(req, "unknown")).status_code == 403
    client.refs["ref-1"] = True
    r = post(client, body(req))
    assert r.status_code == 200 and r.json()["status"] == "approved"
    bad = request(draft(S("s1", "Goblet cells are in the colon.", ["p1"])))
    r2 = post(client, body(bad))
    assert r2.status_code == 200 and r2.json()["response_code"] == "A5"


def test_operational_error_is_503(client):
    client.refs["ref-1"] = True
    client.kit.registry.down = True
    r = post(client, body(request(draft(S("s1", TEXTS["p1"], ["p1"])))))
    assert r.status_code == 503 and r.json()["reply_key"] == "service_unavailable"


def test_health_exposes_no_text(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and set(r.json()) == {"status", "operating_mode", "student_release_ready"}
