"""Integration acceptance I01-I20 (Integration plan v0.3, section 13). Synthetic fixtures only."""

from __future__ import annotations

import json
import uuid

import pytest

from brain.schema import ErrorCode as BrainCode
from contracts.models import DraftAnswer
from tutor_app.fixtures import FixtureBrain, fixture_passage

from .conftest import COURSE, NOTICE, Q, scores


def body(**over):
    b = {"text": Q, "requested_mode": None, "session_id": None, "expected_session_revision": None,
         "notice_version": NOTICE}
    b.update(over)
    return b


def draft(*sentences, status="draft"):
    objs = [{"sentence_id": f"s{i + 1}", "text": t, "kind_hint": "factual", "visibility": vis, "cites": list(c),
             "depends_on": []} for i, (t, c, vis) in enumerate(s if len(s) == 3 else (*s, "student")
                                                              for s in sentences)]
    used = sorted({c for o in objs for c in o["cites"]})
    return DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": status, "sentences": objs,
                                       "used_passage_ids": used})


NO_CALLS = {"retrieve": 0, "brain": 0, "verify": 0}


# ----------------------------------------------------------------------------- I01-I04 admission

@pytest.mark.parametrize("raw", [
    json.dumps(body(extra=1)),
    json.dumps(body())[:-1] + ',"text":"again"}',
    json.dumps(body(expected_session_revision="0", session_id=str(uuid.uuid4()))),
    json.dumps(body(requested_mode="grade")),
    json.dumps(body(session_id=str(uuid.uuid4()))),          # revision pair missing
    '{"text": "x", ',
    json.dumps({k: v for k, v in body().items() if k != "session_id"}),
    json.dumps(body(text=1)),
])
def test_I01_malformed_bodies_are_422_without_downstream_work(h, raw):
    status, out = h.ask(None, raw=raw)
    assert (status, out["error_code"]) == (422, "INVALID_REQUEST") and set(out) == {"error_code", "request_id"}
    assert h.calls == NO_CALLS


def test_I01_bad_idempotency_key_and_content_type(h):
    assert h.ask(Q, key="not-a-uuid")[0] == 422
    status, _ = h.ask(Q, headers={"Content-Type": "text/plain"})
    assert status == 422 and h.calls == NO_CALLS


def test_I02_auth_course_and_session_scope(h):
    from fastapi.testclient import TestClient

    r = h._client.post(f"/v1/courses/{COURSE}/interactions", json=body(), headers={"Idempotency-Key": str(uuid.uuid4())})
    assert (r.status_code, r.json()["error_code"]) == (401, "UNAUTHORIZED")
    assert h.ask(Q, course="other-course")[0] == 403
    _, a = h.ask(Q)
    h.login("student-b")
    status, out = h.ask(Q, session=a["session"], user="student-b")
    assert (status, out["error_code"]) == (403, "FORBIDDEN")
    assert h.users["student-b"].get(f"/v1/sessions/{a['session']['session_id']}").status_code == 403
    forged = h._client.post(f"/v1/courses/{COURSE}/interactions", json=body(),
                       headers={"Idempotency-Key": str(uuid.uuid4()), "X-Gate-Owner": "owner-a",
                                "Cookie": "tutor_dev_identity=c3R1ZGVudC1h.deadbeef"})
    assert forged.status_code == 401


def test_I03_oversize_body_and_text(h):
    status, out = h.ask(None, raw=json.dumps(body(text="x" * 40000)))
    assert (status, out["error_code"]) == (413, "INPUT_TOO_LONG")
    status, out = h.ask("word " * 900)
    assert (status, out["error_code"]) == (413, "INPUT_TOO_LONG")
    assert h.calls == NO_CALLS


def test_I04_notice_required(h):
    h.login("student-c", accept=False)
    status, out = h.ask(Q, user="student-c")
    assert (status, out["error_code"]) == (403, "NOTICE_REQUIRED")
    status, out = h.ask(None, raw=json.dumps(body(notice_version="old-notice")))
    assert (status, out["error_code"]) == (403, "NOTICE_REQUIRED")
    assert h.calls == NO_CALLS


# ----------------------------------------------------------------------------- I05-I10 fixed routes

def test_I05_unclear_topic_and_bare_option(h):
    for text in ("B", "Quiz me on astrophysics"):
        status, out = h.ask(text)
        assert status == 200 and out["response_code"] == "A3" and out["session"] is None
    assert h.calls == NO_CALLS


@pytest.mark.parametrize("sc,code", [(scores(topic={"course_related": 0.02, "outside_course": 0.95, "nonmedical": 0.01,
                                                   "unclear": 0.02}), "A5"),
                                    (scores(risks={"real_person_advice": 0.9}), "A6")])
def test_I06_gate_reply_routes_make_no_teaching_calls(make, sc, code):
    h = make(scorer=lambda canonical: sc)
    status, out = h.ask(Q)
    assert status == 200 and out["response_code"] == code and out["content_type"] == "fixed"
    assert out["citations"] == [] and out["activity"] is None and h.calls == NO_CALLS


def test_I06_A6_pauses_existing_session(make):
    h = make()
    _, a = h.ask(Q)
    h.comps.gate.fixture_scorer = lambda c: scores(risks={"real_person_advice": 0.9})
    _, b = h.ask("What should my father take?", session=a["session"])
    assert b["response_code"] == "A6" and b["session"]["state"] == "paused"
    assert b["session"]["revision"] == a["session"]["revision"] + 1


def test_I07_hard_crisis_with_inference_failure_is_A7_with_local_alert(make):
    def broken(canonical):
        raise RuntimeError("inference down")
    h = make(scorer=broken)
    status, out = h.ask("I want to kill myself tonight")
    assert status == 200 and out["response_code"] == "A7" and h.calls == NO_CALLS
    rows = h.q("SELECT category, delivery_status FROM welfare_outbox")
    assert rows == [{"category": "self_harm", "delivery_status": "pending"}]  # queued, never "sent"


def test_I08_generated_fictional_crisis_creates_no_welfare_event(make):
    brain = FixtureBrain(lambda r, c: draft(("The fictional patient said he wanted to die.", ("fixture-p1",))))
    h = make(brain=brain)
    status, out = h.ask(Q)
    assert status == 200 and out["response_code"] != "A7"
    assert h.q("SELECT * FROM welfare_outbox") == []


def test_I09_A7_still_displays_when_audit_and_outbox_fail(make):
    h = make()
    h.store.fail_audit = True
    status, out = h.ask("I want to kill myself tonight")
    assert status == 200 and out["response_code"] == "A7"
    assert "a7_audit_outbox" in h.tutor.readiness()["degraded"]
    assert h.q("SELECT delivery_status FROM welfare_outbox") == [{"delivery_status": "pending"}]


def test_I10_non_A7_audit_failure_blocks_authorization(make):
    h = make()
    h.store.fail_audit = True
    status, out = h.ask(Q)
    assert (status, out["error_code"]) == (503, "SERVICE_UNAVAILABLE")
    assert h.q("SELECT * FROM delivery_authorizations") == []
    status, out = h.ask("Quiz me on epithelium")
    assert status == 503


# ----------------------------------------------------------------------------- I11-I17 evidence and Brain

def test_I11_scope_violation_never_reaches_brain(h):
    h.retriever.ignore_filter = True  # a broken retriever returning a disabled lib3 passage
    h.retriever.passages = [p for p in h.retriever.passages if p.library_id == "lib3"]
    status, out = h.ask(Q)
    assert status == 503 and h.calls["brain"] == 0


def test_I12_no_evidence_and_retrieval_fault(h):
    h.retriever.status = "no_evidence"
    status, out = h.ask(Q)
    assert out["response_code"] == "A5" and h.calls["brain"] == 0
    h.retriever.status = "error"
    status, out = h.ask(Q)
    assert (status, out["error_code"]) == (503, "SERVICE_UNAVAILABLE") and h.calls["brain"] == 0


def test_I13_exact_passages_reach_brain_and_verifier(make):
    seen = {}

    def capture(req, ctx):
        seen["value"] = ctx.evidence.fresh_value()
        return None
    h = make(brain=FixtureBrain(capture))
    status, out = h.ask(Q)
    assert status == 200
    original = {p.passage_id: p for p in h.retriever.passages}
    brain_passages = seen["value"]["passages"]
    for p in brain_passages:
        o = original[p["passage_id"]]
        assert p["text"] == o.text and p["text_sha256"] == o.text_sha256 and p["source_version"] == o.source_version
    vctx = h.verifier.calls[0][1]
    assert [p.text for p in vctx.evidence.passages] == [p["text"] for p in brain_passages]


def test_I14_binding_mismatch_rejects_delivery(make):
    h = make()
    h.verifier.mutate = lambda r: r.model_copy(update={"evidence_digest": "0" * 64})
    status, out = h.ask(Q)
    assert status == 503 and h.q("SELECT * FROM delivery_authorizations") == []


def test_I14_brain_result_for_other_evidence_rejected(make):
    class WrongDigestBrain(FixtureBrain):
        async def generate(self, request, context, *, cancel=None):
            r = await super().generate(request, context, cancel=cancel)
            return r.model_copy(update={"evidence_digest": "f" * 64})
    h = make(brain=WrongDigestBrain())
    status, _ = h.ask(Q)
    assert status == 503 and h.calls["verify"] == 0


def test_I15_brain_no_evidence_and_invalid_output_are_fixed_A5(make):
    h = make(brain=FixtureBrain(lambda r, c: draft(status="no_evidence")))
    status, out = h.ask(Q)
    assert out["response_code"] == "A5" and h.calls["verify"] == 0
    h2 = make(brain=FixtureBrain(lambda r, c: BrainCode.INVALID_OUTPUT))
    status, out = h2.ask(Q)
    assert out["response_code"] == "A5" and h2.calls == {"retrieve": 1, "brain": 1, "verify": 0}


@pytest.mark.parametrize("code", [BrainCode.DEADLINE_EXCEEDED, BrainCode.MEMORY_LIMIT, BrainCode.TRANSPORT_FAILURE,
                                  BrainCode.OUTPUT_LIMIT, BrainCode.MODEL_UNAVAILABLE])
def test_I16_brain_operational_failure_is_503_without_retry(make, code):
    h = make(brain=FixtureBrain(lambda r, c: code))
    status, out = h.ask(Q)
    assert (status, out["error_code"]) == (503, "SERVICE_UNAVAILABLE")
    assert h.calls["brain"] == 1 and h.calls["verify"] == 0


def test_I17_verifier_rejected_and_error(make):
    h = make(brain=FixtureBrain(lambda r, c: draft(("Cilia beat at fifty hertz in the colon.", ("fixture-p1",)))))
    status, out = h.ask(Q)
    assert out["response_code"] == "A5" and "fifty hertz" not in json.dumps(out)
    h2 = make()
    h2.verifier.mutate = lambda r: r  # passthrough
    h2.comps.registry.down = True     # verifier REGISTRY_UNAVAILABLE -> error
    status, out = h2.ask(Q)
    assert status == 503


# ----------------------------------------------------------------------------- I18-I20 output

def test_I18_coverage_unknown_is_A2_full_is_A1(make):
    h = make()
    status, out = h.ask(Q)
    assert out["response_code"] == "A2"
    orig = h.retriever.retrieve

    def full(req, ctx):
        r = orig(req, ctx)
        return r.model_copy(update={"coverage": "full"})
    h.retriever.retrieve = full
    status, out = h.ask(Q)
    assert out["response_code"] == "A1"
    assert h.cfg.features.a4_educational is False


def test_I19_formatter_exact_sentences_and_authoritative_metadata(make):
    h = make()
    h.retriever.passages[0] = fixture_passage("fixture-p1", "lib1", "Simple squamous epithelium is a single layer of "
                                              "flat cells.", 1).model_copy(update={"title": "<script>alert(1)</script>"})
    status, out = h.ask(Q)
    assert out["body"].startswith("Simple squamous epithelium is a single layer of flat cells. [1]")
    c = out["citations"][0]
    assert c["title"] == "<script>alert(1)</script>"  # text, rendered with textContent by the UI
    assert c["url"].startswith("/v1/evidence/fixture-p1?kb_version=") and c["locator"]["label"] == "Slide 1"
    assert set(c) == {"citation_id", "passage_id", "title", "source_version", "locator", "url"}


def test_I20_hidden_keys_internal_sentences_and_rejected_text_absent(make):
    h = make(brain=FixtureBrain(lambda r, c: draft(
        ("Simple squamous epithelium is a single layer of flat cells.", ("fixture-p1",), "student"),
        ("INTERNAL-NOTE flat cells.", ("fixture-p1",), "internal"))))
    _, a = h.ask(Q)
    _, t = h.ask("Tutor me on epithelium")
    _, x = h.ask("show explanation", session=t["session"])
    _, qz = h.ask("Quiz me on epithelium")
    blob = json.dumps([a, t, x, qz])
    for secret in ("FIXTURE-HIDDEN-KEY", "INTERNAL-NOTE", "correct_option_id", "expected_answer"):
        assert secret not in blob
    audit = json.dumps(h.q("SELECT * FROM audit_events"))
    assert "FIXTURE-HIDDEN-KEY" not in audit and Q not in audit and "INTERNAL-NOTE" not in audit
