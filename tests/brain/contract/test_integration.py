"""B36: Brain result reaches the independent verifier and authorized formatter only (fake runtime, fixture verifier)."""

from __future__ import annotations

import json

import pytest

from brain.adapter import LocalBrainAdapter
from gate_classifier.schema import ErrorCode, ResponseCode
from tests.contract.test_acceptance_gate import COURSE_Q

from ..conftest import draft_body, envelope, make_service


@pytest.fixture
def adapter(runtime):
    a = LocalBrainAdapter(make_service(runtime))
    yield a
    a.close()


def test_B36_generated_draft_is_verified_before_delivery(make_harness, runtime, adapter):
    h = make_harness(brain=adapter)
    r = h.ask(COURSE_Q)
    assert r.http_status == 200 and r.response_code in (ResponseCode.A1, ResponseCode.A2)
    assert h.verifier.calls == 1 and len(runtime.completions) == 1
    # the delivered text is the verifier's approved text, bound to the draft the Brain produced
    # (the authorized formatter adds citation markers and fixed notices around it)
    assert r.text.startswith(h.verifier.results[0].verified_text)
    record = json.loads(runtime.completions[0]["messages"][1]["content"])
    shown = [p["passage_id"] for p in record["passages"]]
    assert shown == list(h.verifier.seen[0].evidence.shown_passage_ids)


def test_B36_invalid_output_is_fixed_A5_without_verifier(make_harness, runtime, adapter):
    runtime.completion_script = lambda body: envelope("Plain prose answer about epithelium.")
    h = make_harness(brain=adapter)
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A5 and r.reason == "INVALID_DRAFT"
    assert h.verifier.calls == 0 and "Plain prose" not in (r.text or "")


def test_B36_unsupported_citation_rejected_by_verifier(make_harness, runtime, adapter):
    def script(body):
        ids = body["response_format"]["schema"]["properties"]["used_passage_ids"]["items"]["enum"]
        return envelope(draft_body([{"sentence_id": "s1", "text": "Cilia beat at fifty hertz in the colon.",
                                     "kind_hint": "factual", "visibility": "student", "cites": [ids[0]],
                                     "depends_on": []}]))

    runtime.completion_script = script
    h = make_harness(brain=adapter)
    r = h.ask(COURSE_Q)
    assert h.verifier.calls == 1 and r.response_code == ResponseCode.A5
    assert "fifty hertz" not in (r.text or "")


def test_B36_brain_no_evidence_is_A5(make_harness, runtime, adapter):
    runtime.completion_script = lambda body: envelope(draft_body([], [], status="no_evidence"))
    h = make_harness(brain=adapter)
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A5 and h.verifier.calls == 0


def test_B36_brain_infrastructure_failure_is_service_unavailable(make_harness, runtime, adapter):
    runtime.completion_script = lambda body: envelope(json.dumps(draft_body())[:30], finish="length")
    h = make_harness(brain=adapter)
    r = h.ask(COURSE_Q)
    assert (r.http_status, r.error_code) == (503, ErrorCode.SERVICE_UNAVAILABLE)
    assert h.verifier.calls == 0
