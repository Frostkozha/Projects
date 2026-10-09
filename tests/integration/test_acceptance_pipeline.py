"""Acceptance tests T33-T45, T49, T51, T58, T59 (Gate spec section 17): orchestration with spy adapters.

Verification runs the real verifier pipeline in fixture mode (injected NLI scores, unit tests only).
"""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import ValidationError

from gate_classifier.adapters import Passage, RetrievalRequest
from gate_classifier.audit import AlertEvent, InMemoryAlertQueue
from gate_classifier.config import AI_NOTICE
from gate_classifier.orchestrator import FixtureBrain, FixtureRetriever, fixture_verifier, sentence_draft
from gate_classifier.schema import ErrorCode, ResponseCode
from tests.conftest import KB, Harness, SpyVerifier, passages
from tests.contract.test_acceptance_gate import COURSE_Q, quiz_session

UNAVAILABLE = (503, ErrorCode.SERVICE_UNAVAILABLE)
P1 = "Simple squamous epithelium is a single layer of flat cells (fixture text)."


def drafting(*sentences):
    """Scripted Brain returning a fixed sentence-level draft; the real fixture-mode verifier judges it."""
    return FixtureBrain(lambda req: sentence_draft(*sentences))


def test_T33_concurrent_replies_advance_once(make_harness):
    barrier = threading.Barrier(2, timeout=5)

    def brain_script(req):
        barrier.wait()  # both requests have read the same session revision
        return FixtureBrain().draft(req)

    h = make_harness(brain=FixtureBrain(brain_script))
    s = quiz_session(h)
    results = []
    threads = [threading.Thread(target=lambda: results.append(h.ask("B", session_id=s.session_id))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    codes = sorted((r.http_status, r.error_code.value if r.error_code else None) for r in results)
    assert codes == [(200, None), (409, "SESSION_CONFLICT")]
    final = h.sessions.get(s.session_id)
    assert final.answered_items == ("item-e1",) and final.revision == s.revision + 1


def test_T34_no_evidence_is_A5_without_brain(make_harness):
    h = make_harness(retriever=FixtureRetriever(passages(), KB, status="no_evidence"))
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A5 and r.text.startswith("I could not find enough support")
    assert h.brain.calls == []


def test_T35_retrieval_error_is_unavailable(make_harness):
    for retriever in (FixtureRetriever(passages(), KB, status="error"), FixtureRetriever(passages(), KB, raise_error=True)):  # noqa: E501
        h = make_harness(retriever=retriever)
        r = h.ask(COURSE_Q)
        assert (r.http_status, r.error_code) == UNAVAILABLE
        assert h.brain.calls == []


def test_T36_unauthorized_or_stale_evidence_is_unavailable(make_harness):
    h = make_harness(retriever=FixtureRetriever(passages(), KB, ignore_filter=True))
    h.retriever.passages = [p for p in passages() if p.library_id == "lib3"]
    assert (h.ask(COURSE_Q).http_status, h.brain.calls) == (503, [])
    h2 = make_harness(retriever=FixtureRetriever(passages(), "old-kb"))
    assert (h2.ask(COURSE_Q).http_status, h2.brain.calls) == (503, [])
    bad = passages()[0].model_copy(update={"review_status": "under_review"})  # only live evidence is allowed
    h3 = make_harness(retriever=FixtureRetriever([bad], KB))
    assert (h3.ask(COURSE_Q).http_status, h3.brain.calls) == (503, [])


def test_T37_fabricated_citation_or_unevidenced_draft_rejected(make_harness):
    h = make_harness(brain=drafting(("Cilia beat at 50 Hz.", ("made-up-7",))))
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A5 and AI_NOTICE not in r.text
    h2 = make_harness(brain=drafting(("Cilia beat fast.", ()), ("Goblet cells secrete mucus.", ())))
    assert h2.ask(COURSE_Q).response_code == ResponseCode.A5


def test_T38_verifier_missing_error_or_timeout(make_harness):
    h = make_harness(verifier=None)
    assert (h.ask(COURSE_Q).http_status, h.ask(COURSE_Q).error_code) == UNAVAILABLE

    class BrokenPolicy:
        version = "broken"

        def evaluate(self, *a):
            raise RuntimeError("policy adapter down")

    inner = fixture_verifier()
    inner.policy = BrokenPolicy()
    h2 = make_harness(verifier=SpyVerifier(inner))
    r2 = h2.ask(COURSE_Q)
    assert r2.error_code == ErrorCode.SERVICE_UNAVAILABLE and h2.verifier.results[0].error_code.value == \
        "POLICY_UNAVAILABLE"

    class Boom:
        calls = 0

        def verify(self, *a):
            raise RuntimeError("verifier crashed")

    h3 = make_harness(verifier=Boom())
    assert h3.ask(COURSE_Q).error_code == ErrorCode.SERVICE_UNAVAILABLE

    class Slow(SpyVerifier):
        def verify(self, request, context):
            time.sleep(0.5)
            return super().verify(request, context)

    h4 = make_harness(verifier=Slow())
    h4.orch.adapter_timeout = 0.1
    r = h4.ask(COURSE_Q)
    assert (r.http_status, r.error_code) == UNAVAILABLE and r.text is None


def test_T39_single_unsafe_removal_rejects_whole_draft(make_harness):
    h = make_harness(brain=drafting((P1, ("fixture-p1",)),
                                    ("Flat cells line every duct except in the trachea.", ("fixture-p1",))))
    assert h.ask(COURSE_Q).response_code == ResponseCode.A5


def test_T40_multiple_unsupported_sentences_reject(make_harness):
    h = make_harness(brain=drafting((P1, ("fixture-p1",)), ("Goblet cells are found in the colon.", ("fixture-p1",)),
                                    ("Cilia line the trachea.", ("fixture-p1",))))
    assert h.ask(COURSE_Q).response_code == ResponseCode.A5


def test_T41_unknown_coverage_or_safe_trim_is_A2(make_harness):
    h = make_harness()
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A2 and r.text.endswith(AI_NOTICE)
    assert "covers only what the supplied course material states" in r.text
    h2 = make_harness(brain=drafting((P1, ("fixture-p1",)), ("Goblet cells are found in the colon.", ("fixture-p1",))))
    r2 = h2.ask(COURSE_Q)
    assert r2.response_code == ResponseCode.A2 and "verify only part" in r2.text and "colon" not in r2.text


def test_T42_supported_answer_has_citations_and_notice(harness):
    r = harness.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A2  # unknown coverage: never A1 without an approved coverage signal
    assert r.text.endswith(AI_NOTICE)
    assert r.citations and r.citations[0]["passage_id"] == "fixture-p1" and r.citations[0]["number"] == 1
    view = r.student_view()
    assert set(view) == {"request_id", "response_code", "text", "citations", "session_id"}
    assert "prediction" not in str(view) and "config_sha256" not in str(view)


def test_T43_real_person_advice_in_draft_is_suppressed(make_harness):
    h = make_harness(brain=drafting(("You should take 5 mg of the drug tonight.", ("fixture-p1",))))
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A6 and "fixture-p1" not in r.text and "5 mg" not in r.text
    assert any(rec.event == "incident" for rec in h.audit.records)
    # a generated fictional emergency never creates a welfare alert (A7 is judged on the current request only)
    h2 = make_harness(brain=drafting(("The patient collapsed and is not breathing.", ("fixture-p1",))))
    r2 = h2.ask(COURSE_Q)
    assert r2.response_code != ResponseCode.A7 and len(h2.alert_queue.events) == 0


class FlakyQueue(InMemoryAlertQueue):
    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def enqueue(self, event):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise OSError("queue unavailable")
        super().enqueue(event)


def test_T44_alert_failure_still_delivers_A7_with_retry_and_no_duplicates(make_harness):
    h = make_harness(alert_sink=FlakyQueue(failures=2))
    r = h.ask("I want to kill myself tonight.")
    assert r.response_code == ResponseCode.A7 and "PLACEHOLDER-STUDENT-SUPPORT" in r.text
    h.dispatcher.wait_idle()
    assert len(h.alert_queue.events) == 1 and h.alert_queue.attempts == 3
    assert any(a["code"] == "alert_write_failed" for a in h.alarms.alarms)
    retries = [rec for rec in h.audit.records if rec.event == "alert_retry"]
    assert retries and all("kill" not in rec.model_dump_json() for rec in retries)
    # idempotent by event_id
    ev = next(iter(h.alert_queue.events.values()))
    h.alert_queue.enqueue(ev)
    assert len(h.alert_queue.events) == 1
    assert AlertEvent.new(ev.request_id, None, "self_harm").event_id == ev.event_id

    h2 = make_harness(alert_sink=FlakyQueue(failures=99))
    r2 = h2.ask("I want to kill myself tonight.")
    assert r2.response_code == ResponseCode.A7
    h2.dispatcher.wait_idle()
    assert h2.dispatcher.failed_events and any(a["code"] == "alert_delivery_exhausted" for a in h2.alarms.alarms)


def test_T45_audit_failure_blocks_generation(make_harness):
    class FailingAudit:
        records = []

        def write(self, record):
            raise OSError("disk full")

    h = make_harness(audit=FailingAudit())
    r = h.ask(COURSE_Q)
    assert (r.http_status, r.error_code) == UNAVAILABLE
    assert h.retriever.calls == [] and h.brain.calls == []
    # emergency still renders its fixed support message when audit fails
    r2 = h.ask("Someone here just collapsed and is not breathing!")
    assert r2.response_code == ResponseCode.A7
    assert any(a["code"] == "audit_write_failed" for a in h.alarms.alarms)


def test_T49_adapter_exception_text_is_suppressed(make_harness):
    secret_q = "Explain simple squamous epithelium for cohort ZETA-UNIQUE-TOKEN."
    h = make_harness(retriever=FixtureRetriever(passages(), KB, raise_error=True))
    r = h.ask(secret_q)
    assert (r.http_status, r.error_code) == UNAVAILABLE
    assert "ZETA-UNIQUE-TOKEN" not in str(r.student_view())
    assert all("ZETA-UNIQUE-TOKEN" not in rec.model_dump_json() for rec in h.audit.records)


def test_T51_kb_change_invalidates_pending_quiz(harness):
    s = quiz_session(harness, kb_version="fixture-kb-000")
    r = harness.ask("B", session_id=s.session_id)
    assert (r.http_status, r.error_code) == (409, ErrorCode.SESSION_CONFLICT)
    after = harness.sessions.get(s.session_id)
    assert after.pending_item_id is None and after.state == "idle" and after.kb_version == KB
    assert harness.retriever.calls == [] and harness.scorer.seen == []


def test_T58_empty_library_filter_rejected_by_adapter_contract():
    with pytest.raises(ValidationError):
        RetrievalRequest(query_text="x", allowed_libraries=(), course_id="c", kb_version="k")
    with pytest.raises(ValidationError):
        RetrievalRequest(query_text="x", allowed_libraries=("lib1",), preferred_libraries=("lib3",),
                         course_id="c", kb_version="k")


def test_T58_gate_decision_cannot_carry_empty_plan(harness):
    from gate_classifier.schema import GateDecision, GateRequest, new_request_id

    d = harness.service.evaluate(GateRequest(text=COURSE_Q), harness.ctx(), new_request_id()).decision
    data = d.model_dump()
    data["retrieval_plan"]["allowed_libraries"] = ()
    with pytest.raises(ValidationError):
        GateDecision.model_validate(data)


def test_T59_commit_conflict_returns_409_without_duplicate_progress(make_harness):
    holder = {}

    def brain_script(req):
        s = holder["h"].sessions.get(holder["sid"])
        holder["h"].sessions.compare_and_swap(s.session_id, s.revision, state="awaiting_response")  # concurrent writer
        return FixtureBrain().draft(req)

    h = make_harness(brain=FixtureBrain(brain_script))
    s = quiz_session(h)
    holder.update(h=h, sid=s.session_id)
    r = h.ask("B", session_id=s.session_id)
    assert (r.http_status, r.error_code) == (409, ErrorCode.SESSION_CONFLICT)
    current = h.sessions.get(s.session_id)
    assert current.answered_items == () and current.pending_item_id == "item-e1"


def test_fixture_retriever_never_returns_disabled_library(harness):
    harness.ask(COURSE_Q)
    assert all(isinstance(p, Passage) and p.library_id != "lib3" for p in harness.brain.calls[0].evidence.passages)


def test_quiz_session_created_and_advanced_end_to_end(harness: Harness):
    r = harness.ask("Quiz me on simple squamous epithelium.")
    assert r.response_code == ResponseCode.A2 and r.session_id
    s = harness.sessions.get(r.session_id)
    assert s.mode.value == "quiz" and s.state == "awaiting_response" and s.pending_item_id == "item-e1"
    r2 = harness.ask("B", session_id=r.session_id)
    assert r2.response_code == ResponseCode.A1
    assert harness.sessions.get(r.session_id).answered_items == ("item-e1",)
