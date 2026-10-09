"""T16, T58, T59: delivery authorization, revocation races, tampering and stale sessions."""

from __future__ import annotations

import pytest

from contracts.models import ContentReason, ResponseCode
from gate_classifier.schema import ErrorCode
from tests.conftest import SpyVerifier
from tests.contract.test_acceptance_gate import COURSE_Q
from tests.verifier.conftest import TEXTS, S, draft, request
from verifier.formatter import DeliveryAuthorizer, DeliveryRefused, compile_payload


def approved(kit):
    v, req, ctx = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", TEXTS["p2"], ["p2"])))
    assert v.result.status == "approved"
    return v.result, req.draft, ctx.evidence


@pytest.fixture
def auth(kit):
    return DeliveryAuthorizer(kit.verifier.rules.notices)


def test_T16_revocation_before_verification(kit):
    kit.registry.revoke("p1")
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert v.result.response_code == ResponseCode.A5
    assert ContentReason.SOURCE_INELIGIBLE in v.result.sentence_results[0].reason_codes


def test_T16_revocation_between_verification_and_authorization(kit, auth):
    result, d, bundle = approved(kit)
    kit.registry.revoke("p2")
    with pytest.raises(DeliveryRefused) as exc:
        auth.authorize(result, d, bundle, kit.registry)
    assert exc.value.code == "source_ineligible" and not exc.value.unavailable


def test_T16_revocation_after_authorization(kit, auth):
    result, d, bundle = approved(kit)
    payload, record = auth.authorize(result, d, bundle, kit.registry)
    kit.registry.revoke("p1")
    assert auth.accept_for_display(payload)  # authorization is the linearization point
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])))  # new requests deny the source
    assert v.result.status == "rejected" and record.revocation_epoch == 0


def test_registry_outage_at_delivery_is_unavailable(kit, auth):
    result, d, bundle = approved(kit)
    kit.registry.down = True
    with pytest.raises(DeliveryRefused) as exc:
        auth.authorize(result, d, bundle, kit.registry)
    assert exc.value.unavailable


def test_T58_only_authorized_untampered_payload_displays(kit, auth):
    result, d, bundle = approved(kit)
    unauthorized = compile_payload(result, d, bundle, kit.verifier.rules.notices)
    assert not auth.accept_for_display(unauthorized)
    payload, _ = auth.authorize(result, d, bundle, kit.registry)
    assert auth.accept_for_display(payload)
    from dataclasses import replace

    for forged in (replace(payload, text=payload.text + " Extra unverified claim."),
                   replace(payload, citations=()),
                   replace(payload, notices=payload.notices[:-1])):
        assert not auth.accept_for_display(forged)


@pytest.mark.parametrize("update", [{"verified_text": "Simple squamous epithelium is thick."},
                                    {"citation_map": {"s1": ("p3",), "s2": ("p2",)}},
                                    {"final_sentence_ids": ("s2", "s1")},
                                    {"response_code": ResponseCode.A1}])
def test_T58_tampered_result_never_compiles(kit, update):
    result, d, bundle = approved(kit)
    with pytest.raises(DeliveryRefused):
        compile_payload(result.model_copy(update=update), d, bundle, kit.verifier.rules.notices)


def test_T58_rejected_result_and_swapped_draft_never_compile(kit):
    result, d, bundle = approved(kit)
    other = draft(S("s1", TEXTS["p3"], ["p3"]))
    with pytest.raises(DeliveryRefused):
        compile_payload(result, other, bundle, kit.verifier.rules.notices)
    v, req, ctx = kit.run(draft(S("s1", "Goblet cells are in the colon.", ["p1"])))
    with pytest.raises(DeliveryRefused):
        compile_payload(v.result, req.draft, ctx.evidence, kit.verifier.rules.notices)


def test_formatter_escapes_and_uses_registry_metadata(make_kit):
    from tests.verifier.conftest import passage

    text = "Simple squamous epithelium lines the alveoli & capillaries."
    kit = make_kit(passages=[passage("p1", text)])
    v, req, ctx = kit.run(draft(S("s1", text, ["p1"])))
    payload = compile_payload(v.result, req.draft, ctx.evidence, kit.verifier.rules.notices)
    assert "&amp;" in payload.text and payload.citations[0]["title"] == "Fixture source p1"
    assert payload.citations[0]["evidence_uri"].startswith("/v1/evidence/p1")
    assert payload.text.endswith(kit.verifier.rules.notices["ai_notice"])


def test_T59_old_snapshot_cannot_undo_live_block(kit):
    kit.registry.revoke("p1")
    kit.registry.revoke("p2")  # epoch moves on; the bundle still carries revocation_epoch 0
    d = draft(S("s1", TEXTS["p1"], ["p1"]))
    req = request(d)
    bundle = kit.bundle(req.request_id)
    assert bundle.retrieval.revocation_epoch == 0
    v = kit.verifier.run(req, kit.context(req, bundle))
    assert v.result.status == "rejected" and v.epoch == 2


def test_T59_stale_session_stops_delivery(kit, auth):
    result, d, bundle = approved(kit)
    with pytest.raises(DeliveryRefused) as exc:
        auth.authorize(result, d, bundle, kit.registry, session_check=lambda: False)
    assert exc.value.code == "stale_session"


# ----------------------------------------------------------------------------- orchestrator-level races


def test_orchestrator_revocation_between_verify_and_authorization_suppresses(make_harness):
    holder = {}

    class RevokeAfterVerify(SpyVerifier):
        def verify(self, request, context):
            result = super().verify(request, context)
            holder["h"].retriever.registry.revoke("fixture-p1")
            return result

    h = make_harness(verifier=RevokeAfterVerify())
    holder["h"] = h
    r = h.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A5 and r.reason == "DELIVERY_SUPPRESSED"
    assert h.verifier.results[0].status == "approved" and "fixture text" not in r.text


def test_orchestrator_registry_outage_is_unavailable(make_harness):
    h = make_harness()
    h.retriever.registry.down = True
    r = h.ask(COURSE_Q)
    assert (r.http_status, r.error_code) == (503, ErrorCode.SERVICE_UNAVAILABLE)


def test_orchestrator_delivers_only_compiled_payload(harness):
    r = harness.ask(COURSE_Q)
    assert r.response_code == ResponseCode.A2 and r.text.endswith(harness.verifier.rules.notices["ai_notice"])
    assert r.citations[0]["passage_id"] == "fixture-p1" and "[1]" in r.text
    assert any(rec.service_status == "delivery_authorized" for rec in harness.audit.records)
