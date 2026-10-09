"""T42-T53: answer-level counting, trimming, hidden answers, coverage, conflicts, policy and modes."""

from __future__ import annotations

import pytest

from contracts.models import ContentReason, OperationalError, ResponseCode
from tests.verifier.conftest import CONTRA, NEUTRAL, TEXTS, Kit, S, draft, request
from verifier.formatter import compile_payload

GOOD1, GOOD2, GOOD3 = TEXTS["p1"], TEXTS["p2"], TEXTS["p3"]
BAD = "Goblet cells are found in the colon."  # neutral against every fixture passage


def test_T42_two_rejected_sentences_reject_whole_draft(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", BAD, ["p1"]), S("s3", "Cilia beat in the lung.", ["p2"])))
    r = v.result
    assert (r.status, r.response_code, r.disposition) == ("rejected", ResponseCode.A5, "fallback")
    assert r.unsupported_claims == 2 and r.verified_text == "" and r.final_sentence_ids == ()


def test_T43_safe_trim_of_terminal_ordinary_sentence_is_A2(kit):
    v, req, ctx = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", GOOD2, ["p2"]), S("s3", BAD, ["p1"])))
    r = v.result
    assert (r.status, r.disposition, r.response_code) == ("approved", "trimmed", ResponseCode.A2)
    assert r.final_sentence_ids == ("s1", "s2") and BAD not in r.verified_text and r.partial_support
    payload = compile_payload(r, req.draft, ctx.evidence, kit.verifier.rules.notices)
    assert kit.verifier.rules.notices["partial"] in payload.text and BAD not in payload.text


@pytest.mark.parametrize("bad", [
    "Goblet cells are found in 3 layers of the colon.",          # number
    "Goblet cells are not found in the colon.",                  # negation
    "Goblet cells appear if the colon is inflamed.",             # condition
    "Goblet cells line the colon except in the rectum.",         # exception
    "The left colon contains goblet cells.",                     # laterality
    "Goblet cells are found in the colon after puberty.",        # sequence
])
def test_T44_risk_cue_in_failed_sentence_rejects_whole_draft(kit, bad):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", bad, ["p1"])))
    assert v.result.response_code == ResponseCode.A5
    assert ContentReason.UNSAFE_TRIM in v.answer_reasons or v.result.unsupported_claims >= 1


def test_T44_trim_not_last_or_question_rejects(kit):
    v, _, _ = kit.run(draft(S("s1", BAD, ["p1"]), S("s2", GOOD1, ["p1"])))
    assert ContentReason.UNSAFE_TRIM in v.answer_reasons
    v2, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", "Are goblet cells found in the colon?", ["p1"],
                                                         kind="question")))
    assert ContentReason.UNSAFE_TRIM in v2.answer_reasons


def test_T45_failed_hidden_answer_rejects_and_never_leaks(kit):
    hidden = "The expected answer is stratified columnar epithelium."
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("k1", hidden, ["p1"], visibility="internal")), mode="tutor")
    r = v.result
    assert r.response_code == ResponseCode.A5 and ContentReason.HIDDEN_ANSWER_FAILED in v.answer_reasons
    assert hidden not in r.model_dump_json() and r.internal_sentences == ()


def test_T45_approved_hidden_answer_kept_separate(kit):
    hidden = TEXTS["p2"]
    v, req, ctx = kit.run(draft(S("s1", GOOD1, ["p1"]), S("k1", hidden, ["p2"], visibility="internal")),
                          mode="tutor")
    r = v.result
    assert r.status == "approved" and r.internal_sentence_ids == ("k1",)
    assert hidden not in r.verified_text and "k1" not in r.citation_map
    payload = compile_payload(r, req.draft, ctx.evidence, kit.verifier.rules.notices)
    assert hidden not in payload.text and all(s[0] != "k1" for s in payload.sentences)


def test_T46_kept_sentence_depending_on_deletion_rejects(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", BAD, ["p1"]),
                            S("k1", GOOD2, ["p2"], visibility="internal", depends_on=["s2"])))
    assert v.result.response_code == ResponseCode.A5
    v2, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", "It lines the alveoli of the lung.", ["p2"]),
                             S("s3", BAD, ["p1"])))
    kit.scores.hyp["It lines the alveoli of the lung."] = (0.01, 0.97, 0.02)
    v3, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", "It lines the alveoli of the lung.", ["p2"]),
                             S("s3", BAD, ["p1"])))
    assert v2.result.response_code == ResponseCode.A5 and v3.result.response_code == ResponseCode.A5
    assert ContentReason.UNSAFE_TRIM in v3.answer_reasons


def test_T47_unknown_coverage_is_A2(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])))
    assert v.result.response_code == ResponseCode.A2 and v.result.partial_support
    assert ContentReason.COVERAGE_PARTIAL in v.answer_reasons


def test_T48_trusted_full_coverage_is_A1(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])), coverage="full")
    assert v.result.response_code == ResponseCode.A1 and not v.result.partial_support


def test_T49_known_conflict_needs_every_side(make_kit):
    kit = make_kit(conflicts=[("p1", "p3")])
    v, req, ctx = kit.run(draft(S("s1", GOOD1, ["p1"]), S("s2", GOOD3, ["p3"])), coverage="full")
    r = v.result
    assert (r.response_code, r.source_conflict, r.partial_support) == (ResponseCode.A2, True, True)
    payload = compile_payload(r, req.draft, ctx.evidence, kit.verifier.rules.notices)
    assert kit.verifier.rules.notices["conflict"] in payload.text and len(payload.citations) == 2
    v2, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])))
    assert v2.result.response_code == ResponseCode.A5 and ContentReason.CONFLICT_INCOMPLETE in v2.answer_reasons


def test_T50_real_person_advice_is_A6(kit):
    v, _, _ = kit.run(draft(S("s1", "You should take 5 mg of the drug tonight.", ["p1"])))
    r = v.result
    assert (r.status, r.response_code, r.disposition, r.reply_key) == ("rejected", ResponseCode.A6, "refuse",
                                                                       "real_person")
    assert r.output_policy_violations == ("real_person_advice",) and kit.backend.calls == 0


def test_T50_policy_uncertainty_is_A5_and_failure_is_unavailable(kit):
    v, _, _ = kit.run(draft(S("s1", "If you have chest pain, the epithelium is thin.", ["p1"])))
    assert ContentReason.OUTPUT_POLICY_UNCERTAIN in v.answer_reasons

    class Broken:
        version = "x"

        def evaluate(self, *a):
            raise RuntimeError("down")

    kit.verifier.policy = Broken()
    v2, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])))
    assert v2.result.error_code == OperationalError.POLICY_UNAVAILABLE


def test_T51_crisis_attribution(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])), redacted_request="I want to kill myself tonight.")
    assert (v.result.response_code, v.result.disposition, v.result.reply_key) == (ResponseCode.A7, "escalate",
                                                                                  "self_harm")
    fiction = "In the case the patient collapsed and is not breathing."
    kit.scores.hyp[fiction] = NEUTRAL
    v2, _, _ = kit.run(draft(S("s1", fiction, ["p1"])))
    assert v2.result.response_code != ResponseCode.A7


def test_T52_A4_requires_authorization_and_feature(kit):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])), a4_authorized=True)
    assert v.result.error_code == OperationalError.CONTEXT_MISMATCH
    kit.verifier.profile = kit.verifier.profile.model_copy(
        update={"features": {**kit.verifier.profile.features, "a4_educational": True}})
    v2, req, ctx = kit.run(draft(S("s1", GOOD1, ["p1"])), a4_authorized=True)
    assert v2.result.response_code == ResponseCode.A4
    payload = compile_payload(v2.result, req.draft, ctx.evidence, kit.verifier.rules.notices)
    assert kit.verifier.rules.notices["a4_disclaimer"] in payload.text
    v3, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])))
    assert v3.result.response_code == ResponseCode.A2  # no automatic A4 without gate authorization


@pytest.mark.parametrize("mode", ["virtual_patient", "open_ended", "image", "flashcard"])
def test_T53_disabled_modes_fail_closed(kit, mode):
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])), mode=mode)
    assert v.result.error_code == OperationalError.CONTEXT_MISMATCH and kit.backend.calls == 0


def test_contradicted_single_sentence_never_approved(kit):
    kit.scores.hyp[GOOD1] = CONTRA
    v, _, _ = kit.run(draft(S("s1", GOOD1, ["p1"])))
    assert v.result.status == "rejected" and v.result.near_miss  # "one layer" carries a number/unit cue


def test_result_invariants_and_digest(kit):
    from verifier import digests

    v, req, ctx = kit.run(draft(S("s1", GOOD1, ["p1"])))
    r = v.result
    assert digests.result_digest_valid(r)
    assert r.draft_digest == digests.draft_digest(req.draft)
    assert r.evidence_digest == digests.evidence_digest(ctx.evidence)
    assert r.verifier_version.startswith("fixture-verifier-0.2+")
    tampered = r.model_copy(update={"verified_text": "Different text."})
    assert not digests.result_digest_valid(tampered)


def test_request_context_id_mismatch(kit):
    d = draft(S("s1", GOOD1, ["p1"]))
    req, other = request(d), request(d)
    v = kit.verifier.run(req, kit.context(other, kit.bundle(req.request_id)))
    assert v.result.error_code == OperationalError.CONTEXT_MISMATCH


def test_kit_isolated():
    assert Kit().verifier.readiness()["student_release_ready"] is False
