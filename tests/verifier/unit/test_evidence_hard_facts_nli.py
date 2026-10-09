"""T13-T41: evidence trust, live eligibility, hard facts, NLI boundaries (injected fixture scores only)."""

from __future__ import annotations

import numpy as np
import pytest

from contracts.models import ContentReason, OperationalError
from tests.verifier.conftest import (CONTRA, GRAY, NEUTRAL, PROFILE, ROOT, SUPPORT, TEXTS, Kit, S, draft, passage,
                                     request)
from verifier.service import InMemoryVerifierAudit, Verifier
from verifier.config import ThresholdProfile, load_rules
from verifier.decision import classify_pair
from verifier.hard_facts import HardFactChecker
from verifier.nli import FixtureNLI, NLIOutputInvalid, NLIUnavailable, to_scores, validate_label_mapping


def one(kit: Kit, text: str, cites=("p1",), **kw):
    v, req, ctx = kit.run(draft(S("s1", text, cites)), **kw)
    return v


# ----------------------------------------------------------------------------- T13-T18 evidence trust


def test_T13_source_hash_tamper_is_integrity_error(make_kit):
    bad = passage("p1", TEXTS["p1"], text_sha256="0" * 64)
    kit = make_kit(passages=[bad, passage("p2", TEXTS["p2"])])
    v = one(kit, TEXTS["p1"])
    assert (v.result.status, v.result.error_code) == ("error", OperationalError.EVIDENCE_INTEGRITY)
    assert v.result.reply_key == "service_unavailable" and kit.backend.calls == 0


@pytest.mark.parametrize("override", [{"tenant": "other-tenant"}, {"course": "other-course"},
                                      {"bundle_rid": "00000000-0000-4000-8000-0000000000ff"}])
def test_T14_scope_mismatch_is_context_error(kit, override):
    d = draft(S("s1", TEXTS["p1"], ["p1"]))
    req = request(d)
    bundle = kit.bundle(req.request_id, **override)
    v = kit.verifier.run(req, kit.context(req, bundle))
    assert v.result.error_code == OperationalError.CONTEXT_MISMATCH and kit.backend.calls == 0


def test_T15_source_expired_before_verification_is_ineligible(kit):
    kit.registry.revoke("p2")  # expired/revoked in the live registry
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", TEXTS["p2"], ["p2"])))
    s2 = v.result.sentence_results[1]
    assert ContentReason.SOURCE_INELIGIBLE in s2.reason_codes and s2.nli_scores == {}
    assert "p2" not in str(v.result.citation_map)


def test_T17_registry_outage_is_error_without_cached_allow(kit):
    kit.registry.down = True
    v = one(kit, TEXTS["p1"])
    assert v.result.error_code == OperationalError.REGISTRY_UNAVAILABLE and kit.backend.calls == 0
    v2 = one(kit, TEXTS["p1"], registry=None)
    assert v2.result.error_code == OperationalError.REGISTRY_UNAVAILABLE


def test_T18_under_review_passage_excluded(make_kit):
    kit = make_kit(passages=[passage("p1", TEXTS["p1"], review_status="under_review"), passage("p2", TEXTS["p2"])])
    v = one(kit, TEXTS["p1"])
    assert ContentReason.SOURCE_INELIGIBLE in v.result.sentence_results[0].reason_codes


# ----------------------------------------------------------------------------- T19-T30 hard facts

DRUGS = "Drug A is given at 5 mg and drug B is given at 10 mg."


@pytest.fixture(scope="module")
def hf() -> HardFactChecker:
    return HardFactChecker(load_rules(ROOT / "config/verifier"))


def hard_case(make_kit, premise, hypothesis):
    kit = make_kit(passages=[passage("p1", premise)])
    kit.scores.hyp[hypothesis] = SUPPORT  # even a perfect NLI score cannot rescue a mismatch
    return one(kit, hypothesis)


@pytest.mark.parametrize("hyp", ["Drug A is given at 10 mg.", "Drug B is given at 5 mg."])
def test_T19_entity_number_swap(make_kit, hyp):
    v = hard_case(make_kit, DRUGS, hyp)
    assert ContentReason.HARD_FACT_MISMATCH in v.result.sentence_results[0].reason_codes


@pytest.mark.parametrize("hyp", ["Drug A is given at 5 mg.", "Drug B is given at 10 mg."])
def test_T19_correct_binding_passes_to_nli(make_kit, hyp):
    v = hard_case(make_kit, DRUGS, hyp)
    assert v.result.status == "approved" and v.result.sentence_results[0].checks.hard_fact == "pass"


@pytest.mark.parametrize("premise,hyp", [("The dose is 5 mg.", "The dose is 5 mcg."),
                                         ("The dose is 5 mg/kg.", "The dose is 5 mg.")])
def test_T20_unit_swap(make_kit, premise, hyp):
    assert hard_case(make_kit, premise, hyp).result.status == "rejected"


def test_T21_reviewed_alias_passes_constraints_then_runs_nli(make_kit):
    kit = make_kit(passages=[passage("p1", "The membrane is 50 nm thick.")])
    hyp = "The membrane is 50nm thick."
    kit.scores.hyp[hyp] = NEUTRAL
    v = one(kit, hyp)
    s = v.result.sentence_results[0]
    assert s.checks.hard_fact == "pass" and s.checks.entailment == "fail" and "p1" in s.nli_scores


@pytest.mark.parametrize("premise,hyp", [
    ("The basement membrane is 48 nm thick.", "The basement membrane is 50 nm thick."),
    ("The value is 2.5 mm.", "The value is 2 mm."),
])
def test_T22_rounding_not_exact(make_kit, premise, hyp):
    assert hard_case(make_kit, premise, hyp).result.status == "rejected"


def test_T23_range_endpoint_change(make_kit):
    v = hard_case(make_kit, "The diameter is less than 10 um.", "The diameter is at most 10 um.")
    assert v.result.status == "rejected"


def test_T24_wrong_entity_same_property(make_kit):
    v = hard_case(make_kit, "The trachea is lined by 3 layers of cells and the bronchus by 2 layers of cells.",
                  "The trachea is lined by 2 layers of cells.")
    assert ContentReason.HARD_FACT_MISMATCH in v.result.sentence_results[0].reason_codes


def test_T25_laterality(make_kit):
    v = hard_case(make_kit, "The left lung has two lobes.", "The right lung has two lobes.")
    assert v.result.status == "rejected"


@pytest.mark.parametrize("premise,hyp", [
    ("Hyaline cartilage has no blood vessels.", "Hyaline cartilage has blood vessels."),
    ("Hyaline cartilage has blood vessels.", "Hyaline cartilage has no blood vessels."),
    ("Not all epithelia are keratinized.", "No epithelia are keratinized."),
])
def test_T26_negation_scope(make_kit, premise, hyp):
    assert hard_case(make_kit, premise, hyp).result.status == "rejected"


def test_T27_exception_omission(make_kit):
    v = hard_case(make_kit, "The drug is safe unless the kidney fails.", "The drug is safe.")
    assert v.result.status == "rejected"


@pytest.mark.parametrize("premise,hyp", [("Most goblet cells secrete mucus.", "All goblet cells secrete mucus."),
                                         ("Mast cells sometimes degranulate.", "Mast cells always degranulate.")])
def test_T28_quantifier_strengthening(make_kit, premise, hyp):
    assert hard_case(make_kit, premise, hyp).result.status == "rejected"


@pytest.mark.parametrize("premise,hyp", [
    ("Afferent arterioles enter the glomerulus.", "Efferent arterioles enter the glomerulus."),
    ("Insulin will increase glucose uptake.", "Insulin will decrease glucose uptake."),
])
def test_T29_antonym_inversion(make_kit, premise, hyp):
    assert hard_case(make_kit, premise, hyp).result.status == "rejected"


def test_T30_unresolved_high_risk_relation(make_kit):
    kit = make_kit(passages=[passage("p1", "Chondrocytes sit in lacunae within the cartilage matrix.")])
    hyp = "The left drug dose always causes toxic injury."
    kit.scores.hyp[hyp] = SUPPORT
    v = one(kit, hyp)
    reasons = v.result.sentence_results[0].reason_codes
    assert ContentReason.HARD_FACT_UNRESOLVED in reasons or ContentReason.HARD_FACT_MISMATCH in reasons
    assert v.result.status == "rejected" and v.result.near_miss


def test_hyaline_cartilage_true_paraphrase_not_flagged(hf):
    premise = "It has no blood vessels and is nourished by diffusion from the perichondrium."
    assert hf.check("Hyaline cartilage has no blood vessels.", premise).status != "fail"


# ----------------------------------------------------------------------------- T31-T41 NLI


def test_T31_pair_order_is_premise_passage_hypothesis_sentence(kit):
    one(kit, TEXTS["p2"], cites=("p2",))
    assert kit.backend.pairs_seen == [(TEXTS["p2"], TEXTS["p2"])]
    asym = Kit(passages=[passage("p1", "Simple squamous epithelium lines the alveoli and the blood vessels.")])
    asym.scores.pairs[("Simple squamous epithelium lines the alveoli and the blood vessels.",
                       "Simple squamous epithelium lines the alveoli.")] = SUPPORT
    asym.scores.pairs[("Simple squamous epithelium lines the alveoli.",
                       "Simple squamous epithelium lines the alveoli and the blood vessels.")] = SUPPORT
    v = one(asym, "Simple squamous epithelium lines the alveoli.")
    assert v.result.status == "approved"
    assert asym.backend.pairs_seen[-1][0].endswith("blood vessels.")  # premise is the passage, never reversed


@pytest.mark.parametrize("mapping", [{"0": "entailment", "1": "contradiction", "2": "neutral"},
                                     {"0": "contradiction", "1": "entailment"},
                                     {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}])
def test_T32_label_mapping_mismatch_fails(mapping):
    with pytest.raises(NLIUnavailable):
        validate_label_mapping(mapping, {"0": "contradiction", "1": "entailment", "2": "neutral"})


def test_T32_swapped_fixture_labels_fail_readiness(make_kit):
    kit = make_kit()
    swapped = FixtureNLI(lambda p, h: SUPPORT, label_index={"contradiction": 0, "entailment": 1})
    v = Verifier.from_profile(PROFILE, audit=InMemoryVerifierAudit(), fixture_backend=swapped)
    assert v.readiness()["ready"] is False
    req = request(draft(S("s1", TEXTS["p1"], ["p1"])))
    r = v.verify(req, kit.context(req))
    assert (r.status, r.error_code) == ("error", OperationalError.MODEL_UNAVAILABLE)


@pytest.mark.parametrize("arr", [np.array([[np.nan, 0, 0]]), np.array([[np.inf, 0, 0]]), np.zeros((1, 2)),
                                 np.zeros((2, 3))])
def test_T33_non_finite_or_wrong_shape(arr):
    with pytest.raises(NLIOutputInvalid):
        to_scores(arr, {"contradiction": 0, "entailment": 1, "neutral": 2}, 1)


def test_T33_worker_bad_output_is_worker_failed(kit):
    kit.verifier.worker.backend.logits = lambda pairs: np.full((len(pairs), 3), np.nan)
    v = one(kit, TEXTS["p1"])
    assert v.result.error_code == OperationalError.WORKER_FAILED


TH = ThresholdProfile(profile_id="t", Tc=0.30, T_lo=0.50, T_hi=0.90)


def test_T34_contradiction_equality_wins():
    assert classify_pair({"contradiction": 0.30, "entailment": 0.95, "neutral": 0.0}, TH) == "contradiction"


@pytest.mark.parametrize("c,e,label", [(0.05, 0.90, "supported"), (0.05, 0.50, "uncertain"),
                                       (0.05, 0.4999, "unsupported"), (0.2999, 0.70, "uncertain")])
def test_T35_entailment_boundaries(c, e, label):
    assert classify_pair({"contradiction": c, "entailment": e, "neutral": 1 - c - e}, TH) == label


def test_T35_boundaries_through_softmax(make_kit):
    kit = make_kit()
    kit.scores.hyp[TEXTS["p1"]] = (0.05, 0.91, 0.04)  # exact equality is tested on classify_pair (float softmax)
    assert one(kit, TEXTS["p1"]).result.status == "approved"
    kit.scores.hyp[TEXTS["p1"]] = (0.30, 0.65, 0.05)
    assert ContentReason.CONTRADICTED in one(kit, TEXTS["p1"]).result.sentence_results[0].reason_codes


def test_T36_true_but_unsourced_rejects(kit):
    v = one(kit, "The heart has four chambers.")
    assert ContentReason.NOT_ENTAILED in v.result.sentence_results[0].reason_codes


def test_T37_composite_sources_do_not_combine(make_kit):
    kit = make_kit(passages=[passage("p1", "Goblet cells secrete mucus."), passage("p2", "Cilia move mucus.")])
    hyp = "Goblet cells secrete mucus that cilia move."
    kit.scores.pairs[("Goblet cells secrete mucus.", hyp)] = GRAY
    kit.scores.pairs[("Cilia move mucus.", hyp)] = GRAY
    v = one(kit, hyp, cites=("p1", "p2"))
    assert v.result.status == "rejected"
    assert ContentReason.UNCERTAIN_SUPPORT in v.result.sentence_results[0].reason_codes


def test_T38_one_contradictory_cite_rejects(make_kit):
    kit = make_kit()
    kit.scores.pairs[(TEXTS["p5"], TEXTS["p1"])] = CONTRA
    v = one(kit, TEXTS["p1"], cites=("p1", "p5"))
    assert ContentReason.CONTRADICTED in v.result.sentence_results[0].reason_codes


def test_T39_premise_beyond_nli_bound_is_evidence_limit(make_kit):
    long_text = " ".join(["Epithelial cells rest on a basement membrane."] * 60)
    kit = make_kit(passages=[passage("p1", long_text)])
    v = one(kit, "Epithelial cells rest on a basement membrane.")
    assert v.result.error_code == OperationalError.EVIDENCE_LIMIT and kit.backend.calls == 0


def test_T40_overlong_hypothesis_rejects_without_truncated_call(kit):
    hyp = "Simple squamous epithelium " + "very " * 100 + "thin."
    v = one(kit, hyp)
    assert v.result.response_code.value == "A5" and ContentReason.INVALID_DRAFT in v.answer_reasons
    assert kit.backend.calls == 0


def test_T41_gray_band_rejects_and_judge_is_disabled(kit):
    kit.scores.hyp[TEXTS["p1"]] = GRAY
    v = one(kit, TEXTS["p1"])
    assert ContentReason.UNCERTAIN_SUPPORT in v.result.sentence_results[0].reason_codes
    assert kit.verifier.profile.features["experimental_judge"] is False
