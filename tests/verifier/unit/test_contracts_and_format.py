"""T01-T11: strict contracts, segmentation, allowlist and citation integrity (injected fixture scores only)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts.models import ContentReason, ContractError, DraftAnswer, VerifyRequest, strict_json_loads
from tests.verifier.conftest import TEXTS, S, draft, passage

GOOD = {"schema_version": "verify-request-0.2", "request_id": "00000000-0000-4000-8000-000000000003",
        "prompt_version": "brain-prompt-0.2",
        "draft": {"schema_version": "brain-draft-0.2", "status": "draft",
                  "sentences": [S("s1", TEXTS["p1"], ["p1"])], "used_passage_ids": ["p1"]}}


def _req(mutate) -> dict:
    data = json.loads(json.dumps(GOOD))
    mutate(data)
    return data


def test_synthetic_valid_request_parses():
    req = VerifyRequest.model_validate(strict_json_loads(json.dumps(GOOD)))
    assert req.draft.sentences[0].cites == ("p1",)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra_field=1),
    lambda d: d.update(request_id=3),
    lambda d: d.update(request_id="not-a-uuid"),
    lambda d: d["draft"]["sentences"][0].update(cites="p1"),
    lambda d: d["draft"]["sentences"][0].update(kind_hint="summary"),
    lambda d: d["draft"].update(draft_text="legacy free text"),
    lambda d: d.update(thresholds={"T_hi": 0.1}),
])
def test_T01_strict_request_rejected(mutate, kit):
    with pytest.raises(ValidationError):
        VerifyRequest.model_validate(_req(mutate))
    assert kit.backend.calls == 0


def test_T01_duplicate_keys_and_nan_rejected_before_validation():
    with pytest.raises(ContractError):
        strict_json_loads('{"a": 1, "a": 2}')
    with pytest.raises(ContractError):
        strict_json_loads('{"a": NaN}')
    with pytest.raises(ContractError):
        strict_json_loads(b"\xff")


def test_T02_used_ids_must_equal_cites_union():
    with pytest.raises(ValidationError):
        VerifyRequest.model_validate(_req(lambda d: d["draft"].update(used_passage_ids=["p1", "p2"])))
    with pytest.raises(ValidationError):
        VerifyRequest.model_validate(_req(lambda d: d["draft"].update(used_passage_ids=[])))


@pytest.mark.parametrize("sentences", [
    [S("s1", TEXTS["p1"], ["p1"]), S("s1", TEXTS["p2"], ["p2"])],                   # duplicate id
    [S("s1", TEXTS["p1"], ["p1"], depends_on=["s1"])],                               # self dependency
    [S("s1", TEXTS["p1"], ["p1"], depends_on=["s2"]), S("s2", TEXTS["p2"], ["p2"])],  # forward reference
    [S("s1", TEXTS["p1"], ["p1"], visibility="internal")],                           # no visible sentence
    [S("s1", TEXTS["p1"], ["p1", "p1"])],                                            # duplicate cite
])
def test_T03_sentence_id_and_dependency_invariants(sentences):
    with pytest.raises(ValidationError):
        draft(*sentences)


def test_contract_limits():
    with pytest.raises(ValidationError):
        draft(*[S(f"s{i}", TEXTS["p1"], ["p1"]) for i in range(17)])
    with pytest.raises(ValidationError):
        draft(S("s1", "A" * 599 + "aa.", ["p1"]))
    with pytest.raises(ValidationError):
        draft(S("s1", TEXTS["p1"], ["a", "b", "c", "d", "e", "f"]))


@pytest.mark.parametrize("text", [
    "Simple squamous epithelium is thin. It lines the alveoli.",
    "Simple squamous epithelium is thin.\nIt lines the alveoli.",
    "# Epithelium heading text here.",
    "- Simple squamous epithelium is thin.",
    "Simple squamous epithelium is `thin`.",
    "See [the slide](http://example.org) for epithelium.",
    "Simple squamous epithelium is <b>thin</b>.",
    "Simple squamous epithelium is thin",
    "Простой плоский эпителий тонкий.",
])
def test_T04_hidden_sentence_merge_and_markup_reject_whole_draft(kit, text):
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", text, ["p2"])))
    r = v.result
    assert (r.status, r.response_code.value) == ("rejected", "A5")
    assert ContentReason.INVALID_DRAFT in v.answer_reasons
    assert kit.backend.calls == 0 and r.verified_text == ""


def test_T04_latin_terms_and_abbreviations_accepted(kit):
    text = "Simple squamous epithelium lines the alveoli of the lung."
    kit.scores.hyp["The tunica intima of the aorta is lined by endothelium."] = (0.01, 0.97, 0.02)
    v, _, _ = kit.run(draft(S("s1", text, ["p2"]),
                            S("s2", "The tunica intima of the aorta is lined by endothelium.", ["p2"])))
    assert v.result.status == "approved"


def test_T05_no_evidence_draft_is_A5_without_nli(kit):
    d = DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": "no_evidence",
                                    "sentences": [], "used_passage_ids": []})
    v, _, _ = kit.run(d)
    assert v.result.response_code.value == "A5" and kit.backend.calls == 0


def test_T06_transition_label_cannot_bypass_checking(kit):
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]),
                            S("s2", "Goblet cells secrete insulin.", (), kind="transition")))
    s2 = v.result.sentence_results[1]
    assert ContentReason.MISSING_CITATION in s2.reason_codes and s2.factual_checked


def test_T07_claim_without_digits_or_vocabulary_is_checked(kit):
    v, _, _ = kit.run(draft(S("s1", "Mesothelium wraps the viscera snugly.", ())))
    r = v.result
    assert r.status == "rejected" and ContentReason.MISSING_CITATION in r.sentence_results[0].reason_codes


def test_T08_allowlisted_prose_skips_nli_but_cannot_approve_alone(kit):
    filler = "Let us examine the concept."
    v, _, _ = kit.run(draft(S("s1", filler, ()), S("s2", TEXTS["p1"], ["p1"])))
    r = v.result
    assert r.status == "approved" and r.final_sentence_ids == ("s1", "s2")
    assert r.sentence_results[0].factual_checked is False and r.sentence_results[0].nli_scores == {}
    assert all(h != filler for _, h in kit.backend.pairs_seen)
    v2, _, _ = kit.run(draft(S("s1", filler, ())))
    assert ContentReason.NO_VISIBLE_SUPPORT in v2.answer_reasons and v2.result.response_code.value == "A5"
    # an allowlisted string that carries citations is checked like any other sentence
    v3, _, _ = kit.run(draft(S("s1", filler, ["p1"]), S("s2", TEXTS["p1"], ["p1"])))
    assert v3.result.sentence_results[0].factual_checked is True


def test_T09_missing_citation_fails_before_nli(kit):
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", TEXTS["p2"], ())))
    r = v.result
    s2 = r.sentence_results[1]
    assert s2.checks.citation == "fail" and s2.checks.entailment == "not_run" and s2.nli_scores == {}
    assert any(c.passage_id is None and c.reason == ContentReason.MISSING_CITATION for c in r.invalid_citations)
    assert all(h != TEXTS["p2"] for _, h in kit.backend.pairs_seen)


def test_T10_invented_citation_fails(kit):
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", TEXTS["p2"], ["made-up-7"])))
    r = v.result
    assert ContentReason.UNKNOWN_CITATION in r.sentence_results[1].reason_codes
    assert r.status == "approved" and "made-up-7" not in json.dumps(r.citation_map)  # trimmed, never displayed


def test_T11_valid_index_passage_not_shown_cannot_support(kit):
    rid_draft = draft(S("s1", TEXTS["p1"], ["p1"]), S("s2", TEXTS["p2"], ["p2"]))
    from tests.verifier.conftest import request

    req = request(rid_draft)
    bundle = kit.bundle(req.request_id, shown=["p1", "p3"])  # p2 is in the retrieval but was not shown
    v = kit.verifier.run(req, kit.context(req, bundle))
    assert ContentReason.UNKNOWN_CITATION in v.result.sentence_results[1].reason_codes


def test_T12_neutral_extra_citation_dropped_and_logged(kit):
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1", "p4"])))
    r = v.result
    assert r.status == "approved" and r.citation_map == {"s1": ("p1",)}
    assert any(c.passage_id == "p4" and c.reason == ContentReason.IRRELEVANT_CITATION for c in r.invalid_citations)


def test_passage_fixture_hash_is_computed():
    import hashlib

    p = passage("x", "Text.")
    assert p.text_sha256 == hashlib.sha256(b"Text.").hexdigest()
