"""Gate -> real RetrieverService -> spy Brain/verifier integration (Retriever spec v0.2, sections 11-12, 16)."""

from __future__ import annotations

import pytest

from gate_classifier.adapters import DraftAnswer
from gate_classifier.orchestrator import FixtureBrain, FixtureVerifier
from gate_classifier.schema import ErrorCode, ResponseCode
from retriever.schema import Status
from tests.conftest import Harness, ScriptedScorer, scores
from tests.retriever.helpers import KB, find

Q = "Explain what simple squamous epithelium lines."


def wire(env, **kw) -> Harness:
    h = Harness(retriever=env.service, **kw)
    h.orch.kb_version = KB
    return h


def items_for(env):
    snap = env.snapshot
    return [{"item_id": "item-e1", "item_version": "v1", "approved": True,
             "evidence_sets": [{"passage_ids": [find(snap, "Practice item E1"), find(snap, "single layer of flat cells")],
                                "coverage": "full"}]}]


def test_end_to_end_answer_with_real_retriever(env):
    h = wire(env)
    r = h.ask(Q)
    assert r.response_code == ResponseCode.A1, (r.http_status, r.error_code)
    assert r.citations and r.citations[0]["number"] == 1 and r.citations[0]["evidence_uri"].startswith("/v1/evidence/")
    brain_req = h.brain.calls[0]
    assert "relevance" not in brain_req.prompt and "6.25" not in brain_req.prompt  # no scores in the prompt
    assert h.verifier.seen[0].digest == brain_req.evidence.digest  # identical evidence map


def test_no_evidence_maps_to_A5_without_brain(make_env):
    env = make_env(score_fn=lambda q, d: -9.0)
    h = wire(env)
    r = h.ask(Q)
    assert r.response_code == ResponseCode.A5 and h.brain.calls == [] and h.verifier.calls == 0


def test_retrieval_error_is_unavailable_without_brain(make_env):
    env = make_env(score_fn=lambda q, d: (_ for _ in ()).throw(RuntimeError("x")))
    h = wire(env)
    r = h.ask(Q)
    assert (r.http_status, r.error_code) == (503, ErrorCode.SERVICE_UNAVAILABLE) and h.brain.calls == []


def test_T45_T47_quiz_flow_uses_item_fetch(make_env):
    probe = make_env()
    env = make_env(items=items_for(probe))
    h = wire(env)
    first = h.ask("Quiz me on simple squamous epithelium alveoli practice item")
    assert first.response_code == ResponseCode.A1 and first.session_id
    s = h.sessions.get(first.session_id)
    assert s.pending_item_id == "item-e1" and s.state == "awaiting_response"
    searches = env.service.audit.records[:]
    reply = h.ask("B", session_id=first.session_id)
    assert reply.response_code == ResponseCode.A1
    new = env.service.audit.records[len(searches):]
    assert [r.operation for r in new] == ["item_fetch"]  # no semantic search for "B"
    # T47: a bare reply without pending state never reaches the retriever
    before = len(env.service.audit.records)
    assert h.ask("B").response_code == ResponseCode.A3
    assert len(env.service.audit.records) == before


def test_T47_gate_blocked_turn_never_calls_retriever(env):
    h = wire(env)
    r = h.ask("My mother has a lump in her neck, what should she take?")
    assert r.response_code == ResponseCode.A6 and env.service.audit.records == []


def test_T52_context_budget_whole_passage_removal(make_env):
    env = make_env(score_fn=lambda q, d: 3.0)
    full = wire(env)
    full.ask(Q)
    n_full = len(full.brain.calls[0].evidence.passages)
    assert n_full >= 2
    prompt_tokens = full.brain.count_tokens(full.brain.calls[0].prompt)
    small = wire(env, brain=FixtureBrain(context_limit=prompt_tokens - 1 + 256))
    r = small.ask(Q)
    assert r.response_code == ResponseCode.A1
    kept = small.brain.calls[0].evidence.passages
    assert 1 <= len(kept) < n_full
    assert all(p.text == env.snapshot.by_id[p.passage_id].text for p in kept)  # never trimmed mid-passage
    tiny = wire(env, brain=FixtureBrain(context_limit=300))
    assert tiny.ask(Q).error_code == ErrorCode.SERVICE_UNAVAILABLE and tiny.brain.calls == []


def test_T52_required_item_set_cannot_be_thinned(make_env):
    probe = make_env()
    env = make_env(items=items_for(probe))
    h = wire(env)
    first = h.ask("Quiz me on simple squamous epithelium alveoli practice item")
    h.brain.context_limit = 256 + 120  # cannot fit both required passages
    r = h.ask("B", session_id=first.session_id)
    assert r.error_code == ErrorCode.SERVICE_UNAVAILABLE
    assert h.sessions.get(first.session_id).answered_items == ()


def test_T53_evidence_map_mismatch_blocks_delivery(env):
    from gate_classifier.adapters import VerificationResult

    class WrongMap(FixtureVerifier):
        def verify(self, draft, evidence, *a):
            self.calls += 1
            return VerificationResult(status="approved", evidence_digest="0" * 64, verified_text=draft.draft_text,
                                      verifier_version="spy")

    h = wire(env, verifier=WrongMap())
    r = h.ask(Q)
    assert r.error_code == ErrorCode.SERVICE_UNAVAILABLE and r.text is None


def test_T40_revocation_before_brain_blocks_generation(env):
    h = wire(env)
    original = env.service.revalidate

    def revoke_then_check(result, ctx):
        for p in result.passages:
            env.revocations.revoke(p.source_id)
        return original(result, ctx)

    env.service.revalidate = revoke_then_check
    r = h.ask(Q)
    assert r.error_code == ErrorCode.SERVICE_UNAVAILABLE and h.brain.calls == []


def test_T59_retriever_audit_failure_blocks_generation(env):
    class Broken:
        def write(self, record):
            raise OSError("disk full")

    env.service.audit = Broken()
    h = wire(env)
    r = h.ask(Q)
    assert r.error_code == ErrorCode.SERVICE_UNAVAILABLE and h.brain.calls == []


def test_gate_vector_is_not_reused_by_retriever(env):
    """The gate encodes 'Current request: ...' so its vector is incompatible; the retriever recomputes."""
    from gate_classifier.encoder import FixtureEncoder
    from gate_classifier.schema import EmbeddingRef

    h = wire(env)
    h.ask(Q)
    assert env.service.encode_calls == 1
    rec = env.service.audit.records[-1]
    assert rec.embedding_reused is False
    assert EmbeddingRef is __import__("retriever.schema", fromlist=["EmbeddingRef"]).EmbeddingRef
    _ = FixtureEncoder


def test_scope_from_gate_matches_retriever(env):
    """Gate's all_active plan equals the retriever's trusted active set (no SCOPE_MISMATCH)."""
    h = wire(env)
    h.ask(Q)
    rec = env.service.audit.records[-1]
    assert rec.status == Status.ok.value and rec.libraries_searched == ("lib1", "lib2", "lib5", "lib6")


@pytest.mark.parametrize("text", ["What does the slide say about cartilage?"])
def test_brain_receives_no_scores_or_paths(env, text):
    h = wire(env, scorer=ScriptedScorer(scores()))
    h.ask(text)
    if h.brain.calls:
        prompt = h.brain.calls[0].prompt
        assert "artifacts" not in prompt and "cross_encoder_logit" not in prompt
    _ = DraftAnswer
