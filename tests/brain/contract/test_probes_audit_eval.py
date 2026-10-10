"""B05, B10, B13 capability probes; B29 metadata-only audit; B31-B33 evaluation tooling (fake runtime)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from brain.audit import InMemoryBrainAudit, JsonlBrainAudit
from brain.capabilities import run_probes
from brain.evaluate import load_dataset, summarize, wilson
from brain.manifest import Manifest
from brain.parse import SentencePolicy
from brain.transport import LocalTransport

from ..conftest import BASE, KEY, draft_body, envelope, make_request, make_service, run
from ..unit.test_config_manifest_supervisor import manifest_fields


def anon_transport():
    return httpx.MockTransport(lambda r: httpx.Response(401) if r.url.path != "/" else httpx.Response(404))


def probes(runtime, cfg, manifest=None):
    async def go():
        t = LocalTransport(BASE, KEY, http_transport=httpx.MockTransport(runtime.handler))
        try:
            return await run_probes(cfg, manifest, t, api_key=KEY, sentence_policy=SentencePolicy(),
                                    anon_http_transport=anon_transport())
        finally:
            await t.close()
    return run(go())


def test_probes_pass_on_conforming_runtime(runtime, fixture_cfg):
    rep = probes(runtime, fixture_cfg)
    assert rep["passed"], {k: v for k, v in rep["probes"].items() if not v["pass"]}


def test_B05_known_count_mismatch_fails_readiness(runtime, fixture_cfg):
    rep = probes(runtime, fixture_cfg)
    good = rep["observed"]["known_prompt_tokens"]
    m = Manifest(**manifest_fields(gguf_filename="Qwen3.5-9B-Q4_K_M.gguf",
                                   capability_report={"known_prompt_tokens": good + 1}))
    rep2 = probes(runtime, fixture_cfg, m)
    assert not rep2["passed"] and rep2["probes"]["known_token_count"]["pass"] is False


def test_B05_template_hash_and_identity_mismatch_fail(runtime, fixture_cfg):
    m = Manifest(**manifest_fields(gguf_filename="Qwen3.5-9B-Q4_K_M.gguf", chat_template_sha256="f" * 64))
    rep = probes(runtime, fixture_cfg, m)
    assert rep["probes"]["chat_template_hash"]["pass"] is False and not rep["passed"]
    runtime.props["total_slots"] = 4
    rep2 = probes(runtime, fixture_cfg)
    assert rep2["probes"]["one_slot_context"]["pass"] is False


def test_B10_reasoning_leak_fails_probe(runtime, fixture_cfg):
    runtime.completion_script = lambda body: envelope(draft_body([], [], status="no_evidence"),
                                                      message_extra={"reasoning_content": "I think..."})
    rep = probes(runtime, fixture_cfg)
    assert rep["probes"]["schema_enforced_hostile_format"]["pass"] is False and not rep["passed"]


def test_B10_think_scaffold_missing_fails_probe(runtime, fixture_cfg, monkeypatch):
    original = runtime.render
    monkeypatch.setattr(runtime, "render", lambda messages: original(messages).replace("<think>\n\n</think>\n\n", ""))
    rep = probes(runtime, fixture_cfg)
    assert rep["probes"]["reasoning_off_template"]["pass"] is False


def test_B13_runtime_ignoring_bad_grammar_fails_capability(runtime, fixture_cfg):
    runtime.accept_bad_schema = True
    rep = probes(runtime, fixture_cfg)
    assert rep["probes"]["unsupported_grammar_rejected"]["pass"] is False and not rep["passed"]


def test_unauthenticated_runtime_fails_probe(runtime, fixture_cfg):
    async def go():
        t = LocalTransport(BASE, KEY, http_transport=httpx.MockTransport(runtime.handler))
        try:
            return await run_probes(fixture_cfg, None, t, api_key=KEY, sentence_policy=SentencePolicy(),
                                    anon_http_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        finally:
            await t.close()
    rep = run(go())
    assert rep["probes"]["local_authentication"]["pass"] is False


# ----------------------------------------------------------------------------- B29 audit is metadata only

SENTINEL_Q = "SENTINEL-QUESTION-7f3a"
SENTINEL_P = "SENTINEL-PASSAGE-91bc cells are flat."
SENTINEL_OUT = "SENTINEL-OUTPUT-c0de"


def test_B29_audit_records_hold_no_content(runtime, tmp_path):
    runtime.completion_script = lambda body: envelope(draft_body([{
        "sentence_id": "s1", "text": f"The {SENTINEL_OUT} layer is flat.", "kind_hint": "factual",
        "visibility": "student", "cites": ["fixture-passage-001"], "depends_on": []}]))
    jsonl = JsonlBrainAudit(tmp_path / "audit.jsonl")
    svc = make_service(runtime, audit=jsonl)
    req, ctx = make_request((("fixture-passage-001", SENTINEL_P),), question=f"What is {SENTINEL_Q}?")
    r = run(svc.generate(req, ctx))
    assert r.status == "ok"
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    for s in (SENTINEL_Q, SENTINEL_P, SENTINEL_OUT, KEY):
        assert s not in text
    rec = json.loads(text.splitlines()[-1])
    assert rec["request_id"] == str(req.request_id) and rec["passage_ids"] == ["fixture-passage-001"]


def test_B29_audit_refuses_content_fields():
    with pytest.raises(ValueError):
        InMemoryBrainAudit().write({"event": "x", "question": "raw text"})


def test_B29_transport_repr_hides_key():
    t = LocalTransport(BASE, KEY)
    assert KEY not in repr(t)
    asyncio.run(t.close())


# ----------------------------------------------------------------------------- B31-B33 evaluation tooling

def test_B31_dataset_must_be_marked_synthetic(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"case_id": "a", "content": "student"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(p)
    p.write_text(json.dumps({"case_id": "a", "content": "synthetic"}) + "\n" +
                 json.dumps({"case_id": "a", "content": "synthetic"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(p)


def test_B31_committed_synthetic_dataset_loads_and_counterfactuals_share_groups():
    from pathlib import Path

    rows = load_dataset(Path(__file__).resolve().parents[2] / "fixtures/brain/synthetic.jsonl")
    cf = [r for r in rows if r.get("kind") == "counterfactual"]
    assert cf, "counterfactual pairs expected"
    for r in cf:
        parents = [x for x in rows if x["case_id"] == r["paired_with"]]
        assert parents and parents[0]["group_id"] == r["group_id"]  # B32: mutation shares its parent group
    assert all(r["partition"] == "development" for r in rows)


def _row(case, status, cites=(), sentences=0, err=None, total=100.0):
    return {"case_id": case, "status": status, "error_code": err, "as_expected": status == "ok",
            "cites": list(cites), "sentences": sentences,
            "metrics": {"finish_reason": "stop" if status != "error" else None, "total_ms": total,
                        "prompt_tokens": 400, "completion_tokens": 100, "prompt_ms": 200.0, "generation_ms": 1000.0}}


def test_B33_repeat_consistency_reports_variability_not_determinism():
    rows = [_row("c1", "ok", ["p1"], 2), _row("c1", "ok", ["p1", "p2"], 3), _row("c1", "no_evidence"),
            _row("c2", "ok", ["p1"], 1)]
    s = summarize(rows)
    assert s["consistency"]["c1"] == {"repeats": 3, "distinct_statuses": 2, "distinct_cite_sets": 3,
                                      "sentence_count_range": [0, 3]}
    assert "c2" not in s["consistency"]
    assert s["status_counts"] == {"ok": 3, "no_evidence": 1}
    assert s["decode_tokens_per_s"] == 100.0 and s["latency_ms"]["n"] == 4


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
