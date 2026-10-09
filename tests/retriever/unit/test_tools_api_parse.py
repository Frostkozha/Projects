"""Evaluation tooling (T60), HTTP wrapper, parser safety and real-model code paths."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from retriever.api import create_app
from retriever.parse import ParseError, parse_json, parse_markdown
from retriever.register import SourceRecord
from retriever.schema import ErrorCode
from tests.retriever.helpers import KB, _record, find
from tools.evaluate import DatasetError, evaluate, validate_dataset
from tools.tune_profile import sweep


def rec(i, group, partition="development", answerability="answerable", rel=("p",), **kw):
    r = {"id": f"q{i}", "group_id": group, "partition": partition, "query_text": "q", "course_id": "histology-dev",
         "kb_version": KB, "topic": "epithelium", "answerability": answerability,
         "relevant_passage_ids": list(rel), "sufficient_evidence_sets": [list(rel)] if rel else [],
         "expected_conflict": None, "annotator_ids": ["a", "b"], "approval_status": "approved"}
    if answerability == "unanswerable":
        r.update(relevant_passage_ids=[], sufficient_evidence_sets=[], missing_scope="reviewed: not in corpus")
    r.update(kw)
    return r


def test_T60_group_crossing_partitions_rejected():
    with pytest.raises(DatasetError, match="cross partitions"):
        validate_dataset([rec(1, "g1"), rec(2, "g1", partition="final")])
    with pytest.raises(DatasetError, match="not approved"):
        validate_dataset([rec(1, "g1", approval_status="draft")])
    with pytest.raises(DatasetError, match="missing_scope"):
        validate_dataset([{**rec(1, "g1", answerability="unanswerable"), "missing_scope": ""}])


def test_T60_tuning_refuses_final_partition(env):
    with pytest.raises(DatasetError, match="development"):
        sweep(env.service, env.ctx(), [rec(1, "g1", partition="final")], list(env.request("x")["allowed_libraries"]),
              [0.0])


def test_evaluate_and_sweep_on_synthetic(env):
    snap = env.snapshot
    target = find(snap, "single layer of flat cells")
    data = [rec(1, "g1", query_text="simple squamous single layer flat cells alveoli", rel=(target,)),
            rec(2, "g2", answerability="unanswerable", query_text="zzqx unrelated")]
    validate_dataset(data)
    active = ["lib1", "lib2", "lib5", "lib6"]
    report = evaluate(env.service, lambda r: env.ctx(), data, active)
    assert report["hit_at_5"]["successes"] == 1 and report["candidate_hit_at_30"]["successes"] == 1
    assert report["access_citation_violations"] == 0 and report["operational_errors"] == 0
    rows = sweep(env.service, env.ctx(), data, active, [-10.0, 0.0, 100.0])
    assert rows[-1]["false_no_evidence"] == 1.0 and rows[-1]["correct_no_evidence"] == 1.0


# --------------------------------------------------------------------------- API


@pytest.fixture
def client(env):
    app = create_app(env.service, lambda sid, req: env.ctx(service_id=sid), {"secret-token": "orchestrator"})
    return TestClient(app)


AUTH = {"Authorization": "Bearer secret-token"}


def test_api_ok_and_error_mapping(env, client):
    r = client.post("/v1/retrieve", json=env.request("simple squamous epithelium"), headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "ok"
    for body, status, code in ((env.request("x", allowed_libraries=[]), 422, "INVALID_SCOPE"),
                               (env.request("x", allowed_libraries=["lib1", "lib2", "lib5"]), 409, "SCOPE_MISMATCH"),
                               (env.request("x", kb_version="fixture-kb-001"), 409, "VERSION_MISMATCH"),
                               (env.request("epithelium " * 330), 413, "INPUT_TOO_LONG"),
                               ({"query_text": "x"}, 422, "INVALID_REQUEST")):
        r = client.post("/v1/retrieve", json=body, headers=AUTH)
        assert r.status_code == status and r.json()["error_code"] == code
        assert set(r.json()) == {"error_code", "request_id"}
    assert client.post("/v1/retrieve", json=env.request("x")).status_code == 401
    big = client.post("/v1/retrieve", content=b"{" + b" " * 40000 + b"}", headers={**AUTH, "content-type": "application/json"})
    assert big.status_code == 413
    no_ev = client.post("/v1/retrieve", json=env.request("zzqx"), headers=AUTH)
    assert no_ev.status_code == 200 and no_ev.json()["status"] == "no_evidence"


def test_api_evidence_endpoint(env, client):
    pid = find(env.snapshot, "single layer of flat cells")
    ok = client.get(f"/v1/evidence/{pid}", params={"kb_version": KB}, headers=AUTH)
    assert ok.status_code == 200 and ok.json()["passage_id"] == pid and "file_ref" not in ok.text and "sources/" not in ok.text
    assert client.get("/v1/evidence/psg-guess", params={"kb_version": KB}, headers=AUTH).status_code == 403
    env.revocations.revoke("slides")
    assert client.get(f"/v1/evidence/{pid}", params={"kb_version": KB}, headers=AUTH).status_code == 403
    assert client.get("/health/ready").json() == {"status": "fixture_only"}


# --------------------------------------------------------------------------- parser safety


def _src(fmt="markdown"):
    return SourceRecord.model_validate(_record("s", "x.md", fmt, "lib1", "0" * 64))


@pytest.mark.parametrize("text,reason", [("# A\n\n<script>alert(1)</script>", "active_html"),
                                         ("# A\n\n![img](https://evil.example/x.png)", "remote_resource"),
                                         ("# A\n\n[link](javascript:alert(1))", "active_html")])
def test_markdown_rejects_active_content(text, reason):
    with pytest.raises(ParseError, match=reason):
        parse_markdown(text, _src())


def test_markdown_structure_and_normalization():
    res = parse_markdown("# Bone\n\n## Osteon\n\nH₂O and Ca²⁺ at 10 µm.  Not   vascular. {#b-osteon}\n\n- one\n- two\n",
                         _src())
    para, lst = res.blocks
    assert para.block_id == "b-osteon" and para.section_path == ("Bone", "Osteon")
    assert para.text == "H₂O and Ca²⁺ at 10 µm. Not vascular." and lst.kind == "list"


def test_json_schema_strict():
    with pytest.raises(ParseError):
        parse_json(json.dumps({"blocks": [{"block_id": "a", "section_path": [], "kind": "video", "text": "x"}]}),
                   _src("json"))
    with pytest.raises(ParseError):
        parse_json(json.dumps({"blocks": [], "extra": 1}), _src("json"))


# --------------------------------------------------------------------------- real-model code paths (tiny local models)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    d = tmp_path_factory.mktemp("tiny")
    words = ("[PAD] [UNK] [CLS] [SEP] [MASK] query passage : simple squamous epithelium lines alveoli "
             "cartilage bone layer cells the a of").split()
    (d / "vocab.txt").write_text("\n".join(words))
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    enc_dir, rr_dir = d / "enc", d / "rr"
    cfg = dict(vocab_size=len(words), hidden_size=384, num_hidden_layers=1, num_attention_heads=4,
               intermediate_size=64, max_position_embeddings=512)
    tok.save_pretrained(enc_dir)
    transformers.BertModel(transformers.BertConfig(**cfg)).save_pretrained(enc_dir)
    tok.save_pretrained(rr_dir)
    transformers.BertForSequenceClassification(transformers.BertConfig(**cfg, num_labels=1)).save_pretrained(rr_dir)
    bad_dir = d / "bad"
    tok.save_pretrained(bad_dir)
    transformers.BertForSequenceClassification(transformers.BertConfig(**cfg, num_labels=2)).save_pretrained(bad_dir)
    return enc_dir, rr_dir, bad_dir


def test_real_reranker_raw_logits_and_config_checks(tiny_models):
    from retriever.rerank import CrossEncoderReranker
    from retriever.schema import RetrievalError

    _, rr_dir, bad_dir = tiny_models
    rr = CrossEncoderReranker(str(rr_dir), "tiny", "a" * 40, 512)
    out = rr.score([("simple squamous", "epithelium lines alveoli"), ("bone", "cartilage cells")])
    assert out.shape == (2, 1)
    assert rr.tokenizer.pair_special_tokens() == 3 and rr.tokenizer.count_pair("bone", "cells") == 5
    with pytest.raises(RetrievalError) as exc:
        CrossEncoderReranker(str(bad_dir), "tiny", "a" * 40, 512)  # two labels: not a single-logit reranker
    assert exc.value.code == ErrorCode.MODEL_UNAVAILABLE
    with pytest.raises(RetrievalError):
        CrossEncoderReranker(str(rr_dir), "tiny", "a" * 40, 1024)  # pair ceiling beyond model positions
    with pytest.raises(RetrievalError):
        CrossEncoderReranker(str(rr_dir), "tiny", None, 512)  # unpinned


def test_real_e5_encoder_path_with_tiny_model(tiny_models):
    from gate_classifier.encoder import E5Encoder

    enc_dir, _, _ = tiny_models
    e = E5Encoder(str(enc_dir), "tiny", "b" * 40, "b" * 40)
    v = e.encode(["passage: simple squamous epithelium", "query: bone"])
    assert v.shape == (2, 384) and abs(float((v[0] ** 2).sum()) - 1.0) < 1e-4
