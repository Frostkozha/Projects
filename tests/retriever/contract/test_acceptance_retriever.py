"""Retriever acceptance scenarios (Retriever spec v0.2, section 16).

Deterministic synthetic models and corpus. These verify contracts and behavior, not retrieval quality.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from gate_classifier.encoder import FixtureEncoder
from retriever import service as service_mod
from retriever.build import BuildError
from retriever.dense import dense_search, make_embedding_ref
from retriever.fetch import ItemMapping
from retriever.fuse import rrf_ids
from retriever.lexical import build_match_query
from retriever.parse import ImportRejected, resolve_import
from retriever.register import SourceRegister
from retriever.schema import ErrorCode, Reason, Status
from retriever.select import Candidate, select
from retriever.service import BoundedWorker, build_service
from retriever.snapshot import Snapshot, SnapshotError
from tests.retriever.helpers import ACTIVE, COURSE, KB, NOW, TENANT, Env, find, fixture_profile

Q = "What does simple squamous epithelium line?"


def err(result, code):
    assert result.status == Status.error and result.error_code == code, (result.status, result.error_code)
    assert result.passages == () and result.conflicts == ()


def nothing_ran(env):
    assert env.service.encode_calls == 0 and env.models.reranker.calls == 0


# --------------------------------------------------------------------------- T01-T06 validation and scope


@pytest.mark.parametrize("patch", [
    {"query_text": ""}, {"query_text": "   "}, {"query_text": 42}, {"request_id": "not-a-uuid"},
    {"strategy": "selective"}, {"course_id": "bad id!"}, {"unknown_field": 1}, {"allowed_libraries": "lib1"},
])
def test_T01_invalid_request(env, patch):
    body = env.request(Q)
    body.update(patch)
    err(env.service.retrieve(body, env.ctx()), ErrorCode.INVALID_REQUEST)
    nothing_ran(env)


def test_T01_missing_field(env):
    body = env.request(Q)
    del body["preferred_libraries"]
    err(env.service.retrieve(body, env.ctx()), ErrorCode.INVALID_REQUEST)


def test_T02_query_limits_no_truncation(env):
    err(env.service.retrieve(env.request("epithelium " * 330), env.ctx()), ErrorCode.INPUT_TOO_LONG)
    err(env.service.retrieve(env.request("a" * 4001), env.ctx()), ErrorCode.INPUT_TOO_LONG)
    nothing_ran(env)


def test_T03_empty_scope_is_never_all_libraries(env):
    err(env.service.retrieve(env.request(Q, allowed_libraries=[]), env.ctx()), ErrorCode.INVALID_SCOPE)
    nothing_ran(env)


@pytest.mark.parametrize("allowed,code", [
    (["lib1", "lib2", "lib5", "lib9"], ErrorCode.INVALID_SCOPE),
    (["lib1", "lib1", "lib2", "lib5", "lib6"], ErrorCode.INVALID_SCOPE),
    (["lib1", "lib2", "lib3", "lib5", "lib6"], ErrorCode.INVALID_SCOPE),
])
def test_T04_unknown_duplicate_disabled(env, allowed, code):
    err(env.service.retrieve(env.request(Q, allowed_libraries=allowed), env.ctx()), code)
    nothing_ran(env)


def test_T04_unauthorized_library(env):
    ctx = env.ctx(authorized_libraries=frozenset({"lib1", "lib2", "lib5"}))
    err(env.service.retrieve(env.request(Q), ctx), ErrorCode.FORBIDDEN)


def test_T05_subset_is_scope_mismatch(env):
    err(env.service.retrieve(env.request(Q, allowed_libraries=["lib1", "lib2", "lib5"]), env.ctx()),
        ErrorCode.SCOPE_MISMATCH)


def test_T06_preferences_do_not_change_search(env):
    a = env.service.retrieve(env.request(Q), env.ctx())
    b = env.service.retrieve(env.request(Q, preferred_libraries=["lib6"]), env.ctx())
    assert a.status == Status.ok and [p.passage_id for p in a.passages] == [p.passage_id for p in b.passages]
    assert b.libraries_searched == ACTIVE
    err(env.service.retrieve(env.request(Q, preferred_libraries=["lib3"]), env.ctx()), ErrorCode.INVALID_SCOPE)


def test_context_must_be_trusted_and_gate_permitted(env):
    err(env.service.retrieve(env.request(Q), env.ctx(service_id="student-ui")), ErrorCode.UNAUTHORIZED)
    err(env.service.retrieve(env.request(Q), env.ctx(gate_permitted=False)), ErrorCode.FORBIDDEN)
    err(env.service.retrieve(env.request(Q), env.ctx(tenant_id="other-tenant")), ErrorCode.FORBIDDEN)


# --------------------------------------------------------------------------- T07-T11 fusion and search


def test_T07_both_branches_add_contributions():
    out = dict(rrf_ids([["a", "b"], ["a", "c"]]))
    assert out["a"] == pytest.approx(2 / 61) and out["b"] == pytest.approx(1 / 62)


def test_T07_one_rerank_pair_per_id(env):
    seen = []
    env.models.reranker.score_fn = lambda q, d: (seen.append(d), 5.0)[1]
    env.service.retrieve(env.request(Q), env.ctx())
    assert len(seen) == len(set(seen))


def test_T08_branch_duplicates_count_once():
    out = dict(rrf_ids([["a", "a", "b"]]))
    assert out["a"] == pytest.approx(1 / 61) and out["b"] == pytest.approx(1 / 62)  # ranks recomputed


def test_T09_stable_tie_breaking(env):
    assert rrf_ids([["b"], ["a"]]) == rrf_ids([["a"], ["b"]]) == [("a", 1 / 61), ("b", 1 / 61)]
    env.models.reranker.score_fn = lambda q, d: 1.0  # every logit ties
    runs = {tuple(p.passage_id for p in env.service.retrieve(env.request(Q), env.ctx()).passages) for _ in range(3)}
    assert len(runs) == 1


def test_T10_filter_before_top_k():
    m = np.eye(4, dtype=np.float32)
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    ids = ["unauth-a", "allowed-b", "allowed-c", "allowed-d"]
    out = dense_search(m, ids, np.array([1, 2, 3]), q, 2)
    assert [pid for pid, _ in out] == ["allowed-b", "allowed-c"]


def test_T11_empty_lexical_branch_is_normal(make_env):
    env = make_env(score_fn=lambda q, d: 5.0)
    r = env.service.retrieve(env.request("?! ..."), env.ctx())
    assert build_match_query("?! ...") is None
    assert r.status == Status.ok and env.models.reranker.calls >= 1
    rec = env.service.audit.records[-1]
    assert rec.lexical_count == 0 and rec.dense_count > 0


# --------------------------------------------------------------------------- T12-T17 failures and vectors


def test_T12_operation_failures_are_errors(env, monkeypatch):
    env.models.encoder.encode = lambda texts: (_ for _ in ()).throw(RuntimeError("boom"))
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.SEARCH_FAILED)


def test_T12_lexical_failure(env, monkeypatch):
    def broken(*a, **k):
        from retriever.schema import RetrievalError
        raise RetrievalError(ErrorCode.SEARCH_FAILED)
    monkeypatch.setattr(service_mod, "lexical_search", broken)
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.SEARCH_FAILED)
    assert env.models.reranker.calls == 0  # no alternative (dense-only) profile


@pytest.mark.parametrize("score_fn", [
    lambda q, d: (_ for _ in ()).throw(RuntimeError("reranker crashed")),
])
def test_T13_reranker_failure(make_env, score_fn):
    env = make_env(score_fn=score_fn)
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.RERANK_FAILED)


def test_T13_missing_score_and_timeout(env):
    env.models.reranker.score = lambda pairs: np.zeros(len(pairs) - 1)
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.RERANK_FAILED)
    env.models.reranker.score = lambda pairs: (time.sleep(0.4), np.zeros(len(pairs)))[1]
    env.service.profile = env.service.profile.model_copy(
        update={"runtime": env.service.profile.runtime.model_copy(update={"deadline_seconds": 0.2})})
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.DEADLINE_EXCEEDED)


def test_T14_nonfinite_or_bad_shapes(env):
    env.models.reranker.score = lambda pairs: np.full(len(pairs), np.nan)
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.RERANK_FAILED)
    env.models.reranker.score = lambda pairs: np.zeros((len(pairs), 2))
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.RERANK_FAILED)
    env.models.encoder.encode = lambda texts: np.ones((1, 383), dtype=np.float32)
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.SEARCH_FAILED)


def test_T15_compatible_embedding_ref_skips_encoding(env):
    base = env.service.retrieve(env.request(Q), env.ctx())
    assert env.service.encode_calls == 1
    text = env.profile.encoder.query_prefix + Q
    ref = make_embedding_ref(env.models.encoder.encode([text])[0], text, env.models.encoder, env.profile)
    reused = env.service.retrieve(env.request(Q), env.ctx(embedding_ref=ref))
    assert env.service.encode_calls == 1  # skipped
    assert [(p.passage_id, p.relevance_score) for p in reused.passages] == \
        [(p.passage_id, p.relevance_score) for p in base.passages]
    assert env.service.audit.records[-1].embedding_reused is True


def test_T16_incompatible_refs_are_recomputed(env):
    enc = env.models.encoder
    gate_text = "query: Current request: " + Q
    for ref in (make_embedding_ref(enc.encode([gate_text])[0], gate_text, enc, env.profile),
                make_embedding_ref(enc.encode(["query: " + Q])[0], "query: " + Q, enc,
                                   env.profile.model_copy(update={"encoder": env.profile.encoder.model_copy(
                                       update={"query_prefix": "q: "})}))):
        before = env.service.encode_calls
        r = env.service.retrieve(env.request(Q), env.ctx(embedding_ref=ref))
        assert r.status == Status.ok and env.service.encode_calls == before + 1


def test_T17_malformed_embedding_ref(env):
    enc = env.models.encoder
    good = make_embedding_ref(enc.encode(["query: " + Q])[0], "query: " + Q, enc, env.profile)
    from dataclasses import replace
    for bad in (replace(good, vector=good.vector.astype(np.float64)), replace(good, vector=good.vector * 2),
                replace(good, vector=np.full(384, np.nan, dtype=np.float32)), replace(good, vector=good.vector[:10])):
        err(env.service.retrieve(env.request(Q), env.ctx(embedding_ref=bad)), ErrorCode.INVALID_EMBEDDING_REF)


# --------------------------------------------------------------------------- T18-T23 thresholds and selection


def _cand(pid, logit, src="s", section=("A",), sha=None, spans=None, rrf=0.01):
    return Candidate(pid, logit, rrf, (src, "1"), section, sha or pid, spans or ((pid, 0, 10),))


def test_T18_raw_logits(make_env):
    scores = {"alveoli": 8.0, "Stratified": 0.0}
    env = make_env(score_fn=lambda q, d: next((v for k, v in scores.items() if k in d), -2.0))
    p = env.profile.model_copy(update={"threshold": env.profile.threshold.model_copy(update={"t_rerank": -1.0})})
    env.service.profile = p
    r = env.service.retrieve(env.request(Q), env.ctx())
    got = sorted({p.relevance_score for p in r.passages})
    assert 8.0 in got and 0.0 in got and -2.0 not in got  # raw logits, nothing forced into [0,1]
    assert all(p.score_type == "cross_encoder_logit" for p in r.passages)


def test_T19_equality_passes():
    sel = select([_cand("a", 1.5), _cand("b", 1.49, src="t")], 1.5, 5, 0.8)
    assert [c.passage_id for c in sel.passages] == ["a"]


def test_T20_all_below_threshold(make_env):
    env = make_env(score_fn=lambda q, d: -5.0)
    r = env.service.retrieve(env.request(Q), env.ctx())
    assert (r.status, r.reason, r.error_code, r.passages) == (Status.no_evidence, Reason.BELOW_THRESHOLD, None, ())


def test_T21_no_eligible_passages(env):
    for sid in ("notes", "slides", "quizbank", "atlas-a", "atlas-b", "textbook"):
        env.revocations.revoke(sid)
    r = env.service.retrieve(env.request(Q), env.ctx())
    assert (r.status, r.reason) == (Status.no_evidence, Reason.NO_ELIGIBLE_PASSAGES)
    nothing_ran(env)


def test_T22_no_branch_candidates(env, monkeypatch):
    monkeypatch.setattr(service_mod, "dense_search", lambda *a, **k: [])
    r = env.service.retrieve(env.request("?!"), env.ctx())
    assert (r.status, r.reason) == (Status.no_evidence, Reason.NO_MATCHES)
    assert env.models.reranker.calls == 0


def test_T23_at_most_five_deterministic(make_env):
    env = make_env(score_fn=lambda q, d: 3.0)
    runs = [env.service.retrieve(env.request("epithelium cartilage bone cells layer"), env.ctx()) for _ in range(2)]
    assert len(runs[0].passages) == 5
    assert [p.passage_id for p in runs[0].passages] == [p.passage_id for p in runs[1].passages]


def test_T24_overlap_suppression():
    a = _cand("a", 5, spans=(("b1", 0, 100),))
    dup = _cand("b", 4, spans=(("b1", 10, 100),))       # 0.90 overlap with a
    comp = _cand("c", 3, spans=(("b1", 90, 200),))      # 0.10 overlap: complementary
    sel = select([a, dup, comp], 0, 5, 0.8)
    assert [c.passage_id for c in sel.passages] == ["a", "c"] and sel.suppressed == ["b"]
    same_text = _cand("d", 2, sha="a")  # exact same-source duplicate text
    assert "d" in select([a, same_text], 0, 5, 0.8).suppressed


def test_T25_conflicting_sources_not_deduplicated(env):
    r = env.service.retrieve(env.request("articular cartilage thickness in adults"), env.ctx())
    sources = {p.source_id for p in r.passages}
    assert {"atlas-a", "atlas-b"} <= sources


# --------------------------------------------------------------------------- T26-T31 index integrity and parsing


def _reseal(snapshot_dir: Path):
    from retriever.build import file_sha256
    m = json.loads((snapshot_dir / "manifest.json").read_text())
    for name in ("snapshot.sqlite", "vectors.npy"):
        m["artifacts"][name] = file_sha256(snapshot_dir / name)
    (snapshot_dir / "manifest.json").write_text(json.dumps(m))


def _open(env, path):
    from retriever.chunk import chunking_fingerprint
    from retriever.dense import preprocessing_fingerprint
    return Snapshot.open(path, profile=env.profile, preprocessing_fp=preprocessing_fingerprint(env.models.encoder,
                         env.profile), chunking_fp=chunking_fingerprint(env.profile), operating_mode="fixture")


def test_T26_same_id_different_text_is_corruption(env):
    import sqlite3
    path = env.build("fixture-index-x", KB, activate=False)
    conn = sqlite3.connect(path / "snapshot.sqlite")
    conn.execute("UPDATE passages SET text = text || ' tampered' WHERE rowid = 1")
    conn.commit()
    conn.close()
    _reseal(path)
    with pytest.raises(SnapshotError, match="passage_text_hash_mismatch"):
        _open(env, path)
    with pytest.raises(SnapshotError):
        env.service.snapshots.activate(COURSE, "fixture-index-x")
    assert env.service.snapshots.active_version(COURSE) == "fixture-index-002"


def test_T27_section_locator_without_page_or_year(env):
    snap = env.snapshot
    row = snap.by_id[find(snap, "Simple cuboidal epithelium has one layer")]
    assert row.locator["kind"] == "section" and row.locator["anchor"] and row.locator["start"] is None
    assert row.publication_year is None and row.edition is None
    slide = snap.by_id[find(snap, "single layer of flat cells")]
    assert slide.locator == {"kind": "slide", "start": 4, "end": 4, "label": "Slide 4", "anchor": None}


def test_T28_pdf_physical_index_and_printed_label(env):
    row = env.snapshot.by_id[find(env.snapshot, "Transitional epithelium")]
    assert row.locator["kind"] == "pdf_page" and row.locator["start"] == 1 and row.locator["end"] == 3
    assert row.locator["label"] == "pp. i-2"  # printed labels differ from physical pages 1-3


def test_T29_incomplete_records_never_live(env):
    srcs = {p.source_id for p in env.snapshot.rows}
    assert "norights" not in srcs and "pending" not in srcs
    q = env.snapshot.conn.execute("SELECT source_id, reason FROM quarantine").fetchall()
    assert ("norights", "rights_missing") in q and ("pending", "status_under_review") in q
    data = json.loads(env.register_path.read_text())
    del data["sources"][0]["source_version"]
    with pytest.raises(Exception):
        SourceRegister([__import__("retriever.register", fromlist=["SourceRecord"]).SourceRecord.model_validate(
            data["sources"][0])])


def test_T30_long_sentence_and_scanned_page_quarantined(make_env, tmp_path):
    from tests.retriever.helpers import _record
    import hashlib
    env = make_env(build=False)
    long = ("Epithelial cells " + "and more cells " * 200).strip() + "."
    md = f"# Long\n\nShort intro sentence here. {long} Final short sentence.\n".encode()
    (env.imports / "long.md").write_bytes(md)
    data = json.loads(env.register_path.read_text())
    data["sources"].append(_record("longsrc", "long.md", "markdown", "lib1", hashlib.sha256(md).hexdigest()))
    env.register_path.write_text(json.dumps(data))
    env.register = SourceRegister.load(env.register_path)
    env.build("fixture-index-002", KB)
    q = env.snapshot.conn.execute("SELECT source_id, reason FROM quarantine").fetchall()
    assert ("longsrc", "long_sentence") in q and ("textbook", "blank_or_scanned") in q
    texts = [p.text for p in env.snapshot.rows if p.source_id == "longsrc"]
    assert texts and all("and more cells" not in t for t in texts)
    assert not any("Short intro sentence here. Final" in t for t in texts)  # never joined across the gap


def test_T31_units_negations_tables_preserved(env):
    texts = [p.text for p in env.snapshot.rows if p.source_id == "notes"]
    assert any("< 1 µm" in t for t in texts) and any("is not vascular" in t for t in texts)
    table = env.snapshot.by_id[find(env.snapshot, "| Feature |")]
    assert table.locator["kind"] == "table" and "| Layers | 1 | 1 |" in table.text


def test_T31_failed_extraction_blocks_activation(make_env):
    env = make_env()
    data = json.loads(env.register_path.read_text())
    data["sources"][0]["extraction_checks"].append("this reviewed phrase is missing")
    env.register_path.write_text(json.dumps(data))
    with pytest.raises(BuildError, match="extraction_check_failed"):
        env.build("fixture-index-003", "fixture-kb-003", register=SourceRegister.load(env.register_path))
    assert env.service.snapshots.active_version(COURSE) == "fixture-index-002"


def test_T32_pair_length_checked(env):
    env.models.reranker.tokenizer.count_pair = lambda q, p: 513
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.INPUT_TOO_LONG)


# --------------------------------------------------------------------------- T33-T41 snapshots and revocation


def test_T33_matrix_mapping_mismatch_not_ready(env):
    path = env.build("fixture-index-y", KB, activate=False)
    m = np.load(path / "vectors.npy")
    np.save(path / "vectors.npy", m[:-1], allow_pickle=False)
    _reseal(path)
    with pytest.raises(SnapshotError, match="vector_row_mapping_mismatch"):
        _open(env, path)
    pointer = env.snapshots_root / COURSE / "ACTIVE.json"
    pointer.write_text(json.dumps({"index_version": "fixture-index-y", "kb_version": KB}))
    fresh = build_service(env.profile, env.registry, env.models.encoder, env.models.reranker, courses=(COURSE,),
                          revocations=env.revocations, snapshots_root=str(env.snapshots_root), now=lambda: NOW)
    r = fresh.readiness()
    assert not r.ready and any("vector_row_mapping_mismatch" in x for x in r.reasons)
    err(fresh.retrieve(env.request(Q), env.ctx()), ErrorCode.INDEX_UNAVAILABLE)


def test_T34_representation_change_requires_rebuild(env):
    path = env.snapshots_root / COURSE / "fixture-index-002"
    other = env.profile.model_copy(update={"encoder": env.profile.encoder.model_copy(update={"passage_prefix": "doc: "})})
    from retriever.chunk import chunking_fingerprint
    from retriever.dense import preprocessing_fingerprint
    with pytest.raises(SnapshotError, match="rebuild_required"):
        Snapshot.open(path, profile=other, preprocessing_fp=preprocessing_fingerprint(env.models.encoder, other),
                      chunking_fp=chunking_fingerprint(other), operating_mode="fixture")
    new_enc = FixtureEncoder(384, "query: ")
    new_enc.preprocessing = new_enc.preprocessing.__class__("fixture-hash-encoder", "v2", "fixture", "mean",
                                                            "query: ", "l2", 384)
    with pytest.raises(SnapshotError, match="encoder_changed_rebuild_required"):
        Snapshot.open(path, profile=env.profile, preprocessing_fp=preprocessing_fingerprint(new_enc, env.profile),
                      chunking_fp=chunking_fingerprint(env.profile), operating_mode="fixture")


def test_T35_stable_ids_across_builds(env):
    first = {p.passage_id for p in env.snapshot.rows}
    env.build("fixture-index-again", KB, activate=False)
    second = {p.passage_id for p in _open(env, env.snapshots_root / COURSE / "fixture-index-again").rows}
    assert first == second


def test_T36_changed_source_gets_new_ids(env):
    data = json.loads(env.register_path.read_text())
    for s in data["sources"]:
        if s["source_id"] == "slides":
            s["source_version"] = "2"
    env.register_path.write_text(json.dumps(data))
    env.build("fixture-index-v3", "fixture-kb-003", register=SourceRegister.load(env.register_path))
    old = {p.passage_id for p in env.service.snapshots._loaded[(COURSE, "fixture-index-002")].rows
           if p.source_id == "slides"}
    new = {p.passage_id for p in env.snapshot.rows if p.source_id == "slides"}
    assert old and new and not (old & new)
    assert env.snapshot.kb_version == "fixture-kb-003"


def test_T37_failed_build_leaves_active_and_no_staging(make_env):
    env = make_env()
    with pytest.raises(BuildError):
        env.build("fixture-index-002", KB)  # index_version already exists
    assert env.service.snapshots.active_version(COURSE) == "fixture-index-002"
    assert not [p for p in (env.snapshots_root / COURSE).iterdir() if p.name.startswith(".staging")]


def test_T38_request_keeps_its_pinned_snapshot(make_env):
    gate = threading.Event()
    release = threading.Event()

    def score(q, d):
        gate.set()
        release.wait(5)
        return 5.0

    env = make_env(score_fn=score)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", env.service.retrieve(env.request(Q), env.ctx())))
    t.start()
    assert gate.wait(5)
    env.build("fixture-index-new", "fixture-kb-new")  # activates while the request is running
    release.set()
    t.join(10)
    r = out["r"]
    assert r.status == Status.ok and r.kb_version == KB and r.index_version == "fixture-index-002"
    assert env.service.snapshots.active_version(COURSE) == "fixture-index-new"


def test_T39_stale_kb_version(env):
    err(env.service.retrieve(env.request(Q, kb_version="fixture-kb-001"), env.ctx()), ErrorCode.VERSION_MISMATCH)


def test_T40_revocation_during_request(make_env):
    holder = {}

    def score(q, d):
        if not holder.get("done"):
            holder["done"] = True
            holder["env"].revocations.revoke("slides")
        return 5.0

    env = make_env(score_fn=score)
    holder["env"] = env
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.SOURCE_STATE_CHANGED)


def test_T41_rollback_keeps_revocation(env):
    env.build("fixture-index-003", "fixture-kb-003")
    env.revocations.revoke("slides")
    env.service.snapshots.activate(COURSE, "fixture-index-002")  # rollback
    r = env.service.retrieve(env.request("simple squamous single layer flat cells alveoli"), env.ctx())
    assert r.status == Status.ok and all(p.source_id != "slides" for p in r.passages)
    reloaded = type(env.revocations)(env.tmp / "revocations.json")
    assert reloaded.is_revoked("slides", "1") and reloaded.epoch() == env.revocations.epoch()


def test_T42_review_expiry_excludes_at_runtime(env):
    env.service.now = lambda: NOW + timedelta(days=400)  # review_due_at passed: treated as under_review
    r = env.service.retrieve(env.request(Q), env.ctx())
    assert (r.status, r.reason) == (Status.no_evidence, Reason.NO_ELIGIBLE_PASSAGES)


# --------------------------------------------------------------------------- T43-T44 readiness


def test_T43_missing_or_unevaluated_threshold(env):
    prod = env.profile.model_copy(update={"operating_mode": "production",
                                          "threshold": env.profile.threshold.model_copy(update={"status": "provisional"})})
    assert "threshold_not_evaluated" in prod.readiness_problems()
    none = env.profile.model_copy(update={"threshold": env.profile.threshold.model_copy(update={"t_rerank": None})})
    env.service.profile = none
    assert "threshold_missing" in env.service.readiness().reasons
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.MODEL_UNAVAILABLE)
    with pytest.raises(Exception):
        select([_cand("a", 1.0)], None, 5, 0.8)


def test_T44_fixture_artifacts_refused_outside_fixture_mode(env):
    dev = env.profile.model_copy(update={"operating_mode": "real_model_development",
                                         "threshold": env.profile.threshold.model_copy(update={"status": "provisional"})})
    svc = build_service(dev, env.registry, env.models.encoder, env.models.reranker, courses=(COURSE,),
                        revocations=env.revocations, snapshots_root=str(env.snapshots_root), now=lambda: NOW)
    reasons = svc.readiness().reasons
    assert not svc.readiness().ready
    assert {"fixture_encoder_not_allowed", "fixture_reranker_not_allowed"} <= set(reasons)
    assert any("fixture_snapshot_not_allowed" in r for r in reasons)


# --------------------------------------------------------------------------- T45-T51 items, coverage, conflicts


def _items_env(make_env, sets):
    probe = make_env()
    snap = probe.snapshot
    ids = {k: find(snap, v) for k, v in {"quiz": "Practice item E1", "slide": "single layer of flat cells",
                                         "notes": "Simple squamous epithelium has one layer"}.items()}
    items = [{"item_id": "item-e1", "item_version": "v1", "approved": True,
              "evidence_sets": [{"passage_ids": [ids[k] for k in s["keys"]], "coverage": s.get("coverage", "full")}
                                for s in sets]}]
    return make_env(items=items), ids


def _item_req(**kw):
    body = {"request_id": "00000000-0000-4000-8000-000000000003", "item_id": "item-e1", "item_version": "v1",
            "course_id": COURSE, "kb_version": KB}
    body.update(kw)
    return body


def test_T45_item_fetch_no_semantic_search(make_env):
    env, ids = _items_env(make_env, [{"keys": ["quiz", "slide"]}])
    r = env.service.fetch_item_evidence(_item_req(), env.ctx())
    assert (r.status, r.reason, r.coverage) == (Status.ok, Reason.APPROVED_ITEM_EVIDENCE, "full")
    assert [p.passage_id for p in r.passages] == [ids["quiz"], ids["slide"]]
    assert all(p.relevance_score is None and p.score_type is None for p in r.passages)
    assert r.libraries_searched == ()
    nothing_ran(env)


def test_T46_item_errors(make_env):
    env, ids = _items_env(make_env, [{"keys": ["quiz", "slide"]}])
    err(env.service.fetch_item_evidence(_item_req(item_id="item-missing"), env.ctx()),
        ErrorCode.ITEM_EVIDENCE_UNAVAILABLE)
    err(env.service.fetch_item_evidence(_item_req(item_version="v0"), env.ctx()), ErrorCode.VERSION_MISMATCH)
    err(env.service.fetch_item_evidence(_item_req(kb_version="fixture-kb-001"), env.ctx()), ErrorCode.VERSION_MISMATCH)
    err(env.service.fetch_item_evidence(_item_req(), env.ctx(authorized_libraries=frozenset({"lib1", "lib2", "lib6"}))),
        ErrorCode.ITEM_EVIDENCE_UNAVAILABLE)
    env.revocations.revoke("quizbank")
    err(env.service.fetch_item_evidence(_item_req(), env.ctx()), ErrorCode.ITEM_EVIDENCE_UNAVAILABLE)


def test_T48_item_sets_over_five_rejected():
    with pytest.raises(Exception, match="five-passage"):
        ItemMapping.model_validate({"item_id": "i", "item_version": "1", "approved": True,
                                    "evidence_sets": [{"passage_ids": [f"p{i}" for i in range(6)]}]})


def test_T49_free_text_coverage_unknown(env):
    r = env.service.retrieve(env.request(Q), env.ctx())
    assert r.status == Status.ok and r.coverage == "unknown"


def test_T50_alternative_sets_and_partial(make_env):
    env, ids = _items_env(make_env, [{"keys": ["quiz", "slide"]}, {"keys": ["quiz", "notes"], "coverage": "partial"}])
    env.revocations.revoke("slides")  # first set unavailable -> second complete set is used
    r = env.service.fetch_item_evidence(_item_req(), env.ctx())
    assert r.status == Status.ok and [p.passage_id for p in r.passages] == [ids["quiz"], ids["notes"]]
    assert r.coverage == "partial"


def _conflict(env, approved=True):
    a = find(env.snapshot, "2 to 4 mm")
    b = find(env.snapshot, "1 to 6 mm")
    return [{"conflict_id": "cf-1", "topic_id": "cartilage", "faculty_review_status": "approved",
             "description": "Sources report different adult articular cartilage thickness ranges.",
             "description_approved": approved, "policy_reference": "proposed-conflict-policy-0.2",
             "sides": [{"source_id": "atlas-a", "source_version": "1", "passage_ids": [a]},
                       {"source_id": "atlas-b", "source_version": "1", "passage_ids": [b]}]}]


def test_T51_conflict_presented_or_fails_conservatively(make_env):
    probe = make_env()
    conflicts = _conflict(probe)
    env = make_env(conflicts=conflicts)
    r = env.service.retrieve(env.request("articular cartilage thickness in adults"), env.ctx())
    assert r.status == Status.ok and r.conflicts and len(r.conflicts[0].passage_ids) == 2
    # one side below threshold -> cannot present both sides
    env2 = make_env(conflicts=conflicts, score_fn=lambda q, d: -9.0 if "1 to 6" in d else 5.0)
    err(env2.service.retrieve(env2.request("articular cartilage thickness in adults"), env2.ctx()),
        ErrorCode.CONFLICT_EVIDENCE_INCOMPLETE)
    env3 = make_env(conflicts=_conflict(probe, approved=False))
    err(env3.service.retrieve(env3.request("articular cartilage thickness in adults"), env3.ctx()),
        ErrorCode.CONFLICT_EVIDENCE_INCOMPLETE)


# --------------------------------------------------------------------------- T54-T59 access, safety, runtime


def test_T54_evidence_lookup_is_access_checked(env):
    from retriever.schema import RetrievalError
    pid = find(env.snapshot, "single layer of flat cells")
    assert env.service.get_evidence(pid, KB, env.ctx()).passage_id == pid
    for args, code in (((pid, "fixture-kb-001"), ErrorCode.VERSION_MISMATCH),
                       (("psg-guessed", KB), ErrorCode.FORBIDDEN)):
        with pytest.raises(RetrievalError) as exc:
            env.service.get_evidence(*args, env.ctx())
        assert exc.value.code == code
    env.revocations.revoke("slides")
    with pytest.raises(RetrievalError) as exc:
        env.service.get_evidence(pid, KB, env.ctx())
    assert exc.value.code == ErrorCode.FORBIDDEN


@pytest.mark.parametrize("q", ['body: "epithelium" OR NEAR(cell layer)', "epith* AND NOT bone", "x'); DROP TABLE passages;--",
                               '"""', "^cartilage {heading}: col"])
def test_T55_query_operators_are_literals(env, q):
    m = build_match_query(q)
    assert m is None or all(t.startswith('"') and t.endswith('"') for t in m.split(" OR "))
    r = env.service.retrieve(env.request(q), env.ctx())
    assert r.status != Status.error
    assert env.snapshot.conn.execute("SELECT count(*) FROM passages").fetchone()[0] == len(env.snapshot.rows)


def test_T56_import_path_rules(tmp_path):
    root = tmp_path / "imports"
    root.mkdir()
    (root / "ok.md").write_text("# A\n\nText.")
    (tmp_path / "secret.md").write_text("outside")
    try:
        (root / "link.md").symlink_to(tmp_path / "secret.md")
        has_link = True
    except OSError:  # Windows without symlink privilege: the traversal cases below still run
        has_link = False
    (root / "big.md").write_bytes(b"x" * 2048)
    (root / "notes.exe").write_text("x")
    for ref, fmt, reason in (("../secret.md", "markdown", "path_escapes_import_root"),
                             ("/etc/passwd", "markdown", "not_a_relative_local_path"),
                             ("https://example.org/a.md", "markdown", "not_a_relative_local_path"),
                             ("notes.exe", "markdown", "unsupported_format"),
                             ("big.md", "markdown", "file_too_large")):
        with pytest.raises(ImportRejected, match=reason):
            resolve_import(root, ref, fmt, 1024)
    if has_link:
        with pytest.raises(ImportRejected, match="path_escapes_import_root"):
            resolve_import(root, "link.md", "markdown", 1024)
    assert resolve_import(root, "ok.md", "markdown", 1024).name == "ok.md"


def test_T57_capacity_and_deadline():
    from retriever.schema import RetrievalError
    w = BoundedWorker(capacity=1, stuck_seconds=30, max_restarts=1)
    block = threading.Event()
    threading.Thread(target=lambda: _swallow(w.run, lambda: block.wait(5), 5)).start()
    time.sleep(0.3)  # first job running
    threading.Thread(target=lambda: _swallow(w.run, lambda: None, 5)).start()  # queued
    time.sleep(0.1)
    with pytest.raises(RetrievalError) as exc:
        w.run(lambda: None, 1)
    assert exc.value.code == ErrorCode.CAPACITY_EXCEEDED
    block.set()
    w2 = BoundedWorker(capacity=4, stuck_seconds=30, max_restarts=1)
    with pytest.raises(RetrievalError) as exc:
        w2.run(lambda: time.sleep(0.5), 0.1)
    assert exc.value.code == ErrorCode.DEADLINE_EXCEEDED


def _swallow(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


def test_T58_late_result_suppressed_and_bounded_recovery():
    from retriever.schema import RetrievalError
    w = BoundedWorker(capacity=4, stuck_seconds=0.2, max_restarts=1)
    effects = []
    with pytest.raises(RetrievalError):
        w.run(lambda: (time.sleep(0.6), effects.append("late"))[1], 0.1)
    time.sleep(0.3)
    assert w.run(lambda: "fresh", 2) == "fresh" and w.restarts == 1  # stuck worker replaced
    time.sleep(0.5)
    assert effects == ["late"]  # the late job finished but its result was never returned to anyone
    hang = threading.Event()
    with pytest.raises(RetrievalError):
        w.run(lambda: hang.wait(5), 0.1)
    time.sleep(0.3)
    with pytest.raises(RetrievalError):
        w.run(lambda: None, 1)
    assert not w.healthy  # bounded recovery exhausted -> readiness false
    hang.set()


def test_T59_no_text_leak_and_audit_failure(env):
    secret = "Aigerim Bekova 990101300123"
    env.models.encoder.encode = lambda texts: (_ for _ in ()).throw(RuntimeError(f"failed on {texts} {secret}"))
    r = env.service.retrieve(env.request(Q + " " + secret), env.ctx())
    err(r, ErrorCode.SEARCH_FAILED)
    assert secret not in r.model_dump_json()
    assert all(secret not in rec.model_dump_json() and Q not in rec.model_dump_json()
               for rec in env.service.audit.records)


def test_T59_audit_failure_blocks_evidence(make_env):
    env = make_env()

    class Broken:
        def write(self, record):
            raise OSError("disk full")

    env.service.audit = Broken()
    err(env.service.retrieve(env.request(Q), env.ctx()), ErrorCode.AUDIT_UNAVAILABLE)


# --------------------------------------------------------------------------- property-style invariants


def test_property_invariants(make_env):
    import random
    rng = random.Random(7)
    vocab = "simple squamous cuboidal epithelium cartilage bone kidney alveoli layer cells matrix collagen".split()
    env = make_env()
    for _ in range(30):
        q = " ".join(rng.sample(vocab, rng.randint(1, 5)))
        r1 = env.service.retrieve(env.request(q), env.ctx())
        r2 = env.service.retrieve(env.request(q), env.ctx())
        assert r1.status != Status.error
        ids = [p.passage_id for p in r1.passages]
        assert ids == [p.passage_id for p in r2.passages] and len(ids) == len(set(ids)) <= 5
        assert set(r1.libraries_searched) <= set(ACTIVE)
        for p in r1.passages:
            assert p.library_id in ACTIVE and np.isfinite(p.relevance_score) and p.review_status == "live"
            assert p.relevance_score >= env.profile.threshold.t_rerank
            assert env.snapshot.by_id[p.passage_id].text == p.text
        assert r1.kb_version == KB and r1.revocation_epoch == env.revocations.epoch()


def test_T44_artifact_hash_mismatch_refused(env):
    path = env.build("fixture-index-z", KB, activate=False)
    with open(path / "vectors.npy", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(SnapshotError, match="artifact_checksum"):
        _open(env, path)
    with pytest.raises(SnapshotError):
        env.service.snapshots.activate(COURSE, "fixture-index-z")
    assert env.service.snapshots.active_version(COURSE) == "fixture-index-002"


def test_unpinned_reranker_not_ready(env):
    dev = env.profile.model_copy(update={"operating_mode": "real_model_development",
                                         "reranker": env.profile.reranker.model_copy(update={"revision": None}),
                                         "threshold": env.profile.threshold.model_copy(update={"status": "provisional"})})
    assert "reranker_revision_unpinned" in dev.readiness_problems()
