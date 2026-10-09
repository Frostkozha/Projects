"""Staged, validated snapshot construction (Retriever spec v0.2, sections 8, 9).

Everything is written into a fresh staging directory, validated, then renamed into place. The live
snapshot is never modified; activation is a separate atomic pointer update (snapshot.py).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .chunk import Chunk, Chunker, sha256_text
from .config import RetrieverProfile
from .dense import encoder_identity, preprocessing_fingerprint
from .fetch import ItemMapping
from .lexical import FTS_TOKENIZER, lexical_search
from .parse import PARSER_VERSION, ImportRejected, ParseError, Quarantined, parse_source
from .register import SourceRecord, SourceRegister
from .schema import Locator
from .select import ConflictRecord

SNAPSHOT_SCHEMA = "retriever-snapshot-0.2"
REVIEW_SAMPLE_MIN = 50


class BuildError(RuntimeError):
    """Build failed; the staging directory is removed and the active snapshot is untouched."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def file_sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SCHEMA_SQL = f"""
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources(source_id TEXT NOT NULL, source_version TEXT NOT NULL, record_json TEXT NOT NULL,
                     PRIMARY KEY(source_id, source_version));
CREATE TABLE passages(
  rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL, digest TEXT NOT NULL,
  source_id TEXT NOT NULL, source_version TEXT NOT NULL, library_id TEXT NOT NULL, title TEXT NOT NULL,
  edition TEXT, publication_year INTEGER, locator_json TEXT NOT NULL, section_path_json TEXT NOT NULL,
  heading TEXT NOT NULL, text TEXT NOT NULL, text_sha256 TEXT NOT NULL, spans_json TEXT NOT NULL,
  rights_reference TEXT NOT NULL, kind TEXT NOT NULL, vector_row INTEGER UNIQUE NOT NULL);
CREATE VIRTUAL TABLE passages_fts USING fts5(heading, body, tokenize='{FTS_TOKENIZER}');
CREATE TABLE items(item_id TEXT PRIMARY KEY, item_version TEXT NOT NULL, mapping_json TEXT NOT NULL);
CREATE TABLE conflicts(conflict_id TEXT PRIMARY KEY, record_json TEXT NOT NULL);
CREATE TABLE quarantine(source_id TEXT, source_version TEXT, ref TEXT, reason TEXT);
"""


def _review_sample(chunks: list[tuple[SourceRecord, Chunk]]) -> list[str]:
    """>=50 passages (or all), stratified by source/format, always including tables and numeric chunks."""
    if len(chunks) <= REVIEW_SAMPLE_MIN:
        return sorted(c.passage_id for _, c in chunks)
    picked: list[str] = []
    special = [c for _, c in chunks if c.kind == "table" or any(ch.isdigit() for ch in c.text)]
    picked.extend(c.passage_id for c in special[: REVIEW_SAMPLE_MIN // 2])
    by_source = defaultdict(list)
    for r, c in chunks:
        by_source[(r.source_id, r.format)].append(c.passage_id)
    keys = sorted(by_source)
    i = 0
    while len(set(picked)) < REVIEW_SAMPLE_MIN:
        bucket = by_source[keys[i % len(keys)]]
        idx = i // len(keys)
        if idx < len(bucket):
            picked.append(bucket[idx])
        i += 1
        if i > len(chunks) * (len(keys) + 1):
            break
    return sorted(set(picked))


def build_snapshot(
    *,
    profile: RetrieverProfile,
    register: SourceRegister,
    import_root: str | Path,
    snapshots_root: str | Path,
    tenant_id: str,
    course_id: str,
    kb_version: str,
    index_version: str,
    encoder,
    tokenizers: Sequence,
    reranker_identity: dict,
    items: Sequence[dict] = (),
    conflicts: Sequence[dict] = (),
    synthetic_fixture: bool = False,
    now: Optional[datetime] = None,
) -> Path:
    """Build, validate and publish an inactive snapshot directory. Returns its final path."""
    now = now or datetime.now(timezone.utc)
    course_root = Path(snapshots_root) / course_id
    course_root.mkdir(parents=True, exist_ok=True)
    final = course_root / index_version
    if final.exists():
        raise BuildError("index_version_exists")
    staging = course_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _build_into(staging, profile=profile, register=register, import_root=import_root, tenant_id=tenant_id,
                    course_id=course_id, kb_version=kb_version, index_version=index_version, encoder=encoder,
                    tokenizers=tokenizers, reranker_identity=reranker_identity, items=items, conflicts=conflicts,
                    synthetic_fixture=synthetic_fixture, now=now)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _build_into(staging: Path, *, profile, register, import_root, tenant_id, course_id, kb_version, index_version,
                encoder, tokenizers, reranker_identity, items, conflicts, synthetic_fixture, now) -> None:
    chunker = Chunker(profile, tokenizers)
    quarantined: list[Quarantined] = []
    indexed: list[tuple[SourceRecord, Chunk]] = []
    used_sources: list[SourceRecord] = []
    for key in sorted(register.records):
        record = register.records[key]
        if course_id not in record.course_ids or record.tenant_id != tenant_id:
            continue
        if record.synthetic_fixture and not synthetic_fixture:
            quarantined.append(Quarantined(record.source_id, record.source_version, "-", "synthetic_fixture_source"))
            continue
        problems = record.build_problems()
        if problems:
            quarantined.extend(Quarantined(record.source_id, record.source_version, "-", p) for p in problems)
            continue
        try:
            parsed = parse_source(record, import_root, profile.limits.max_file_bytes, profile.limits.max_pdf_pages)
        except (ImportRejected, ParseError) as exc:
            quarantined.append(Quarantined(record.source_id, record.source_version, "-", str(exc)))
            continue
        quarantined.extend(parsed.quarantined)
        chunks, q = chunker.chunk(record, parsed.blocks)
        quarantined.extend(q)
        texts = [c.text for c in chunks]
        for check in record.extraction_checks:
            if not any(check in t for t in texts):
                raise BuildError(f"extraction_check_failed:{record.source_id}")
        indexed.extend((record, c) for c in chunks)
        used_sources.append(record)

    if len(indexed) > profile.limits.max_passages_per_snapshot:
        raise BuildError("too_many_passages")
    if not indexed:
        raise BuildError("no_indexable_passages")

    # identity and collisions
    seen: dict[str, Chunk] = {}
    for _, c in indexed:
        prev = seen.get(c.passage_id)
        if prev is not None and (prev.digest != c.digest or prev.text_sha256 != c.text_sha256):
            raise BuildError("passage_id_collision")
        if prev is not None:
            raise BuildError("duplicate_passage")
        seen[c.passage_id] = c
    for _, c in indexed:
        Locator.model_validate(c.locator)  # every citation locator must be valid

    indexed.sort(key=lambda rc: rc[1].passage_id)  # row order is deterministic but never public identity

    # vectors
    reps = [f"{profile.encoder.passage_prefix}{c.heading}\n{c.text}" if c.heading else
            f"{profile.encoder.passage_prefix}{c.text}" for _, c in indexed]
    vecs = []
    for i in range(0, len(reps), 32):
        out = np.asarray(encoder.encode(reps[i:i + 32]), dtype=np.float32)
        vecs.append(out)
    matrix = np.vstack(vecs).astype(np.float32)
    if matrix.shape != (len(indexed), profile.encoder.dimension) or not np.all(np.isfinite(matrix)):
        raise BuildError("vector_shape_invalid")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise BuildError("vectors_not_normalized")
    with open(staging / "vectors.npy", "wb") as fh:
        np.save(fh, matrix, allow_pickle=False)

    # items and conflicts
    passage_ids = set(seen)
    item_rows = []
    for raw in items:
        mapping = ItemMapping.model_validate(raw)  # rejects sets larger than five passages
        if not mapping.approved:
            raise BuildError(f"item_not_approved:{mapping.item_id}")
        for s in mapping.evidence_sets:
            if any(p not in passage_ids for p in s.passage_ids):
                raise BuildError(f"item_evidence_missing:{mapping.item_id}")
        item_rows.append(mapping)
    conflict_rows = [ConflictRecord.model_validate(c) for c in conflicts]
    for c in conflict_rows:
        if any(p not in passage_ids for p in c.passage_ids):
            raise BuildError(f"conflict_passage_missing:{c.conflict_id}")

    # sqlite
    db = staging / "snapshot.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(SCHEMA_SQL)
        for r in used_sources:
            conn.execute("INSERT INTO sources VALUES (?,?,?)", (r.source_id, r.source_version, r.model_dump_json()))
        for row, (r, c) in enumerate(indexed):
            conn.execute(
                "INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row + 1, c.passage_id, c.digest, r.source_id, r.source_version, r.library_id, r.title, r.edition,
                 r.publication_year, json.dumps(c.locator, sort_keys=True), json.dumps(list(c.section_path)),
                 c.heading, c.text, c.text_sha256, json.dumps([list(s) for s in c.spans]),
                 r.rights.rights_reference, c.kind, row))
            conn.execute("INSERT INTO passages_fts(rowid, heading, body) VALUES (?,?,?)", (row + 1, c.heading, c.text))
        for m in item_rows:
            conn.execute("INSERT INTO items VALUES (?,?,?)", (m.item_id, m.item_version, m.model_dump_json()))
        for c in conflict_rows:
            conn.execute("INSERT INTO conflicts VALUES (?,?)", (c.conflict_id, c.model_dump_json()))
        for q in quarantined:
            conn.execute("INSERT INTO quarantine VALUES (?,?,?,?)", (q.source_id, q.source_version, q.ref, q.reason))
        meta = {"kb_version": kb_version, "index_version": index_version, "tenant_id": tenant_id,
                "course_id": course_id}
        for k, v in meta.items():
            conn.execute("INSERT INTO meta VALUES (?,?)", (k, v))
        conn.commit()
        # validation 5: FTS consistency and deterministic self-retrieval fixtures
        conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('integrity-check')")
        n_fts = conn.execute("SELECT count(*) FROM passages_fts").fetchone()[0]
        if n_fts != len(indexed):
            raise BuildError("fts_count_mismatch")
        all_rows = [i + 1 for i in range(len(indexed))]
        for _, c in indexed[:5]:
            hits = lexical_search(conn, c.text, all_rows, profile.search.lexical_k)
            if c.passage_id not in [h[0] for h in hits]:
                raise BuildError("self_retrieval_fixture_failed")
        conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        conn.close()

    # validation 1-2: mapping and hashes
    for _, c in indexed:
        if sha256_text(c.text) != c.text_sha256:
            raise BuildError("text_hash_mismatch")

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "index_version": index_version,
        "kb_version": kb_version,
        "tenant_id": tenant_id,
        "course_id": course_id,
        "created_at": now.isoformat(),
        "operating_mode": "fixture" if synthetic_fixture or getattr(encoder, "is_fixture", False) else profile.operating_mode,
        "synthetic_fixture": bool(synthetic_fixture),
        "profile": {"profile_version": profile.profile_version,
                    "representation_fingerprint": profile.representation_fingerprint(),
                    "chunking_fingerprint": chunker.fingerprint,
                    "query_builder_version": profile.query_builder_version,
                    "retrieval_policy_version": profile.retrieval_policy_version},
        "encoder": {**encoder_identity(encoder, profile),
                    "preprocessing_fingerprint": preprocessing_fingerprint(encoder, profile)},
        "reranker": reranker_identity,
        "parser_version": PARSER_VERSION,
        "dimension": profile.encoder.dimension,
        "passage_count": len(indexed),
        "sources": [{"source_id": r.source_id, "source_version": r.source_version, "file_sha256": r.file_sha256}
                    for r in used_sources],
        "quarantine_count": len(quarantined),
        "review": {"sample_passage_ids": _review_sample(indexed), "approved": bool(synthetic_fixture),
                   "reviewer": "synthetic-fixture" if synthetic_fixture else None},
        "evaluation_ref": profile.threshold.evaluation_ref,
        "artifacts": {"snapshot.sqlite": file_sha256(db), "vectors.npy": file_sha256(staging / "vectors.npy")},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def approve_review(snapshot_dir: str | Path, reviewer: str) -> None:
    """Record that the stratified review sample was checked by a named reviewer (offline operator step)."""
    path = Path(snapshot_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["review"]["approved"] = True
    manifest["review"]["reviewer"] = reviewer
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
