"""Immutable snapshot loading, version pinning and atomic activation (Retriever spec v0.2, section 9).

A loaded Snapshot is read-only and self-consistent; a request pins one Snapshot object for its whole
life, so a concurrent activation can never mix a new lexical index with old vectors.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .build import SNAPSHOT_SCHEMA, file_sha256
from .chunk import passage_identity, sha256_text
from .config import RetrieverProfile
from .fetch import ItemMapping
from .register import SourceRecord
from .select import ConflictRecord

ACTIVE_POINTER = "ACTIVE.json"


class SnapshotError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PassageRow:
    rowid: int
    vector_row: int
    passage_id: str
    digest: str
    source_id: str
    source_version: str
    library_id: str
    title: str
    edition: Optional[str]
    publication_year: Optional[int]
    locator: dict
    section_path: tuple[str, ...]
    heading: str
    text: str
    text_sha256: str
    spans: tuple[tuple[str, int, int], ...]
    rights_reference: str
    kind: str

    @property
    def source_key(self) -> tuple[str, str]:
        return (self.source_id, self.source_version)


class Snapshot:
    def __init__(self, path: Path, manifest: dict, conn: sqlite3.Connection, matrix: np.ndarray,
                 passages: list[PassageRow], sources: dict, items: dict, conflicts: list):
        self.path = path
        self.manifest = manifest
        self.conn = conn
        self.matrix = matrix
        self.rows = passages                      # ordered by vector_row
        self.by_id = {p.passage_id: p for p in passages}
        self.ids_by_row = [p.passage_id for p in passages]
        self.sources: dict[tuple[str, str], SourceRecord] = sources
        self.items: dict[str, ItemMapping] = items
        self.conflicts: list[ConflictRecord] = conflicts
        self.lock = threading.Lock()  # serialize use of the read-only sqlite connection

    kb_version = property(lambda self: self.manifest["kb_version"])
    index_version = property(lambda self: self.manifest["index_version"])
    tenant_id = property(lambda self: self.manifest["tenant_id"])
    course_id = property(lambda self: self.manifest["course_id"])
    is_fixture = property(lambda self: self.manifest.get("operating_mode") == "fixture")

    @classmethod
    def open(cls, path: str | Path, *, profile: RetrieverProfile, preprocessing_fp: str, chunking_fp: str,
             operating_mode: str) -> "Snapshot":
        path = Path(path)
        mpath = path / "manifest.json"
        if not mpath.is_file():
            raise SnapshotError("manifest_missing")
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SnapshotError("manifest_unreadable") from None
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
            raise SnapshotError("manifest_schema")
        if operating_mode != "fixture" and manifest.get("operating_mode") == "fixture":
            raise SnapshotError("fixture_snapshot_not_allowed")
        if operating_mode != "fixture" and not manifest.get("review", {}).get("approved"):
            raise SnapshotError("review_sample_not_approved")
        if operating_mode == "production" and not manifest.get("evaluation_ref"):
            raise SnapshotError("evaluation_record_missing")
        prof = manifest.get("profile", {})
        if prof.get("representation_fingerprint") != profile.representation_fingerprint():
            raise SnapshotError("representation_changed_rebuild_required")
        if manifest.get("encoder", {}).get("preprocessing_fingerprint") != preprocessing_fp:
            raise SnapshotError("encoder_changed_rebuild_required")
        if prof.get("chunking_fingerprint") != chunking_fp:
            raise SnapshotError("chunking_changed_rebuild_required")
        for name in ("snapshot.sqlite", "vectors.npy"):
            f = path / name
            if not f.is_file() or file_sha256(f) != manifest.get("artifacts", {}).get(name):
                raise SnapshotError(f"artifact_checksum:{name}")
        try:
            with open(path / "vectors.npy", "rb") as fh:
                matrix = np.load(fh, allow_pickle=False)
        except Exception:
            raise SnapshotError("vectors_unreadable") from None
        dim = profile.encoder.dimension
        if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != dim or manifest.get("dimension") != dim:
            raise SnapshotError("vector_dimension_mismatch")
        if not np.all(np.isfinite(matrix)) or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
            raise SnapshotError("vectors_not_normalized")
        uri = f"file:{(path / 'snapshot.sqlite').as_posix()}?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            raw = conn.execute(
                "SELECT rowid, vector_row, passage_id, digest, source_id, source_version, library_id, title, edition, "
                "publication_year, locator_json, section_path_json, heading, text, text_sha256, spans_json, "
                "rights_reference, kind FROM passages ORDER BY vector_row").fetchall()
            n_fts = conn.execute("SELECT count(*) FROM passages_fts").fetchone()[0]
            sources = {(s, v): SourceRecord.model_validate_json(j)
                       for s, v, j in conn.execute("SELECT source_id, source_version, record_json FROM sources")}
            items = {i: ItemMapping.model_validate_json(j) for i, _, j in conn.execute("SELECT * FROM items")}
            conflicts = [ConflictRecord.model_validate_json(j) for _, j in conn.execute("SELECT * FROM conflicts")]
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.Error:
            raise SnapshotError("sqlite_unreadable") from None
        if meta.get("kb_version") != manifest["kb_version"] or meta.get("index_version") != manifest["index_version"]:
            raise SnapshotError("meta_manifest_mismatch")
        rows: list[PassageRow] = []
        for r in raw:
            rows.append(PassageRow(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], json.loads(r[10]),
                                   tuple(json.loads(r[11])), r[12], r[13], r[14],
                                   tuple(tuple(s) for s in json.loads(r[15])), r[16], r[17]))
        # validation: one vector row per passage, one passage per vector row, FTS complete
        if len(rows) != matrix.shape[0] or [p.vector_row for p in rows] != list(range(matrix.shape[0])):
            raise SnapshotError("vector_row_mapping_mismatch")
        if n_fts != len(rows) or len(rows) != manifest.get("passage_count"):
            raise SnapshotError("lexical_mapping_mismatch")
        for p in rows:
            if sha256_text(p.text) != p.text_sha256:
                raise SnapshotError("passage_text_hash_mismatch")
            pid, digest = passage_identity(p.source_id, p.source_version, p.locator, p.spans, p.text_sha256,
                                           chunking_fp)
            if pid != p.passage_id or digest != p.digest:
                raise SnapshotError("passage_identity_mismatch")
            if p.source_key not in sources:
                raise SnapshotError("passage_without_source")
        return cls(path, manifest, conn, matrix, rows, sources, items, conflicts)


class SnapshotManager:
    """Per-course active pointer with atomic activation under one lock."""

    def __init__(self, root: str | Path, profile: RetrieverProfile, preprocessing_fp: str, chunking_fp: str):
        self.root = Path(root)
        self.profile = profile
        self.preprocessing_fp = preprocessing_fp
        self.chunking_fp = chunking_fp
        self._lock = threading.Lock()
        self._loaded: dict[tuple[str, str], Snapshot] = {}
        self._active: dict[str, str] = {}

    def _open(self, course_id: str, index_version: str) -> Snapshot:
        key = (course_id, index_version)
        if key not in self._loaded:
            self._loaded[key] = Snapshot.open(self.root / course_id / index_version, profile=self.profile,
                                              preprocessing_fp=self.preprocessing_fp, chunking_fp=self.chunking_fp,
                                              operating_mode=self.profile.operating_mode)
        return self._loaded[key]

    def activate(self, course_id: str, index_version: str) -> Snapshot:
        """Validate the complete snapshot, then atomically repoint. A failure leaves the old pointer."""
        with self._lock:
            snap = self._open(course_id, index_version)
            course_dir = self.root / course_id
            fd, tmp = tempfile.mkstemp(dir=course_dir, prefix=".active-")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"index_version": index_version, "kb_version": snap.kb_version}, fh)
            os.replace(tmp, course_dir / ACTIVE_POINTER)
            self._active[course_id] = index_version
            return snap

    def pin(self, course_id: str) -> Snapshot:
        """Return the active snapshot object; the caller keeps this handle for the whole request."""
        with self._lock:
            version = self._active.get(course_id)
            if version is None:
                pointer = self.root / course_id / ACTIVE_POINTER
                if not pointer.is_file():
                    raise SnapshotError("no_active_snapshot")
                try:
                    version = json.loads(pointer.read_text(encoding="utf-8"))["index_version"]
                except (OSError, ValueError, KeyError):
                    raise SnapshotError("active_pointer_unreadable") from None
                self._active[course_id] = version
            return self._open(course_id, version)

    def active_version(self, course_id: str) -> Optional[str]:
        with self._lock:
            return self._active.get(course_id)
