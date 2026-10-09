"""SYNTHETIC retriever fixtures: fake course text, fake approvals, deterministic models.

Nothing here is reviewed course material or a real approval. Used only by tests and smoke runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from gate_classifier.config import load_registry
from gate_classifier.encoder import FixtureEncoder, FixtureTokenizer
from retriever.build import build_snapshot
from retriever.config import RetrieverProfile, load_profile
from retriever.register import RevocationRegistry, SourceRegister
from retriever.rerank import FixtureReranker, overlap_score
from retriever.schema import RetrievalContext
from retriever.service import build_service

ROOT = Path(__file__).resolve().parents[2]
COURSE = "histology-dev"
TENANT = "dev-tenant"
KB = "fixture-kb-002"
ACTIVE = ("lib1", "lib2", "lib5", "lib6")
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)

NOTES_MD = """# Epithelium

## Simple squamous

Simple squamous epithelium has one layer of flattened cells. It lines the alveoli of the lung and the endothelium of blood vessels. Its thin barrier allows rapid diffusion of gases.

The basement membrane under it is about 50 nm thick and is not vascular. Nutrients reach the cells by diffusion from underlying connective tissue.

## Simple cuboidal

Simple cuboidal epithelium has one layer of cube-shaped cells with round central nuclei. It lines kidney tubules and small gland ducts.

Caption: Kidney tubule cross-section showing simple cuboidal cells around a central lumen.

| Feature | Squamous | Cuboidal |
| Layers | 1 | 1 |
| Cell height | < 1 µm | ~10 µm |

# Connective tissue

## Hyaline cartilage

Hyaline cartilage contains chondrocytes in lacunae surrounded by a matrix rich in type II collagen. It has no blood vessels and is nourished by diffusion from the perichondrium.
"""

SLIDES_JSON = {"blocks": [
    {"block_id": "s4", "section_path": ["Epithelium", "Simple squamous"], "kind": "paragraph", "slide": 4,
     "text": "Simple squamous epithelium is a single layer of flat cells found in alveoli."},
    {"block_id": "s5", "section_path": ["Epithelium", "Stratified squamous"], "kind": "paragraph", "slide": 5,
     "text": "Stratified squamous epithelium has many layers and protects against abrasion in the oesophagus."},
]}

QUIZ_JSON = {"blocks": [
    {"block_id": "q1", "section_path": ["Practice", "Epithelium"], "kind": "paragraph",
     "text": "Practice item E1: Which epithelium lines the alveoli? A) stratified squamous B) simple squamous. Key: B."},
]}

CONFLICT_A = {"blocks": [{"block_id": "c1", "section_path": ["Cartilage", "Thickness"], "kind": "paragraph",
                          "text": "Articular cartilage thickness in adults is typically 2 to 4 mm."}]}
CONFLICT_B = {"blocks": [{"block_id": "c1", "section_path": ["Cartilage", "Thickness"], "kind": "paragraph",
                          "text": "Articular cartilage thickness in adults is typically 1 to 6 mm."}]}


def _record(source_id, file_ref, fmt, library, sha, *, status="live", rights=True, approval=True,
            review_due=NOW + timedelta(days=365), version="1", checks=(), labels=False, synthetic=True, title=None):
    return {
        "source_id": source_id, "source_version": version, "title": title or f"Synthetic {source_id}",
        "owner": "synthetic-fixture-owner", "edition": None, "publication_year": None, "source_tier": 1,
        "library_id": library, "course_ids": [COURSE], "tenant_id": TENANT, "topic_ids": ["epithelium"],
        "language": "en", "file_ref": file_ref, "file_sha256": sha, "format": fmt,
        "acquisition_record": "synthetic-fixture",
        "rights": {"rights_reference": f"fixture-rights-{source_id}", "allows_local_indexing": True,
                   "allows_excerpt_display": True, "reviewer": "fixture", "reviewed_at": NOW.isoformat(),
                   "supporting_document": "fixture-doc"} if rights else None,
        "content_approval": {"approver": "fixture", "approved_at": NOW.isoformat()} if approval else None,
        "status": status, "review_due_at": review_due.isoformat(),
        "effective_from": (NOW - timedelta(days=30)).isoformat(),
        "parser": {"parser_id": {"markdown": "markdown", "json": "json", "pdf": "pdf_text"}[fmt],
                   "parser_version": "parsers-0.2", "normalization_version": "nfc-ws-0.2",
                   "parsed_output_approved": True, "verified_page_labels": labels},
        "extraction_checks": list(checks), "synthetic_fixture": synthetic,
    }


def make_pdf(pages: list[str], roman_first: int = 0) -> bytes:
    """Minimal text PDF. ``roman_first`` leading pages get roman page labels (printed != physical)."""
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    labels = b""
    if roman_first:
        labels = b" /PageLabels << /Nums [0 << /S /r >> %d << /S /D /St 1 >>] >>" % roman_first
    objs.append(b"<< /Type /Catalog /Pages 2 0 R" + labels + b" >>")
    objs.append(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % i for i in page_ids) + b"] /Count %d >>" % n)
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines = text.split("\n")
        ops = [b"BT /F1 11 Tf 50 750 Td 14 TL"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")
            ops.append(b"(" + safe + b") Tj T*")
        ops.append(b"ET")
        stream = b"\n".join(ops)
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>"
                    b" /Contents %d 0 R >>" % (page_ids[i] + 1))
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


PDF_PAGES = [
    "Preface\n\nThis synthetic textbook is a test fixture and is not course material.",
    "Chapter 1 Epithelium\n\nTransitional epithelium lines the urinary bladder and can stretch.\nIt is not keratinised.",
    "Chapter 2 Bone\n\nCompact bone is organised into osteons around central Haversian canals.",
    "",
]


def write_corpus(root: Path, extra_sources: Optional[list] = None) -> tuple[Path, Path]:
    imports = root / "sources"
    imports.mkdir(parents=True, exist_ok=True)
    files = {
        "notes.md": NOTES_MD.encode("utf-8"),
        "slides.json": json.dumps(SLIDES_JSON).encode(),
        "quiz.json": json.dumps(QUIZ_JSON).encode(),
        "conflict_a.json": json.dumps(CONFLICT_A).encode(),
        "conflict_b.json": json.dumps(CONFLICT_B).encode(),
        "textbook.pdf": make_pdf(PDF_PAGES, roman_first=1),
    }
    for name, data in files.items():
        (imports / name).write_bytes(data)
    sha = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}
    sources = [
        _record("notes", "notes.md", "markdown", "lib1", sha["notes.md"],
                checks=("< 1 µm", "is not vascular", "| Layers | 1 | 1 |")),
        _record("slides", "slides.json", "json", "lib1", sha["slides.json"]),
        _record("quizbank", "quiz.json", "json", "lib5", sha["quiz.json"]),
        _record("atlas-a", "conflict_a.json", "json", "lib6", sha["conflict_a.json"]),
        _record("atlas-b", "conflict_b.json", "json", "lib6", sha["conflict_b.json"]),
        _record("textbook", "textbook.pdf", "pdf", "lib2", sha["textbook.pdf"], labels=True,
                checks=("It is not keratinised.",)),
        _record("pending", "notes.md", "markdown", "lib1", sha["notes.md"], status="under_review"),
        _record("norights", "notes.md", "markdown", "lib1", sha["notes.md"], rights=False),
    ] + list(extra_sources or [])
    reg = root / "register.json"
    reg.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return imports, reg


def fixture_profile(**updates) -> RetrieverProfile:
    p = load_profile(ROOT / "config/retriever_development.yaml")
    return p.model_copy(update=updates) if updates else p


class FixtureModels:
    def __init__(self, score_fn: Callable[[str, str], float] = overlap_score):
        self.tokenizer = FixtureTokenizer()
        self.encoder = FixtureEncoder(384, "query: ")
        self.reranker = FixtureReranker(score_fn, self.tokenizer)


def find(snapshot, needle: str) -> str:
    return next(p.passage_id for p in snapshot.rows if needle in p.text)


class Env:
    """Built and activated synthetic snapshot plus a wired service."""

    def __init__(self, tmp: Path, *, profile=None, score_fn=overlap_score, items=None, conflicts=None,
                 index_version="fixture-index-002", kb=KB, build=True, register_extra=None):
        self.tmp = tmp
        self.profile = profile or fixture_profile()
        self.models = FixtureModels(score_fn)
        self.registry = load_registry(ROOT / "config/library_registry.yaml")
        self.imports, self.register_path = write_corpus(tmp, register_extra)
        self.register = SourceRegister.load(self.register_path)
        self.snapshots_root = tmp / "snapshots"
        self.revocations = RevocationRegistry(tmp / "revocations.json")
        self.service = build_service(self.profile, self.registry, self.models.encoder, self.models.reranker,
                                     courses=(COURSE,), revocations=self.revocations,
                                     snapshots_root=str(self.snapshots_root), now=lambda: NOW)
        if build:
            self.build(index_version, kb, items=items, conflicts=conflicts)

    def build(self, index_version, kb, items=None, conflicts=None, activate=True, register=None):
        path = build_snapshot(
            profile=self.profile, register=register or self.register, import_root=self.imports,
            snapshots_root=self.snapshots_root, tenant_id=TENANT, course_id=COURSE, kb_version=kb,
            index_version=index_version, encoder=self.models.encoder,
            tokenizers=[self.models.tokenizer], reranker_identity={"model_id": "fixture-reranker",
                                                                    "revision": "fixture"},
            items=items or (), conflicts=conflicts or (), synthetic_fixture=True, now=NOW)
        if activate:
            self.service.snapshots.activate(COURSE, index_version)
        return path

    @property
    def snapshot(self):
        return self.service.snapshots.pin(COURSE)

    def ctx(self, **kw) -> RetrievalContext:
        base = dict(service_id="orchestrator", tenant_id=TENANT, course_id=COURSE,
                    authorized_libraries=frozenset(ACTIVE), registry_version="histology-registry-dev-0.1",
                    gate_permitted=True)
        base.update(kw)
        return RetrievalContext(**base)

    def request(self, query: str, **kw) -> dict:
        body = {"request_id": "00000000-0000-4000-8000-000000000002", "query_text": query,
                "allowed_libraries": list(ACTIVE), "preferred_libraries": [], "course_id": COURSE,
                "kb_version": KB, "strategy": "all_active"}
        body.update(kw)
        return body
