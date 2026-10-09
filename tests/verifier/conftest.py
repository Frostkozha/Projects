"""Verifier fixtures. Every NLI score here is an injected synthetic value (unit tests only), never model output."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Callable, Optional

import pytest

from contracts.models import DraftAnswer, VerifyRequest
from retriever.schema import ConflictOut, EvidencePassage, Reason, RetrievalResult, Status
from verifier.evidence import EvidenceBundle, RegistryUnavailable
from verifier.nli import FixtureNLI
from verifier.service import InMemoryVerifierAudit, Verifier, VerifyContext

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/verifier_development.yaml"
TENANT, COURSE, KB = "dev-tenant", "histology-dev", "fixture-kb-001"

SUPPORT = (0.01, 0.97, 0.02)
NEUTRAL = (0.05, 0.05, 0.90)
CONTRA = (0.90, 0.05, 0.05)
GRAY = (0.05, 0.70, 0.25)

TEXTS = {
    "p1": "Simple squamous epithelium has one layer of flattened cells.",
    "p2": "Simple squamous epithelium lines the alveoli of the lung.",
    "p3": "Stratified squamous epithelium protects the oral cavity.",
    "p4": "Goblet cells secrete mucus in the respiratory epithelium.",
    "p5": "Transitional epithelium lines the urinary bladder.",
}


def passage(pid: str, text: str, *, lib: str = "lib1", review_status: str = "live",
            text_sha256: Optional[str] = None) -> EvidencePassage:
    return EvidencePassage(
        passage_id=pid, source_id=f"src-{pid}", source_version="v1", title=f"Fixture source {pid}", library_id=lib,
        locator={"kind": "slide", "start": 1, "end": 1, "label": "Slide 1", "anchor": None},
        section_path=("Epithelium",), edition=None, publication_year=None, text=text,
        text_sha256=text_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest(), review_status=review_status,
        rights_reference="fixture-rights", relevance_score=5.0, score_type="cross_encoder_logit",
        evidence_uri=f"/v1/evidence/{pid}?kb_version={KB}")


def retrieval(request_id: str, passages, conflicts=()) -> RetrievalResult:
    return RetrievalResult(request_id=request_id, status=Status.ok, reason=Reason.QUALIFYING_PASSAGES,
                           error_code=None, passages=tuple(passages), coverage="unknown",
                           conflicts=tuple(ConflictOut(conflict_id=f"c{i}", topic_id="t1", passage_ids=tuple(c),
                                                       description="fixture conflict", policy_reference="pol-1")
                                           for i, c in enumerate(conflicts)),
                           libraries_searched=("lib1",), kb_version=KB, index_version="fixture-index",
                           profile_version="fixture-profile", revocation_epoch=0)


class FakeRegistry:
    """Trusted live registry fixture: revocable ids, epoch and outage switch."""

    def __init__(self):
        self.revoked: set[str] = set()
        self._epoch = 0
        self.down = False
        self.calls = 0

    def revoke(self, pid: str):
        self.revoked.add(pid)
        self._epoch += 1

    def eligible(self, passages, kb_version):
        self.calls += 1
        if self.down:
            raise RegistryUnavailable("down")
        return {p.passage_id: p.passage_id not in self.revoked for p in passages}

    def epoch(self):
        if self.down:
            raise RegistryUnavailable("down")
        return self._epoch


def S(sid: str, text: str, cites=(), *, kind="factual", visibility="student", depends_on=()) -> dict:
    return {"sentence_id": sid, "text": text, "kind_hint": kind, "visibility": visibility, "cites": list(cites),
            "depends_on": list(depends_on)}


def draft(*sentences: dict, status="draft") -> DraftAnswer:
    used = sorted({c for s in sentences for c in s["cites"]})
    return DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": status,
                                       "sentences": list(sentences), "used_passage_ids": used})


def request(d: DraftAnswer, rid: Optional[str] = None) -> VerifyRequest:
    return VerifyRequest(schema_version="verify-request-0.2", request_id=rid or str(uuid.uuid4()), draft=d,
                         prompt_version="brain-prompt-0.2")


class Scores:
    """score_fn: exact (premise, hypothesis) table, then hypothesis-only table, then default."""

    def __init__(self, default=NEUTRAL):
        self.pairs: dict[tuple[str, str], tuple] = {}
        self.hyp: dict[str, tuple] = {}
        self.default = default

    def __call__(self, premise, hypothesis):
        if (premise, hypothesis) in self.pairs:
            return self.pairs[(premise, hypothesis)]
        if hypothesis in self.hyp:
            return self.hyp[hypothesis]
        if hypothesis == premise:
            return SUPPORT
        return self.default


class Kit:
    def __init__(self, passages=None, conflicts=(), audit=None):
        self.scores = Scores()
        self.backend = FixtureNLI(self.scores)
        self.audit = audit if audit is not None else InMemoryVerifierAudit()
        self.verifier = Verifier.from_profile(PROFILE, audit=self.audit, fixture_backend=self.backend)
        self.backend.calls = 0
        self.backend.pairs_seen.clear()
        self.registry = FakeRegistry()
        self.passages = passages if passages is not None else [passage(k, v) for k, v in TEXTS.items()]
        self.conflicts = conflicts

    def bundle(self, rid: str, shown=None, *, tenant=TENANT, course=COURSE, bundle_rid=None, passages=None):
        ps = passages or self.passages
        r = retrieval(bundle_rid or rid, ps, self.conflicts)
        return EvidenceBundle(request_id=bundle_rid or rid, tenant_id=tenant, course_id=course, retrieval=r,
                              shown_passage_ids=tuple(shown or [p.passage_id for p in ps]))

    def context(self, req: VerifyRequest, bundle=None, **kw) -> VerifyContext:
        base = dict(request_id=req.request_id, route="retrieve", tenant_id=TENANT, course_id=COURSE,
                    redacted_request="Explain simple squamous epithelium.", mode="answer",
                    evidence=bundle or self.bundle(req.request_id), registry=self.registry)
        base.update(kw)
        return VerifyContext(**base)

    def run(self, d: DraftAnswer, **kw):
        req = request(d)
        bundle = kw.pop("bundle", None)
        ctx = self.context(req, bundle, **kw)
        return self.verifier.run(req, ctx), req, ctx


@pytest.fixture
def kit() -> Kit:
    return Kit()


@pytest.fixture
def make_kit() -> Callable[..., Kit]:
    return Kit
