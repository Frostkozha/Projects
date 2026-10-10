"""Explicit fixture adapters and synthetic data (fixture profile and the development gate only).

None of these is a model. ``RuleBackedDevScorer`` assigns a fixed course-related, low-risk prediction so
routing relies on the gate's deterministic rule engine; it exists because no trained gate bundle is
available yet and is refused by real_model/student_release readiness.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Callable, Optional

from brain.schema import RESULT_SCHEMA, BrainMetrics, BrainResult, ErrorCode
from contracts.models import DraftAnswer
from retriever.schema import EvidencePassage

FIXTURE_KB = "fixture-kb-001"
_SPLIT = re.compile(r"(?<=[.!?])\s+")


class RuleBackedDevScorer:
    is_fixture = True

    def __call__(self, canonical: str) -> dict:
        return {
            "mode": {"answer": 0.90, "tutor": 0.05, "quiz": 0.05},
            "topic_scope": {"course_related": 0.95, "outside_course": 0.02, "nonmedical": 0.01, "unclear": 0.02},
            "risks": {"real_person_advice": 0.01, "imminent_emergency": 0.01, "self_harm_crisis": 0.01,
                      "assessed_work": 0.01, "instruction_override": 0.01, "private_data_request": 0.01},
            "libraries": {"lib1": 0.90, "lib2": 0.60, "lib3": None, "lib4": None, "lib5": 0.40, "lib6": 0.20},
        }


def fixture_passage(pid: str, lib: str, text: str, n: int, *, source_id: Optional[str] = None) -> EvidencePassage:
    return EvidencePassage(
        passage_id=pid, source_id=source_id or f"src-{lib}", source_version="v1", title=f"Fixture {lib} notes",
        library_id=lib, locator={"kind": "slide", "start": n, "end": n, "label": f"Slide {n}", "anchor": None},
        section_path=("Epithelium",), edition=None, publication_year=None, text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(), review_status="live",
        rights_reference="fixture-rights", relevance_score=6.25, score_type="cross_encoder_logit",
        evidence_uri=f"/v1/evidence/{pid}?kb_version={FIXTURE_KB}")


def fixture_corpus() -> list[EvidencePassage]:
    return [
        fixture_passage("fixture-p1", "lib1", "Simple squamous epithelium is a single layer of flat cells.", 1),
        fixture_passage("fixture-p2", "lib2", "Simple squamous epithelium lines alveoli and blood vessels.", 2),
        fixture_passage("fixture-p3", "lib1", "Goblet cells secrete mucus.", 3),
        fixture_passage("fixture-p9", "lib3", "DISABLED clinical protocol passage that must never be returned.", 4),
    ]


def fixture_sources() -> list[dict]:
    return [{"source_id": s, "source_version": "v1", "status": "live"} for s in ("src-lib1", "src-lib2", "src-lib3",
                                                                              "src-lib5", "src-lib6")]


class FixtureBrain:
    """Deterministic draft: the first sentence of each of the first two passages, each citing its passage.

    ``script(request, context) -> DraftAnswer | ErrorCode | None`` overrides the output for tests.
    """

    is_fixture = True

    def __init__(self, script: Optional[Callable] = None, delay: float = 0.0):
        self.script = script
        self.delay = delay
        self.calls: list = []
        self.cfg = None

    def readiness(self) -> dict:
        return {"ready": True, "profile": "fixture", "model_version": None, "runtime_version": None}

    async def startup(self) -> dict:
        return {"passed": True, "probes": {}}

    async def shutdown(self) -> None:
        return None

    async def generate(self, request, context, *, cancel=None) -> BrainResult:
        import asyncio  # noqa: PLC0415

        self.calls.append(request)
        t0 = time.monotonic()
        end = t0 + self.delay
        while time.monotonic() < end:
            if cancel is not None and cancel.is_set():
                return self._result(request, context, ErrorCode.CANCELLED, t0)
            await asyncio.sleep(min(0.02, end - time.monotonic()))
        out = self.script(request, context) if self.script else None
        if out is None:
            value = context.evidence.fresh_value()
            by_id = {p["passage_id"]: p for p in value["passages"]}
            sentences = []
            for i, pid in enumerate(request.passage_ids[:2]):
                first = _SPLIT.split(by_id[pid]["text"].strip())[0]
                sentences.append({"sentence_id": f"s{i + 1}", "text": first, "kind_hint": "factual",
                                  "visibility": "student", "cites": [pid], "depends_on": []})
            out = DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": "draft",
                                              "sentences": sentences,
                                              "used_passage_ids": sorted({s["cites"][0] for s in sentences})})
        if isinstance(out, ErrorCode):
            return self._result(request, context, out, t0)
        status = "ok" if out.status == "draft" else "no_evidence"
        return BrainResult(schema_version=RESULT_SCHEMA, request_id=request.request_id, status=status, draft=out,
                           error_code=None, model_version=None, runtime_version=None,
                           prompt_version=request.prompt_version,
                           sampling_profile_version=request.sampling_profile_version,
                           evidence_digest=context.evidence.evidence_digest,
                           metrics=BrainMetrics(queue_ms=0.0, prompt_ms=None, generation_ms=None,
                                                total_ms=round((time.monotonic() - t0) * 1000, 3), prompt_tokens=None,
                                                completion_tokens=None, finish_reason="stop"))

    @staticmethod
    def _result(request, context, code, t0) -> BrainResult:
        return BrainResult(schema_version=RESULT_SCHEMA, request_id=request.request_id, status="error", draft=None,
                           error_code=code, model_version=None, runtime_version=None,
                           prompt_version=request.prompt_version,
                           sampling_profile_version=request.sampling_profile_version,
                           evidence_digest=context.evidence.evidence_digest, metrics=BrainMetrics.empty())


def new_uuid() -> str:
    return str(uuid.uuid4())
