"""Shared deterministic fixtures. All scores here are synthetic test values, not model output."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pytest

from gate_classifier.adapters import Passage
from gate_classifier.audit import AlertDispatcher, InMemoryAlertQueue, InMemoryAuditSink, OperationalAlarms
from gate_classifier.config import load_config, load_registry
from gate_classifier.encoder import FixtureTokenizer
from gate_classifier.orchestrator import FixtureBrain, FixtureRetriever, FixtureVerifier, Orchestrator, Principal
from gate_classifier.schema import LIBRARY_IDS, RISK_LABELS
from gate_classifier.service import GateService
from gate_classifier.session import InMemorySessionStore

ROOT = Path(__file__).resolve().parents[1]
KB = "fixture-kb-001"
COURSE = "histology-dev"
TENANT = "dev-tenant"


def scores(*, mode: Optional[dict] = None, topic: Optional[dict] = None, risks: Optional[dict] = None,
           libs: Optional[dict] = None) -> dict:
    """Full prediction dict; every risk defaults below its uncertainty threshold."""
    m = mode or {"answer": 0.90, "tutor": 0.05, "quiz": 0.05}
    t = topic or {"course_related": 0.95, "outside_course": 0.02, "nonmedical": 0.01, "unclear": 0.02}
    r = {k: 0.01 for k in RISK_LABELS}
    r.update(risks or {})
    lib = {"lib1": 0.85, "lib2": 0.65, "lib3": None, "lib4": None, "lib5": 0.90, "lib6": 0.10}
    lib.update(libs or {})
    return {"mode": m, "topic_scope": t, "risks": r, "libraries": lib}


TOPIC_OUTSIDE = {"course_related": 0.05, "outside_course": 0.90, "nonmedical": 0.03, "unclear": 0.02}
TOPIC_NONMED = {"course_related": 0.02, "outside_course": 0.03, "nonmedical": 0.93, "unclear": 0.02}
TOPIC_UNCLEAR = {"course_related": 0.40, "outside_course": 0.10, "nonmedical": 0.10, "unclear": 0.40}


class ScriptedScorer:
    """Maps canonical-text substrings to synthetic predictions. Records every input it sees."""

    def __init__(self, default: Optional[dict] = None):
        self.default = default or scores()
        self.rules: list[tuple[str, object]] = []
        self.seen: list[str] = []

    def when(self, substring: str, result) -> "ScriptedScorer":
        self.rules.append((substring, result))
        return self

    def __call__(self, canonical: str) -> dict:
        self.seen.append(canonical)
        for sub, result in self.rules:
            if sub in canonical:
                return result(canonical) if callable(result) else result
        return self.default


def passages() -> list[Passage]:
    import hashlib

    def p(pid, lib, text, slide):
        return Passage(passage_id=pid, source_id=f"src-{lib}", source_version="v1", title=f"Fixture {lib}",
                       library_id=lib, locator={"kind": "slide", "start": slide, "end": slide,
                                                "label": f"Slide {slide}", "anchor": None},
                       section_path=("Epithelium",), edition=None, publication_year=None, text=text,
                       text_sha256=hashlib.sha256(text.encode()).hexdigest(), review_status="live",
                       rights_reference="fixture-rights", relevance_score=6.25, score_type="cross_encoder_logit",
                       evidence_uri=f"/v1/evidence/{pid}?kb_version={KB}")

    return [
        p("fixture-p1", "lib1", "Simple squamous epithelium is a single layer of flat cells (fixture text).", 1),
        p("fixture-p2", "lib2", "It lines alveoli and blood vessels (fixture text).", 2),
        p("fixture-q1", "lib5", "Practice item: Which epithelium lines alveoli? A) stratified B) simple squamous.", 3),
        p("fixture-p3", "lib3", "DISABLED clinical protocol passage that must never be returned.", 4),
    ]


ITEMS = {"item-e1": ("v1", ("fixture-q1", "fixture-p1"))}


class Harness:
    def __init__(self, scorer: Optional[ScriptedScorer] = None, *, config_overrides: Optional[dict] = None,
                 audit=None, alert_sink=None, verifier="default", retriever=None, brain=None,
                 fixture_scorer: Optional[Callable] = "default"):
        cfg = load_config(ROOT / "config/development.yaml")
        if config_overrides:
            cfg = cfg.model_copy(update=config_overrides)
        self.config = cfg
        self.registry = load_registry(ROOT / "config/library_registry.yaml")
        self.scorer = scorer or ScriptedScorer()
        self.audit = audit if audit is not None else InMemoryAuditSink()
        self.alarms = OperationalAlarms()
        self.alert_queue = alert_sink if alert_sink is not None else InMemoryAlertQueue()
        self.dispatcher = AlertDispatcher(self.alert_queue, self.alarms, self.audit, max_attempts=3,
                                          base_backoff=0.001)
        self.service = GateService(
            cfg, self.registry, tokenizer=FixtureTokenizer(),
            fixture_scorer=self.scorer if fixture_scorer == "default" else fixture_scorer,
            audit=self.audit, alert_dispatcher=self.dispatcher, alarms=self.alarms,
        )
        self.sessions = InMemorySessionStore(cfg.runtime.session_ttl_minutes)
        self.retriever = retriever or FixtureRetriever(passages(), KB, items=ITEMS)
        self.brain = brain or FixtureBrain()
        self.verifier = FixtureVerifier() if verifier == "default" else verifier
        self.orch = Orchestrator(self.service, self.sessions, self.retriever, self.brain, self.verifier, kb_version=KB)
        self.principal = Principal("owner-a", TENANT, COURSE)

    def ask(self, text, principal=None, **body):
        return self.orch.handle(principal or self.principal, {"text": text, **body})

    def ctx(self, session_id=None, owner="owner-a"):
        return self.orch.build_context(owner, TENANT, COURSE, session_id)


@pytest.fixture
def harness():
    return Harness()


@pytest.fixture
def make_harness():
    return Harness


__all__ = ["scores", "ScriptedScorer", "Harness", "LIBRARY_IDS"]
