"""Component wiring: gate, retriever, Brain and verifier adapters (Integration plan v0.3, section 3, appendix F).

Existing packages are reused through their own types (``GateService``/``GateDecision``,
``RetrieverService``/``RetrievalResult``, ``BrainService``/``BrainResult``, ``Verifier``/
``VerificationResult``). Fixture adapters are explicit and labelled; they never satisfy real_model or
student_release readiness. Synchronous components run on bounded worker threads, never on the event loop.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

from gate_classifier.audit import InMemoryAlertQueue, InMemoryAuditSink, OperationalAlarms, AlertDispatcher
from gate_classifier.config import load_config as load_gate_config
from gate_classifier.config import load_registry
from gate_classifier.encoder import FixtureTokenizer
from gate_classifier.service import GateService
from retriever.schema import RetrievalContext
from verifier.evidence import RegistryUnavailable

from .config import TutorConfig
from .fixtures import FIXTURE_KB, FixtureBrain, RuleBackedDevScorer, fixture_corpus, fixture_sources
from .store import Store


@dataclass
class ComponentStatus:
    name: str
    kind: str          # real | fixture | fixture_scoring
    ready: bool
    reasons: list[str] = field(default_factory=list)


class AppLiveRegistry:
    """Verifier ``SourceRegistry`` over the application's live registry (authoritative outside the index)."""

    def __init__(self, store: Store):
        self.store = store
        self.down = False

    def eligible(self, passages, kb_version):
        if self.down:
            raise RegistryUnavailable("registry down")
        try:
            def run(conn):
                now = self.store.now()
                ok = Store.sources_eligible(conn, [(p.source_id, p.source_version) for p in passages], now)
                return {p.passage_id: ok[(p.source_id, p.source_version)] for p in passages}
            return self.store.run_sync(run)
        except Exception:
            raise RegistryUnavailable("registry lookup failed") from None

    def epoch(self) -> int:
        if self.down:
            raise RegistryUnavailable("registry down")
        try:
            return self.store.run_sync(Store.registry_epoch)
        except Exception:
            raise RegistryUnavailable("registry epoch unavailable") from None


class Components:
    def __init__(self, cfg: TutorConfig, store: Store):
        self.cfg = cfg
        self.store = store
        self.statuses: dict[str, ComponentStatus] = {}
        c = cfg.components
        self.gate_cfg = load_gate_config(cfg.path(c.gate_config))
        self.library_registry = load_registry(cfg.path(c.library_registry))
        self.gate_audit = InMemoryAuditSink()
        self.alarms = OperationalAlarms()
        self.registry = AppLiveRegistry(store)
        self.gate: Optional[GateService] = None
        self.retriever = None
        self.brain = None
        self.verifier = None
        self.kb_version: Optional[str] = None
        self.sources: list[dict] = []

    # ------------------------------------------------------------------ construction

    def build(self, *, brain=None, verifier_audit=None, fixture_verifier_scores=None) -> None:
        self._build_gate()
        self._build_retriever()
        self._build_verifier(verifier_audit, fixture_verifier_scores)
        self._build_brain(brain)

    def _build_gate(self) -> None:
        c = self.cfg.components
        dispatcher = AlertDispatcher(InMemoryAlertQueue(), self.alarms, self.gate_audit, 1, 0.0)
        if c.gate_scoring == "fixture":
            if self.gate_cfg.operating_mode != "fixture":
                raise ValueError("fixture gate scoring requires a gate config in fixture operating mode")
            tokenizer = FixtureTokenizer()
            reasons = []
            if c.gate_tokenizer == "e5":
                from gate_classifier.encoder import HFTokenizer  # noqa: PLC0415

                enc = self.gate_cfg.encoder
                try:
                    tokenizer = HFTokenizer(str(self.cfg.path(enc.local_path)), enc.tokenizer_revision)
                except Exception:  # noqa: BLE001
                    reasons.append("e5_tokenizer_unavailable")
            self.gate = GateService(self.gate_cfg, self.library_registry, tokenizer=tokenizer,
                                    fixture_scorer=RuleBackedDevScorer(), audit=self.gate_audit, alarms=self.alarms,
                                    alert_dispatcher=dispatcher)
            r = self.gate.readiness()
            self.statuses["gate"] = ComponentStatus("gate", "fixture_scoring", r.ready and not reasons,
                                                    r.reasons + reasons)
        else:
            self.gate = GateService.from_config(self.gate_cfg, self.library_registry, audit=self.gate_audit,
                                                alarms=self.alarms, alert_dispatcher=dispatcher)
            r = self.gate.readiness()
            self.statuses["gate"] = ComponentStatus("gate", "real", r.ready, r.reasons)

    def _build_retriever(self) -> None:
        c = self.cfg.components
        if c.retriever == "fixture":
            from gate_classifier.orchestrator import FixtureRetriever  # noqa: PLC0415

            self.retriever = FixtureRetriever(fixture_corpus(), FIXTURE_KB)
            self.kb_version = FIXTURE_KB
            self.sources = fixture_sources()
            self.statuses["retriever"] = ComponentStatus("retriever", "fixture", True)
            return
        from retriever.config import load_profile  # noqa: PLC0415
        from retriever.models import load_models  # noqa: PLC0415
        from retriever.service import build_service  # noqa: PLC0415

        profile = load_profile(self.cfg.path(c.retriever_profile))
        encoder, reranker, errors = load_models(profile)
        self.retriever = build_service(profile, self.library_registry, encoder, reranker, courses=(self.cfg.course_id,))
        reasons = list(errors)
        try:
            snap = self.retriever.snapshots.pin(self.cfg.course_id)
            self.kb_version = snap.kb_version
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"snapshot:{getattr(exc, 'reason', type(exc).__name__)}")
        r = self.retriever.readiness()
        reasons += list(getattr(r, "reasons", []) or [])
        self.statuses["retriever"] = ComponentStatus("retriever", "real", not reasons and bool(r.ready), reasons)
        register = self.cfg.path(profile.import_root).parent / "real" / "register.json"
        self.sources = _register_sources(register)

    def _build_verifier(self, audit, fixture_scores) -> None:
        from verifier.service import InMemoryVerifierAudit, Verifier  # noqa: PLC0415

        audit = audit if audit is not None else InMemoryVerifierAudit()
        path = self.cfg.path(self.cfg.components.verifier_profile)
        if self.cfg.profile == "fixture":
            from verifier.nli import FixtureNLI, substring_scores  # noqa: PLC0415

            self.verifier = Verifier.from_profile(path, audit=audit,
                                                  fixture_backend=FixtureNLI(fixture_scores or substring_scores))
            kind = "fixture"
        else:
            self.verifier = Verifier.from_profile(path, audit=audit)
            kind = "real"
        r = self.verifier.readiness()
        failed = [k for k, v in r.get("checks", {}).items() if not v]
        self.statuses["verifier"] = ComponentStatus("verifier", kind, bool(r.get("ready")), failed)

    def _build_brain(self, brain) -> None:
        if brain is not None:
            self.brain = brain
        elif self.cfg.components.brain == "fixture":
            self.brain = FixtureBrain()
        else:
            from brain.config import load_config as load_brain_config  # noqa: PLC0415
            from brain.service import build_service  # noqa: PLC0415

            self.brain = build_service(load_brain_config(self.cfg.path(self.cfg.components.brain_config)))
        kind = "fixture" if getattr(self.brain, "is_fixture", False) else "real"
        self.statuses["brain"] = ComponentStatus("brain", kind, False, ["not_started"])

    async def start_brain(self) -> None:
        st = self.statuses["brain"]
        try:
            report = await self.brain.startup()
            st.ready = bool(report.get("passed"))
            st.reasons = [] if st.ready else [k for k, v in report.get("probes", {}).items() if not v.get("pass")]
        except Exception as exc:  # noqa: BLE001
            st.ready, st.reasons = False, [f"startup:{getattr(exc, 'code', type(exc).__name__)}"]

    def brain_ready(self) -> bool:
        if getattr(self.brain, "is_fixture", False):
            return self.statuses["brain"].ready
        r = self.brain.readiness()
        return bool(r.get("ready"))

    # ------------------------------------------------------------------ calls (bounded threads)

    def retrieval_context(self, embedding_ref=None, allowed=None, session_ref=None) -> RetrievalContext:
        active = self.library_registry.active_libraries(self.cfg.course_id)
        return RetrievalContext(service_id="orchestrator", tenant_id=self.cfg.tenant_id, course_id=self.cfg.course_id,
                                authorized_libraries=frozenset(allowed or active),
                                registry_version=self.library_registry.registry_version, gate_permitted=True,
                                embedding_ref=embedding_ref, session_ref=session_ref)

    async def thread(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    async def shutdown(self) -> None:
        if self.brain is not None:
            try:
                await self.brain.shutdown()
            except Exception:  # noqa: BLE001
                pass
        worker = getattr(self.verifier, "worker", None)
        if worker is not None and hasattr(worker, "close"):
            try:
                worker.close()
            except Exception:  # noqa: BLE001
                pass


def _register_sources(path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for s in data.get("sources", []):
        rights = s.get("rights") or {}
        out.append({"source_id": s["source_id"], "source_version": s["source_version"],
                    "status": s.get("status", "pending_review"), "review_due_at": _utc(s.get("review_due_at")),
                    "rights_expires_at": _utc(rights.get("expires_at"))})
    return out


def _utc(value):
    if not value:
        return None
    from datetime import datetime, timezone  # noqa: PLC0415

    from .store import utc_str  # noqa: PLC0415

    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return utc_str(dt)
