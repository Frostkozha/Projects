"""Application assembly: configuration -> store -> components -> coordinator (Integration plan v0.3, section 11).

Migrations, startup recovery and live-registry loading finish before readiness. Provisioning never runs
here; the Brain process is started only when requested and only from validated local artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .auth import IdentityProvider, build_identity
from .components import Components
from .config import TutorConfig, read_secret
from .items import load_bank, load_topics
from .orchestrator import Coordinator
from .readiness import readiness
from .store import Store


@dataclass
class TutorApp:
    cfg: TutorConfig
    store: Store
    components: Components
    coordinator: Coordinator
    identity: IdentityProvider
    cookie_key: bytes

    def readiness(self) -> dict:
        r = readiness(self.cfg, self.store, self.components, identity_institutional=self.identity.institutional)
        alarms = {a["code"] for a in self.components.alarms.alarms}
        r["degraded"] = sorted(self.coordinator.degraded | alarms)
        return r

    async def start(self) -> None:
        await self.components.start_brain()

    async def stop(self) -> None:
        await self.components.shutdown()
        self.store.close()


def build_app(cfg: TutorConfig, *, brain=None, fixture_verifier_scores=None, verifier_audit=None,
              store: Optional[Store] = None, clock=None) -> TutorApp:
    dev = cfg.profile in ("fixture", "development")
    fp_key = read_secret(cfg.path(cfg.fingerprint_key_file), create=dev)
    cookie_key = read_secret(cfg.path(cfg.cookie_key_file), create=dev)
    store = store or (Store(cfg.path(cfg.db_path), clock=clock) if clock else Store(cfg.path(cfg.db_path)))
    store.migrate()
    store.recover_startup()
    comps = Components(cfg, store)
    comps.build(brain=brain, verifier_audit=verifier_audit, fixture_verifier_scores=fixture_verifier_scores)
    reg = comps.library_registry
    store.load_registry(registry_version=reg.registry_version, sources=comps.sources, course_id=cfg.course_id)
    bank = load_bank(cfg.path(cfg.item_bank))
    topics = load_topics(cfg.path(cfg.topic_registry))
    if bank.course_id != cfg.course_id or topics.course_id != cfg.course_id or bank.tenant_id != cfg.tenant_id:
        raise ValueError("item bank / topic registry scope does not match the configured course")
    coord = Coordinator(cfg, store, comps, fingerprint_key=fp_key, bank=bank, topics=topics)
    return TutorApp(cfg, store, comps, coord, build_identity(cfg, cookie_key), cookie_key)
