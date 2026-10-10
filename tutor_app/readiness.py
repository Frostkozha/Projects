"""Readiness and release checks (Integration plan v0.3, sections 10, 11).

Liveness is process operation only. Readiness also requires migrations, the live registry, deployment
state, every component's own readiness and the profile's rules. Fixture adapters, missing artifacts or
unsigned/fixture-only manifests can never pass real_model or student_release readiness (I37).
"""

from __future__ import annotations

from pathlib import Path

from .components import Components
from .store import Store

STATUS_LABEL = {"fixture": "fixture_only", "development": "development_mixed", "real_model": "ready",
                "student_release": "ready"}


def readiness(cfg, store: Store, comps: Components, *, identity_institutional: bool) -> dict:
    reasons: list[str] = []
    try:
        versions = store.schema_versions()
        expected = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "migrations").glob("*.sql"))
        if versions != expected:
            reasons.append("migrations_incomplete")
        state, revision = store.run_sync(lambda c: Store.deployment(c, cfg.course_id))
        store.run_sync(Store.registry_epoch)
    except Exception:  # noqa: BLE001
        reasons.append("database_unavailable")
        state, revision = "unknown", -1
    components = {}
    for name, st in comps.statuses.items():
        ready = st.ready if name != "brain" else comps.brain_ready()
        components[name] = {"kind": st.kind, "ready": ready}
        if not ready:
            reasons.append(f"{name}_not_ready")
    kinds = {n: s.kind for n, s in comps.statuses.items()}
    if cfg.profile in ("real_model", "student_release"):
        for n, k in kinds.items():
            if k != "real":
                reasons.append(f"{n}_fixture_adapter_not_allowed")
        manifest = getattr(getattr(comps.brain, "manifest", None), "release_ready", None)
        if manifest is None:
            reasons.append("brain_manifest_missing")
    if cfg.profile == "development" and kinds.get("gate") != "fixture_scoring":
        reasons.append("development_profile_expects_gate_fixture_scoring")
    if cfg.profile == "student_release":
        if not identity_institutional:
            reasons.append("institutional_identity_missing")
        reasons += ["release_manifest_missing", "approved_retention_record_missing", "welfare_procedure_missing",
                    "evaluation_record_missing"]
        if not getattr(getattr(comps.brain, "manifest", None), "release_ready", False):
            reasons.append("brain_not_release_ready")
    if state != "active":
        reasons.append(f"deployment_{state}")
    ready = not reasons
    return {"ready": ready, "status": STATUS_LABEL[cfg.profile] if ready else "not_ready", "profile": cfg.profile,
            "deployment": {"state": state, "revision": revision}, "components": components,
            "reasons": sorted(set(reasons))}
