"""Release bundle manifest, activation and rollback (Integration plan v0.3, section 11).

Activation switches one validated bundle pointer under the store's transaction lock and keeps the previous
pointer for rollback. A bundle must declare the database schema versions it supports; activating a bundle
that does not support the migrations already applied is refused (no pointer-only rollback into an
incompatible schema). Activation never touches live source blocks, notices or suspensions (I39).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from contracts.draft import SHA256, MachineID, StrictModel
from contracts.json_codec import digest, load_object

from .transactions import immediate_transaction


class ReleaseManifest(StrictModel):
    schema_version: Literal["release-bundle-0.3"]
    bundle_id: MachineID
    profile: Literal["fixture", "development", "real_model", "student_release"]
    shared_schemas: dict[str, str]
    application_commit: Optional[str]
    dependency_lock_sha256: SHA256
    supported_migrations: list[str]
    components: dict[str, dict]
    item_bank_digest: Optional[SHA256]
    evaluation_reference: Optional[str]
    approval_reference: Optional[str]


def load_release(path: Path) -> ReleaseManifest:
    return ReleaseManifest.model_validate(load_object(path.read_bytes(), max_bytes=1024 * 1024))


def activate(cfg, store, path: Path) -> dict:
    m = load_release(path)
    if cfg.profile == "student_release" and (m.profile != "student_release" or not m.approval_reference
                                             or not m.evaluation_reference or not m.application_commit):
        raise ValueError("student_release activation requires an approved, evaluated release record")
    pointer = cfg.path(str(Path(cfg.db_path).parent / "ACTIVE_BUNDLE.json"))

    def run(conn):
        applied = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
        if not set(applied) <= set(m.supported_migrations):
            raise ValueError("bundle does not support the applied database migrations")
        with immediate_transaction(conn):
            previous = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None
            record = {"active": {"bundle_id": m.bundle_id, "manifest_digest": digest(m.model_dump(mode="json")),
                                 "path": str(path)},
                      "previous": previous.get("active") if previous else None}
            pointer.parent.mkdir(parents=True, exist_ok=True)
            tmp = pointer.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(pointer)
            store.audit(conn, event="bundle_activated", reason=m.bundle_id)
        return record

    return store.run_sync(run)
