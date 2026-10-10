"""Operator CLI (Integration plan v0.3, sections 11, 12).

    python -m tutor_app.cli validate --config config/fixture.yaml
    python -m tutor_app.cli start --config config/local-dev.yaml          # chat at http://127.0.0.1:8000
    python -m tutor_app.cli status --config config/local-dev.yaml
    python -m tutor_app.cli suspend --config ... --expected-revision 0 --actor operator-a [--exam]
    python -m tutor_app.cli resume  --config ... --expected-revision 1 --actor operator-a [--release-record PATH]
    python -m tutor_app.cli inspect-outbox --config ...
    python -m tutor_app.cli ack-outbox --config ... --event-id ID --actor operator-a
    python -m tutor_app.cli revoke-source --config ... --source-id ID [--source-version V]
    python -m tutor_app.cli activate-bundle --config ... --bundle PATH
    python -m tutor_app.cli evaluate --suite integration --out reports
    python -m tutor_app.cli export-schemas --out docs/schemas

Provisioning never runs here. Operational commands are never reachable from the student UI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT, TutorConfigError, load_tutor_config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _store(cfg):
    from .store import Store  # noqa: PLC0415

    store = Store(cfg.path(cfg.db_path))
    store.migrate()
    return store


def cmd_validate(args) -> int:
    cfg = load_tutor_config(args.config)
    from .app import build_app  # noqa: PLC0415

    report = {"config": "valid", "profile": cfg.profile}
    tutor = build_app(cfg)
    try:
        import asyncio  # noqa: PLC0415

        if args.start_brain:
            asyncio.run(tutor.start())
        r = tutor.readiness()
        report.update(r)
    finally:
        import asyncio  # noqa: PLC0415

        asyncio.run(tutor.components.shutdown())
        tutor.store.close()
    _print(report)
    return 0 if report.get("ready") or not args.start_brain else 1


def cmd_start(args) -> int:
    cfg = load_tutor_config(args.config)
    import uvicorn  # noqa: PLC0415

    from .api import create_app  # noqa: PLC0415
    from .app import build_app  # noqa: PLC0415

    print(f"Loading the {cfg.profile} profile (models, index, database) ...", flush=True)
    tutor = build_app(cfg)
    app = create_app(tutor)

    @app.on_event("startup")
    async def _announce():  # pragma: no cover - interactive
        r = tutor.readiness()
        print(json.dumps({"ready": r["ready"], "status": r["status"], "components": r["components"],
                          "reasons": r["reasons"]}, indent=2), flush=True)
        print(f"Open http://{cfg.host}:{cfg.port}/ in your browser. Press Ctrl+C to stop.", flush=True)

    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1, log_level="warning", access_log=False)
    return 0


def cmd_status(args) -> int:
    cfg = load_tutor_config(args.config)
    store = _store(cfg)
    try:
        from .store import Store  # noqa: PLC0415

        def run(conn):
            state, rev = Store.deployment(conn, cfg.course_id) if conn.execute(
                "SELECT 1 FROM deployment_state WHERE course_id=?", (cfg.course_id,)).fetchone() else ("unset", -1)
            counts = {r[0]: r[1] for r in conn.execute("SELECT state, COUNT(*) FROM requests GROUP BY state")}
            outbox = conn.execute("SELECT COUNT(*) FROM welfare_outbox WHERE delivery_status='pending'").fetchone()[0]
            epoch = conn.execute("SELECT epoch FROM registry_meta").fetchone()
            return {"deployment": {"state": state, "revision": rev}, "requests": counts, "pending_welfare_events": outbox,
                    "registry_epoch": epoch[0] if epoch else None}
        _print({"profile": cfg.profile, "migrations": store.schema_versions(), **store.run_sync(run)})
    finally:
        store.close()
    return 0


def cmd_deploy(args, state: str) -> int:
    cfg = load_tutor_config(args.config)
    if state == "active" and cfg.profile == "student_release" and not args.release_record:
        print("student_release resume requires --release-record", file=sys.stderr)
        return 2
    store = _store(cfg)
    try:
        from .store import Store  # noqa: PLC0415

        has = store.run_sync(lambda c: c.execute("SELECT 1 FROM deployment_state WHERE course_id=?",
                                                 (cfg.course_id,)).fetchone())
        if not has:
            store.run_sync(lambda c: c.execute("INSERT INTO deployment_state VALUES (?, 'active', 0, ?, 'bootstrap')",
                                               (cfg.course_id, store.now())))
        rev = store.set_deployment(cfg.course_id, state, args.expected_revision, args.actor)
        _print({"deployment": state, "revision": rev})
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


def cmd_inspect_outbox(args) -> int:
    cfg = load_tutor_config(args.config)
    store = _store(cfg)
    try:
        rows = store.run_sync(lambda c: [dict(r) for r in c.execute(
            "SELECT event_id, request_id, category, support_ref, event_time, delivery_status FROM welfare_outbox "
            "ORDER BY event_time")])
        _print({"welfare_outbox": rows, "note": "Local events only. No external message is sent by this build."})
    finally:
        store.close()
    return 0


def cmd_ack_outbox(args) -> int:
    cfg = load_tutor_config(args.config)
    store = _store(cfg)
    try:
        n = store.run_sync(lambda c: c.execute(
            "UPDATE welfare_outbox SET delivery_status='acknowledged', reviewed_by=?, reviewed_at=? WHERE event_id=?",
            (args.actor, store.now(), args.event_id)).rowcount)
        _print({"acknowledged": n})
    finally:
        store.close()
    return 0 if n else 1


def cmd_revoke(args) -> int:
    cfg = load_tutor_config(args.config)
    store = _store(cfg)
    try:
        n = store.set_source_status(args.source_id, args.source_version, "revoked")
        _print({"revoked_rows": n})
    finally:
        store.close()
    return 0


def cmd_activate_bundle(args) -> int:
    """Switch the validated bundle pointer; keep the previous one for rollback. Migrations must be compatible."""
    from .bundle import activate  # noqa: PLC0415

    cfg = load_tutor_config(args.config)
    store = _store(cfg)
    try:
        _print(activate(cfg, store, Path(args.bundle)))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


def cmd_evaluate(args) -> int:
    suites = {"integration": ["tests/tutor", "tests/test_integration_primitives.py"],
              "brain": ["tests/brain", "tests/test_brain_primitives.py"], "all": ["tests"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    junit = out / f"{args.suite}-{stamp}.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *suites[args.suite]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1:] or [""]
    report = {"suite": args.suite, "command": " ".join(cmd[2:]), "exit_code": proc.returncode, "summary": tail[0],
              "junit": str(junit), "record_type": "engineering fixture tests (synthetic); not faculty evaluation"}
    (out / f"{args.suite}-{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)
    return proc.returncode


def cmd_export_schemas(args) -> int:
    from .input_schema import CloseSessionInput, InteractionInput, NoticeAcceptInput, ReportInput  # noqa: PLC0415
    from .public import ErrorEnvelope, EvidenceView, StudentReply  # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in {"interaction_input": InteractionInput, "close_session_input": CloseSessionInput,
                        "notice_accept_input": NoticeAcceptInput, "report_input": ReportInput,
                        "student_reply": StudentReply, "error_envelope": ErrorEnvelope,
                        "evidence_view": EvidenceView}.items():
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"schemas written to {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tutor_app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--config", required=True)
    v.add_argument("--start-brain", action="store_true", help="also start the Brain and require readiness")
    for n in ("start", "status", "inspect-outbox"):
        sub.add_parser(n).add_argument("--config", required=True)
    for n in ("suspend", "resume"):
        s = sub.add_parser(n)
        s.add_argument("--config", default="config/local-dev.yaml")
        s.add_argument("--expected-revision", type=int, required=True)
        s.add_argument("--actor", default="operator")
        s.add_argument("--exam", action="store_true")
        s.add_argument("--release-record")
    a = sub.add_parser("ack-outbox")
    a.add_argument("--config", required=True)
    a.add_argument("--event-id", required=True)
    a.add_argument("--actor", required=True)
    r = sub.add_parser("revoke-source")
    r.add_argument("--config", required=True)
    r.add_argument("--source-id", required=True)
    r.add_argument("--source-version")
    b = sub.add_parser("activate-bundle")
    b.add_argument("--config", required=True)
    b.add_argument("--bundle", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--suite", choices=("integration", "brain", "all"), default="integration")
    e.add_argument("--out", default="reports")
    x = sub.add_parser("export-schemas")
    x.add_argument("--out", default="docs/schemas")
    args = p.parse_args(argv)
    try:
        if args.cmd == "suspend":
            return cmd_deploy(args, "exam_shutdown" if args.exam else "suspended")
        if args.cmd == "resume":
            return cmd_deploy(args, "active")
        return {"validate": cmd_validate, "start": cmd_start, "status": cmd_status,
                "inspect-outbox": cmd_inspect_outbox, "ack-outbox": cmd_ack_outbox, "revoke-source": cmd_revoke,
                "activate-bundle": cmd_activate_bundle, "evaluate": cmd_evaluate,
                "export-schemas": cmd_export_schemas}[args.cmd](args)
    except TutorConfigError as exc:
        print(f"CONFIG INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
