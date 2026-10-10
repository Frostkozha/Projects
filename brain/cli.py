"""Brain command interface (Brain plan v0.3, section 14).

    python -m brain.cli provision --candidate qwen35-9b-q4 [--download]
    python -m brain.cli validate  --config config/brain-local-9b.yaml
    python -m brain.cli start     --config config/brain-local-9b.yaml
    python -m brain.cli smoke     --config config/brain-local-9b.yaml
    python -m brain.cli benchmark --dataset tests/fixtures/brain/synthetic.jsonl --config ... --out reports/brain

Only ``provision`` touches the network, and only when explicitly invoked. ``start``/``smoke``/``benchmark``
use existing local files and fail closed. Output is metadata; smoke/benchmark content is synthetic only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid

from .config import ConfigError, load_config
from .schema import BrainError


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_provision(args) -> int:
    from .provision import ProvisionError, default_config_for, provision  # noqa: PLC0415

    cfg = load_config(args.config or default_config_for(args.candidate))
    if cfg.candidate_id != args.candidate:
        print("config candidate_id does not match --candidate", file=sys.stderr)
        return 2
    try:
        m = provision(cfg, download=args.download)
    except (ProvisionError, BrainError) as exc:
        print(f"PROVISION FAILED: {exc}", file=sys.stderr)
        return 1
    _print({"manifest": cfg.manifest_path, "gguf_sha256": m.gguf_sha256, "conversion_revision": m.conversion_revision,
            "runtime_commit": m.runtime_commit, "placement": m.measured_placement,
            "capabilities": m.capability_report, "release_ready": m.release_ready})
    return 0


def cmd_validate(args) -> int:
    from .manifest import ManifestError, load_manifest, verify_artifacts  # noqa: PLC0415
    from .service import build_service  # noqa: PLC0415
    from .supervisor import Supervisor, check_flags  # noqa: PLC0415

    cfg = load_config(args.config)
    report: dict = {"config": "valid", "profile": cfg.profile}
    try:
        manifest = load_manifest(cfg.path(cfg.manifest_path))
        verify_artifacts(manifest, model_path=cfg.path(cfg.artifacts.model_path),
                         server_binary=cfg.path(cfg.artifacts.server_binary))
        report["artifacts"] = "verified"
    except ManifestError as exc:
        report["artifacts"] = str(exc)
        _print(report)
        return 1
    missing = check_flags(cfg)
    report["flags"] = "supported" if not missing else {"missing": missing}
    if missing:
        _print(report)
        return 1
    placement = Supervisor(cfg).placement_preflight()
    report["placement"] = {"offloaded": placement.offloaded_layers, "total": placement.total_layers,
                           "matches_manifest": (manifest.measured_placement or {}).get("offloaded_layers")
                           == placement.offloaded_layers and placement.full}
    if not report["placement"]["matches_manifest"]:
        _print(report)
        return 1

    async def probe():
        svc = build_service(cfg)
        try:
            return await svc.startup(verify_bytes=False)
        finally:
            await svc.shutdown()

    probes = asyncio.run(probe())
    report["capabilities"] = {k: v["pass"] for k, v in probes["probes"].items()}
    report["ready"] = probes["passed"]
    _print(report)
    return 0 if probes["passed"] else 1


def cmd_start(args) -> int:
    from .service import build_service  # noqa: PLC0415

    cfg = load_config(args.config)

    async def run():
        svc = build_service(cfg)
        try:
            report = await svc.startup()
            _print({"readiness": svc.readiness(), "probes_passed": report["passed"]})
            if not report["passed"]:
                return 1
            print("Brain ready on loopback; press Ctrl+C to stop.", flush=True)
            while svc.supervisor.running:
                await asyncio.sleep(1)
            print("runtime exited", file=sys.stderr)
            return 1
        finally:
            await svc.shutdown()

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 0


def cmd_smoke(args) -> int:
    from .evaluate import SMOKE_CASES, run_cases  # noqa: PLC0415

    cfg = load_config(args.config)
    results = asyncio.run(run_cases(cfg, SMOKE_CASES, show_content=True))
    _print(results)
    return 0 if results.get("ready") and all(r["status"] == r["expected_status"] for r in results["cases"]) else 1


def cmd_benchmark(args) -> int:
    from .evaluate import benchmark  # noqa: PLC0415

    cfg = load_config(args.config)
    summary = asyncio.run(benchmark(cfg, args.dataset, args.out, repeats=args.repeats))
    _print(summary)
    return 0 if summary.get("ready") else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="brain.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("provision")
    pv.add_argument("--candidate", required=True)
    pv.add_argument("--config")
    pv.add_argument("--download", action="store_true")
    for name in ("validate", "start", "smoke"):
        sub.add_parser(name).add_argument("--config", required=True)
    bm = sub.add_parser("benchmark")
    bm.add_argument("--config", required=True)
    bm.add_argument("--dataset", required=True)
    bm.add_argument("--out", required=True)
    bm.add_argument("--repeats", type=int, default=1)
    args = p.parse_args(argv)
    try:
        return {"provision": cmd_provision, "validate": cmd_validate, "start": cmd_start, "smoke": cmd_smoke,
                "benchmark": cmd_benchmark}[args.cmd](args)
    except ConfigError as exc:
        print(f"CONFIG INVALID: {exc}", file=sys.stderr)
        return 2
    except BrainError as exc:
        print(f"BRAIN ERROR: {exc.code.value}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
