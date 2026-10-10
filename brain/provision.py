"""Explicit, online provisioning of one selected candidate (Brain plan v0.3, sections 2 and 5).

Resolves the conversion repository to a full immutable revision, selects one exact GGUF file, checks the
repository digest against the complete local SHA-256 and byte count, records the runtime release/commit
and executable hash, measures actual GPU layer placement and runs the capability probes, then writes a
candidate manifest with ``release_ready: false``. A file is downloaded only with ``download=True``; an
already-present file is adopted only if its bytes match the pinned digest. Nothing here runs at serve time.
"""

from __future__ import annotations

import asyncio
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .capabilities import run_probes
from .config import BrainConfig
from .grammar import schema_hash
from .manifest import MANIFEST_SCHEMA, Manifest, save_manifest, sha256_file
from .parse import SentencePolicy
from .prompt import prompt_hashes
from .resources import gpu_memory, snapshot
from .schema import BrainError, ErrorCode
from .supervisor import Supervisor, check_flags, redact_args, runtime_version, server_args

CANDIDATES = {
    "qwen35-9b-q4": {"repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf",
                     "upstream": "Qwen/Qwen3.5-9B", "role": "retained main candidate"},
    "qwen35-4b-q4": {"repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_K_M.gguf",
                     "upstream": "Qwen/Qwen3.5-4B", "role": "smaller capacity candidate"},
    "qwen3-4b-q4": {"repo": "Qwen/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q4_K_M.gguf",
                    "upstream": "Qwen/Qwen3-4B", "role": "independent model-generation comparison"},
}
RUNTIME_REPO = "https://github.com/ggml-org/llama.cpp"


class ProvisionError(RuntimeError):
    pass


def resolve_model(candidate_id: str) -> dict:
    from huggingface_hub import HfApi  # noqa: PLC0415 - online provisioning only

    spec = CANDIDATES[candidate_id]
    api = HfApi()
    info = api.model_info(spec["repo"], files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    if spec["file"] not in files:
        raise ProvisionError("selected GGUF file not present at the resolved revision")
    f = files[spec["file"]]
    upstream = api.model_info(spec["upstream"])
    lic = (upstream.card_data or {}).get("license") if upstream.card_data else None
    return {"conversion_repo": spec["repo"], "conversion_revision": info.sha, "gguf_filename": spec["file"],
            "repo_bytes": f.size, "repo_sha256": f.lfs.sha256 if f.lfs else None,
            "model_id": spec["upstream"], "upstream_revision": upstream.sha,
            "license_reference": f"{spec['upstream']}@{upstream.sha} card license: {lic}; GGUF conversion "
                                 f"{spec['repo']}@{info.sha}"}


def resolve_runtime_commit(short: str, build: str) -> str:
    out = subprocess.run(["git", "ls-remote", RUNTIME_REPO, f"refs/tags/b{build}"], capture_output=True,
                         text=True, timeout=60, shell=False).stdout.split()
    if not out or not out[0].startswith(short):
        raise ProvisionError("runtime tag does not resolve to the executable's reported commit")
    return out[0]


def provision(cfg: BrainConfig, *, download: bool = False, log=print) -> Manifest:
    if cfg.candidate_id not in CANDIDATES:
        raise ProvisionError("unknown candidate")
    model_path = cfg.path(cfg.artifacts.model_path)
    binary = cfg.path(cfg.artifacts.server_binary)

    log("resolving model revision ...")
    model = resolve_model(cfg.candidate_id)
    if model_path.name != model["gguf_filename"]:
        raise ProvisionError("configured model file name differs from the selected candidate file")
    if not model_path.exists():
        if not download:
            raise ProvisionError("model file missing; rerun with --download to fetch the pinned revision")
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        log("downloading the selected file at the pinned revision ...")
        hf_hub_download(model["conversion_repo"], model["gguf_filename"], revision=model["conversion_revision"],
                        local_dir=str(model_path.parent))
    log("hashing local model bytes ...")
    sha, size = sha256_file(model_path)
    if model["repo_sha256"] and sha != model["repo_sha256"]:
        raise ProvisionError("local model bytes do not match the repository digest")
    if model["repo_bytes"] and size != model["repo_bytes"]:
        raise ProvisionError("local model size does not match the repository")

    log("checking runtime ...")
    rv = runtime_version(cfg)
    commit = resolve_runtime_commit(rv["commit_short"], rv["build"])
    missing = check_flags(cfg)
    if missing:
        raise ProvisionError("runtime lacks configured flags: " + ", ".join(missing))
    exe_sha, _ = sha256_file(binary)

    log("measuring GPU layer placement ...")
    before = gpu_memory()
    sup = Supervisor(cfg)
    placement = sup.placement_preflight()
    if not placement.full:
        raise BrainError(ErrorCode.MEMORY_LIMIT)

    manifest_fields = dict(
        schema_version=MANIFEST_SCHEMA, candidate_id=cfg.candidate_id, model_id=model["model_id"],
        upstream_revision=model["upstream_revision"], conversion_repo=model["conversion_repo"],
        conversion_revision=model["conversion_revision"], gguf_filename=model["gguf_filename"], gguf_bytes=size,
        gguf_sha256=sha, gguf_upstream_sha256=model["repo_sha256"], license_reference=model["license_reference"],
        runtime_release=f"b{rv['build']}-{rv['commit_short']}", runtime_commit=commit, executable_sha256=exe_sha,
        build_environment={"source": f"official prebuilt release {binary.parent.name}", "compiler": rv["compiler"],
                           "gpu": (before or {}).get("name"), "driver": (before or {}).get("driver"),
                           "os": platform.platform(), "runtime_library_dirs": [Path(d).name for d in
                                                                              cfg.artifacts.runtime_library_dirs]},
        chat_template_sha256=None, launch_args_redacted=redact_args(server_args(cfg)), schema_hash=schema_hash(),
        prompt_hashes=prompt_hashes(),
        sampling_profiles={cfg.sampling.profile_version: cfg.sampling.model_dump(mode="json")},
        limits={**cfg.limits.model_dump(mode="json"), "context_tokens": cfg.runtime.context_tokens,
                "parallel": cfg.runtime.parallel},
        measured_placement={"offloaded_layers": placement.offloaded_layers, "total_layers": placement.total_layers,
                            "buffers_mib": placement.buffers_mib},
        measured_resources=None, capability_report=None, evaluation_report=None, release_ready=False,
    )

    log("starting server and running capability probes ...")
    report = asyncio.run(_first_probe(cfg, Manifest(**manifest_fields)))
    obs = report["observed"]
    manifest_fields["chat_template_sha256"] = obs.get("chat_template_sha256")
    manifest_fields["capability_report"] = {"known_prompt_tokens": obs.get("known_prompt_tokens"),
                                            "passed": report["passed"],
                                            "probes": {k: v["pass"] for k, v in report["probes"].items()},
                                            "hostile_probe": obs.get("hostile_probe")}
    manifest_fields["measured_resources"] = {"idle_after_load": report.get("resources")}
    manifest = Manifest(**manifest_fields)
    save_manifest(manifest, cfg.path(cfg.manifest_path))
    if not report["passed"]:
        failed = [k for k, v in report["probes"].items() if not v["pass"]]
        raise ProvisionError("capability probes failed: " + ", ".join(failed))
    return manifest


async def _first_probe(cfg: BrainConfig, manifest: Manifest) -> dict:
    from .service import read_api_key  # noqa: PLC0415
    from .transport import LocalTransport  # noqa: PLC0415

    key = read_api_key(cfg.path(cfg.runtime.api_key_file))
    sup = Supervisor(cfg)
    await asyncio.to_thread(sup.start, key)
    transport = LocalTransport(f"http://{cfg.runtime.host}:{cfg.runtime.port}", key)
    try:
        report = await run_probes(cfg, manifest, transport, api_key=key, sentence_policy=SentencePolicy())
        report["resources"] = snapshot(cfg.path("."))
        return report
    finally:
        await transport.close()
        await asyncio.to_thread(sup.stop)


def default_config_for(candidate_id: str) -> Optional[str]:
    return {"qwen35-9b-q4": "config/brain-local-9b.yaml", "qwen35-4b-q4": "config/brain-local-4b.yaml",
            "qwen3-4b-q4": "config/brain-comparison-qwen3-4b.yaml"}.get(candidate_id)
