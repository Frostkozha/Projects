"""B27, B28, B30, B35: configuration, manifest identity, launch command and placement parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.config import ConfigError, load_config
from brain.manifest import Manifest, ManifestError, load_manifest, save_manifest, sha256_file, verify_artifacts
from brain.schema import BrainError, ErrorCode
from brain.supervisor import parse_load_log, redact_args, server_args

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "config/brain-local-9b.yaml"


def write_cfg(tmp_path, mutate):
    import yaml

    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    mutate(data)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- configuration

@pytest.mark.parametrize("name", ["brain-local-9b.yaml", "brain-local-4b.yaml", "brain-comparison-qwen3-4b.yaml",
                                  "brain-fixture.yaml"])
def test_shipped_profiles_validate(name):
    cfg = load_config(ROOT / "config" / name)
    assert cfg.runtime.parallel == 1 and cfg.runtime.context_tokens == 4096
    assert cfg.limits.prompt_tokens + cfg.limits.output_tokens + cfg.limits.reserve_tokens == 4096


@pytest.mark.parametrize("mutate", [
    lambda d: d["runtime"].update(parallel=4),
    lambda d: d["runtime"].update(threads=True),
    lambda d: d["runtime"].update(unknown_flag=1),
    lambda d: d["runtime"].update(host="0.0.0.0"),
    lambda d: d["runtime"].update(cache_prompt=True),
    lambda d: d["runtime"].update(reasoning="on"),
    lambda d: d["runtime"].update(fit="on"),
    lambda d: d["runtime"].update(cache_type_k="q8_0"),
    lambda d: d["limits"].update(prompt_tokens=3000),
    lambda d: d["limits"].update(queue_capacity=9),
    lambda d: d["limits"].update(brain_deadline_seconds=31),
    lambda d: d.update(manifest_path="C:/Windows/manifest.json"),
    lambda d: d["artifacts"].update(model_path="C:/Users/someone/model.gguf"),
    lambda d: d["features"].update(tutor_clue_enabled=True),
    lambda d: d.update(schema_version="brain-config-0.2"),
])
def test_config_rejections(tmp_path, mutate):
    with pytest.raises(ConfigError):
        load_config(write_cfg(tmp_path, mutate))


def test_config_duplicate_yaml_keys_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(CFG.read_text(encoding="utf-8") + "\nprofile: fixture\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


# ----------------------------------------------------------------------------- B30 offline launch, B11 text-only

def test_launch_command_is_the_single_tested_profile():
    cfg = load_config(CFG)
    args = server_args(cfg)
    joined = " ".join(args)
    for flag in ("--offline", "--no-mmproj", "--no-webui", "--no-cache-prompt", "--no-context-shift", "--jinja",
                 "--metrics", "--api-key-file"):
        assert flag in args
    pairs = dict(zip(args[::1], args[1::1]))
    assert pairs["--reasoning"] == "off" and pairs["--fit"] == "off" and pairs["--cache-ram"] == "0"
    assert pairs["--parallel"] == "1" and pairs["--ctx-size"] == "4096" and pairs["--host"] == "127.0.0.1"
    assert pairs["--cache-type-k"] == "f16" and pairs["--cache-type-v"] == "f16"
    assert "--reasoning-format" not in args  # not a thinking-off switch
    assert "--props" not in args and "--slot-save-path" not in args and "--mmproj" not in joined.replace(
        "--no-mmproj", "")
    red = redact_args(args)
    assert "<redacted>" in red and "secrets" not in " ".join(red) and red[red.index("--model") + 1] == \
        "Qwen3.5-9B-Q4_K_M.gguf"


def test_supervisor_env_forces_offline_and_strips_runtime_overrides(monkeypatch):
    from brain.supervisor import _env

    monkeypatch.setenv("LLAMA_ARG_FIT", "on")
    monkeypatch.setenv("HF_TOKEN", "x")
    env = _env(load_config(CFG))
    assert "LLAMA_ARG_FIT" not in env and "HF_TOKEN" not in env and env["HF_HUB_OFFLINE"] == "1"


# ----------------------------------------------------------------------------- B27 placement / memory

LOG_FULL = ["load_tensors: offloaded 33/33 layers to GPU\n", "load_tensors:        CUDA0 model buffer size =  4861.28 MiB\n",
            "llama_kv_cache:      CUDA0 KV buffer size =   128.00 MiB\n"]


def test_B27_full_and_partial_placement():
    p = parse_load_log(LOG_FULL)
    assert p.full and p.buffers_mib["CUDA0:model"] == 4861.28
    partial = parse_load_log(["load_tensors: offloaded 20/33 layers to GPU\n"])
    assert not partial.full


@pytest.mark.parametrize("lines,code", [
    (["ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: cudaMalloc failed: out of memory"],
     ErrorCode.MEMORY_LIMIT),
    (["model loaded\n"], ErrorCode.RUNTIME_INCOMPATIBLE),
])
def test_B27_oom_or_unknown_placement_fails(lines, code):
    with pytest.raises(BrainError) as e:
        parse_load_log(lines)
    assert e.value.code == code


# ----------------------------------------------------------------------------- B28 manifest

def manifest_fields(**over):
    base = dict(schema_version="brain-manifest-0.3", candidate_id="qwen35-9b-q4", model_id="Qwen/Qwen3.5-9B",
                upstream_revision="c" * 40, conversion_repo="unsloth/Qwen3.5-9B-GGUF", conversion_revision="a" * 40,
                gguf_filename="model.gguf", gguf_bytes=5, gguf_sha256="0" * 64, gguf_upstream_sha256=None,
                license_reference="apache-2.0", runtime_release="b11435-43fe9c642", runtime_commit="4" * 40,
                executable_sha256="0" * 64, build_environment={}, chat_template_sha256=None, launch_args_redacted=[],
                schema_hash=None, prompt_hashes={}, sampling_profiles={}, limits={}, measured_placement=None,
                measured_resources=None, capability_report=None, evaluation_report=None, release_ready=False)
    base.update(over)
    return base


@pytest.mark.parametrize("over", [{"conversion_revision": "main"}, {"conversion_revision": "3885219"},
                                  {"upstream_revision": "latest"}, {"gguf_sha256": "XYZ"}, {"gguf_bytes": 0},
                                  {"schema_version": "brain-manifest-0.2"}])
def test_B28_mutable_or_malformed_manifest_rejected(over):
    with pytest.raises(ValueError):
        Manifest(**manifest_fields(**over))


def test_B28_artifact_bytes_verified(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"hello")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"exe")
    m = Manifest(**manifest_fields(gguf_sha256=sha256_file(model)[0], executable_sha256=sha256_file(exe)[0]))
    verify_artifacts(m, model_path=model, server_binary=exe)
    model.write_bytes(b"hellp")
    with pytest.raises(ManifestError):
        verify_artifacts(m, model_path=model, server_binary=exe)
    model.write_bytes(b"hello")
    exe.write_bytes(b"exe2")
    with pytest.raises(ManifestError):
        verify_artifacts(m, model_path=model, server_binary=exe)
    with pytest.raises(ManifestError):
        verify_artifacts(m, model_path=tmp_path / "missing.gguf", server_binary=exe)


def test_B28_manifest_roundtrip_and_duplicate_keys(tmp_path):
    m = Manifest(**manifest_fields())
    p = tmp_path / "manifest.json"
    save_manifest(m, p)
    assert load_manifest(p) == m
    p.write_text(p.read_text()[:-2] + ',"release_ready": true}\n')
    with pytest.raises(ManifestError):
        load_manifest(p)
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "absent.json")


def test_B28_committed_manifest_is_pinned_and_not_release_ready():
    p = ROOT / "artifacts/brain/qwen35-9b-q4/manifest.json"
    if not p.exists():
        pytest.skip("manifest not provisioned on this machine")
    m = load_manifest(p)
    assert m.release_ready is False and len(m.conversion_revision) == 40 and len(m.runtime_commit) == 40
    assert m.gguf_sha256 == m.gguf_upstream_sha256
    assert m.measured_placement["offloaded_layers"] == m.measured_placement["total_layers"]


# ----------------------------------------------------------------------------- B35 no runtime routing switch

def test_B35_startup_rejects_manifest_for_other_candidate(tmp_path):
    import asyncio

    from brain.service import BrainService
    from brain.parse import SentencePolicy
    from brain.transport import LocalTransport

    cfg = load_config(CFG)
    m = Manifest(**manifest_fields(candidate_id="qwen3-4b-q4"))
    svc = BrainService(cfg, transport=LocalTransport("http://127.0.0.1:8080", "k" * 30), manifest=m,
                       sentence_policy=SentencePolicy(), supervisor=object(), api_key="k" * 30)
    with pytest.raises(BrainError) as e:
        asyncio.run(svc.startup(verify_bytes=False))
    assert e.value.code == ErrorCode.ARTIFACT_MISMATCH


def test_B35_each_candidate_has_its_own_manifest_path():
    paths = {load_config(ROOT / "config" / n).manifest_path for n in
             ("brain-local-9b.yaml", "brain-local-4b.yaml", "brain-comparison-qwen3-4b.yaml")}
    assert len(paths) == 3
