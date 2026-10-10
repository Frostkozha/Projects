"""One owned llama-server process (Brain plan v0.3, sections 5, 6 and 11).

Starts only the configured executable with an argument array (shell=False), in its own process group,
with bounded startup/shutdown. Never kills a process it did not start: an occupied port is a failure.
Every launch flag is confirmed against the executable's ``--help`` before use; a missing flag fails
validation rather than being silently dropped. The CLI ``start`` command and the coordinator-managed
service share this single implementation.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .config import BrainConfig
from .schema import BrainError, ErrorCode

IS_WINDOWS = sys.platform == "win32"
_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_BUFFER = re.compile(r"(CUDA\d+|CPU_Mapped|CUDA_Host|CPU) (model|KV|RS|compute) buffer size =\s*([\d.]+) MiB")
_OOM = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|ErrorOutOfDeviceMemory", re.IGNORECASE)


def server_args(cfg: BrainConfig, *, model_path: Optional[Path] = None, port: Optional[int] = None) -> list[str]:
    """The single capability-tested command (section 6), mapped from validated configuration."""
    rt = cfg.runtime
    return [
        "--model", str(model_path or cfg.path(cfg.artifacts.model_path)),
        "--alias", rt.alias, "--host", rt.host, "--port", str(port or rt.port),
        "--ctx-size", str(rt.context_tokens), "--parallel", str(rt.parallel),
        "--n-gpu-layers", str(rt.gpu_layers_requested), "--fit", rt.fit,
        "--threads", str(rt.threads), "--threads-batch", str(rt.threads_batch),
        "--batch-size", str(rt.batch_size), "--ubatch-size", str(rt.ubatch_size),
        "--flash-attn", rt.flash_attention,
        "--cache-type-k", rt.cache_type_k, "--cache-type-v", rt.cache_type_v,
        "--jinja", "--reasoning", rt.reasoning, "--no-webui",
        "--no-cache-prompt", "--cache-ram", str(rt.cache_ram_mib), "--no-context-shift",
        "--no-mmproj", "--offline",
        "--api-key-file", str(cfg.path(rt.api_key_file)), "--metrics",
    ]


def redact_args(args: list[str]) -> list[str]:
    """Launch args for the manifest: no key path, model path reduced to its file name."""
    out = list(args)
    for flag in ("--api-key-file",):
        if flag in out:
            out[out.index(flag) + 1] = "<redacted>"
    if "--model" in out:
        i = out.index("--model") + 1
        out[i] = Path(out[i]).name
    return out


def _flags(args: list[str]) -> list[str]:
    return [a for a in args if a.startswith("--")]


def _env(cfg: BrainConfig) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "HF_", "HUGGING"))}
    dirs = [str(cfg.path(cfg.artifacts.server_binary).parent)] + [str(cfg.path(d)) for d in
                                                                    cfg.artifacts.runtime_library_dirs]
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _run(cfg: BrainConfig, extra: list[str], timeout: float = 30) -> str:
    binary = cfg.path(cfg.artifacts.server_binary)
    proc = subprocess.run([str(binary), *extra], capture_output=True, text=True, timeout=timeout, shell=False,
                          env=_env(cfg), errors="replace")
    return proc.stdout + proc.stderr


def runtime_version(cfg: BrainConfig) -> dict:
    out = _run(cfg, ["--version"])
    m = re.search(r"version:\s*\S+\s*\(build (\d+), commit ([0-9a-f]+)\)", out)
    c = re.search(r"built with (.+)", out)
    if not m:
        raise BrainError(ErrorCode.RUNTIME_INCOMPATIBLE)
    return {"build": m.group(1), "commit_short": m.group(2), "compiler": c.group(1).strip() if c else None}


def check_flags(cfg: BrainConfig) -> list[str]:
    """Return missing flags (empty list = every configured flag is supported by this executable)."""
    help_text = _run(cfg, ["--help"])
    return [f for f in sorted(set(_flags(server_args(cfg)))) if not re.search(re.escape(f) + r"(?![\w-])", help_text)]


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@dataclass
class Placement:
    offloaded_layers: int
    total_layers: int
    buffers_mib: dict

    @property
    def full(self) -> bool:
        return self.offloaded_layers == self.total_layers


def parse_load_log(lines: list[str]) -> Placement:
    """Actual placement from loader log lines. OOM -> MEMORY_LIMIT; no placement line -> RUNTIME_INCOMPATIBLE."""
    offload, buffers = None, {}
    for line in lines:
        if _OOM.search(line):
            raise BrainError(ErrorCode.MEMORY_LIMIT)
        if m := _OFFLOAD.search(line):
            offload = (int(m.group(1)), int(m.group(2)))
        if m := _BUFFER.search(line):
            buffers[f"{m.group(1)}:{m.group(2)}"] = float(m.group(3))
    if offload is None:
        raise BrainError(ErrorCode.RUNTIME_INCOMPATIBLE)
    return Placement(offload[0], offload[1], buffers)


class Supervisor:
    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self.started_at: Optional[float] = None

    # ---------------------------------------------------------------- placement preflight

    def placement_preflight(self, *, timeout: Optional[float] = None) -> Placement:
        """Load the model once with verbose loader logs, read the actual layer placement, then stop.

        No request is ever sent to this preflight process, so its verbose log holds no prompt content.
        """
        rt = self.cfg.runtime
        port = rt.port + 1
        if port_in_use(rt.host, port):
            raise BrainError(ErrorCode.MODEL_UNAVAILABLE)
        args = server_args(self.cfg, port=port) + ["-lv", "4"]
        proc = self._spawn(args, stdout=subprocess.PIPE)
        deadline = time.monotonic() + (timeout or rt.startup_timeout_seconds)
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                lines.append(line)
                if _OOM.search(line) or "listening on" in line or "model loaded" in line:
                    break
                if time.monotonic() > deadline:
                    raise BrainError(ErrorCode.MODEL_UNAVAILABLE)
        finally:
            self._terminate(proc)
        return parse_load_log(lines)

    # ---------------------------------------------------------------- owned server

    def _spawn(self, args: list[str], *, stdout) -> subprocess.Popen:
        binary = self.cfg.path(self.cfg.artifacts.server_binary)
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen([str(binary), *args], stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                shell=False, env=_env(self.cfg), cwd=str(binary.parent), **kwargs)

    def start(self, api_key: str) -> None:
        rt = self.cfg.runtime
        if self.running:
            return
        if port_in_use(rt.host, rt.port):
            # Never kill an unrelated process because its port is occupied.
            raise BrainError(ErrorCode.MODEL_UNAVAILABLE)
        log_path = self.cfg.path(self.cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        self.proc = self._spawn(server_args(self.cfg), stdout=self._log)
        self.started_at = time.monotonic()
        deadline = self.started_at + rt.startup_timeout_seconds
        url = f"http://{rt.host}:{rt.port}/health"
        with httpx.Client(trust_env=False, follow_redirects=False,
                          headers={"Authorization": "Bearer " + api_key}) as client:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    code = ErrorCode.MEMORY_LIMIT if self._log_mentions_oom(log_path) else ErrorCode.MODEL_UNAVAILABLE
                    self.stop()
                    raise BrainError(code)
                try:
                    if client.get(url, timeout=1.0).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
        self.stop()
        raise BrainError(ErrorCode.MODEL_UNAVAILABLE)

    @staticmethod
    def _log_mentions_oom(path: Path) -> bool:
        try:
            tail = path.read_bytes()[-65536:].decode("utf-8", errors="replace")
        except OSError:
            return False
        return bool(_OOM.search(tail))

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def _terminate(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                if IS_WINDOWS:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=self.cfg.runtime.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                if IS_WINDOWS:
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=self.cfg.runtime.shutdown_timeout_seconds)
        if proc.stdout:
            proc.stdout.close()

    def stop(self) -> None:
        """Stop only the owned process (bounded graceful, then forced) and wait for its exit."""
        if self.proc is not None:
            self._terminate(self.proc)
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None
