"""Strict local profile configuration (Brain plan v0.3, sections 3, 6 and appendix F).

Unknown keys, booleans where integers are expected, multiple slots, token-budget mismatches and paths
outside the approved roots are rejected. The configuration does not establish hardware fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from pydantic import Field, ValidationError, model_validator

from contracts.draft import MachineID, StrictModel

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "brain-config-0.3"

PosInt = Annotated[int, Field(gt=0)]


class ConfigError(ValueError):
    """Configuration rejected; message is a stable reason."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise ConfigError(f"duplicate configuration key: {key}")
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class RuntimeConfig(StrictModel):
    host: Literal["127.0.0.1", "::1"]
    port: Annotated[int, Field(ge=1024, le=65535)]
    alias: MachineID
    api_key_file: str
    parallel: Literal[1]
    context_tokens: PosInt
    gpu_layers_requested: PosInt
    threads: Annotated[int, Field(ge=1, le=64)]
    threads_batch: Annotated[int, Field(ge=1, le=64)]
    batch_size: PosInt
    ubatch_size: PosInt
    flash_attention: Literal["on", "off"]
    cache_type_k: Literal["f16"]
    cache_type_v: Literal["f16"]
    reasoning: Literal["off"]
    cache_prompt: Literal[False]
    context_shift: Literal[False]
    web_ui: Literal[False]
    fit: Literal["off"]
    cache_ram_mib: Literal[0]
    startup_timeout_seconds: Annotated[float, Field(gt=0, le=600)]
    shutdown_timeout_seconds: Annotated[float, Field(gt=0, le=60)]
    cancel_idle_check_seconds: Annotated[float, Field(gt=0, le=10)]
    idle_probe_interval_seconds: Annotated[float, Field(ge=60)]

    @model_validator(mode="after")
    def _batches(self) -> Self:
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size exceeds batch_size")
        return self


class LimitsConfig(StrictModel):
    prompt_tokens: PosInt
    output_tokens: PosInt
    reserve_tokens: PosInt
    queue_capacity: Annotated[int, Field(ge=0, le=8)]
    queue_wait_seconds: Annotated[float, Field(gt=0, le=8)]
    brain_deadline_seconds: Annotated[float, Field(gt=0, le=30)]
    response_bytes: Annotated[int, Field(gt=0, le=262144)]
    min_free_vram_mib: Annotated[int, Field(ge=0)]


class SamplingConfig(StrictModel):
    profile_version: MachineID
    temperature: Annotated[float, Field(ge=0, le=2)]
    top_p: Annotated[float, Field(gt=0, le=1)]
    top_k: Annotated[int, Field(ge=0)]
    min_p: Annotated[float, Field(ge=0, le=1)]
    presence_penalty: float
    frequency_penalty: float
    repeat_penalty: Annotated[float, Field(gt=0)]
    seed: Optional[Annotated[int, Field(ge=0)]] = None


class FeaturesConfig(StrictModel):
    tutor_clue_enabled: bool


class ArtifactsConfig(StrictModel):
    model_path: str
    server_binary: str
    runtime_library_dirs: list[str]


class BrainConfig(StrictModel):
    schema_version: Literal["brain-config-0.3"]
    profile: Literal["real_model", "fixture"]
    candidate_id: MachineID
    manifest_path: str
    prompt_version: MachineID
    approved_roots: Annotated[list[str], Field(min_length=1)]
    artifacts: ArtifactsConfig
    runtime: RuntimeConfig
    limits: LimitsConfig
    sampling: SamplingConfig
    features: FeaturesConfig
    log_path: str

    @model_validator(mode="after")
    def _budget(self) -> Self:
        lim, rt = self.limits, self.runtime
        if lim.prompt_tokens + lim.output_tokens + lim.reserve_tokens != rt.context_tokens:
            raise ValueError("token budget mismatch: prompt + output + reserve must equal context_tokens")
        return self

    # ---- path resolution (once, against the repository root; must stay under approved roots)

    def roots(self) -> list[Path]:
        return [_abs(r) for r in self.approved_roots]

    def path(self, value: str) -> Path:
        p = _abs(value)
        if not any(p == r or r in p.parents for r in self.roots()):
            raise ConfigError("path outside approved roots")
        return p


def _abs(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_config(path: str | Path) -> BrainConfig:
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"configuration unreadable: {type(exc).__name__}") from None
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a mapping")
    try:
        cfg = BrainConfig.model_validate(raw, strict=True)
    except ValidationError as exc:
        locs = sorted({".".join(str(x) for x in e["loc"]) for e in exc.errors()})
        raise ConfigError("invalid configuration fields: " + ", ".join(locs)) from None
    for p in (cfg.manifest_path, cfg.runtime.api_key_file, cfg.artifacts.model_path, cfg.artifacts.server_binary,
              cfg.log_path, *cfg.artifacts.runtime_library_dirs):
        cfg.path(p)
    if cfg.features.tutor_clue_enabled and not (ROOT / "prompts" / "tutor_clue.reviewed").exists():
        raise ConfigError("tutor_clue requires a reviewed clue prompt and leakage evaluation record")
    return cfg
