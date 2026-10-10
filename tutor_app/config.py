"""Validated coordinator configuration (Integration plan v0.3, sections 2, 10, 11).

Profiles are operator configuration; there is no automatic downgrade:

* ``fixture``     synthetic data and deterministic adapters (tests, CI);
* ``development`` real retriever, Brain and verifier with the gate in its fixture scoring mode, because no
                  trained gate bundle exists yet. Clearly labelled; can never report real_model readiness;
* ``real_model``  every component on pinned real artifacts (requires a trained gate bundle);
* ``student_release`` real_model plus institutional identity, privacy, retention, welfare and evaluation records.
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
CONFIG_SCHEMA = "tutor-config-0.3"
Profile = Literal["fixture", "development", "real_model", "student_release"]


class TutorConfigError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in seen:
            raise TutorConfigError(f"duplicate configuration key: {key}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


class NoticeConfig(StrictModel):
    version: MachineID
    text: Annotated[str, Field(min_length=1, max_length=4000)]


class JwtConfig(StrictModel):
    issuer: str
    audience: str
    algorithm: Literal["RS256", "ES256", "EdDSA"]
    public_key_file: str
    course_roles: dict[str, list[str]]
    leeway_seconds: Annotated[int, Field(ge=0, le=60)] = 30


class IdentityConfig(StrictModel):
    adapter: Literal["dev_cookie", "trusted_proxy_jwt"]
    jwt: Optional[JwtConfig] = None

    @model_validator(mode="after")
    def _jwt(self) -> Self:
        if (self.adapter == "trusted_proxy_jwt") != (self.jwt is not None):
            raise ValueError("trusted_proxy_jwt requires the jwt section (and only then)")
        return self


class ComponentsConfig(StrictModel):
    gate_config: str
    library_registry: str
    gate_scoring: Literal["fixture", "bundle"]
    gate_tokenizer: Literal["fixture", "e5"]
    retriever: Literal["fixture", "real"]
    retriever_profile: Optional[str]
    brain: Literal["fixture", "local"]
    brain_config: Optional[str]
    verifier_profile: str


class CapacityConfig(StrictModel):
    generated_active: Literal[1]
    generated_waiting: Annotated[int, Field(ge=0, le=8)]
    generated_wait_seconds: Annotated[float, Field(gt=0, le=8)]
    per_session_pending: Literal[1]
    per_identity_pending: Annotated[int, Field(ge=1, le=2)]
    input_capacity: Annotated[int, Field(ge=1, le=64)]


class DeadlineConfig(StrictModel):
    whole_seconds: Annotated[float, Field(gt=0, le=60)]
    gate_seconds: Annotated[float, Field(gt=0, le=2)]
    context_seconds: Annotated[float, Field(gt=0, le=2)]
    retriever_seconds: Annotated[float, Field(gt=0, le=5)]
    brain_seconds: Annotated[float, Field(gt=0, le=30)]
    verifier_seconds: Annotated[float, Field(gt=0, le=60)]
    delivery_seconds: Annotated[float, Field(gt=0, le=2)]


class BreakerConfig(StrictModel):
    failures: Literal[3]
    window_seconds: Literal[60]
    cooldown_seconds: Literal[30]


class FeatureConfig(StrictModel):
    tutor_clue: Literal[False]
    topic_indicator: bool
    a4_educational: Literal[False]


class TutorConfig(StrictModel):
    schema_version: Literal["tutor-config-0.3"]
    profile: Profile
    tenant_id: MachineID
    course_id: MachineID
    host: Literal["127.0.0.1", "::1"]
    port: Annotated[int, Field(ge=1024, le=65535)]
    approved_roots: Annotated[list[str], Field(min_length=1)]
    db_path: str
    fingerprint_key_file: str
    cookie_key_file: str
    notice: NoticeConfig
    identity: IdentityConfig
    components: ComponentsConfig
    item_bank: str
    topic_registry: str
    capacity: CapacityConfig
    deadlines: DeadlineConfig
    breaker: BreakerConfig
    features: FeatureConfig
    response_buffer_seconds: Annotated[int, Field(ge=0, le=300)]
    session_ttl_minutes: Literal[30]

    @model_validator(mode="after")
    def _profile_rules(self) -> Self:
        c = self.components
        if self.profile == "fixture":
            if c.retriever != "fixture" or c.brain != "fixture" or c.gate_scoring != "fixture":
                raise ValueError("fixture profile uses fixture adapters only")
        else:
            if c.retriever != "real" or c.brain != "local" or not c.retriever_profile or not c.brain_config:
                raise ValueError("non-fixture profiles need the real retriever and local Brain")
        if self.profile in ("real_model", "student_release") and (c.gate_scoring != "bundle"
                                                                  or c.gate_tokenizer != "e5"):
            raise ValueError("real_model/student_release require the trained gate bundle and E5 tokenizer")
        if self.profile == "student_release" and self.identity.adapter != "trusted_proxy_jwt":
            raise ValueError("student_release requires the institutional identity adapter")
        return self

    def roots(self) -> list[Path]:
        return [_abs(r) for r in self.approved_roots]

    def path(self, value: str) -> Path:
        p = _abs(value)
        if not any(p == r or r in p.parents for r in self.roots()):
            raise TutorConfigError("path outside approved roots")
        return p


def _abs(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_tutor_config(path: str | Path) -> TutorConfig:
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506
    except (OSError, yaml.YAMLError) as exc:
        raise TutorConfigError(f"configuration unreadable: {type(exc).__name__}") from None
    if not isinstance(raw, dict):
        raise TutorConfigError("configuration must be a mapping")
    try:
        cfg = TutorConfig.model_validate(raw, strict=True)
    except ValidationError as exc:
        locs = sorted({".".join(str(x) for x in e["loc"]) for e in exc.errors()})
        raise TutorConfigError("invalid configuration fields: " + ", ".join(locs)) from None
    c = cfg.components
    for p in (cfg.db_path, cfg.fingerprint_key_file, cfg.cookie_key_file, c.gate_config, c.library_registry,
              c.verifier_profile, cfg.item_bank, cfg.topic_registry,
              *(x for x in (c.retriever_profile, c.brain_config) if x)):
        cfg.path(p)
    if cfg.identity.jwt:
        cfg.path(cfg.identity.jwt.public_key_file)
    return cfg


def read_secret(path: Path, *, create: bool = False) -> bytes:
    """Operator secret (32+ random bytes, hex/urlsafe text). Development profiles may create it locally."""
    if not path.exists():
        if not create:
            raise TutorConfigError("secret file missing")
        import secrets  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48) + "\n", encoding="ascii")
    value = path.read_text(encoding="ascii").strip()
    if len(value) < 32 or "\n" in value:
        raise TutorConfigError("secret file invalid")
    return value.encode("ascii")
