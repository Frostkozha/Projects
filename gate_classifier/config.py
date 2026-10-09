"""Configuration validation, library registry and config checksum (spec sections 8.1, 10, 12, 13)."""

from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import LIBRARY_IDS, RISK_LABELS

OperatingMode = Literal["fixture", "development", "production"]

REQUIRED_REPLY_KEYS = (
    "real_person",
    "assessed_work",
    "private_data",
    "instruction_override",
    "outside_course",
    "nonmedical",
    "clarify_topic",
    "safety_clarify",
    "unsupported_feature",
    "no_source",
    "service_unavailable",
    "maintenance",
    "exam_disabled",
    "emergency",
    "self_harm",
)
ALLOWED_PLACEHOLDERS = {"approved_emergency_contact", "approved_student_support_contact"}
AI_NOTICE = "AI-generated study aid. Check against your course material."
_PLACEHOLDER_RE = re.compile(r"placeholder|example|todo|tbd|xxx|\{|\}", re.IGNORECASE)


class ConfigError(ValueError):
    pass


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskThreshold(_Cfg):
    uncertainty: float
    block: float

    @model_validator(mode="after")
    def _order(self):
        for v in (self.uncertainty, self.block):
            if not (0.0 <= v <= 1.0):
                raise ValueError("thresholds must lie in [0,1]")
        if self.uncertainty >= self.block:
            raise ValueError("uncertainty threshold must be below block threshold")
        return self


class Thresholds(_Cfg):
    risks: dict[str, RiskThreshold]
    topic: float = 0.70
    mode: float = 0.70
    library_preference: float = 0.50
    max_preferred_libraries: int = 3

    @field_validator("risks")
    @classmethod
    def _keys(cls, v):
        if set(v) != set(RISK_LABELS):
            raise ValueError(f"risk thresholds must cover exactly {RISK_LABELS}")
        return v

    @model_validator(mode="after")
    def _range(self):
        for name in ("topic", "mode", "library_preference"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} threshold must lie in [0,1]")
        if not (0 <= self.max_preferred_libraries <= 6):
            raise ValueError("max_preferred_libraries must be 0..6")
        return self


class EncoderConfig(_Cfg):
    model_id: str = "intfloat/e5-small-v2"
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    local_path: Optional[str] = None
    prefix: str = "query: "
    pooling: Literal["mean"] = "mean"
    normalization: Literal["l2"] = "l2"
    dimension: int = 384
    max_tokens: int = 512
    torch_threads: int = 2


class PrivacyConfig(_Cfg):
    name_detector: Literal["rules", "spacy"] = "rules"
    spacy_model_path: Optional[str] = None
    spacy_model_version: Optional[str] = None


class Limits(_Cfg):
    max_body_bytes: int = 32 * 1024
    max_chars: int = 4000
    max_current_tokens: int = 320
    max_context_tokens: int = 128
    max_total_tokens: int = 512


class RuntimeConfig(_Cfg):
    deadline_seconds: float = 2.0
    queue_capacity: int = 32
    workers: Literal[1] = 1
    session_ttl_minutes: int = 30
    stuck_worker_seconds: float = 10.0
    max_worker_restarts: int = 3


class AuditConfig(_Cfg):
    metadata_sink: Literal["memory", "jsonl"] = "memory"
    metadata_path: Optional[str] = None
    restricted_text_store_enabled: bool = False
    audit_metadata_retention_days: Optional[int] = None
    restricted_text_retention_days: Optional[int] = None


class AlertConfig(_Cfg):
    max_attempts: int = 4
    base_backoff_seconds: float = 0.05
    adapter: Literal["memory", "jsonl"] = "memory"
    queue_path: Optional[str] = None


class GateConfig(_Cfg):
    config_schema: Literal["gate-config-0.2"] = "gate-config-0.2"
    operating_mode: OperatingMode = "fixture"
    policy_version: str = "gate-policy-0.2"
    rules_version: str = "gate-rules-0.2"
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    limits: Limits = Field(default_factory=Limits)
    thresholds: Thresholds
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    bundle_path: Optional[str] = None
    replies: dict[str, str]
    contacts: dict[str, str]
    service_tokens_env: str = "GATE_SERVICE_TOKEN"

    @field_validator("replies")
    @classmethod
    def _replies(cls, v):
        missing = set(REQUIRED_REPLY_KEYS) - set(v)
        if missing:
            raise ValueError(f"missing reply keys: {sorted(missing)}")
        for key, text in v.items():
            names = {f[1] for f in string.Formatter().parse(text) if f[1]}
            if names - ALLOWED_PLACEHOLDERS:
                raise ValueError(f"reply {key} uses unknown placeholders")
        return v

    @field_validator("contacts")
    @classmethod
    def _contacts(cls, v):
        if set(v) != ALLOWED_PLACEHOLDERS:
            raise ValueError(f"contacts must define exactly {sorted(ALLOWED_PLACEHOLDERS)}")
        return v

    # -- checksum covers every setting the evaluated bundle depends on
    def decision_payload(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "rules_version": self.rules_version,
            "encoder": self.encoder.model_dump(exclude={"local_path", "torch_threads"}),
            "limits": self.limits.model_dump(),
            "thresholds": self.thresholds.model_dump(),
        }

    def config_sha256(self) -> str:
        blob = json.dumps(self.decision_payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def placeholder_contacts(self) -> list[str]:
        return [k for k, v in self.contacts.items() if not v.strip() or _PLACEHOLDER_RE.search(v)]

    def render_reply(self, key: str) -> str:
        return self.replies[key].format(**self.contacts)

    def production_problems(self) -> list[str]:
        problems = []
        if self.placeholder_contacts():
            problems.append("placeholder_contacts")
        if not self.encoder.revision or not self.encoder.tokenizer_revision:
            problems.append("encoder_revision_unpinned")
        if self.privacy.name_detector != "spacy" or not self.privacy.spacy_model_version:
            problems.append("privacy_model_unpinned")
        if self.audit.metadata_sink != "jsonl" or self.audit.audit_metadata_retention_days is None:
            problems.append("audit_retention_unconfigured")
        if self.audit.restricted_text_store_enabled and self.audit.restricted_text_retention_days is None:
            problems.append("restricted_text_retention_unconfigured")
        if self.alerts.adapter == "memory":
            problems.append("alert_adapter_not_configured")
        return problems


def load_config(path: str | Path) -> GateConfig:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return GateConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"invalid gate configuration: {type(exc).__name__}") from exc


# --------------------------------------------------------------------------- library registry


class LibraryEntry(_Cfg):
    category: str
    enabled: bool
    indexed: bool = False
    approved: bool = False
    archived: bool = False

    @property
    def active(self) -> bool:
        return self.enabled and self.indexed and self.approved and not self.archived


class CourseEntry(_Cfg):
    tenant_id: str
    course_kind: Literal["histology_pilot", "reviewed_release"] = "histology_pilot"
    libraries: dict[str, LibraryEntry]

    @field_validator("libraries")
    @classmethod
    def _ids(cls, v):
        if set(v) != set(LIBRARY_IDS):
            raise ValueError("each course must declare lib1..lib6")
        return v

    @model_validator(mode="after")
    def _clinical_disabled(self):
        if self.course_kind == "histology_pilot":
            for lib in ("lib3", "lib4"):
                if self.libraries[lib].enabled:
                    raise ValueError("clinical protocol libraries lib3/lib4 must stay disabled in the histology pilot")
        return self


class LibraryRegistry(_Cfg):
    registry_version: str
    courses: dict[str, CourseEntry]

    def active_libraries(self, course_id: str, authorized: Optional[set[str]] = None) -> tuple[str, ...]:
        course = self.courses.get(course_id)
        if course is None:
            return ()
        libs = [lib for lib, e in course.libraries.items() if e.active]
        if authorized is not None:
            libs = [lib for lib in libs if lib in authorized]
        return tuple(sorted(libs))

    def enabled_labels(self, course_id: str) -> tuple[str, ...]:
        course = self.courses.get(course_id)
        if course is None:
            return ()
        return tuple(sorted(lib for lib, e in course.libraries.items() if e.enabled))


def load_registry(path: str | Path) -> LibraryRegistry:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return LibraryRegistry.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"invalid library registry: {type(exc).__name__}") from exc
