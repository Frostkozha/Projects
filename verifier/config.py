"""Verifier profile, threshold profile and versioned rule files (Verifier spec v0.2, sections 5, 7, 8, 11, 12)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class ProfileError(ValueError):
    pass


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ThresholdProfile(_Cfg):
    """Tc / T_lo / T_hi for three-class NLI. Fixture values must never serve students."""

    profile_id: str
    model_digest: Optional[str] = None
    tokenizer_digest: Optional[str] = None
    calibration_version: Optional[str] = None
    Tc: float
    T_lo: float
    T_hi: float
    development_split_digest: Optional[str] = None
    approval_record_id: Optional[str] = None
    evaluation_report_digest: Optional[str] = None
    release_ready: bool = False

    @model_validator(mode="after")
    def _ranges(self):
        for v in (self.Tc, self.T_lo, self.T_hi):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError("thresholds must be finite numbers")
        if not 0 < self.Tc < 1:
            raise ValueError("0 < Tc < 1 required")
        if not 0 <= self.T_lo < self.T_hi <= 1:
            raise ValueError("0 <= T_lo < T_hi <= 1 required")
        if self.release_ready and not (self.approval_record_id and self.evaluation_report_digest
                                       and self.development_split_digest and self.model_digest):
            raise ValueError("release_ready requires approval, evaluation, split and model digests")
        return self

    def version(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class NLIProfile(_Cfg):
    model_id: str = "cross-encoder/nli-deberta-v3-xsmall"
    revision: Optional[str] = None
    local_path: Optional[str] = None
    expected_labels: dict[str, str] = Field(default_factory=lambda: {"0": "contradiction", "1": "entailment",
                                                                      "2": "neutral"})
    weights_sha256: Optional[str] = None
    max_premise_tokens: int = 384
    max_hypothesis_tokens: int = 96
    max_pair_tokens: int = 512
    reserve_tokens: int = 32
    batch_size: int = Field(8, ge=1, le=8)
    torch_threads: int = 4

    @model_validator(mode="after")
    def _budget(self):
        if self.max_premise_tokens + self.max_hypothesis_tokens + self.reserve_tokens > self.max_pair_tokens:
            raise ValueError("premise + hypothesis + reserve exceed the pair ceiling")
        return self


class SegmenterProfile(_Cfg):
    kind: Literal["spacy_model", "spacy_sentencizer"] = "spacy_sentencizer"
    model_path: Optional[str] = None  # pinned en_core_web_sm package path for production
    model_version: Optional[str] = None


class RuntimeProfile(_Cfg):
    deadline_seconds: float = 5.0
    queue_capacity: int = 16
    max_pairs: int = 80
    worker: Literal["in_process", "subprocess"] = "in_process"
    hard_recovery_seconds: float = 20.0
    max_body_bytes: int = 256 * 1024


class VerifierProfile(_Cfg):
    schema_version: Literal["verifier-profile-0.2"] = "verifier-profile-0.2"
    profile_version: str
    operating_mode: Literal["fixture", "real_model_development", "production"] = "fixture"
    git_commit: Optional[str] = None
    nli: NLIProfile = Field(default_factory=NLIProfile)
    segmenter: SegmenterProfile = Field(default_factory=SegmenterProfile)
    runtime: RuntimeProfile = Field(default_factory=RuntimeProfile)
    thresholds_path: str
    rules_dir: str = "config/verifier"
    features: dict[str, bool] = Field(default_factory=lambda: {"a4_educational": False, "virtual_patient": False,
                                                               "open_ended": False, "images": False,
                                                               "experimental_judge": False})

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q


class Rules(_Cfg):
    """Versioned rule files (allowlist, units, antonyms, cues, notices). Hashes enter verifier_version."""

    allowlist: tuple[str, ...]
    units: dict
    antonyms: tuple[tuple[str, str], ...]
    cues: dict
    notices: dict[str, str]
    reviewed: bool
    digests: dict[str, str]


RULE_FILES = ("allowlist.json", "units.json", "antonyms.json", "cues.json", "notices.json")


def load_rules(rules_dir: Path) -> Rules:
    data, digests = {}, {}
    for name in RULE_FILES:
        p = rules_dir / name
        data[name] = json.loads(p.read_text(encoding="utf-8"))
        digests[name] = file_sha256(p)
    reviewed = all(data[n].get("review", {}).get("status") == "reviewed" for n in RULE_FILES)
    return Rules(
        allowlist=tuple(data["allowlist.json"]["phrases"]),
        units=data["units.json"],
        antonyms=tuple(tuple(p) for p in data["antonyms.json"]["pairs"]),
        cues=data["cues.json"],
        notices=data["notices.json"]["notices"],
        reviewed=reviewed,
        digests=digests,
    )


def load_profile(path: str | Path) -> VerifierProfile:
    try:
        return VerifierProfile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ProfileError(f"invalid verifier profile: {type(exc).__name__}") from exc


def load_thresholds(path: str | Path) -> ThresholdProfile:
    try:
        return ThresholdProfile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ProfileError(f"invalid threshold profile: {type(exc).__name__}") from exc
