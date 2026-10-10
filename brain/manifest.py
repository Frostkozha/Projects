"""Candidate manifest: artifact identity, checksums and measured records (Brain plan v0.3, section 13).

Runtime loading verifies actual bytes. ``release_ready`` stays false until evaluated records exist.
No wildcard/latest revision, automatic update or user-uploaded bundle is accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field, ValidationError, field_validator

from contracts.draft import SHA256, MachineID, StrictModel
from contracts.json_codec import load_object

MANIFEST_SCHEMA = "brain-manifest-0.3"
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Manifest(StrictModel):
    schema_version: str = Field(pattern=r"^brain-manifest-0\.3$")
    candidate_id: MachineID
    model_id: str
    upstream_revision: Optional[str]
    conversion_repo: str
    conversion_revision: str
    gguf_filename: str
    gguf_bytes: Annotated[int, Field(gt=0)]
    gguf_sha256: SHA256
    gguf_upstream_sha256: Optional[SHA256]
    license_reference: str
    runtime_release: str
    runtime_commit: str
    executable_sha256: SHA256
    build_environment: dict[str, Any]
    chat_template_sha256: Optional[SHA256]
    launch_args_redacted: list[str]
    schema_hash: Optional[SHA256]
    prompt_hashes: dict[str, SHA256]
    sampling_profiles: dict[str, dict[str, Any]]
    limits: dict[str, Any]
    measured_placement: Optional[dict[str, Any]]
    measured_resources: Optional[dict[str, Any]]
    capability_report: Optional[dict[str, Any]]
    evaluation_report: Optional[str]
    release_ready: bool

    @field_validator("conversion_revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not _FULL_REVISION.match(v):
            raise ValueError("conversion_revision must be a full 40-hex commit, never a branch or tag")
        return v

    @field_validator("upstream_revision")
    @classmethod
    def _pinned_upstream(cls, v):
        if v is not None and not _FULL_REVISION.match(v):
            raise ValueError("upstream_revision must be a full 40-hex commit")
        return v


def load_manifest(path: Path) -> Manifest:
    try:
        raw = path.read_bytes()
        return Manifest.model_validate(load_object(raw, max_bytes=4 * 1024 * 1024))
    except FileNotFoundError:
        raise ManifestError("ARTIFACT_MISMATCH: manifest missing") from None
    except (ValueError, ValidationError):
        raise ManifestError("ARTIFACT_MISMATCH: manifest invalid") from None


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_artifacts(manifest: Manifest, *, model_path: Path, server_binary: Path) -> None:
    """Recompute the actual bytes of the model and executable. Raises ManifestError(ARTIFACT_MISMATCH)."""
    if not model_path.is_file() or model_path.name != manifest.gguf_filename:
        raise ManifestError("ARTIFACT_MISMATCH: model file")
    if model_path.stat().st_size != manifest.gguf_bytes:
        raise ManifestError("ARTIFACT_MISMATCH: model size")
    if sha256_file(model_path)[0] != manifest.gguf_sha256:
        raise ManifestError("ARTIFACT_MISMATCH: model checksum")
    if not server_binary.is_file() or sha256_file(server_binary)[0] != manifest.executable_sha256:
        raise ManifestError("ARTIFACT_MISMATCH: runtime executable")
