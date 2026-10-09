"""Write a model bundle: JSON manifest + NPZ arrays with SHA-256 per file (spec section 18)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gate_classifier.config import GateConfig
from gate_classifier.heads import BUNDLE_SCHEMA, LinearHead, sha256_file
from gate_classifier.policy_rules import RULES_VERSION

ROOT = Path(__file__).resolve().parents[1]


def build_encoder(cfg: GateConfig, kind: str):
    from gate_classifier.encoder import E5Encoder, FixtureEncoder  # noqa: PLC0415

    if kind == "fixture":
        return FixtureEncoder(cfg.encoder.dimension, cfg.encoder.prefix)
    enc = cfg.encoder
    return E5Encoder(enc.local_path or "", enc.model_id, enc.revision or "", enc.tokenizer_revision or "",
                     enc.prefix, enc.dimension, enc.max_tokens, enc.torch_threads)


def _lockfile_sha() -> Optional[str]:
    lock = ROOT / "requirements.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


def write_bundle(out_dir: str | Path, heads: dict[str, LinearHead], *, cfg: GateConfig, encoder,
                 operating_mode: str, dataset: dict, training: Optional[dict] = None,
                 evaluation: Optional[dict] = None, privacy: Optional[dict] = None,
                 bundle_version: Optional[str] = None, approvals: Optional[list] = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, head in sorted(heads.items()):
        fname = f"{name.replace('.', '_')}.npz"
        digest = head.save(out / fname)
        entries[name] = {"file": fname, "sha256": digest, "kind": head.kind, "classes": list(head.classes),
                         "calibration": "sigmoid", "calibration_data_version": dataset.get("calibration_data_version")}
    pre = encoder.preprocessing
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_version": bundle_version or f"{operating_mode}-{created[:19].replace(':', '')}",
        "created_at": created,
        "operating_mode": operating_mode,
        "encoder": {"encoder_id": pre.encoder_id, "revision": pre.revision, "tokenizer_revision": pre.tokenizer_revision,
                    "pooling": pre.pooling, "dimension": pre.dimension, "prefix": pre.prefix,
                    "normalization": pre.normalization, "preprocessing_sha256": pre.fingerprint()},
        "privacy": privacy or {"name_detector": cfg.privacy.name_detector, "model_version": cfg.privacy.spacy_model_version,
                               "rules_version": RULES_VERSION},
        "heads": entries,
        "enabled_libraries": sorted(n.split(".", 1)[1] for n in heads if n.startswith("library.")),
        "config_sha256": cfg.config_sha256(),
        "policy_version": cfg.policy_version,
        "dataset": dataset,
        "training": training or {},
        "evaluation": evaluation or {"report": None, "production_acceptance_passed": False},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "lockfile_sha256": _lockfile_sha()},
        "approvals": approvals or [],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = ["build_encoder", "write_bundle", "sha256_file"]
