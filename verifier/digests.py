"""Canonical result binding (Verifier spec v0.2, section 12).

Canonical JSON: UTF-8, sorted keys, no insignificant whitespace, NaN/Infinity forbidden. The digests
detect accidental mutation; an unkeyed hash is not authentication.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _plain(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _plain(obj.model_dump(mode="python"))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> bytes:
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def draft_digest(draft) -> str:
    """Exact text, sentence order, cites and dependencies."""
    return sha256_hex(draft)


def evidence_digest(bundle) -> str:
    return sha256_hex(bundle.binding())


def decision_digest(fields: dict) -> str:
    """Covers all decision fields except decision_digest itself."""
    return sha256_hex({k: v for k, v in fields.items() if k != "decision_digest"})


def result_digest_valid(result) -> bool:
    return decision_digest(result.model_dump(mode="python")) == result.decision_digest
