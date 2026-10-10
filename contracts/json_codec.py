"""Strict JSON wire codec shared by every component (Brain plan v0.3, shared primitives).

Duplicate keys, malformed UTF-8, lone surrogates, NaN/Infinity and trailing data are rejected before
any Pydantic validation. Never log the exceptions' context: inputs can contain student text.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class InvalidJSON(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidJSON("duplicate_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise InvalidJSON("nonfinite_number")


def _check_values(value: Any) -> None:
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")  # lone surrogates fail here
    elif isinstance(value, float) and not math.isfinite(value):
        raise InvalidJSON("nonfinite_number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_values(key)
            _check_values(item)
    elif isinstance(value, list):
        for item in value:
            _check_values(item)


def load_object(raw: bytes, *, max_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise InvalidJSON("byte_limit")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise InvalidJSON("object_required")
        _check_values(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise InvalidJSON("invalid_json") from None


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8", errors="strict")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_wire(model: type[T], raw: bytes, *, max_bytes: int) -> T:
    value = load_object(raw, max_bytes=max_bytes)
    # JSON-mode strict validation permits UUID strings on the wire.
    # Duplicate-key and nonfinite checks already ran before re-encoding.
    return model.model_validate_json(canonical_json(value), strict=True)
