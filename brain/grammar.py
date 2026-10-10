"""Inline generation schema derived from the shared DraftAnswer contract (Brain plan v0.3, section 9).

Flat (no ``$ref``), ``additionalProperties: false`` everywhere, primitive enums, bounded strings/arrays.
The generation schema is narrower than the contract: cites are restricted to the current passage IDs and
pilot sentences to factual/student. Cross-field rules (sorted citation union, earlier dependencies,
uniqueness) remain Python/verifier checks.
"""

from __future__ import annotations

import hashlib

from contracts.json_codec import canonical_json
from contracts.models import DRAFT_SCHEMA, MAX_CITES, MAX_DEPENDENCIES, MAX_SENTENCE_CHARS

SENTENCE_ID_PATTERN = "^s[1-9][0-9]?$"


def inline_schema(passage_ids: tuple[str, ...], max_sentences: int) -> dict:
    if not 1 <= len(passage_ids) <= 5 or len(set(passage_ids)) != len(passage_ids):
        raise ValueError("INVALID_EVIDENCE")
    if not 1 <= max_sentences <= 16:
        raise ValueError("INVALID_REQUEST")
    ids = list(passage_ids)
    sentence = {
        "type": "object",
        "properties": {
            "sentence_id": {"type": "string", "pattern": SENTENCE_ID_PATTERN},
            "text": {"type": "string", "minLength": 1, "maxLength": MAX_SENTENCE_CHARS},
            "kind_hint": {"type": "string", "enum": ["factual"]},
            "visibility": {"type": "string", "enum": ["student"]},
            "cites": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1,
                      "maxItems": min(MAX_CITES, len(ids))},
            "depends_on": {"type": "array", "items": {"type": "string", "pattern": SENTENCE_ID_PATTERN},
                           "maxItems": MAX_DEPENDENCIES},
        },
        "required": ["sentence_id", "text", "kind_hint", "visibility", "cites", "depends_on"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [DRAFT_SCHEMA]},
            "status": {"type": "string", "enum": ["draft", "no_evidence"]},
            "sentences": {"type": "array", "items": sentence, "maxItems": max_sentences},
            "used_passage_ids": {"type": "array", "items": {"type": "string", "enum": ids},
                                 "maxItems": len(ids)},
        },
        "required": ["schema_version", "status", "sentences", "used_passage_ids"],
        "additionalProperties": False,
    }


def schema_hash() -> str:
    """Hash of the schema template (placeholder IDs) for the manifest."""
    return hashlib.sha256(canonical_json(inline_schema(("PASSAGE",), 4))).hexdigest()
