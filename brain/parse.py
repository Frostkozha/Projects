"""Strict completion parser (Brain plan v0.3, section 9 and appendix C).

Accepts exactly one complete chat-completion envelope whose single assistant message content is exactly
one DraftAnswer JSON object. No brace searching, substring extraction, repair or second-model repair.
Validation errors are never logged: they can contain original content.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Optional

from pydantic import ValidationError

from contracts.draft import DraftAnswer
from contracts.json_codec import InvalidJSON, load_object, validate_wire

MAX_BODY = 256 * 1024
_ALLOWED_MESSAGE_KEYS = {"role", "content", "reasoning_content", "tool_calls", "function_call"}


class CompletionFailure(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_completion(
    raw: bytes, *, expected_alias: str, supplied_ids: tuple[str, ...],
    sentence_policy: Callable[[DraftAnswer], None], max_visible_sentences: int = 4,
) -> DraftAnswer:
    if not 1 <= len(supplied_ids) <= 5 or len(set(supplied_ids)) != len(supplied_ids):
        raise CompletionFailure("INVALID_EVIDENCE")
    if type(max_visible_sentences) is not int or not 1 <= max_visible_sentences <= 4:
        raise CompletionFailure("INVALID_REQUEST")
    try:
        envelope = load_object(raw, max_bytes=MAX_BODY)
        if envelope.get("model") != expected_alias:
            raise CompletionFailure("RUNTIME_INCOMPATIBLE")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise CompletionFailure("INVALID_OUTPUT")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise CompletionFailure("INVALID_OUTPUT")
        finish = choice.get("finish_reason")
        if finish == "length":
            raise CompletionFailure("OUTPUT_LIMIT")
        if finish != "stop":
            raise CompletionFailure("INVALID_OUTPUT")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise CompletionFailure("INVALID_OUTPUT")
        if set(message) - _ALLOWED_MESSAGE_KEYS:
            raise CompletionFailure("INVALID_OUTPUT")
        if message.get("reasoning_content") not in (None, ""):
            raise CompletionFailure("INVALID_OUTPUT")
        if message.get("tool_calls") not in (None, []):
            raise CompletionFailure("INVALID_OUTPUT")
        if message.get("function_call") is not None:
            raise CompletionFailure("INVALID_OUTPUT")
        content = message.get("content")
        if not isinstance(content, str):
            raise CompletionFailure("INVALID_OUTPUT")
        draft = validate_wire(DraftAnswer, content.encode("utf-8"), max_bytes=MAX_BODY)
        if not set(draft.used_passage_ids) <= set(supplied_ids):
            raise CompletionFailure("INVALID_OUTPUT")
        if any(s.kind_hint != "factual" or s.visibility != "student" for s in draft.sentences):
            raise CompletionFailure("INVALID_OUTPUT")
        if len(draft.sentences) > max_visible_sentences:
            raise CompletionFailure("INVALID_OUTPUT")
        sentence_policy(draft)  # the pinned verifier-compatible sentence policy
        return draft
    except CompletionFailure:
        raise
    except (InvalidJSON, ValidationError, UnicodeError, ValueError, TypeError, KeyError):
        raise CompletionFailure("INVALID_OUTPUT") from None


def observed_metrics(raw: bytes) -> dict:
    """Runtime-reported token counts/timings from an envelope, validated; unobserved values stay None."""
    out: dict[str, Optional[float | int | str]] = {"prompt_ms": None, "generation_ms": None,
                                                  "prompt_tokens": None, "completion_tokens": None,
                                                  "finish_reason": None}
    try:
        env = load_object(raw, max_bytes=MAX_BODY)
    except InvalidJSON:
        return out
    usage = env.get("usage") if isinstance(env.get("usage"), dict) else {}
    timings = env.get("timings") if isinstance(env.get("timings"), dict) else {}
    for key, src in (("prompt_tokens", usage.get("prompt_tokens")), ("completion_tokens", usage.get("completion_tokens"))):
        if type(src) is int and src >= 0:
            out[key] = src
    for key, src in (("prompt_ms", timings.get("prompt_ms")), ("generation_ms", timings.get("predicted_ms"))):
        if type(src) in (int, float) and math.isfinite(src) and src >= 0:
            out[key] = float(src)
    choices = env.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        fr = choices[0].get("finish_reason")
        out["finish_reason"] = fr if fr in ("stop", "length", "tool_calls") else "unknown"
    return out


# ----------------------------------------------------------------------------- sentence policy

_TRUSTED_METADATA = re.compile(r"\b(?:A[1-7]|reply_key|response_code|passage_id|sentence_id)\b")


class SentencePolicy:
    """Verifier-compatible one-sentence / plain-text policy (``verifier.format``) plus Brain metadata rules.

    Uses the same pinned segmenter class the verifier uses, so a draft that passes here is not re-segmented
    differently downstream. The verifier still performs support, coverage, harm and live-source checks.
    """

    def __init__(self, segmenter=None):
        from verifier.format import SpacySentencizer  # noqa: PLC0415

        self.segmenter = segmenter or SpacySentencizer()
        self.version = getattr(self.segmenter, "version", "unknown")

    def __call__(self, draft: DraftAnswer) -> None:
        from verifier.format import sentence_problem  # noqa: PLC0415

        for s in draft.sentences:
            if sentence_problem(s.text, self.segmenter) is not None:
                raise CompletionFailure("INVALID_OUTPUT")
            if _TRUSTED_METADATA.search(s.text):
                raise CompletionFailure("INVALID_OUTPUT")
            if any(re.search(r"(?<![\w.-])" + re.escape(c) + r"(?![\w-])", s.text) for c in s.cites):
                raise CompletionFailure("INVALID_OUTPUT")
