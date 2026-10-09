"""Canonicalize request text (spec section 5.2).

English prose is the supported distribution, but validation is not ASCII-only: Greek letters,
scientific symbols, units and names are legitimate in English questions.
"""

from __future__ import annotations

import re
import unicodedata

from .schema import ErrorCode, GateError

_ALLOWED_WHITESPACE_CONTROLS = {"\t", "\n"}
_HORIZONTAL_WS = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return canonical text or raise GateError(EMPTY_INPUT / INVALID_REQUEST).

    Casing, punctuation, numbers, units and negations are preserved.
    """
    if not isinstance(text, str):
        raise GateError(ErrorCode.INVALID_REQUEST)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch == "\x00":
            raise GateError(ErrorCode.INVALID_REQUEST)
        cat = unicodedata.category(ch)
        if cat == "Cc":
            if ch in _ALLOWED_WHITESPACE_CONTROLS:
                out.append(ch)
                continue
            # C0/C1 controls other than ordinary whitespace are unsupported.
            raise GateError(ErrorCode.INVALID_REQUEST)
        if cat == "Cf":
            # zero-width/bidi formatting controls are removed, never interpreted
            continue
        if cat in ("Zl", "Zp"):
            out.append("\n")
            continue
        out.append(ch)
    text = "".join(out)
    text = "\n".join(_HORIZONTAL_WS.sub(" ", line).strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text).strip()
    if not text or not any(ch.isalnum() for ch in text):
        raise GateError(ErrorCode.EMPTY_INPUT)
    return text


def casefold_view(text: str) -> str:
    """View for rule matching only; never sent to the encoder or downstream."""
    return (
        text.casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
