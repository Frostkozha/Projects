"""Layer 0a: segmentation and coverage validation (Verifier spec v0.2, sections 4.1, 5.1).

A sentence object is not proof of valid segmentation. Every text field is re-segmented with a pinned
spaCy pipeline; hidden extra sentences, headings, code, markup, lists, links or unaccounted text reject
the whole draft (INVALID_DRAFT -> A5). Ambiguous segmentation rejects rather than truncates.
"""

from __future__ import annotations

import re
from typing import Optional, Protocol

from contracts.models import DraftAnswer

_MARKUP = [
    re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>"),          # HTML tags
    re.compile(r"`"),                                 # code
    re.compile(r"^\s*#{1,6}\s"),                      # headings
    re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)"),         # list markers
    re.compile(r"\|"),                                # tables
    re.compile(r"\[[^\]]*\]\([^)]*\)"),               # markdown links
    re.compile(r"(?i)\b(?:https?://|www\.)"),        # URLs
    re.compile(r"\*\*|__"),                           # emphasis markup
]
_TERMINAL = re.compile(r"[.!?][\"')\]]*$")
_LATIN = re.compile(r"[A-Za-z]")
_FOREIGN_LETTER = re.compile(r"[Ѐ-ӿ؀-ۿ一-鿿぀-ヿ가-힯]")


class Segmenter(Protocol):
    version: str
    is_fixture: bool

    def sentences(self, text: str) -> list[tuple[int, int]]: ...


class SpacySentencizer:
    """Rule-based spaCy sentencizer (no model download). Development segmenter."""

    is_fixture = False

    def __init__(self):
        import spacy  # noqa: PLC0415

        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")
        self.version = f"spacy-{spacy.__version__}-sentencizer"
        self.production_grade = False

    def sentences(self, text: str) -> list[tuple[int, int]]:
        return [(s.start_char, s.end_char) for s in self._nlp(text).sents]


class SpacyModelSegmenter:
    """Pinned en_core_web_sm pipeline loaded from a local path (production segmenter)."""

    is_fixture = False

    def __init__(self, model_path: str, expected_version: Optional[str]):
        import spacy  # noqa: PLC0415

        self._nlp = spacy.load(model_path)
        version = self._nlp.meta.get("version")
        if expected_version and version != expected_version:
            raise RuntimeError("segmenter version mismatch")
        self.version = f"spacy-{self._nlp.meta.get('name')}-{version}"
        self.production_grade = True

    def sentences(self, text: str) -> list[tuple[int, int]]:
        return [(s.start_char, s.end_char) for s in self._nlp(text).sents]


def sentence_problem(text: str, segmenter: Segmenter) -> Optional[str]:
    """Return a stable problem code, or None when the text is exactly one well-formed sentence."""
    if "\n" in text or "\r" in text or "\t" in text:
        return "multiline"
    if text != text.strip():
        return "surrounding_whitespace"
    for pat in _MARKUP:
        if pat.search(text):
            return "markup"
    if not _TERMINAL.search(text):
        return "no_terminal_punctuation"
    foreign = len(_FOREIGN_LETTER.findall(text))
    if foreign and foreign >= len(_LATIN.findall(text)):
        return "non_english_prose"
    spans = segmenter.sentences(text)
    if len(spans) != 1:
        return "multiple_sentences"
    start, end = spans[0]
    if text[:start].strip() or text[end:].strip():
        return "unaccounted_text"
    return None


def validate_draft_format(draft: DraftAnswer, segmenter: Segmenter) -> dict[str, Optional[str]]:
    """Per-sentence format problems (any problem rejects the whole draft)."""
    return {s.sentence_id: sentence_problem(s.text, segmenter) for s in draft.sentences}
