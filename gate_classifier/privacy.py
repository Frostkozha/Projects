"""Local identifier redaction (spec section 6).

Redaction reduces exposure; it does not prove anonymity. Raw matches never leave this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

CATEGORIES = ("PERSON", "ID", "EMAIL", "PHONE", "ADDRESS")


class PrivacyError(RuntimeError):
    """Adapter failure. Callers must block downstream work (service unavailable)."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    category: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    categories: tuple[str, ...]

    @property
    def detected(self) -> bool:
        return bool(self.categories)

    def __repr__(self) -> str:  # do not print text accidentally into logs
        return f"RedactionResult(categories={self.categories})"


class NameDetector(Protocol):
    def find(self, text: str) -> list[Span]: ...


# ----------------------------------------------------------------------------- structured patterns

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}\b")
# IIN-like: bounded 12-digit run, optionally grouped as 6-6
_IIN = re.compile(r"(?<![\d.,])(?:\d{12}|\d{6}[ -]\d{6})(?![\d.,]\d)")
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?![\w])"
)
_LABELED_PHONE = re.compile(r"(?i)\b(?:phone|tel|telephone|mobile|cell)(?:\s+(?:no|number))?\s*[:#]?\s*(\+?[\d][\d\s().-]{5,18}\d)")
_LABELED_ID = re.compile(
    r"(?i)\b(?:IIN|student\s+(?:id|number|no)|patient\s+(?:id|number|no)|MRN|medical\s+record(?:\s+number)?|"
    r"passport(?:\s+(?:no|number))?|ID\s+(?:no|number))\s*[:#]?\s*([A-Z0-9][A-Z0-9-]{3,20})"
)
_LABELED_ADDRESS = re.compile(r"(?i)\b(?:address|lives at|living at|home is at)\s*[:]?\s*([^\n.;]{4,120})")
_STREET = re.compile(
    r"(?i)\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'-]+\s+){1,4}"
    r"(?:street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd\.|lane|ln\.|drive|dr\.|prospekt|microdistrict|mkr\.?)"
    r"(?:\s*,?\s*(?:apt|apartment|flat)\.?\s*\d{1,5})?"
)
_POSTCODE = re.compile(r"(?i)\b(?:postcode|postal code|zip(?: code)?)\s*[:#]?\s*([A-Z0-9][A-Z0-9 -]{2,9})")


class RuleNameDetector:
    """Explicit identifier-label rules for person names (development baseline)."""

    version = "rule-names-0.2"
    _PATTERNS = [
        re.compile(r"\b(?:[Mm]y name is|[Ii] am called|[Cc]all me)\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})"),
        re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})"),
        re.compile(
            r"(?i:\b(?:patient|student|name|named|classmate|colleague)\b)\s*[:]?\s*([A-Z][a-z'-]+\s+[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?)"
        ),
        re.compile(
            r"(?i:\bmy\s+(?:mother|father|mom|mum|dad|brother|sister|son|daughter|wife|husband|friend|"
            r"grandmother|grandfather|grandma|grandpa|aunt|uncle|cousin|roommate|neighbou?r|partner|child|baby))"
            r",?\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})"
        ),
    ]
    # tokens that look capitalized but are course terms or sentence words; reduces false positives
    _STOP = {
        "Has", "Is", "Was", "Had", "Took", "Takes", "Gets", "Got", "Said", "The", "A", "An", "And", "With",
        "Simple", "Squamous", "Epithelium", "Cell", "Cells", "Tissue", "Histology",
    }

    def find(self, text: str) -> list[Span]:
        spans = []
        for pat in self._PATTERNS:
            for m in pat.finditer(text):
                words = m.group(1).split()
                kept = []
                for w in words:
                    if w in self._STOP:
                        break
                    kept.append(w)
                if not kept:
                    continue
                start = m.start(1)
                end = start + len(" ".join(kept))
                spans.append(Span(start, end, "PERSON"))
        return spans


class SpacyNameDetector:
    """PERSON entities from a locally installed, pinned spaCy pipeline. Never downloads."""

    def __init__(self, model_path: str, expected_version: str | None = None):
        try:
            import spacy  # noqa: PLC0415  (optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PrivacyError("spaCy unavailable") from exc
        try:
            self._nlp = spacy.load(model_path)
        except Exception as exc:  # pragma: no cover - depends on environment
            raise PrivacyError("spaCy model unavailable") from exc
        meta_version = self._nlp.meta.get("version")
        if expected_version and meta_version != expected_version:
            raise PrivacyError("spaCy model version mismatch")
        self.version = f"spacy-{self._nlp.meta.get('name')}-{meta_version}"
        self._rules = RuleNameDetector()

    def find(self, text: str) -> list[Span]:
        doc = self._nlp(text)
        spans = [Span(e.start_char, e.end_char, "PERSON") for e in doc.ents if e.label_ == "PERSON"]
        return spans + self._rules.find(text)


class Redactor:
    def __init__(self, name_detector: NameDetector | None = None):
        self.name_detector = name_detector or RuleNameDetector()

    @property
    def version(self) -> str:
        return f"redactor-0.2+{getattr(self.name_detector, 'version', 'custom')}"

    def _structured(self, text: str) -> list[Span]:
        spans: list[Span] = []

        def add(pattern, category, group=0):
            for m in pattern.finditer(text):
                spans.append(Span(m.start(group), m.end(group), category))

        add(_EMAIL, "EMAIL")
        add(_LABELED_ID, "ID", 1)
        add(_IIN, "ID")
        add(_LABELED_PHONE, "PHONE", 1)
        add(_PHONE, "PHONE")
        add(_LABELED_ADDRESS, "ADDRESS", 1)
        add(_STREET, "ADDRESS")
        add(_POSTCODE, "ADDRESS", 1)
        return spans

    def redact(self, text: str) -> RedactionResult:
        try:
            spans = self._structured(text) + list(self.name_detector.find(text))
        except PrivacyError:
            raise
        except Exception:  # never include the text in the error
            raise PrivacyError("redaction failed") from None
        if not spans:
            return RedactionResult(text, ())
        # merge overlaps; earlier start wins, longer span wins on ties; priority keeps first category
        priority = {c: i for i, c in enumerate(("ID", "EMAIL", "PHONE", "ADDRESS", "PERSON"))}
        spans.sort(key=lambda s: (s.start, -(s.end - s.start), priority[s.category]))
        merged: list[Span] = []
        for s in spans:
            if merged and s.start < merged[-1].end:
                last = merged[-1]
                if s.end > last.end:
                    merged[-1] = Span(last.start, s.end, last.category)
                continue
            merged.append(s)
        out, pos = [], 0
        for s in merged:
            out.append(text[pos:s.start])
            out.append(f"[{s.category}]")
            pos = s.end
        out.append(text[pos:])
        categories = tuple(sorted({s.category for s in merged}))
        return RedactionResult("".join(out), categories)
