"""Local parser adapters (Retriever spec v0.2, section 8.1).

Initial formats: approved structured Markdown/JSON and text PDFs. No URL fetching, OCR, table-cell
inference, archive expansion or active HTML. Anything uncertain is quarantined for reviewed
transcription instead of entering the index. Normalization is NFC + whitespace only, so units,
negations, Greek letters and super/subscripts survive unchanged.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .register import SourceRecord

PARSER_VERSION = "parsers-0.2"
ALLOWED_SUFFIXES = {"markdown": {".md"}, "json": {".json"}, "pdf": {".pdf"}}
BLOCK_KINDS = ("paragraph", "list", "table", "caption")

_ACTIVE_HTML = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>|javascript:", re.IGNORECASE)
_REMOTE = re.compile(r"\]\(\s*(?:https?:|//|ftp:|data:)", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BLOCK_ID = re.compile(r"\s*\{#([A-Za-z0-9._-]{1,64})\}\s*$")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_WS = re.compile(r"[ \t ]+")
_TABLE_LIKE = re.compile(r"\S(?: {2,}|\t)\S+(?: {2,}|\t)\S")


class ImportRejected(ValueError):
    """Rejected before parsing (path, format, size). Message is a stable code."""


class ParseError(ValueError):
    """Whole-source parse failure (stable code)."""


@dataclass(frozen=True)
class Block:
    block_id: str
    kind: str
    text: str
    section_path: tuple[str, ...]
    page_index: Optional[int] = None   # physical 1-based PDF page
    page_label: Optional[str] = None   # verified printed label
    slide: Optional[int] = None


@dataclass(frozen=True)
class Quarantined:
    source_id: str
    source_version: str
    ref: str
    reason: str


@dataclass
class ParseResult:
    blocks: list[Block] = field(default_factory=list)
    quarantined: list[Quarantined] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """NFC (not NFKC: keeps µ, ², ₂ ...), unify newlines, collapse horizontal whitespace per line."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def resolve_import(import_root: str | Path, file_ref: str, fmt: str, max_bytes: int) -> Path:
    """Resolve a declared local path. Rejects traversal, symlink escape, wrong format and oversize."""
    root = Path(import_root).resolve(strict=True)
    if "://" in file_ref or file_ref.startswith(("/", "\\")) or ":" in file_ref:
        raise ImportRejected("not_a_relative_local_path")
    candidate = root / file_ref
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ImportRejected("file_missing") from None
    if not resolved.is_relative_to(root):
        raise ImportRejected("path_escapes_import_root")
    if not resolved.is_file():
        raise ImportRejected("not_a_regular_file")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES[fmt]:
        raise ImportRejected("unsupported_format")
    if resolved.stat().st_size > max_bytes:
        raise ImportRejected("file_too_large")
    return resolved


def _check_safe(text: str) -> None:
    if _ACTIVE_HTML.search(text):
        raise ParseError("active_html")
    if _REMOTE.search(text):
        raise ParseError("remote_resource")


# ----------------------------------------------------------------------------- markdown


def parse_markdown(raw: str, record: SourceRecord) -> ParseResult:
    _check_safe(raw)
    text = normalize_text(raw)
    result = ParseResult()
    path: list[str] = []
    paragraph: list[str] = []
    counters: dict[tuple[str, ...], int] = {}

    def flush():
        if not paragraph:
            return
        lines = list(paragraph)
        paragraph.clear()
        body = "\n".join(lines)
        explicit = None
        m = _BLOCK_ID.search(body)
        if m:
            explicit = m.group(1)
            body = body[: m.start()].rstrip()
        if all(line.startswith("|") for line in lines):
            kind = "table"
        elif all(_LIST_LINE.match(line) for line in lines):
            kind = "list"
        elif body.lower().startswith("caption:"):
            kind, body = "caption", body[len("caption:"):].strip()
        else:
            kind, body = "paragraph", " ".join(body.split("\n"))
        section = tuple(path) or (record.default_section,)
        n = counters.get(section, 0) + 1
        counters[section] = n
        block_id = explicit or f"{_slug(section)}-b{n}"
        result.blocks.append(Block(block_id, kind, body, section))

    for line in text.split("\n"):
        h = _HEADING.match(line)
        if h:
            flush()
            level = len(h.group(1))
            path[:] = path[: level - 1] + [h.group(2)]
            continue
        if not line:
            flush()
            continue
        paragraph.append(line)
    flush()
    _unique_ids(result, record)
    return result


def _slug(section: tuple[str, ...]) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", "-".join(section)).strip("-").lower()
    return (s or "section")[:80]


def _unique_ids(result: ParseResult, record: SourceRecord) -> None:
    ids = [b.block_id for b in result.blocks]
    if len(ids) != len(set(ids)):
        raise ParseError("duplicate_block_ids")


# ----------------------------------------------------------------------------- json


def parse_json(raw: str, record: SourceRecord) -> ParseResult:
    try:
        data = json.loads(raw)
    except ValueError:
        raise ParseError("invalid_json") from None
    if not isinstance(data, dict) or set(data) != {"blocks"} or not isinstance(data["blocks"], list):
        raise ParseError("json_schema")
    result = ParseResult()
    allowed = {"block_id", "section_path", "kind", "text", "slide", "page", "page_label"}
    for item in data["blocks"]:
        if not isinstance(item, dict) or set(item) - allowed or not {"block_id", "section_path", "kind", "text"} <= set(item):
            raise ParseError("json_block_schema")
        if item["kind"] not in BLOCK_KINDS or not isinstance(item["text"], str):
            raise ParseError("json_block_kind")
        _check_safe(item["text"])
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", str(item["block_id"])):
            raise ParseError("json_block_id")
        section = tuple(str(s) for s in item["section_path"]) or (record.default_section,)
        for key in ("slide", "page"):
            v = item.get(key)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise ParseError("json_locator")
        text = normalize_text(item["text"])
        if item["kind"] == "paragraph":
            text = " ".join(text.split("\n"))
        if not text:
            result.quarantined.append(Quarantined(record.source_id, record.source_version, item["block_id"], "empty_block"))
            continue
        result.blocks.append(Block(item["block_id"], item["kind"], text, section, page_index=item.get("page"),
                                   page_label=item.get("page_label"), slide=item.get("slide")))
    _unique_ids(result, record)
    return result


# ----------------------------------------------------------------------------- pdf


def parse_pdf(path: Path, record: SourceRecord, max_pages: int) -> ParseResult:
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ParseError("pypdf_unavailable") from exc
    try:
        reader = PdfReader(str(path), strict=True)
        n = len(reader.pages)
    except Exception:
        raise ParseError("pdf_unreadable") from None
    if n > max_pages:
        raise ImportRejected("too_many_pages")
    labels = None
    if record.parser.verified_page_labels:
        try:
            labels = list(reader.page_labels)
        except Exception:
            raise ParseError("page_labels_unreadable") from None
    result = ParseResult()
    section = (record.default_section,)
    for i, page in enumerate(reader.pages):
        phys = i + 1
        try:
            raw = page.extract_text() or ""
        except Exception:
            result.quarantined.append(Quarantined(record.source_id, record.source_version, f"page-{phys}", "extraction_failed"))
            continue
        text = normalize_text(raw)
        if len(re.sub(r"\s", "", text)) < 20:
            result.quarantined.append(Quarantined(record.source_id, record.source_version, f"page-{phys}", "blank_or_scanned"))
            continue
        if any(_TABLE_LIKE.search(line) for line in raw.split("\n")):
            result.quarantined.append(Quarantined(record.source_id, record.source_version, f"page-{phys}", "table_or_columns"))
            continue
        label = labels[i] if labels else None
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for j, para in enumerate(paragraphs, start=1):
            result.blocks.append(Block(f"p{phys}-b{j}", "paragraph", " ".join(para.split("\n")), section,
                                       page_index=phys, page_label=label))
    return result


def parse_source(record: SourceRecord, import_root: str | Path, max_bytes: int, max_pages: int) -> ParseResult:
    path = resolve_import(import_root, record.file_ref, record.format, max_bytes)
    import hashlib  # noqa: PLC0415

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record.file_sha256:
        raise ParseError("file_hash_mismatch")
    if record.format == "pdf":
        return parse_pdf(path, record, max_pages)
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ParseError("not_utf8") from None
    return parse_markdown(raw, record) if record.format == "markdown" else parse_json(raw, record)
