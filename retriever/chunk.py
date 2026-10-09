"""Token-aware chunking, locators and stable passage IDs (Retriever spec v0.2, sections 7.2, 8.2, 8.3).

Chunks follow section, paragraph and sentence boundaries and never cross sections. Token counts
use the larger of the encoder and reranker tokenizers. Nothing is truncated: a sentence, list or
table that cannot fit is quarantined and the chunk is split around the gap, so text across a
quarantined unit is never joined as if it were adjacent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import RetrieverProfile
from .parse import Block, Quarantined
from .register import SourceRecord

PASSAGE_ID_SCHEME = "passage-id-v1"
_ABBREV = {"e.g.", "i.e.", "fig.", "figs.", "approx.", "vs.", "etc.", "no.", "dr.", "st.", "cf.", "ca.", "al."}
_SENT_END = re.compile(r"[.!?][\"')\]]*\s+")


class Tokenizer(Protocol):
    def count(self, text: str, add_special_tokens: bool) -> int: ...


@dataclass(frozen=True)
class Unit:
    block: Block
    start: int
    end: int
    kind: str  # sentence | list | table | caption


@dataclass(frozen=True)
class Chunk:
    passage_id: str
    digest: str
    text: str
    text_sha256: str
    heading: str
    section_path: tuple[str, ...]
    spans: tuple[tuple[str, int, int], ...]
    locator: dict
    kind: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    """Sorted keys, UTF-8 (no ASCII escaping), compact separators."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def chunking_fingerprint(profile: RetrieverProfile) -> str:
    lim = profile.limits
    payload = {
        "chunking": profile.chunking_version, "normalization": profile.parser_normalization_version,
        "target": lim.chunk_target_tokens, "max": lim.chunk_max_tokens, "heading": lim.heading_max_tokens,
        "overlap": lim.overlap_max_tokens, "encoder_tokenizer": profile.encoder.tokenizer_revision,
        "reranker": profile.reranker.revision,
    }
    return sha256_text(canonical_json(payload))


def passage_identity(source_id: str, source_version: str, locator: dict, spans, text_sha: str,
                     fingerprint: str) -> tuple[str, str]:
    payload = {"scheme": PASSAGE_ID_SCHEME, "source_id": source_id, "source_version": source_version,
               "locator": locator, "spans": [list(s) for s in spans], "text_sha256": text_sha,
               "chunking": fingerprint}
    digest = sha256_text(canonical_json(payload))
    return "psg-" + digest[:40], digest


def split_sentences(block: Block) -> list[Unit]:
    text = block.text
    units, start = [], 0
    for m in _SENT_END.finditer(text):
        candidate = text[start:m.end()].strip()
        last_word = candidate.split()[-1].lower() if candidate.split() else ""
        nxt = text[m.end():m.end() + 1]
        if last_word in _ABBREV or (nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([")):
            continue
        end = m.start() + len(m.group(0).rstrip())
        units.append(Unit(block, start, end, "sentence"))
        start = m.end()
    if start < len(text):
        units.append(Unit(block, start, len(text.rstrip()), "sentence"))
    return [u for u in units if text[u.start:u.end].strip()]


class Chunker:
    def __init__(self, profile: RetrieverProfile, tokenizers: Sequence[Tokenizer]):
        if not tokenizers:
            raise ValueError("at least one tokenizer is required")
        self.profile = profile
        self.tokenizers = list(tokenizers)
        self.fingerprint = chunking_fingerprint(profile)

    def tokens(self, text: str) -> int:
        return max(t.count(text, False) for t in self.tokenizers)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _text(units: list[Unit]) -> str:
        parts: list[str] = []
        i = 0
        while i < len(units):
            j = i
            while j + 1 < len(units) and units[j + 1].block is units[i].block:
                j += 1
            block = units[i].block
            parts.append(block.text[units[i].start:units[j].end])
            i = j + 1
        return "\n\n".join(parts)

    def heading(self, section: tuple[str, ...]) -> str:
        parts = list(section)
        while parts:
            h = " > ".join(parts)
            if self.tokens(h) <= self.profile.limits.heading_max_tokens:
                return h
            parts = parts[1:]
        return ""

    def _locator(self, units: list[Unit], record: SourceRecord, kind: str) -> dict:
        blocks = []
        for u in units:
            if not blocks or blocks[-1] is not u.block:
                blocks.append(u.block)
        section_label = " > ".join(blocks[0].section_path)
        slides = [b.slide for b in blocks if b.slide is not None]
        pages = [b.page_index for b in blocks if b.page_index is not None]
        if kind in ("table", "caption"):
            loc = {"kind": kind, "anchor": blocks[0].block_id,
                   "label": f"{section_label} ({kind}, {blocks[0].block_id})", "start": None, "end": None}
            if pages:
                loc["start"], loc["end"] = min(pages), max(pages)
            return loc
        if slides and len(slides) == len(blocks):
            a, b = min(slides), max(slides)
            return {"kind": "slide", "start": a, "end": b, "anchor": None,
                    "label": f"Slide {a}" if a == b else f"Slides {a}-{b}"}
        if pages and len(pages) == len(blocks):
            a, b = min(pages), max(pages)
            labels = [bl.page_label for bl in blocks if bl.page_label]
            if record.parser.verified_page_labels and labels and len(labels) == len(blocks):
                la, lb = labels[0], labels[-1]
                label = f"p. {la}" if la == lb else f"pp. {la}-{lb}"
            else:
                label = f"PDF page {a}" if a == b else f"PDF pages {a}-{b}"
            return {"kind": "pdf_page", "start": a, "end": b, "anchor": None, "label": label}
        return {"kind": "section", "start": None, "end": None, "anchor": blocks[0].block_id, "label": section_label}

    def _make(self, units: list[Unit], record: SourceRecord, kind: str) -> Chunk:
        text = self._text(units)
        section = units[0].block.section_path
        locator = self._locator(units, record, kind)
        spans = tuple((u.block.block_id, u.start, u.end) for u in units)
        text_sha = sha256_text(text)
        pid, digest = passage_identity(record.source_id, record.source_version, locator, spans, text_sha,
                                       self.fingerprint)
        return Chunk(pid, digest, text, text_sha, self.heading(section), section, spans, locator, kind)

    # ------------------------------------------------------------------ main

    def chunk(self, record: SourceRecord, blocks: list[Block]) -> tuple[list[Chunk], list[Quarantined]]:
        lim = self.profile.limits
        chunks: list[Chunk] = []
        quarantined: list[Quarantined] = []
        q = lambda ref, reason: quarantined.append(Quarantined(record.source_id, record.source_version, ref, reason))  # noqa: E731

        # group consecutive blocks by section; never cross a section boundary
        groups: list[list[Block]] = []
        for b in blocks:
            if groups and groups[-1][0].section_path == b.section_path:
                groups[-1].append(b)
            else:
                groups.append([b])

        for group in groups:
            current: list[Unit] = []

            def flush(allow_overlap: bool) -> list[Unit]:
                if not current:
                    return []
                chunks.append(self._make(list(current), record, "text"))
                carry: list[Unit] = []
                if allow_overlap:
                    budget = lim.overlap_max_tokens
                    for u in reversed(current):
                        if u.kind != "sentence":
                            break
                        cand = [u] + carry
                        if self.tokens(self._text(cand)) > budget:
                            break
                        carry = cand
                    if len(carry) == len(current):
                        carry = []  # overlap may not reproduce the whole previous chunk
                current.clear()
                return carry

            for block in group:
                if block.kind in ("table", "caption"):
                    flush(False)
                    unit = Unit(block, 0, len(block.text), block.kind)
                    if self.tokens(block.text) > lim.chunk_max_tokens:
                        q(block.block_id, f"oversize_{block.kind}")
                        continue
                    chunks.append(self._make([unit], record, block.kind))
                    continue
                units = [Unit(block, 0, len(block.text), "list")] if block.kind == "list" else split_sentences(block)
                for unit in units:
                    utext = block.text[unit.start:unit.end]
                    if self.tokens(utext) > lim.chunk_max_tokens:
                        flush(False)
                        q(f"{block.block_id}:{unit.start}-{unit.end}",
                          "long_sentence" if unit.kind == "sentence" else "oversize_list")
                        continue
                    trial = current + [unit]
                    n = self.tokens(self._text(trial))
                    if n <= lim.chunk_max_tokens and (not current or n <= lim.chunk_target_tokens):
                        current.append(unit)
                        continue
                    carry = flush(True)
                    with_carry = carry + [unit]
                    if carry and self.tokens(self._text(with_carry)) <= lim.chunk_max_tokens:
                        current.extend(with_carry)
                    else:
                        current.append(unit)
            flush(False)

        # every complete E5 representation must fit the encoder ceiling
        enc_max = self.profile.encoder.max_tokens
        kept = []
        for c in chunks:
            rep = f"{self.profile.encoder.passage_prefix}{c.heading}\n{c.text}" if c.heading else \
                f"{self.profile.encoder.passage_prefix}{c.text}"
            if max(t.count(rep, True) for t in self.tokenizers) > enc_max:
                q(c.passage_id, "representation_too_long")
                continue
            kept.append(c)
        return kept, quarantined


def span_overlap(a: Sequence[tuple[str, int, int]], b: Sequence[tuple[str, int, int]]) -> float:
    """Intersection length / shorter span length over original block character spans."""
    def total(spans):
        return sum(e - s for _, s, e in spans)

    inter = 0
    for ba, sa, ea in a:
        for bb, sb, eb in b:
            if ba == bb:
                inter += max(0, min(ea, eb) - max(sa, sb))
    shorter = min(total(a), total(b))
    return inter / shorter if shorter > 0 else 0.0
