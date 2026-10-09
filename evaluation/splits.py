"""Grouped development/final split (Verifier spec v0.2, sections 8-9).

Split by scenario/question family and source-section family before mutation or paraphrasing. All
derivatives stay in their parent group. Exact and near duplicates across splits are reported as leaks.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def group_key(case: dict) -> str:
    """Union of question family and source-section family: both define the independence unit."""
    return f"{case['question_family']}|{case['source_family']}"


def _bucket(key: str, seed: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def grouped_split(cases: Iterable[dict], final_fraction: float = 0.3, seed: str = "verifier-split-0.2"
                  ) -> tuple[list[dict], list[dict]]:
    """Deterministic group-level assignment. Families linked through either key stay together."""
    cases = list(cases)
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cases:  # connect question and source families into components
        a, b = find("q:" + c["question_family"]), find("s:" + c["source_family"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    dev, final = [], []
    for c in cases:
        root = find("q:" + c["question_family"])
        (final if _bucket(root, seed) < final_fraction else dev).append(c)
    return dev, final


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _shingles(text: str, n: int = 3) -> set[str]:
    toks = _norm(text).split()
    return {" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def leaks(dev: list[dict], final: list[dict], near: float = 0.8) -> list[tuple[str, str, str]]:
    """Cross-split group overlap, exact duplicates and near duplicates (Jaccard over word 3-grams)."""
    out = []
    dev_groups = {c["question_family"] for c in dev} | {"s:" + c["source_family"] for c in dev}
    for c in final:
        if c["question_family"] in dev_groups or "s:" + c["source_family"] in dev_groups:
            out.append(("group", c["case_id"], ""))
    dev_text = [(d["case_id"], _norm(d["text"]), _shingles(d["text"])) for d in dev]
    for c in final:
        norm, sh = _norm(c["text"]), _shingles(c["text"])
        for did, dn, dsh in dev_text:
            if norm == dn:
                out.append(("exact", c["case_id"], did))
            elif sh and dsh and len(sh & dsh) / len(sh | dsh) >= near:
                out.append(("near", c["case_id"], did))
    return out
