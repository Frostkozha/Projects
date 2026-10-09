"""Grouped, seeded partitioning (spec section 16.1).

60% train / 15% calibration / 10% tuning / 15% restricted final holdout, by group. Records whose
normalized text is identical are merged into one group so exact/near-exact duplicates cannot
cross partitions; paraphrase families must already share a group_id (reviewers' job).

Usage: python -m training.split_dataset DATA.jsonl --seed 20261009 --out split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict

from .validate_dataset import dataset_sha256, load_jsonl

PARTITIONS = (("train", 0.60), ("calibration", 0.15), ("tuning", 0.10), ("holdout", 0.15))
_NORM = re.compile(r"[\W_]+", re.UNICODE)


def _norm_text(text: str) -> str:
    return _NORM.sub(" ", text.casefold()).strip()


def merged_groups(records: list[dict]) -> dict[str, str]:
    """Union-find over group_id and normalized text. Returns record id -> merged group key."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for r in records:
        union(f"g:{r['group_id']}", f"t:{_norm_text(r['text'])}")
    return {r["id"]: find(f"g:{r['group_id']}") for r in records}


def _stratum(records: list[dict]) -> str:
    risks = sorted({k for r in records for k, v in r["risks"].items() if v})
    if risks:
        return "risk:" + risks[0]
    return "topic:" + Counter(r["topic_scope"] for r in records).most_common(1)[0][0]


def split(records: list[dict], seed: int) -> dict[str, list[str]]:
    gmap = merged_groups(records)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[gmap[r["id"]]].append(r)
    strata: dict[str, list[str]] = defaultdict(list)
    for g, recs in groups.items():
        strata[_stratum(recs)].append(g)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n = len(gs)
        bounds, acc = [], 0.0
        for name, frac in PARTITIONS:
            acc += frac
            bounds.append((name, round(acc * n)))
        i = 0
        for name, upper in bounds:
            while i < upper:
                assignment[gs[i]] = name
                i += 1
    parts: dict[str, list[str]] = {name: [] for name, _ in PARTITIONS}
    for r in records:
        parts[assignment[gmap[r["id"]]]].append(r["id"])
    return parts


def class_counts(records: list[dict], ids: list[str]) -> dict:
    idset = set(ids)
    sub = [r for r in records if r["id"] in idset]
    out = {"n": len(sub), "topic": dict(Counter(r["topic_scope"] for r in sub)),
           "mode": dict(Counter(r["mode"] for r in sub if r["mode"])), "risks": {}}
    for r in sub:
        for k, v in r["risks"].items():
            out["risks"].setdefault(k, 0)
            out["risks"][k] += int(v)
    return out


def build_manifest(data_path: str, records: list[dict], seed: int) -> dict:
    parts = split(records, seed)
    body = {
        "schema": "gate-split-0.2",
        "seed": seed,
        "dataset_sha256": dataset_sha256(data_path),
        "partitions": parts,
        "class_counts": {k: class_counts(records, v) for k, v in parts.items()},
        "note": "holdout is restricted: evaluate only after freezing candidate and thresholds",
    }
    body["split_sha256"] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    records = load_jsonl(args.data)
    manifest = build_manifest(args.data, records, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    for k, v in manifest["class_counts"].items():
        print(k, v["n"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
