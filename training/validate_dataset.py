"""Validate reviewed dataset records (spec section 15).

Usage: python -m training.validate_dataset DATA.jsonl [--enabled-libs lib1,lib2,lib5,lib6] [--allow-fixture]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gate_classifier.privacy import Redactor
from gate_classifier.schema import LIBRARY_IDS, MODES, RISK_LABELS, TOPIC_LABELS

REQUIRED = {
    "id", "group_id", "source_kind", "text", "session_context", "mode", "topic_scope", "risks", "libraries",
    "library_labels_verified", "kb_version", "answering_passage_ids", "annotator_ids", "review_status", "notes",
}
REAL_SOURCE_KINDS = {"synthetic_reviewed", "faculty_reviewed", "participant_permitted", "public_licensed"}
FIXTURE_SOURCE_KINDS = {"synthetic_fixture", "privacy_fixture"}
APPROVED_REVIEW = {"adjudicated", "reviewed"}
DEFAULT_ENABLED = ("lib1", "lib2", "lib5", "lib6")


class DatasetError(ValueError):
    pass


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                raise DatasetError(f"line {n}: invalid JSON") from None
    return records


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_record(rec: dict, enabled_libs=DEFAULT_ENABLED, allow_fixture=False, redactor: Redactor | None = None) -> list[str]:
    problems = []
    rid = rec.get("id", "<missing id>")
    keys = set(rec)
    if keys != REQUIRED:
        problems.append(f"{rid}: keys mismatch (missing={sorted(REQUIRED - keys)}, extra={sorted(keys - REQUIRED)})")
        return problems
    if not isinstance(rec["text"], str) or not rec["text"].strip():
        problems.append(f"{rid}: empty text")
    if not isinstance(rec["group_id"], str) or not rec["group_id"]:
        problems.append(f"{rid}: missing group_id")
    kinds = REAL_SOURCE_KINDS | (FIXTURE_SOURCE_KINDS if allow_fixture else set())
    if rec["source_kind"] not in kinds:
        problems.append(f"{rid}: source_kind {rec['source_kind']!r} not permitted")
    if rec["review_status"] not in APPROVED_REVIEW:
        problems.append(f"{rid}: review_status must be one of {sorted(APPROVED_REVIEW)}")
    if rec["mode"] is not None and rec["mode"] not in MODES:
        problems.append(f"{rid}: invalid mode")
    if rec["topic_scope"] not in TOPIC_LABELS:
        problems.append(f"{rid}: invalid topic_scope")
    risks = rec["risks"]
    if not isinstance(risks, dict) or set(risks) != set(RISK_LABELS) or not all(isinstance(v, bool) for v in risks.values()):
        problems.append(f"{rid}: risks must have six booleans")
    libs = rec["libraries"]
    if not isinstance(libs, dict) or set(libs) != set(LIBRARY_IDS):
        problems.append(f"{rid}: libraries must declare lib1..lib6")
    else:
        for lib, v in libs.items():
            if v is not None and not isinstance(v, bool):
                problems.append(f"{rid}: library {lib} must be true/false/null")
            if lib not in enabled_libs and v is not None:
                problems.append(f"{rid}: disabled library {lib} must be null")
        if any(v is True for v in libs.values()):
            if not rec["library_labels_verified"]:
                problems.append(f"{rid}: positive library labels require library_labels_verified")
            if not rec["answering_passage_ids"] or not rec["kb_version"]:
                problems.append(f"{rid}: positive library labels require answering passages and kb_version")
    ctx = rec["session_context"]
    if ctx is not None and (not isinstance(ctx, dict) or set(ctx) - {"topic", "pending_question"}):
        problems.append(f"{rid}: session_context must be null or {{topic, pending_question}}")
    if rec["source_kind"] != "privacy_fixture" and isinstance(rec["text"], str):
        red = (redactor or Redactor()).redact(rec["text"])
        if red.detected:
            problems.append(f"{rid}: identifiers detected ({','.join(red.categories)}); keep identifiers out of training data")
    if len(rec["annotator_ids"]) < 1:
        problems.append(f"{rid}: annotator_ids required")
    return problems


def validate_dataset(records: list[dict], enabled_libs=DEFAULT_ENABLED, allow_fixture=False) -> list[str]:
    problems: list[str] = []
    seen = set()
    redactor = Redactor()
    for rec in records:
        rid = rec.get("id")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        problems.extend(validate_record(rec, enabled_libs, allow_fixture, redactor))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--enabled-libs", default=",".join(DEFAULT_ENABLED))
    ap.add_argument("--allow-fixture", action="store_true", help="accept synthetic fixtures (smoke tests only)")
    args = ap.parse_args(argv)
    records = load_jsonl(args.data)
    problems = validate_dataset(records, tuple(args.enabled_libs.split(",")), args.allow_fixture)
    for p in problems:
        print(p)
    print(f"{len(records)} records, {len(problems)} problems, sha256={dataset_sha256(args.data)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
