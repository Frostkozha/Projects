"""Grouped retrieval evaluation (Retriever spec v0.2, section 13).

Dataset: JSONL records with id, group_id, partition (development|final), query_text, course_id, kb_version,
topic, answerability (answerable|partial|unanswerable), relevant_passage_ids, sufficient_evidence_sets,
expected_conflict, annotator_ids, approval_status, missing_scope (unanswerable only).

Usage: python -m tools.evaluate DATA.jsonl --partition development|final --profile CFG --course histology-dev \
           --out report.json [--confirm-frozen]
The final partition requires --confirm-frozen and an evaluated threshold. Errors count as failures in
access/hit metrics and never as correct no_evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.metrics import recall_report, wilson_interval

REQUIRED = {"id", "group_id", "partition", "query_text", "course_id", "kb_version", "topic", "answerability",
            "relevant_passage_ids", "sufficient_evidence_sets", "expected_conflict", "annotator_ids",
            "approval_status"}
OPTIONAL = {"missing_scope", "notes"}
PROPOSED_TARGETS = {"hit_at_5": 0.90, "candidate_hit_at_30": 0.95, "sufficient_set_coverage_at_5": 0.90,
                    "context_precision_at_5": 0.80, "correct_no_evidence": 0.90, "false_no_evidence_max": 0.05,
                    "wilson_lower_bound": 0.85, "false_no_evidence_wilson_upper": 0.10}


class DatasetError(ValueError):
    pass


def load_dataset(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(records: list[dict]) -> None:
    """Fails on schema problems, unapproved labels or a paraphrase group crossing partitions."""
    seen, groups = set(), defaultdict(set)
    for r in records:
        keys = set(r)
        if not REQUIRED <= keys or keys - REQUIRED - OPTIONAL:
            raise DatasetError(f"{r.get('id')}: schema mismatch")
        if r["id"] in seen:
            raise DatasetError(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        if r["partition"] not in ("development", "final"):
            raise DatasetError(f"{r['id']}: unknown partition")
        if r["approval_status"] != "approved" or not r["annotator_ids"]:
            raise DatasetError(f"{r['id']}: labels not approved")
        if r["answerability"] not in ("answerable", "partial", "unanswerable"):
            raise DatasetError(f"{r['id']}: invalid answerability")
        if r["answerability"] == "unanswerable":
            if r["relevant_passage_ids"] or not r.get("missing_scope"):
                raise DatasetError(f"{r['id']}: unanswerable needs a reviewed missing_scope and no relevant IDs")
        elif not r["relevant_passage_ids"] or not r["sufficient_evidence_sets"]:
            raise DatasetError(f"{r['id']}: answerable/partial need relevant passages and sufficient sets")
        groups[r["group_id"]].add(r["partition"])
    crossing = [g for g, parts in groups.items() if len(parts) > 1]
    if crossing:
        raise DatasetError(f"paraphrase groups cross partitions: {sorted(crossing)[:5]}")


def evaluate(service, ctx_for, records: list[dict], active: list[str]) -> dict:
    answerable = [r for r in records if r["answerability"] != "unanswerable"]
    unanswerable = [r for r in records if r["answerability"] == "unanswerable"]
    hits = cand_hits = covered = correct_none = false_none = errors = violations = 0
    recalls, precisions, latencies = [], [], []
    for r in records:
        body = {"request_id": str(uuid.uuid4()), "query_text": r["query_text"], "allowed_libraries": active,
                "preferred_libraries": [], "course_id": r["course_id"], "kb_version": r["kb_version"],
                "strategy": "all_active"}
        ctx = ctx_for(r)
        t0 = time.perf_counter()
        res = service.retrieve(body, ctx)
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = [p.passage_id for p in res.passages]
        if res.status.value == "error":
            errors += 1
        for p in res.passages:
            if p.library_id not in active or p.review_status != "live":
                violations += 1
        rel = set(r["relevant_passage_ids"])
        if r["answerability"] == "unanswerable":
            correct_none += res.status.value == "no_evidence"
            continue
        false_none += res.status.value == "no_evidence"
        hits += bool(rel & set(ids))
        recalls.append(len(rel & set(ids)) / len(rel))
        precisions.append(len(rel & set(ids)) / len(ids) if ids else 0.0)
        covered += any(set(s) <= set(ids) for s in r["sufficient_evidence_sets"])
        try:
            diag = service.ranking_diagnostics(body, ctx)
            cand_hits += bool(rel & {pid for pid, _ in diag["fused"]})
        except Exception:
            pass  # an operational failure is a miss
    n_a, n_u = len(answerable), len(unanswerable)
    fn_lo, fn_hi = wilson_interval(false_none, n_a)
    lat = np.array(latencies or [float("nan")])
    return {
        "n_answerable": n_a, "n_unanswerable": n_u, "operational_errors": errors,
        "hit_at_5": recall_report(hits, n_a),
        "candidate_hit_at_30": recall_report(cand_hits, n_a),
        "passage_recall_at_5_mean": float(np.mean(recalls)) if recalls else None,
        "sufficient_set_coverage_at_5": recall_report(covered, n_a),
        "context_precision_at_5_mean": float(np.mean(precisions)) if precisions else None,
        "correct_no_evidence": recall_report(correct_none, n_u),
        "false_no_evidence": {"successes": false_none, "n": n_a, "rate": false_none / n_a if n_a else None,
                              "wilson95": [fn_lo, fn_hi]},
        "access_citation_violations": violations,
        "latency_ms_sequential": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "proposed_targets": PROPOSED_TARGETS,
    }


def main(argv=None) -> int:
    from gate_classifier.config import load_registry  # noqa: PLC0415
    from retriever.config import load_profile  # noqa: PLC0415
    from retriever.models import load_models  # noqa: PLC0415
    from retriever.schema import RetrievalContext  # noqa: PLC0415
    from retriever.service import build_service  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--partition", choices=("development", "final"), required=True)
    ap.add_argument("--confirm-frozen", action="store_true")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--library-registry", default="config/library_registry.yaml")
    ap.add_argument("--course", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    records = load_dataset(args.data)
    try:
        validate_dataset(records)
    except DatasetError as exc:
        print(f"dataset invalid: {exc}")
        return 2
    if args.partition == "final":
        if not args.confirm_frozen or profile.threshold.status != "evaluated" and profile.operating_mode != "fixture":
            print("final partition requires --confirm-frozen and a frozen profile")
            return 2
    subset = [r for r in records if r["partition"] == args.partition and r["course_id"] == args.course]
    encoder, reranker, errors = load_models(profile)
    if errors:
        print("model errors:", errors)
        return 2
    registry = load_registry(args.library_registry)
    service = build_service(profile, registry, encoder, reranker, courses=(args.course,))
    tenant = registry.courses[args.course].tenant_id
    active = sorted(registry.active_libraries(args.course))
    ctx = RetrievalContext(service_id=profile.service_ids[0], tenant_id=tenant, course_id=args.course,
                           authorized_libraries=frozenset(active), registry_version=registry.registry_version,
                           gate_permitted=True)
    report = evaluate(service, lambda r: ctx, subset, active)
    report.update({"partition": args.partition, "profile_version": profile.profile_version,
                   "threshold_status": profile.threshold.status,
                   "label": "UNEVALUATED PROFILE" if profile.threshold.status != "evaluated" else "evaluated profile"})
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_answerable", "hit_at_5", "correct_no_evidence", "label")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
