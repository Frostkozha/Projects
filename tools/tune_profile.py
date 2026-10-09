"""Threshold sweep on the DEVELOPMENT partition only (Retriever spec v0.2, sections 13.2-13.3).

Usage: python -m tools.tune_profile DATA.jsonl --profile CFG --course histology-dev [--out sweep.json]
Reports Hit@5, sufficient-set coverage, context precision and correct/false no_evidence per raw-logit
threshold. It never edits the profile and refuses any record from the final partition.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from retriever.select import Candidate, select

from .evaluate import DatasetError, load_dataset, validate_dataset


def sweep(service, ctx, records: list[dict], active: list[str], thresholds: list[float]) -> list[dict]:
    if any(r["partition"] != "development" for r in records):
        raise DatasetError("tuning may only use the development partition")
    p = service.profile
    cached = []
    for r in records:
        body = {"request_id": str(uuid.uuid4()), "query_text": r["query_text"], "allowed_libraries": active,
                "preferred_libraries": [], "course_id": r["course_id"], "kb_version": r["kb_version"],
                "strategy": "all_active"}
        d = service.ranking_diagnostics(body, ctx)
        snap = d["snapshot"]
        cands = [Candidate(pid, d["logits"][pid], rrf, snap.by_id[pid].source_key, snap.by_id[pid].section_path,
                           snap.by_id[pid].text_sha256, snap.by_id[pid].spans) for pid, rrf in d["fused"]]
        cached.append((r, cands, snap.conflicts))
    rows = []
    for t in thresholds:
        hit = cov = prec_sum = cn = fn = n_a = n_u = 0
        for r, cands, conflicts in cached:
            try:
                ids = [c.passage_id for c in select(cands, t, p.search.max_passages, p.search.overlap_limit,
                                                    conflicts).passages]
            except Exception:
                ids = []
            rel = set(r["relevant_passage_ids"])
            if r["answerability"] == "unanswerable":
                n_u += 1
                cn += not ids
                continue
            n_a += 1
            fn += not ids
            hit += bool(rel & set(ids))
            cov += any(set(s) <= set(ids) for s in r["sufficient_evidence_sets"])
            prec_sum += len(rel & set(ids)) / len(ids) if ids else 0.0
        rows.append({"t_rerank": t, "hit_at_5": hit / n_a if n_a else None,
                     "sufficient_set_coverage": cov / n_a if n_a else None,
                     "context_precision": prec_sum / n_a if n_a else None,
                     "correct_no_evidence": cn / n_u if n_u else None,
                     "false_no_evidence": fn / n_a if n_a else None, "n_answerable": n_a, "n_unanswerable": n_u})
    return rows


def main(argv=None) -> int:
    from gate_classifier.config import load_registry  # noqa: PLC0415
    from retriever.config import load_profile  # noqa: PLC0415
    from retriever.models import load_models  # noqa: PLC0415
    from retriever.schema import RetrievalContext  # noqa: PLC0415
    from retriever.service import build_service  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--library-registry", default="config/library_registry.yaml")
    ap.add_argument("--course", required=True)
    ap.add_argument("--grid", default="-6,-4,-2,-1,0,1,2,4,6")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    records = load_dataset(args.data)
    try:
        validate_dataset(records)
    except DatasetError as exc:
        print(f"dataset invalid: {exc}")
        return 2
    dev = [r for r in records if r["partition"] == "development" and r["course_id"] == args.course]
    profile = load_profile(args.profile)
    encoder, reranker, errors = load_models(profile)
    if errors:
        print("model errors:", errors)
        return 2
    registry = load_registry(args.library_registry)
    service = build_service(profile, registry, encoder, reranker, courses=(args.course,))
    active = sorted(registry.active_libraries(args.course))
    ctx = RetrievalContext(service_id=profile.service_ids[0], tenant_id=registry.courses[args.course].tenant_id,
                           course_id=args.course, authorized_libraries=frozenset(active),
                           registry_version=registry.registry_version, gate_permitted=True)
    rows = sweep(service, ctx, dev, active, [float(x) for x in args.grid.split(",")])
    text = json.dumps({"partition": "development", "profile_version": profile.profile_version, "sweep": rows},
                      indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
