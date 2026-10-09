"""Run one retrieval against the active snapshot and print the evidence (development use only).

Usage: python -m tools.search --profile config/retriever_real_dev.yaml --course histology-dev --query "..."
Prints status, raw reranker logits and citations. Results are UNEVALUATED unless the profile threshold
status is 'evaluated'. Not a student-facing tool.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from gate_classifier.config import load_registry
from retriever.config import load_profile
from retriever.models import load_models
from retriever.schema import RetrievalContext
from retriever.service import build_service
from retriever.snapshot import SnapshotError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--library-registry", default="config/library_registry.yaml")
    ap.add_argument("--course", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    encoder, reranker, errors = load_models(profile)
    if errors:
        print("model errors:", ", ".join(errors))
        return 2
    registry = load_registry(args.library_registry)
    service = build_service(profile, registry, encoder, reranker, courses=(args.course,))
    try:
        snap = service.snapshots.pin(args.course)
    except SnapshotError as exc:
        print(f"no usable active snapshot: {exc.reason}")
        return 2
    active = sorted(registry.active_libraries(args.course))
    ctx = RetrievalContext(service_id=profile.service_ids[0], tenant_id=registry.courses[args.course].tenant_id,
                           course_id=args.course, authorized_libraries=frozenset(active),
                           registry_version=registry.registry_version, gate_permitted=True)
    body = {"request_id": str(uuid.uuid4()), "query_text": args.query, "allowed_libraries": active,
            "preferred_libraries": [], "course_id": args.course, "kb_version": snap.kb_version,
            "strategy": "all_active"}
    r = service.retrieve(body, ctx)
    label = "evaluated" if profile.threshold.status == "evaluated" else "UNEVALUATED (provisional threshold)"
    print(f"status={r.status.value} reason={r.reason.value} error={r.error_code.value if r.error_code else None}")
    print(f"kb={r.kb_version} index={r.index_version} threshold={profile.threshold.t_rerank} [{label}]")
    for i, p in enumerate(r.passages, 1):
        print(f"\n[{i}] logit={p.relevance_score:+.3f}  {p.title} - {p.locator.label} ({p.library_id})")
        print(f"    {p.text[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
