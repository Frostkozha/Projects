"""Warm retrieval latency benchmark (Retriever spec v0.2, sections 4.1, 13.3).

Usage: python -m tools.benchmark --profile CFG --course histology-dev --queries QUERIES.txt \
           [--clients 1,4,8] [--repeats 20]
Measures the full adapter path (validation, encode, both searches, rerank, selection, audit) on the
actual machine. Model loading is reported separately. Results are measurements, not promises.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np


def run(service, ctx, kb, course, active, queries, clients, repeats) -> dict:
    def one(q):
        body = {"request_id": str(uuid.uuid4()), "query_text": q, "allowed_libraries": active,
                "preferred_libraries": [], "course_id": course, "kb_version": kb, "strategy": "all_active"}
        t0 = time.perf_counter()
        r = service.retrieve(body, ctx)
        return (time.perf_counter() - t0) * 1000, r.status.value, r.error_code.value if r.error_code else None

    work = [q for _ in range(repeats) for q in queries]
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        out = list(pool.map(one, work))
    lat = np.array([o[0] for o in out])
    return {"clients": clients, "requests": len(out), "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)), "statuses": dict(Counter(o[1] for o in out)),
            "errors": dict(Counter(o[2] for o in out if o[2]))}


def main(argv=None) -> int:
    from gate_classifier.config import load_registry  # noqa: PLC0415
    from retriever.config import load_profile  # noqa: PLC0415
    from retriever.models import load_models  # noqa: PLC0415
    from retriever.schema import RetrievalContext  # noqa: PLC0415
    from retriever.service import build_service  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--library-registry", default="config/library_registry.yaml")
    ap.add_argument("--course", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--clients", default="1,4,8")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    t0 = time.perf_counter()
    encoder, reranker, errors = load_models(profile)
    load_ms = (time.perf_counter() - t0) * 1000
    if errors:
        print("model errors:", errors)
        return 2
    registry = load_registry(args.library_registry)
    service = build_service(profile, registry, encoder, reranker, courses=(args.course,))
    snap = service.snapshots.pin(args.course)
    active = sorted(registry.active_libraries(args.course))
    ctx = RetrievalContext(service_id=profile.service_ids[0], tenant_id=registry.courses[args.course].tenant_id,
                           course_id=args.course, authorized_libraries=frozenset(active),
                           registry_version=registry.registry_version, gate_permitted=True)
    queries = [q for q in Path(args.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    run(service, ctx, snap.kb_version, args.course, active, queries[:3], 1, 1)  # warm-up
    results = [run(service, ctx, snap.kb_version, args.course, active, queries, int(c), args.repeats)
               for c in args.clients.split(",")]
    try:
        import resource  # noqa: PLC0415  (POSIX only)

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # Windows: measure with Task Manager / Performance Monitor instead
        rss = None
    report = {"model_load_ms": load_ms, "passages": len(snap.rows), "operating_mode": profile.operating_mode,
              "peak_rss_mib": rss, "runs": results}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
