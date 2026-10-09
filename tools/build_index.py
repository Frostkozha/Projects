"""Build (and optionally activate) a retriever snapshot from approved local sources.

Usage:
  python -m tools.build_index --profile config/retriever_development.yaml --register REGISTER.json \
      --import-root DIR --tenant dev-tenant --course histology-dev --kb-version kb-001 \
      --index-version idx-001 [--items ITEMS.json] [--conflicts CONFLICTS.json] \
      [--synthetic-fixture] [--approve-review REVIEWER] [--activate]

The build is staged and validated; on failure the active snapshot is unchanged. Activation in
non-fixture modes requires an approved review sample (--approve-review records the named reviewer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gate_classifier.config import load_registry
from retriever.build import BuildError, approve_review, build_snapshot
from retriever.config import load_profile
from retriever.models import load_models, tokenizers_for
from retriever.register import SourceRegister
from retriever.service import build_service
from retriever.snapshot import SnapshotError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--register", required=True)
    ap.add_argument("--import-root", required=True)
    ap.add_argument("--library-registry", default="config/library_registry.yaml")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--course", required=True)
    ap.add_argument("--kb-version", required=True)
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--items")
    ap.add_argument("--conflicts")
    ap.add_argument("--synthetic-fixture", action="store_true")
    ap.add_argument("--approve-review", metavar="REVIEWER")
    ap.add_argument("--activate", action="store_true")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    encoder, reranker, errors = load_models(profile)
    if errors:
        print("model errors:", ", ".join(errors))
        return 2
    register = SourceRegister.load(args.register)
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))["items"] if args.items else []
    conflicts = json.loads(Path(args.conflicts).read_text(encoding="utf-8"))["conflicts"] if args.conflicts else []
    try:
        path = build_snapshot(
            profile=profile, register=register, import_root=args.import_root, snapshots_root=profile.snapshots_root,
            tenant_id=args.tenant, course_id=args.course, kb_version=args.kb_version, index_version=args.index_version,
            encoder=encoder, tokenizers=tokenizers_for(encoder, reranker),
            reranker_identity={"model_id": reranker.model_id, "revision": reranker.revision},
            items=items, conflicts=conflicts, synthetic_fixture=args.synthetic_fixture)
    except BuildError as exc:
        print(f"build failed: {exc.reason} (active snapshot unchanged)")
        return 1
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    print(f"built {path} passages={manifest['passage_count']} quarantined={manifest['quarantine_count']}")
    print(f"review sample ({len(manifest['review']['sample_passage_ids'])} passages) listed in manifest.json")
    if args.approve_review:
        approve_review(path, args.approve_review)
        print(f"review sample approved by {args.approve_review}")
    if args.activate:
        service = build_service(profile, load_registry(args.library_registry), encoder, reranker)
        try:
            service.snapshots.activate(args.course, args.index_version)
        except SnapshotError as exc:
            print(f"activation refused: {exc.reason}")
            return 1
        print(f"activated {args.index_version} for {args.course}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
