"""Dry-run source validation: register checks, safe import, parsing and chunking (no index written).

Usage: python -m tools.validate_sources --profile config/retriever_development.yaml --register REGISTER.json \
           --import-root DIR [--course histology-dev]
Prints quarantine reasons per source. Automated parsing never declares extraction clinically correct;
reviewed samples are still required before activation.
"""

from __future__ import annotations

import argparse
import sys

from retriever.chunk import Chunker
from retriever.config import load_profile
from retriever.models import load_models, tokenizers_for
from retriever.parse import ImportRejected, ParseError, parse_source
from retriever.register import SourceRegister


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--register", required=True)
    ap.add_argument("--import-root", required=True)
    ap.add_argument("--course")
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    encoder, reranker, errors = load_models(profile)
    if errors:
        print("model errors:", ", ".join(errors))
        return 2
    chunker = Chunker(profile, tokenizers_for(encoder, reranker))
    register = SourceRegister.load(args.register)
    bad = 0
    for key in sorted(register.records):
        rec = register.records[key]
        if args.course and args.course not in rec.course_ids:
            continue
        problems = rec.build_problems()
        if problems:
            print(f"QUARANTINE {rec.source_id}@{rec.source_version}: {', '.join(problems)}")
            bad += 1
            continue
        try:
            parsed = parse_source(rec, args.import_root, profile.limits.max_file_bytes, profile.limits.max_pdf_pages)
        except (ImportRejected, ParseError) as exc:
            print(f"REJECT     {rec.source_id}@{rec.source_version}: {exc}")
            bad += 1
            continue
        chunks, q = chunker.chunk(rec, parsed.blocks)
        quarantined = parsed.quarantined + q
        missing = [c for c in rec.extraction_checks if not any(c in ch.text for ch in chunks)]
        status = "OK        " if not quarantined and not missing else "ATTENTION "
        print(f"{status}{rec.source_id}@{rec.source_version}: blocks={len(parsed.blocks)} chunks={len(chunks)} "
              f"quarantined={len(quarantined)} failed_checks={len(missing)}")
        for item in quarantined:
            print(f"    quarantined {item.ref}: {item.reason}")
        bad += bool(missing)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
