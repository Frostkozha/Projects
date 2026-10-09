"""Add or update one approved source in the source register (computes the file hash for you).

Usage (PowerShell, one line):
  python -m tools.register_source --register artifacts/retriever/real/register.json
      --import-root artifacts/retriever/sources --file histology/epithelium_lecture.md
      --source-id epithelium-lecture --version 1 --title "Epithelium lecture notes"
      --owner "Dept. of Histology" --library lib1 --topics epithelium
      --rights-ref rights-epithelium-2026 --rights-reviewer "Name" --rights-document "email-2026-10-09"
      --approver "Faculty name" --review-due 2027-06-30

Only record what is actually true. Leave --approver / --rights-* out if approval or written permission
does not exist yet: the source is then quarantined (never indexed) until the record is completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from retriever.parse import resolve_import
from retriever.register import SourceRecord

FORMATS = {".md": "markdown", ".json": "json", ".pdf": "pdf"}
PARSERS = {"markdown": "markdown", "json": "json", "pdf": "pdf_text"}


def _date(s: str) -> str:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).isoformat()


def build_record(a) -> dict:
    fmt = FORMATS.get(Path(a.file).suffix.lower())
    if fmt is None:
        raise SystemExit("unsupported format: use .md, .json or a text .pdf")
    path = resolve_import(a.import_root, a.file, fmt, 100 * 1024 * 1024)
    now = datetime.now(timezone.utc).isoformat()
    rights = None
    if a.rights_ref and a.rights_reviewer and a.rights_document:
        rights = {"rights_reference": a.rights_ref, "allows_local_indexing": True, "allows_excerpt_display": True,
                  "allows_model_training": False, "constraints": a.rights_constraints, "reviewer": a.rights_reviewer,
                  "reviewed_at": now, "supporting_document": a.rights_document}
    approval = {"approver": a.approver, "approved_at": now} if a.approver else None
    record = {
        "source_id": a.source_id, "source_version": a.version, "title": a.title, "owner": a.owner,
        "edition": a.edition, "publication_year": a.year, "source_tier": a.tier, "library_id": a.library,
        "course_ids": [a.course], "tenant_id": a.tenant, "topic_ids": a.topics.split(","), "language": "en",
        "file_ref": Path(a.file).as_posix(), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": fmt, "acquisition_record": a.acquisition, "rights": rights, "content_approval": approval,
        "status": "live" if (rights and approval) else "pending_review",
        "review_due_at": _date(a.review_due), "effective_from": now, "effective_to": None,
        "superseded_version": None,
        "parser": {"parser_id": PARSERS[fmt], "parser_version": "parsers-0.2", "normalization_version": "nfc-ws-0.2",
                   "parsed_output_approved": bool(approval), "verified_page_labels": False},
        "default_section": a.section, "extraction_checks": [c for c in a.check], "synthetic_fixture": False,
    }
    SourceRecord.model_validate(record)  # fail early on invalid IDs, dates or library
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", required=True)
    ap.add_argument("--import-root", default="artifacts/retriever/sources")
    ap.add_argument("--file", required=True, help="path relative to --import-root")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--version", default="1")
    ap.add_argument("--title", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--edition")
    ap.add_argument("--year", type=int)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--library", required=True, choices=["lib1", "lib2", "lib5", "lib6"])
    ap.add_argument("--course", default="histology-dev")
    ap.add_argument("--tenant", default="dev-tenant")
    ap.add_argument("--topics", required=True, help="comma-separated topic IDs, e.g. epithelium,glands")
    ap.add_argument("--section", default="Document", help="section name used for PDFs without headings")
    ap.add_argument("--acquisition", default="provided by course faculty")
    ap.add_argument("--rights-ref")
    ap.add_argument("--rights-reviewer")
    ap.add_argument("--rights-document", help="where the written permission is kept, e.g. email date/subject")
    ap.add_argument("--rights-constraints", default="local indexing and excerpt display for enrolled students only")
    ap.add_argument("--approver", help="faculty member who approved the content (omit if not yet approved)")
    ap.add_argument("--review-due", required=True, help="YYYY-MM-DD; after this date the source is excluded")
    ap.add_argument("--check", action="append", default=[],
                    help="exact phrase that must survive extraction (repeatable), e.g. a sentence with units")
    a = ap.parse_args(argv)
    rec = build_record(a)
    reg = Path(a.register)
    data = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {"sources": []}
    data["sources"] = [s for s in data["sources"]
                       if (s["source_id"], s["source_version"]) != (rec["source_id"], rec["source_version"])]
    data["sources"].append(rec)
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    state = "LIVE (indexable)" if rec["status"] == "live" else "PENDING REVIEW (will be quarantined)"
    print(f"{rec['source_id']}@{rec['source_version']} -> {state}; sha256={rec['file_sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
