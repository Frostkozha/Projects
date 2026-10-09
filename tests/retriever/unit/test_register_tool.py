"""tools.register_source: hashes, live vs pending status, replacement of the same id/version."""

import json

from retriever.register import SourceRegister
from tools.register_source import main


def test_register_tool(tmp_path):
    src = tmp_path / "sources"
    src.mkdir()
    (src / "n.md").write_text("# Epithelium\n\nSimple squamous epithelium is one layer of flat cells.\n", encoding="utf-8")
    reg = tmp_path / "register.json"
    base = ["--register", str(reg), "--import-root", str(src), "--file", "n.md", "--title", "Notes", "--owner", "Dept",
            "--library", "lib1", "--topics", "epithelium", "--review-due", "2027-06-30"]
    assert main(base + ["--source-id", "draft"]) == 0
    full = ["--source-id", "notes", "--rights-ref", "r-1", "--rights-reviewer", "me", "--rights-document", "email",
            "--approver", "Dr X"]
    assert main(base + full) == 0
    assert main(base + full) == 0  # same id/version replaces, not duplicates
    data = json.loads(reg.read_text(encoding="utf-8"))
    status = {s["source_id"]: s["status"] for s in data["sources"]}
    assert status == {"draft": "pending_review", "notes": "live"}
    register = SourceRegister.load(reg)
    assert register.get("notes", "1").build_problems() == []
    assert set(register.get("draft", "1").build_problems()) >= {"rights_missing", "content_approval_missing"}
