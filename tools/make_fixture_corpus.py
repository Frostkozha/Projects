"""Write the SYNTHETIC retriever fixture corpus (fake text, fake approvals) for smoke runs.

Usage: python -m tools.make_fixture_corpus --out artifacts/retriever/fixture
Creates sources/, register.json, items.json and queries.txt. Never use this as course material.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.retriever.helpers import write_corpus


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imports, register = write_corpus(out)
    (out / "queries.txt").write_text("\n".join([
        "What does simple squamous epithelium line?", "Compare simple squamous and simple cuboidal epithelium.",
        "How is hyaline cartilage nourished?", "Which epithelium lines the urinary bladder?",
        "What organises compact bone?"]) + "\n", encoding="utf-8")
    print(f"synthetic corpus: {imports}\nregister: {register}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
