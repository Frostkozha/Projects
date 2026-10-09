"""Export JSON Schemas for the source register, profile, item/conflict records and retrieval contract.

Usage: python -m tools.export_schemas --out docs/schemas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retriever.config import RetrieverProfile
from retriever.fetch import ItemMapping
from retriever.register import SourceRecord
from retriever.schema import ItemFetchRequest, RetrievalRequest, RetrievalResult
from retriever.select import ConflictRecord

MODELS = {"source_record": SourceRecord, "retriever_profile": RetrieverProfile, "item_mapping": ItemMapping,
          "conflict_record": ConflictRecord, "retrieval_request": RetrievalRequest,
          "item_fetch_request": ItemFetchRequest, "retrieval_result": RetrievalResult}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n",
                                                 encoding="utf-8")
        print(f"wrote {name}.schema.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
