"""Select risk thresholds on the TUNING partition only (spec section 16.2, step 8).

Targets must be frozen before any final-holdout evaluation. Output is a YAML snippet for review;
it is never applied automatically.

Usage: python -m training.tune_thresholds DATA.jsonl --split split.json --bundle DIR --config CFG \
           --encoder e5|fixture [--block-recall 0.95] [--any-recall 0.99]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from gate_classifier.schema import RISK_LABELS

GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def choose_thresholds(probs: np.ndarray, labels: np.ndarray, block_recall: float, any_recall: float) -> dict:
    """Highest block threshold meeting block_recall; highest lower uncertainty threshold meeting any_recall."""
    pos = labels.astype(bool)
    if pos.sum() == 0:
        raise ValueError("no positives in tuning partition")

    def recall(t):
        return float((probs[pos] >= t).mean())

    block_ok = [t for t in GRID if recall(t) >= block_recall]
    block = float(max(block_ok)) if block_ok else float(GRID[0] + 0.01)
    unc_ok = [t for t in GRID if t < block and recall(t) >= any_recall]
    unc = float(max(unc_ok)) if unc_ok else float(GRID[0])
    if unc >= block:
        unc = round(block - 0.01, 2)
    neg = ~pos
    return {
        "uncertainty": round(unc, 2), "block": round(block, 2),
        "tuning_block_recall": recall(block), "tuning_any_recall": recall(unc),
        "block_target_met": bool(block_ok), "uncertainty_target_met": bool(unc_ok),
        "tuning_false_block_rate": float((probs[neg] >= block).mean()) if neg.any() else None,
        "tuning_clarify_or_block_rate_on_negatives": float((probs[neg] >= unc).mean()) if neg.any() else None,
        "positives": int(pos.sum()), "negatives": int(neg.sum()),
    }


def main(argv=None) -> int:
    from gate_classifier.config import load_config  # noqa: PLC0415
    from gate_classifier.heads import load_bundle  # noqa: PLC0415

    from .bundle import build_encoder  # noqa: PLC0415
    from .train import encode_records  # noqa: PLC0415
    from .validate_dataset import load_jsonl  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--split", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", choices=("e5", "fixture"), required=True)
    ap.add_argument("--block-recall", type=float, default=0.95)
    ap.add_argument("--any-recall", type=float, default=0.99)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    encoder = build_encoder(cfg, args.encoder)
    bundle = load_bundle(args.bundle, operating_mode="fixture" if args.encoder == "fixture" else cfg.operating_mode,
                         config_sha256=cfg.config_sha256(), preprocessing_fingerprint=encoder.preprocessing.fingerprint(),
                         dimension=cfg.encoder.dimension)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in load_jsonl(args.data)}
    tuning = [by_id[i] for i in split["partitions"]["tuning"]]
    X = encode_records(encoder, tuning, cfg.encoder.prefix)
    out = {}
    for risk in RISK_LABELS:
        probs = bundle.heads.risks[risk].predict_proba(X)
        labels = np.array([int(r["risks"][risk]) for r in tuning])
        try:
            out[risk] = choose_thresholds(probs, labels, args.block_recall, args.any_recall)
        except ValueError as exc:
            out[risk] = {"error": str(exc)}
    print(yaml.safe_dump({"proposed_risk_thresholds": out, "source": "tuning partition only"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
