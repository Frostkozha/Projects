"""Train logistic heads on the training partition only (spec section 16.2, steps 3-6).

Usage:
  python -m training.train DATA.jsonl --split split_manifest.json --config config/development.yaml \
      --registry config/library_registry.yaml --course histology-dev --encoder e5 --out artifacts/bundles/candidate
  (--encoder fixture builds an operating_mode=fixture bundle for pipeline smoke tests only)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from gate_classifier.encoder import build_canonical_text
from gate_classifier.privacy import Redactor
from gate_classifier.schema import MODES, RISK_LABELS, TOPIC_LABELS

C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)


class TrainingError(RuntimeError):
    pass


def canonical_inputs(records: list[dict], prefix: str, redactor: Optional[Redactor] = None) -> list[str]:
    """Exactly the runtime format: redacted current request + optional valid context, with prefix."""
    redactor = redactor or Redactor()
    out = []
    for r in records:
        ctx = r.get("session_context") or {}
        red = redactor.redact(r["text"]).text
        topic = redactor.redact(ctx["topic"]).text if ctx.get("topic") else None
        pending = redactor.redact(ctx["pending_question"]).text if ctx.get("pending_question") else None
        out.append(prefix + build_canonical_text(red, topic, pending))
    return out


def encode_records(encoder, records: list[dict], prefix: str, batch: int = 32) -> np.ndarray:
    texts = canonical_inputs(records, prefix)
    chunks = [encoder.encode(texts[i:i + batch]) for i in range(0, len(texts), batch)]
    return np.vstack(chunks).astype(np.float64) if chunks else np.zeros((0, encoder.preprocessing.dimension))


@dataclass
class TargetSpec:
    name: str
    kind: str  # multiclass | binary
    classes: tuple[str, ...]


def targets(records: list[dict], enabled_libs: tuple[str, ...]) -> dict[str, tuple[TargetSpec, np.ndarray, np.ndarray]]:
    """Return name -> (spec, record mask, y). Null labels are masked out of loss and metrics."""
    out = {}
    mode_mask = np.array([r["mode"] is not None for r in records])
    out["mode"] = (TargetSpec("mode", "multiclass", MODES), mode_mask,
                   np.array([MODES.index(r["mode"]) if r["mode"] else -1 for r in records]))
    out["topic_scope"] = (TargetSpec("topic_scope", "multiclass", TOPIC_LABELS), np.ones(len(records), bool),
                          np.array([TOPIC_LABELS.index(r["topic_scope"]) for r in records]))
    for risk in RISK_LABELS:
        out[f"risk.{risk}"] = (TargetSpec(f"risk.{risk}", "binary", ("negative", "positive")),
                               np.ones(len(records), bool), np.array([int(r["risks"][risk]) for r in records]))
    for lib in enabled_libs:
        mask = np.array([r["libraries"].get(lib) is not None and bool(r["library_labels_verified"])
                         and bool(r["answering_passage_ids"]) for r in records])
        y = np.array([int(bool(r["libraries"].get(lib))) for r in records])
        out[f"library.{lib}"] = (TargetSpec(f"library.{lib}", "binary", ("negative", "positive")), mask, y)
    return out


def _require_classes(spec: TargetSpec, y: np.ndarray) -> None:
    present = set(np.unique(y).tolist())
    needed = set(range(len(spec.classes))) if spec.kind == "multiclass" else {0, 1}
    if not needed <= present:
        raise TrainingError(f"{spec.name}: training partition lacks classes {sorted(needed - present)}")


def _fit(X, y, C, class_weight, seed):
    model = LogisticRegression(C=C, max_iter=5000, class_weight=class_weight, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, y)
        except ConvergenceWarning:
            raise TrainingError("convergence warning: adjust regularization or data before continuing") from None
    return model


def select_and_fit(spec: TargetSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    _require_classes(spec, y)
    class_weight = "balanced" if spec.name.startswith("risk.") else None
    n_groups = len(set(groups.tolist()))
    folds = min(5, n_groups)
    best_c, best = C_GRID[2], np.inf
    if folds >= 2:
        for C in C_GRID:
            losses = []
            for tr, va in GroupKFold(n_splits=folds).split(X, y, groups):
                if len(np.unique(y[tr])) < len(np.unique(y)):
                    continue
                m = _fit(X[tr], y[tr], C, class_weight, seed)
                losses.append(log_loss(y[va], m.predict_proba(X[va]), labels=m.classes_))
            if losses and np.mean(losses) < best:
                best, best_c = float(np.mean(losses)), C
    model = _fit(X, y, best_c, class_weight, seed)
    return model, {"C": best_c, "cv_log_loss": None if best == np.inf else best, "solver": model.solver,
                   "class_order": [spec.classes[int(c)] for c in model.classes_], "seed": seed,
                   "class_weight": class_weight, "n": int(len(y)), "positives": int(y.sum()) if spec.kind == "binary" else None}


def train_heads(records: list[dict], X: np.ndarray, enabled_libs: tuple[str, ...], seed: int):
    """Fit every head on the training partition. Mandatory heads must have all classes."""
    models, meta = {}, {}
    groups_all = np.array([r["group_id"] for r in records])
    for name, (spec, mask, y) in targets(records, enabled_libs).items():
        if mask.sum() == 0:
            if name.startswith("library."):
                continue  # optional head stays disabled (null scores)
            raise TrainingError(f"{name}: no labeled training records")
        try:
            model, info = select_and_fit(spec, X[mask], y[mask], groups_all[mask], seed)
        except TrainingError:
            if name.startswith("library."):
                meta[name] = {"skipped": "insufficient classes"}
                continue
            raise
        models[name] = (spec, model)
        meta[name] = info
    return models, meta


def main(argv=None) -> int:
    from gate_classifier.config import load_config, load_registry  # noqa: PLC0415

    from .bundle import build_encoder, write_bundle  # noqa: PLC0415
    from .calibrate import calibrate_heads  # noqa: PLC0415
    from .validate_dataset import dataset_sha256, load_jsonl, validate_dataset  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--course", required=True)
    ap.add_argument("--encoder", choices=("e5", "fixture"), required=True)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    registry = load_registry(args.registry)
    enabled = registry.enabled_labels(args.course)
    records = load_jsonl(args.data)
    problems = validate_dataset(records, enabled, allow_fixture=args.encoder == "fixture")
    if problems:
        print("\n".join(problems[:50]))
        raise TrainingError("dataset validation failed")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if split["dataset_sha256"] != dataset_sha256(args.data):
        raise TrainingError("split manifest does not match dataset")
    by_id = {r["id"]: r for r in records}
    train = [by_id[i] for i in split["partitions"]["train"]]
    cal = [by_id[i] for i in split["partitions"]["calibration"]]
    encoder = build_encoder(cfg, args.encoder)
    Xtr = encode_records(encoder, train, cfg.encoder.prefix)
    Xcal = encode_records(encoder, cal, cfg.encoder.prefix)
    models, meta = train_heads(train, Xtr, enabled, args.seed)
    heads, cal_meta = calibrate_heads(models, cal, Xcal, enabled)
    out = write_bundle(
        args.out, heads, cfg=cfg, encoder=encoder,
        operating_mode="fixture" if args.encoder == "fixture" else "development",
        dataset={"dataset_sha256": split["dataset_sha256"], "split_sha256": split["split_sha256"],
                 "calibration_data_version": split["split_sha256"][:16]},
        training={"heads": meta, "calibration": cal_meta},
    )
    print(f"bundle written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
