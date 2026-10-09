"""Sigmoid calibration on the calibration partition only (spec section 16.2, step 7).

The base estimator is frozen (``FrozenEstimator``); it is never refit on calibration records.
Exported (a, b) parameters must reproduce scikit-learn's probabilities numerically.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from gate_classifier.heads import LinearHead

from .train import TrainingError, targets

MIN_CAL_PER_CLASS = 5
REPRO_TOLERANCE = 1e-6


def calibrate_one(name: str, spec, model, Xcal: np.ndarray, ycal: np.ndarray) -> tuple[LinearHead, dict]:
    present = set(np.unique(ycal).tolist())
    needed = set(range(len(spec.classes))) if spec.kind == "multiclass" else {0, 1}
    if not needed <= present:
        raise TrainingError(f"{name}: calibration partition lacks classes {sorted(needed - present)}")
    counts = np.bincount(ycal, minlength=len(needed))
    if counts.min() < MIN_CAL_PER_CLASS:
        raise TrainingError(f"{name}: fewer than {MIN_CAL_PER_CLASS} calibration examples for a class")
    if list(model.classes_) != sorted(needed):
        raise TrainingError(f"{name}: unexpected class order")
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    cal.fit(Xcal, ycal)
    calibrators = cal.calibrated_classifiers_[0].calibrators
    a = np.array([c.a_ for c in calibrators], dtype=np.float64)
    b = np.array([c.b_ for c in calibrators], dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    head = LinearHead(name, spec.kind, spec.classes, coef, intercept, a, b)
    ours = head.predict_proba(Xcal)
    ref = cal.predict_proba(Xcal)
    ref = ref[:, 1] if spec.kind == "binary" else ref
    err = float(np.max(np.abs(ours - ref))) if len(Xcal) else 0.0
    if err > REPRO_TOLERANCE:
        raise TrainingError(f"{name}: exported calibration does not reproduce scikit-learn (max err {err:.2e})")
    return head, {"method": "sigmoid", "n": int(len(ycal)), "class_counts": counts.tolist(), "repro_max_abs_err": err}


def calibrate_heads(models: dict, cal_records: list[dict], Xcal: np.ndarray, enabled_libs: tuple[str, ...]):
    tgt = targets(cal_records, enabled_libs)
    heads, meta = {}, {}
    for name, (spec, model) in models.items():
        _, mask, y = tgt[name]
        try:
            head, info = calibrate_one(name, spec, model, Xcal[mask], y[mask])
        except TrainingError as exc:
            if name.startswith("library."):
                meta[name] = {"skipped": str(exc)}  # null preferences until evaluated
                continue
            raise
        heads[name] = head
        meta[name] = info
    return heads, meta
