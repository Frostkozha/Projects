"""Evaluation metrics (spec section 16.3). Report denominators and uncertainty, never point values alone."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    """Equal-width ECE: sum over nonempty bins of (count/N) * |mean_pred - positive_fraction|."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, bins - 1)
    total, table = 0.0, []
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            table.append({"bin": b, "count": 0})
            continue
        mp, pf = float(probs[m].mean()), float(labels[m].mean())
        total += cnt / n * abs(mp - pf)
        table.append({"bin": b, "count": cnt, "mean_prediction": mp, "positive_fraction": pf})
    return {"ece": total if n else float("nan"), "bins": table, "n": n}


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - np.asarray(labels, dtype=float)) ** 2)) if len(probs) else float("nan")


def recall_report(hits: int, n: int) -> dict:
    lo, hi = wilson_interval(hits, n)
    return {"successes": hits, "n": n, "recall": hits / n if n else float("nan"), "wilson95": [lo, hi]}


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> dict:
    per = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return {"macro_f1": float(np.mean([v["f1"] for v in per.values()])), "per_class": per}
