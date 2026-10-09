"""Approved classifier heads and calibration artifacts (spec sections 4.2, 16.2, 18).

Bundles are JSON manifests plus NPZ numeric arrays loaded with ``allow_pickle=False``.
Pickle/joblib artifacts are never loaded. Probabilities reproduce scikit-learn's
``CalibratedClassifierCV(method="sigmoid")`` over a frozen ``LogisticRegression``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .schema import LIBRARY_IDS, MODES, RISK_LABELS, TOPIC_LABELS

BUNDLE_SCHEMA = "gate-bundle-0.2"
_EXAMPLE_RE = re.compile(r"example-only|fixture|placeholder", re.IGNORECASE)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class BundleError(RuntimeError):
    """Readiness failure. ``reason`` is a stable code, never file content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class LinearHead:
    """Logistic regression weights plus per-class sigmoid calibration (a, b)."""

    name: str
    kind: str  # "multiclass" | "binary"
    classes: tuple[str, ...]
    coef: np.ndarray
    intercept: np.ndarray
    cal_a: np.ndarray
    cal_b: np.ndarray

    def __post_init__(self):
        k = self.coef.shape[0]
        expected = len(self.classes) if self.kind == "multiclass" else 1
        if k != expected or self.intercept.shape != (k,) or self.cal_a.shape != (k,) or self.cal_b.shape != (k,):
            raise BundleError(f"head_shape_mismatch:{self.name}")
        for arr in (self.coef, self.intercept, self.cal_a, self.cal_b):
            if not np.all(np.isfinite(arr)):
                raise BundleError(f"head_non_finite:{self.name}")

    @property
    def dimension(self) -> int:
        return int(self.coef.shape[1])

    def decision(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef.T + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated probabilities; binary heads return P(positive) with shape (n,)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = self.decision(x)
        p = _expit(-(self.cal_a * z + self.cal_b))
        if self.kind == "binary":
            return p[:, 0]
        denom = p.sum(axis=1, keepdims=True)
        uniform = np.full_like(p, 1.0 / p.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denom == 0, uniform, p / np.where(denom == 0, 1.0, denom))
        out[(out > 1.0) & (out <= 1.0 + 1e-5)] = 1.0
        return out

    def save(self, path: Path) -> str:
        with open(path, "wb") as fh:
            np.savez(fh, coef=self.coef, intercept=self.intercept, cal_a=self.cal_a, cal_b=self.cal_b)
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path, name: str, kind: str, classes: tuple[str, ...]) -> "LinearHead":
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"coef", "intercept", "cal_a", "cal_b"}:
                    raise BundleError(f"head_arrays_mismatch:{name}")
                arrays = {k: np.array(data[k], dtype=np.float64) for k in data.files}
        except BundleError:
            raise
        except Exception:
            raise BundleError(f"head_unreadable:{name}") from None
        return cls(name, kind, classes, arrays["coef"], arrays["intercept"], arrays["cal_a"], arrays["cal_b"])


@dataclass
class HeadSet:
    mode: LinearHead
    topic_scope: LinearHead
    risks: dict[str, LinearHead]
    libraries: dict[str, LinearHead] = field(default_factory=dict)

    def predict_mandatory(self, x: np.ndarray) -> dict:
        pm = self.mode.predict_proba(x)[0]
        pt = self.topic_scope.predict_proba(x)[0]
        return {
            "mode": {c: float(v) for c, v in zip(self.mode.classes, pm)},
            "topic_scope": {c: float(v) for c, v in zip(self.topic_scope.classes, pt)},
            "risks": {r: float(self.risks[r].predict_proba(x)[0]) for r in RISK_LABELS},
        }

    def predict_libraries(self, x: np.ndarray, enabled: tuple[str, ...]) -> dict:
        out: dict[str, Optional[float]] = {lib: None for lib in LIBRARY_IDS}
        for lib in enabled:
            head = self.libraries.get(lib)
            if head is not None:
                out[lib] = float(head.predict_proba(x)[0])
        return out


@dataclass
class Bundle:
    manifest: dict
    heads: HeadSet
    path: Path

    @property
    def version(self) -> str:
        return self.manifest["bundle_version"]

    @property
    def is_fixture(self) -> bool:
        return self.manifest.get("operating_mode") == "fixture"


def head_specs() -> dict[str, tuple[str, tuple[str, ...]]]:
    specs = {"mode": ("multiclass", MODES), "topic_scope": ("multiclass", TOPIC_LABELS)}
    for r in RISK_LABELS:
        specs[f"risk.{r}"] = ("binary", ("negative", "positive"))
    for lib in LIBRARY_IDS:
        specs[f"library.{lib}"] = ("binary", ("negative", "positive"))
    return specs


def load_bundle(path: str | Path, *, operating_mode: str, config_sha256: str, preprocessing_fingerprint: str,
                dimension: int) -> Bundle:
    """Validate and load a bundle. Raises BundleError with a stable reason code."""
    root = Path(path)
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise BundleError("manifest_missing")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BundleError("manifest_unreadable") from None
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleError("manifest_schema_mismatch")
    bundle_mode = manifest.get("operating_mode")
    if operating_mode != "fixture" and bundle_mode == "fixture":
        raise BundleError("fixture_bundle_not_allowed")
    if operating_mode == "production":
        if not manifest.get("evaluation", {}).get("production_acceptance_passed"):
            raise BundleError("production_acceptance_not_passed")
        if bundle_mode != "production":
            raise BundleError("bundle_not_production")
    if operating_mode != "fixture":
        _reject_example_values(manifest)
    if manifest.get("config_sha256") != config_sha256:
        raise BundleError("config_checksum_mismatch")
    enc = manifest.get("encoder", {})
    if enc.get("preprocessing_sha256") != preprocessing_fingerprint:
        raise BundleError("preprocessing_mismatch")
    if enc.get("dimension") != dimension:
        raise BundleError("dimension_mismatch")

    specs = head_specs()
    entries = manifest.get("heads", {})
    unknown = set(entries) - set(specs)
    if unknown:
        raise BundleError("unknown_head")
    loaded: dict[str, LinearHead] = {}
    for name, entry in entries.items():
        kind, classes = specs[name]
        if entry.get("kind") != kind or tuple(entry.get("classes", ())) != classes:
            raise BundleError(f"class_order_mismatch:{name}")
        if entry.get("calibration") != "sigmoid" or not entry.get("calibration_data_version"):
            raise BundleError(f"calibration_missing:{name}")
        fpath = root / entry.get("file", "")
        if fpath.suffix != ".npz" or not fpath.is_file() or fpath.resolve().parent != root.resolve():
            raise BundleError(f"head_file_missing:{name}")
        if sha256_file(fpath) != entry.get("sha256"):
            raise BundleError(f"checksum_mismatch:{name}")
        head = LinearHead.load(fpath, name, kind, classes)
        if head.dimension != dimension:
            raise BundleError(f"dimension_mismatch:{name}")
        loaded[name] = head
    for name in ("mode", "topic_scope", *[f"risk.{r}" for r in RISK_LABELS]):
        if name not in loaded:
            raise BundleError(f"mandatory_head_missing:{name}")
    heads = HeadSet(
        mode=loaded["mode"],
        topic_scope=loaded["topic_scope"],
        risks={r: loaded[f"risk.{r}"] for r in RISK_LABELS},
        libraries={n.split(".", 1)[1]: h for n, h in loaded.items() if n.startswith("library.")},
    )
    return Bundle(manifest=manifest, heads=heads, path=root)


def _reject_example_values(manifest: dict) -> None:
    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and _EXAMPLE_RE.search(v):
            raise BundleError("example_value_in_real_bundle")

    walk({k: v for k, v in manifest.items() if k not in ("notes",)})
    enc = manifest.get("encoder", {})
    if not _REVISION_RE.match(str(enc.get("revision", ""))) or not _REVISION_RE.match(str(enc.get("tokenizer_revision", ""))):
        raise BundleError("encoder_revision_invalid")
    for entry in manifest.get("heads", {}).values():
        if not _SHA_RE.match(str(entry.get("sha256", ""))):
            raise BundleError("head_checksum_invalid")


# ----------------------------------------------------------------------------- fixture scorer

FixtureScorer = Callable[[str], dict]
"""Development fixture: maps canonical text to a raw prediction dict. Never a trained model."""
