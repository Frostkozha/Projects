"""Layer 3: three-class NLI (Verifier spec v0.2, sections 5.4, 7).

premise = exact passage text, hypothesis = complete sentence text (never reversed). Raw logits from
AutoModelForSequenceClassification, numerically stable softmax, label mapping validated at load.
Token limits use the verifier's own tokenizer with truncation disabled: no windowing, summarizing or
clipping. Softmax outputs are model scores, not probabilities of clinical correctness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

LABELS = ("contradiction", "entailment", "neutral")


class NLIUnavailable(RuntimeError):
    """Model missing or incompatible (readiness false / MODEL_UNAVAILABLE)."""


class NLIOutputInvalid(RuntimeError):
    """Wrong shape or non-finite outputs (WORKER_FAILED)."""


class NLIBackend(Protocol):
    is_fixture: bool
    version: str
    label_index: dict[str, int]

    def count(self, text: str) -> int: ...

    def count_pair(self, premise: str, hypothesis: str) -> int: ...

    def logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_scores(logits: np.ndarray, label_index: dict[str, int], n: int) -> list[dict[str, float]]:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.shape != (n, 3) or not np.all(np.isfinite(arr)):
        raise NLIOutputInvalid("nli output shape or values invalid")
    probs = stable_softmax(arr)
    if not np.all(np.isfinite(probs)):
        raise NLIOutputInvalid("non-finite softmax")
    return [{lab: float(row[label_index[lab]]) for lab in LABELS} for row in probs]


def validate_label_mapping(id2label: dict, expected: dict[str, str]) -> dict[str, int]:
    """Map {label: index}; any missing/extra/swapped label fails readiness."""
    got = {str(k): str(v).lower() for k, v in (id2label or {}).items()}
    exp = {str(k): str(v).lower() for k, v in expected.items()}
    if got != exp or set(got.values()) != set(LABELS):
        raise NLIUnavailable("label mapping mismatch")
    return {lab: int(i) for i, lab in got.items()}


# ----------------------------------------------------------------------------- fixture (unit tests only)


class FixtureTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class FixtureNLI:
    """Injected scores for unit tests. ``score_fn(premise, hypothesis) -> (contradiction, entailment, neutral)``.

    Never a production backend; readiness outside fixture mode refuses it.
    """

    is_fixture = True
    version = "fixture-nli-0.2"

    def __init__(self, score_fn: Callable[[str, str], tuple[float, float, float]], label_index=None):
        self.score_fn = score_fn
        self.label_index = label_index or {"contradiction": 0, "entailment": 1, "neutral": 2}
        self._tok = FixtureTokenCounter()
        self.calls = 0
        self.pairs_seen: list[tuple[str, str]] = []

    def count(self, text: str) -> int:
        return self._tok.count(text)

    def count_pair(self, premise: str, hypothesis: str) -> int:
        return self.count(premise) + self.count(hypothesis) + 3

    def logits(self, pairs):
        self.calls += 1
        self.pairs_seen.extend(pairs)
        out = np.zeros((len(pairs), 3))
        for i, (p, h) in enumerate(pairs):
            c, e, n = self.score_fn(p, h)
            vals = {"contradiction": c, "entailment": e, "neutral": n}
            for lab, idx in self.label_index.items():
                out[i, idx] = np.log(max(vals[lab], 1e-12))  # softmax(log p) reproduces p
        return out


def substring_scores(premise: str, hypothesis: str) -> tuple[float, float, float]:
    """Deterministic fixture rule: hypothesis text contained in premise -> entailed, else neutral."""
    norm = lambda s: " ".join(s.lower().replace(".", " ").split())  # noqa: E731
    if norm(hypothesis) and norm(hypothesis) in norm(premise):
        return (0.01, 0.97, 0.02)
    return (0.05, 0.05, 0.90)


# ----------------------------------------------------------------------------- real model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TokenizerCounter:
    """Parent-side verifier tokenizer for limit checks when the model runs in a supervised subprocess."""

    def __init__(self, local_path: str):
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._tok = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        except Exception:
            raise NLIUnavailable("verifier tokenizer failed to load") from None

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False, truncation=False)["input_ids"])

    def count_pair(self, premise: str, hypothesis: str) -> int:
        return len(self._tok(premise, hypothesis, truncation=False)["input_ids"])


class TransformersNLI:
    """Pinned local three-class cross-encoder (default cross-encoder/nli-deberta-v3-xsmall), CPU float32."""

    is_fixture = False

    def __init__(self, local_path: str, model_id: str, revision: Optional[str], expected_labels: dict[str, str],
                 max_pair_tokens: int, weights_sha256: Optional[str] = None, torch_threads: int = 4):
        if not revision or len(revision) != 40:
            raise NLIUnavailable("revision must be a pinned 40-character commit")
        path = Path(local_path)
        weights = path / "model.safetensors"
        if not weights.is_file():
            raise NLIUnavailable("model.safetensors missing")
        digest = _sha256(weights)
        if weights_sha256 and digest != weights_sha256:
            raise NLIUnavailable("weights hash mismatch")
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            raise NLIUnavailable("torch/transformers unavailable") from None
        torch.set_num_threads(max(1, torch_threads))
        self._torch = torch
        try:
            self._tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32)
        except Exception:
            raise NLIUnavailable("model or tokenizer failed to load") from None
        cfg = self._model.config
        if getattr(cfg, "num_labels", None) != 3:
            raise NLIUnavailable("not a three-class model")
        self.label_index = validate_label_mapping(getattr(cfg, "id2label", {}), expected_labels)
        if getattr(cfg, "max_position_embeddings", 0) < max_pair_tokens:
            raise NLIUnavailable("pair ceiling exceeds model positions")
        self._model.eval()
        self.weights_sha256 = digest
        self.version = f"{model_id}@{revision}"
        smoke = self.logits([("Cells line the alveoli.", "Cells line the alveoli.")])
        to_scores(smoke, self.label_index, 1)  # finite smoke test

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False, truncation=False)["input_ids"])

    def count_pair(self, premise: str, hypothesis: str) -> int:
        return len(self._tok(premise, hypothesis, truncation=False)["input_ids"])

    def logits(self, pairs):
        torch = self._torch
        batch = self._tok([p for p, _ in pairs], [h for _, h in pairs], padding=True, truncation=False,
                          return_tensors="pt")
        with torch.inference_mode():
            out = self._model(**batch).logits
        return out.float().cpu().numpy()
