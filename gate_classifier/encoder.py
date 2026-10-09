"""Pinned local encoder and tokenizer (spec section 4).

Runtime loads local files only (``local_files_only=True``); there is no download path here.
The fixture encoder/tokenizer are deterministic test doubles and are never "trained models".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class EncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Preprocessing:
    encoder_id: str
    revision: str | None
    tokenizer_revision: str | None
    pooling: str
    prefix: str
    normalization: str
    dimension: int

    def fingerprint(self) -> str:
        blob = "|".join(str(x) for x in (self.encoder_id, self.revision, self.tokenizer_revision, self.pooling,
                                         repr(self.prefix), self.normalization, self.dimension))
        return hashlib.sha256(blob.encode()).hexdigest()


class Tokenizer(Protocol):
    def count(self, text: str, add_special_tokens: bool) -> int: ...


class Encoder(Protocol):
    preprocessing: Preprocessing
    is_fixture: bool

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


# ----------------------------------------------------------------------------- fixtures

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class FixtureTokenizer:
    """Deterministic word-piece-like counter for fixture mode. Long words count as several pieces."""

    special_tokens = 2

    def count(self, text: str, add_special_tokens: bool) -> int:
        n = 0
        for tok in _TOKEN_RE.findall(text):
            n += max(1, (len(tok) + 5) // 6)
        return n + (self.special_tokens if add_special_tokens else 0)


class FixtureEncoder:
    """Feature-hashing embedding for tests and pipeline smoke runs. Not a trained encoder."""

    is_fixture = True

    def __init__(self, dimension: int = 384, prefix: str = "query: "):
        self.preprocessing = Preprocessing("fixture-hash-encoder", "fixture", "fixture", "mean", prefix, "l2", dimension)
        self.tokenizer = FixtureTokenizer()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        dim = self.preprocessing.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = [t.casefold() for t in _TOKEN_RE.findall(text)]
            feats = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


# ----------------------------------------------------------------------------- real encoder


class HFTokenizer:
    def __init__(self, local_path: str, revision: str | None):
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise EncoderError("transformers unavailable") from exc
        try:
            self._tok = AutoTokenizer.from_pretrained(local_path, revision=revision, local_files_only=True)
        except Exception:  # pragma: no cover - depends on local artifacts
            raise EncoderError("tokenizer unavailable") from None

    def count(self, text: str, add_special_tokens: bool) -> int:
        # never truncate: counting must see the whole input
        return len(self._tok(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])

    def __call__(self, texts, **kw):
        return self._tok(texts, **kw)


class E5Encoder:
    """intfloat/e5-small-v2 frozen, CPU, float32, attention-masked mean pooling, L2-normalized."""

    is_fixture = False

    def __init__(self, local_path: str, model_id: str, revision: str, tokenizer_revision: str,
                 prefix: str = "query: ", dimension: int = 384, max_tokens: int = 512, torch_threads: int = 2):
        if not revision or not tokenizer_revision:
            raise EncoderError("encoder revision must be pinned")
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise EncoderError("torch/transformers unavailable") from exc
        torch.set_num_threads(max(1, torch_threads))
        self._torch = torch
        self.tokenizer = HFTokenizer(local_path, tokenizer_revision)
        try:
            self._model = AutoModel.from_pretrained(local_path, revision=revision, local_files_only=True,
                                                    trust_remote_code=False, torch_dtype=torch.float32)
        except Exception:  # pragma: no cover
            raise EncoderError("encoder weights unavailable") from None
        self._model.eval()
        hidden = getattr(self._model.config, "hidden_size", None)
        if hidden != dimension:
            raise EncoderError("encoder dimension mismatch")
        self.max_tokens = max_tokens
        self.preprocessing = Preprocessing(model_id, revision, tokenizer_revision, "mean", prefix, "l2", dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(list(texts), padding=True, truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > self.max_tokens:
            raise EncoderError("input exceeds encoder limit")
        with torch.inference_mode():
            hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


def build_canonical_text(redacted_text: str, topic: str | None = None, pending_question: str | None = None) -> str:
    parts = [f"Current request: {redacted_text}"]
    if topic:
        parts.append(f"Study topic: {topic}")
    if pending_question:
        parts.append(f"Pending question: {pending_question}")
    return "\n".join(parts)
