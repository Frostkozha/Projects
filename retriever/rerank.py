"""Cross-encoder reranking with raw logits (Retriever spec v0.2, sections 4, 8.2, 10.3).

Logits are scores, not calibrated probabilities: they may be negative or greater than one. Every
pair is length-checked in full before scoring; tokenizer truncation is never used.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from .schema import ErrorCode, RetrievalError


class PairTokenizer(Protocol):
    def count(self, text: str, add_special_tokens: bool) -> int: ...

    def count_pair(self, query: str, passage: str) -> int: ...

    def pair_special_tokens(self) -> int: ...


class Reranker(Protocol):
    is_fixture: bool
    tokenizer: PairTokenizer
    model_id: str
    revision: Optional[str]

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


def pair_document(heading: str, text: str) -> str:
    """Bounded heading representation + exact passage text; no E5 prefixes."""
    return f"{heading}\n{text}" if heading else text


def rerank(query: str, candidates: Sequence[str], documents: dict[str, str], reranker: Reranker,
           max_pair_tokens: int, batch_size: int, check_deadline: Callable[[], None]) -> dict[str, float]:
    """Return passage_id -> finite raw logit for every candidate, or raise a typed error."""
    pairs = []
    for pid in candidates:
        doc = documents[pid]
        try:
            n = reranker.tokenizer.count_pair(query, doc)
        except Exception:
            raise RetrievalError(ErrorCode.RERANK_FAILED) from None
        if n > max_pair_tokens:
            raise RetrievalError(ErrorCode.INPUT_TOO_LONG)
        pairs.append((pid, doc))
    out: dict[str, float] = {}
    for i in range(0, len(pairs), batch_size):
        check_deadline()
        batch = pairs[i:i + batch_size]
        try:
            logits = reranker.score([(query, doc) for _, doc in batch])
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError(ErrorCode.RERANK_FAILED) from None
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.shape != (len(batch),) or not np.all(np.isfinite(arr)):
            raise RetrievalError(ErrorCode.RERANK_FAILED)
        for (pid, _), logit in zip(batch, arr):
            out[pid] = float(logit)
    if set(out) != set(candidates):
        raise RetrievalError(ErrorCode.RERANK_FAILED)
    return out


# ----------------------------------------------------------------------------- fixture


class FixturePairTokenizer:
    """Deterministic counter wrapping a single-text tokenizer; [CLS] q [SEP] p [SEP] = 3 specials."""

    def __init__(self, base):
        self.base = base

    def count(self, text: str, add_special_tokens: bool) -> int:
        return self.base.count(text, add_special_tokens)

    def count_pair(self, query: str, passage: str) -> int:
        return self.base.count(query, False) + self.base.count(passage, False) + 3

    def pair_special_tokens(self) -> int:
        return 3


class FixtureReranker:
    """Deterministic test reranker. ``score_fn(query, document) -> logit``. Not a trained model."""

    is_fixture = True
    model_id = "fixture-reranker"
    revision = "fixture"

    def __init__(self, score_fn: Callable[[str, str], float], tokenizer):
        self.score_fn = score_fn
        self.tokenizer = FixturePairTokenizer(tokenizer)
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        return np.array([self.score_fn(q, d) for q, d in pairs], dtype=np.float64)


def overlap_score(query: str, document: str) -> float:
    """Fixture scoring: shared lowercase word count minus 2 (can be negative, like real logits)."""
    import re  # noqa: PLC0415

    q = set(re.findall(r"[^\W_]+", query.casefold()))
    d = set(re.findall(r"[^\W_]+", document.casefold()))
    return float(len(q & d)) - 2.0


# ----------------------------------------------------------------------------- real model


class HFPairTokenizer:
    def __init__(self, tok):
        self._tok = tok

    def count(self, text: str, add_special_tokens: bool) -> int:
        return len(self._tok(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])

    def count_pair(self, query: str, passage: str) -> int:
        return len(self._tok(query, passage, truncation=False)["input_ids"])

    def pair_special_tokens(self) -> int:
        # measured with real content: empty strings can collapse segments in some tokenizers
        q, p = "query", "passage"
        return self.count_pair(q, p) - self.count(q, False) - self.count(p, False)


class CrossEncoderReranker:
    """cross-encoder/ms-marco-MiniLM-L6-v2 from a pinned local path; single raw logit per pair."""

    is_fixture = False

    def __init__(self, local_path: str, model_id: str, revision: str, max_pair_tokens: int, torch_threads: int = 2):
        if not revision or len(revision) != 40:
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE)
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE) from None
        torch.set_num_threads(max(1, torch_threads))
        self._torch = torch
        try:
            tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                local_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
        except Exception:
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE) from None
        cfg = self._model.config
        if getattr(cfg, "num_labels", None) != 1:
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE)
        if getattr(cfg, "max_position_embeddings", 0) < max_pair_tokens:
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE)
        self._model.eval()
        self._tok = tok
        self.tokenizer = HFPairTokenizer(tok)
        self.model_id = model_id
        self.revision = revision

    def score(self, pairs):
        torch = self._torch
        qs = [q for q, _ in pairs]
        ds = [d for _, d in pairs]
        batch = self._tok(qs, ds, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**batch).logits  # raw logits, identity activation
        return logits.float().cpu().numpy()
