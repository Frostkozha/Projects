"""Local E5 encoding and filtered exact vector search (Retriever spec v0.2, sections 4, 6.1, 10.1)."""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol, Sequence

import numpy as np

from .config import RetrieverProfile
from .schema import EmbeddingRef, ErrorCode, RetrievalError

UNIT_NORM_TOLERANCE = 1e-4


class RetrievalEncoder(Protocol):
    is_fixture: bool
    tokenizer: object

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def encoder_identity(encoder, profile: RetrieverProfile) -> dict:
    """Fields an EmbeddingRef must match exactly. Model names alone never establish compatibility."""
    pre = getattr(encoder, "preprocessing", None)
    return {
        "encoder_id": getattr(pre, "encoder_id", profile.encoder.model_id),
        "encoder_revision": getattr(pre, "revision", None),
        "tokenizer_revision": getattr(pre, "tokenizer_revision", None),
        "pooling": profile.encoder.pooling,
        "dimension": profile.encoder.dimension,
        "dtype": profile.encoder.dtype,
        "normalization": profile.encoder.normalization,
    }


def preprocessing_fingerprint(encoder, profile: RetrieverProfile) -> str:
    ident = encoder_identity(encoder, profile)
    blob = "|".join(f"{k}={ident[k]}" for k in sorted(ident))
    blob += f"|query_prefix={profile.encoder.query_prefix!r}|passage_prefix={profile.encoder.passage_prefix!r}"
    return hashlib.sha256(blob.encode()).hexdigest()


def make_embedding_ref(vector: np.ndarray, encoded_text: str, encoder, profile: RetrieverProfile) -> EmbeddingRef:
    ident = encoder_identity(encoder, profile)
    return EmbeddingRef(vector=np.asarray(vector, dtype=np.float32), **ident,
                        text_sha256=hashlib.sha256(encoded_text.encode()).hexdigest(),
                        preprocessing_fingerprint=preprocessing_fingerprint(encoder, profile))


def validate_vector(vec, dimension: int) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.shape != (dimension,) or arr.dtype != np.float32 or not np.all(np.isfinite(arr)):
        raise RetrievalError(ErrorCode.SEARCH_FAILED)
    if abs(float(np.linalg.norm(arr)) - 1.0) > UNIT_NORM_TOLERANCE:
        raise RetrievalError(ErrorCode.SEARCH_FAILED)
    return arr


def reusable_vector(ref: Optional[EmbeddingRef], query_text: str, encoder, profile: RetrievalEncoder | RetrieverProfile
                    ) -> Optional[np.ndarray]:
    """Return the vector only when every field and the exact encoded text match; None means recompute.

    Raises INVALID_EMBEDDING_REF for a malformed reference (wrong type/shape/dtype/non-finite/not unit).
    """
    if ref is None:
        return None
    if not isinstance(ref, EmbeddingRef):
        raise RetrievalError(ErrorCode.INVALID_EMBEDDING_REF)
    vec = ref.vector
    if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.dtype != np.float32 or not np.all(np.isfinite(vec)):
        raise RetrievalError(ErrorCode.INVALID_EMBEDDING_REF)
    if vec.shape[0] != ref.dimension or abs(float(np.linalg.norm(vec)) - 1.0) > UNIT_NORM_TOLERANCE:
        raise RetrievalError(ErrorCode.INVALID_EMBEDDING_REF)
    expected = encoder_identity(encoder, profile)
    actual = {k: getattr(ref, k) for k in expected}
    encoded = profile.encoder.query_prefix + query_text
    if actual != expected or ref.preprocessing_fingerprint != preprocessing_fingerprint(encoder, profile):
        return None
    if ref.text_sha256 != hashlib.sha256(encoded.encode()).hexdigest():
        return None
    return vec


def encode_query(encoder, query_text: str, profile: RetrieverProfile) -> np.ndarray:
    try:
        out = encoder.encode([profile.encoder.query_prefix + query_text])
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError(ErrorCode.SEARCH_FAILED) from None
    out = np.asarray(out)
    if out.ndim != 2 or out.shape[0] != 1:
        raise RetrievalError(ErrorCode.SEARCH_FAILED)
    return validate_vector(out[0].astype(np.float32, copy=False) if out.dtype == np.float32 else out[0],
                           profile.encoder.dimension)


def dense_search(matrix: np.ndarray, passage_ids: Sequence[str], eligible_rows: np.ndarray, query: np.ndarray,
                 k: int) -> list[tuple[str, float]]:
    """Exact dot-product search over eligible rows only (filter before top-k). Ties: passage_id asc."""
    if eligible_rows.size == 0:
        return []
    try:
        scores = matrix[eligible_rows] @ query
    except Exception:
        raise RetrievalError(ErrorCode.SEARCH_FAILED) from None
    if scores.shape != (eligible_rows.size,) or not np.all(np.isfinite(scores)):
        raise RetrievalError(ErrorCode.SEARCH_FAILED)
    order = sorted(range(eligible_rows.size), key=lambda i: (-float(scores[i]), passage_ids[eligible_rows[i]]))
    return [(passage_ids[eligible_rows[i]], float(scores[i])) for i in order[:k]]
