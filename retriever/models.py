"""Model loading for the retriever: pinned local real models or deterministic fixtures.

Real models load from local paths only (``local_files_only``); there is no download path. Fixture
models are test doubles and are refused by real-model and production readiness.
"""

from __future__ import annotations

from .config import RetrieverProfile
from .rerank import CrossEncoderReranker, FixtureReranker, overlap_score
from .schema import RetrievalError


def load_fixture_models(profile: RetrieverProfile):
    from gate_classifier.encoder import FixtureEncoder, FixtureTokenizer  # noqa: PLC0415

    tok = FixtureTokenizer()
    return FixtureEncoder(profile.encoder.dimension, profile.encoder.query_prefix), FixtureReranker(overlap_score, tok), []


def load_real_models(profile: RetrieverProfile):
    """Return (encoder, reranker, errors). Missing artifacts become readiness errors, never fallbacks."""
    from gate_classifier.encoder import E5Encoder, EncoderError  # noqa: PLC0415

    errors: list[str] = []
    encoder = reranker = None
    enc = profile.encoder
    try:
        encoder = E5Encoder(enc.local_path or "", enc.model_id, enc.revision or "", enc.tokenizer_revision or "",
                            enc.query_prefix, enc.dimension, enc.max_tokens, enc.torch_threads)
    except EncoderError as exc:
        errors.append(f"encoder:{exc}")
    rr = profile.reranker
    try:
        reranker = CrossEncoderReranker(rr.local_path or "", rr.model_id, rr.revision or "", rr.max_pair_tokens,
                                        rr.torch_threads)
    except RetrievalError as exc:
        errors.append(f"reranker:{exc.code.value}")
    return encoder, reranker, errors


def load_models(profile: RetrieverProfile):
    return load_fixture_models(profile) if profile.operating_mode == "fixture" else load_real_models(profile)


def tokenizers_for(encoder, reranker) -> list:
    """Both pinned tokenizers; chunk budgets use the larger count."""
    toks = []
    if encoder is not None:
        toks.append(encoder.tokenizer)
    if reranker is not None:
        toks.append(reranker.tokenizer)
    return toks
