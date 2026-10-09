"""Validated retrieval profile (Retriever spec v0.2, sections 4, 8.2, 10, 14)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

OperatingMode = Literal["fixture", "real_model_development", "production"]


class ProfileError(ValueError):
    pass


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EncoderProfile(_Cfg):
    model_id: str = "intfloat/e5-small-v2"
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    local_path: Optional[str] = None
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    pooling: Literal["mean"] = "mean"
    normalization: Literal["l2"] = "l2"
    dtype: Literal["float32"] = "float32"
    dimension: int = 384
    max_tokens: int = 512
    torch_threads: int = 2


class RerankerProfile(_Cfg):
    model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: Optional[str] = None
    local_path: Optional[str] = None
    max_pair_tokens: int = 512
    batch_size: int = Field(8, ge=1, le=8)
    torch_threads: int = 2


class SearchProfile(_Cfg):
    dense_k: int = Field(30, ge=1)
    lexical_k: int = Field(30, ge=1)
    rrf_constant: int = Field(60, ge=1)
    rerank_k: int = Field(30, ge=1)
    max_passages: int = Field(5, ge=1, le=5)
    overlap_limit: float = Field(0.80, gt=0.0, le=1.0)


class ThresholdProfile(_Cfg):
    t_rerank: Optional[float] = None
    status: Literal["fixture", "provisional", "evaluated"] = "fixture"
    evaluation_ref: Optional[str] = None


class Limits(_Cfg):
    max_query_chars: int = 4000
    max_query_tokens: int = 320
    chunk_target_tokens: int = 144
    chunk_max_tokens: int = 160
    heading_max_tokens: int = 24
    overlap_max_tokens: int = 24
    pair_reserve_tokens: int = 8
    max_passages_per_snapshot: int = 50_000
    max_file_bytes: int = 100 * 1024 * 1024
    max_pdf_pages: int = 2000
    max_body_bytes: int = 32 * 1024


class Runtime(_Cfg):
    deadline_seconds: float = 5.0
    queue_capacity: int = 16
    stuck_worker_seconds: float = 30.0
    max_worker_restarts: int = 3


class RetrieverProfile(_Cfg):
    schema_version: Literal["retriever-profile-0.2"] = "retriever-profile-0.2"
    profile_version: str
    operating_mode: OperatingMode = "fixture"
    retrieval_policy_version: str = "retrieval-policy-0.2"
    query_builder_version: str = "fts5-literal-or-0.2"
    chunking_version: str = "chunk-sentence-0.2"
    parser_normalization_version: str = "nfc-ws-0.2"
    encoder: EncoderProfile = Field(default_factory=EncoderProfile)
    reranker: RerankerProfile = Field(default_factory=RerankerProfile)
    search: SearchProfile = Field(default_factory=SearchProfile)
    threshold: ThresholdProfile = Field(default_factory=ThresholdProfile)
    limits: Limits = Field(default_factory=Limits)
    runtime: Runtime = Field(default_factory=Runtime)
    snapshots_root: str = "artifacts/retriever/snapshots"
    revocations_path: str = "artifacts/retriever/revocations.json"
    import_root: str = "artifacts/retriever/sources"
    service_ids: tuple[str, ...] = ("orchestrator",)

    @model_validator(mode="after")
    def _budgets(self):
        lim = self.limits
        budget = lim.max_query_tokens + lim.chunk_max_tokens + lim.heading_max_tokens + lim.pair_reserve_tokens
        if budget > self.reranker.max_pair_tokens:
            raise ValueError("query + passage + heading + reserve exceed the reranker pair ceiling")
        if lim.chunk_target_tokens > lim.chunk_max_tokens:
            raise ValueError("chunk target exceeds chunk maximum")
        if self.search.rerank_k < self.search.max_passages:
            raise ValueError("rerank_k must be at least max_passages")
        if self.threshold.t_rerank is not None and self.threshold.t_rerank != self.threshold.t_rerank:
            raise ValueError("threshold must be a number")
        return self

    def readiness_problems(self) -> list[str]:
        """Profile-level problems for the configured mode (models/snapshots are checked elsewhere)."""
        problems = []
        th = self.threshold
        if th.t_rerank is None:
            problems.append("threshold_missing")
        if self.operating_mode == "fixture":
            if th.status != "fixture":
                problems.append("fixture_mode_requires_fixture_threshold")
        else:
            if th.status == "fixture":
                problems.append("fixture_threshold_not_allowed")
            for name, rev in (("encoder", self.encoder.revision), ("encoder_tokenizer", self.encoder.tokenizer_revision),
                              ("reranker", self.reranker.revision)):
                if not rev or len(rev) != 40:
                    problems.append(f"{name}_revision_unpinned")
        if self.operating_mode == "production":
            if th.status != "evaluated" or not th.evaluation_ref:
                problems.append("threshold_not_evaluated")
        return problems

    def ranking_fingerprint(self) -> str:
        payload = {
            "profile_version": self.profile_version,
            "policy": self.retrieval_policy_version,
            "query_builder": self.query_builder_version,
            "search": self.search.model_dump(),
            "threshold": self.threshold.t_rerank,
            "reranker": self.reranker.model_dump(exclude={"local_path", "torch_threads"}),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def representation_fingerprint(self) -> str:
        """Everything the stored vectors depend on. A change requires rebuilding the snapshot."""
        payload = {
            "encoder": self.encoder.model_dump(exclude={"local_path", "torch_threads"}),
            "chunking": self.chunking_version,
            "normalization": self.parser_normalization_version,
            "limits": {k: getattr(self.limits, k) for k in ("chunk_target_tokens", "chunk_max_tokens",
                                                             "heading_max_tokens", "overlap_max_tokens")},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_profile(path: str | Path) -> RetrieverProfile:
    try:
        return RetrieverProfile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ProfileError(f"invalid retriever profile: {type(exc).__name__}") from exc
