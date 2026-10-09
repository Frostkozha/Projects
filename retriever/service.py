"""Retriever service: deadlines, orchestration and typed failure results (Retriever spec v0.2, 5-7, 9-12, 14).

Python adapter boundary used by the orchestrator. Every call returns exactly one RetrievalResult;
validation and operational failures are typed errors, never no_evidence.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np
from pydantic import ValidationError

from .audit import InMemoryRetrievalAudit, RetrievalAuditRecord
from .chunk import chunking_fingerprint, sha256_text
from .config import RetrieverProfile
from .dense import dense_search, encode_query, preprocessing_fingerprint, reusable_vector
from .fuse import rrf_ids
from .lexical import lexical_search
from .register import RevocationRegistry, utcnow
from .rerank import pair_document, rerank
from .schema import (
    LIBRARY_IDS,
    ConflictOut,
    ErrorCode,
    EvidencePassage,
    ItemFetchRequest,
    Locator,
    Reason,
    RetrievalContext,
    RetrievalError,
    RetrievalRequest,
    RetrievalResult,
    Status,
)
from .select import Candidate, select
from .snapshot import PassageRow, Snapshot, SnapshotError, SnapshotManager

# ----------------------------------------------------------------------------- worker


class _Job:
    __slots__ = ("fn", "done", "result", "error", "cancelled", "started")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result = None
        self.error: Optional[BaseException] = None
        self.cancelled = False
        self.started: Optional[float] = None


class BoundedWorker:
    """One retrieval worker thread, bounded queue, cancellation and late-result suppression.

    Python threads cannot be killed: a stuck job is abandoned and a fresh worker thread started, up to
    ``max_restarts``; after that the worker is unhealthy and readiness turns false. A production
    deployment should run models in a killable worker process (see README, known limitations).
    """

    def __init__(self, capacity: int, stuck_seconds: float, max_restarts: int):
        self._q: queue.Queue[_Job] = queue.Queue(maxsize=capacity)
        self._stuck = stuck_seconds
        self._max_restarts = max_restarts
        self.restarts = 0
        self._gen = 0
        self._current: Optional[_Job] = None
        self._lock = threading.Lock()
        self._spawn()

    @property
    def healthy(self) -> bool:
        return self.restarts <= self._max_restarts

    def _spawn(self) -> None:
        self._gen += 1
        threading.Thread(target=self._loop, args=(self._gen,), daemon=True, name=f"retriever-{self._gen}").start()

    def _loop(self, gen: int) -> None:
        while gen == self._gen:
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job.cancelled:
                continue
            with self._lock:
                self._current = job
                job.started = time.monotonic()
            try:
                job.result = job.fn()
            except BaseException as exc:  # noqa: BLE001 - carried to the caller as a typed code only
                job.error = exc
            finally:
                with self._lock:
                    self._current = None
                job.done.set()

    def _check_stuck(self) -> None:
        with self._lock:
            cur = self._current
            stuck = cur is not None and cur.started is not None and time.monotonic() - cur.started > self._stuck
            if stuck:
                cur.cancelled = True
                self._current = None
        if stuck:
            self.restarts += 1
            self._spawn()

    def run(self, fn: Callable[[], object], timeout: float):
        self._check_stuck()
        if not self.healthy:
            raise RetrievalError(ErrorCode.SEARCH_FAILED)
        job = _Job(fn)
        try:
            self._q.put_nowait(job)
        except queue.Full:
            raise RetrievalError(ErrorCode.CAPACITY_EXCEEDED) from None
        if not job.done.wait(max(0.0, timeout)):
            job.cancelled = True  # a late result is never read
            raise RetrievalError(ErrorCode.DEADLINE_EXCEEDED)
        if job.error is not None:
            if isinstance(job.error, RetrievalError):
                raise job.error
            raise RetrievalError(ErrorCode.SEARCH_FAILED)
        return job.result


# ----------------------------------------------------------------------------- service


@dataclass
class Readiness:
    ready: bool
    status: str  # ready | fixture_only | not_ready
    operating_mode: str
    reasons: list[str] = field(default_factory=list)


def evidence_uri(passage_id: str, kb_version: str) -> str:
    """Server-side route builder; IDs are validated machine IDs and still URL-quoted."""
    return f"/v1/evidence/{quote(passage_id, safe='')}?kb_version={quote(kb_version, safe='')}"


class RetrieverService:
    def __init__(self, profile: RetrieverProfile, snapshots: SnapshotManager, revocations: RevocationRegistry,
                 library_registry, encoder, reranker, *, courses: tuple[str, ...] = (), audit=None,
                 now: Callable[[], datetime] = utcnow, monotonic: Callable[[], float] = time.monotonic,
                 worker: Optional[BoundedWorker] = None, load_errors: tuple[str, ...] = ()):
        self.profile = profile
        self.snapshots = snapshots
        self.revocations = revocations
        self.registry = library_registry
        self.encoder = encoder
        self.reranker = reranker
        self.courses = tuple(courses)
        self.audit = audit if audit is not None else InMemoryRetrievalAudit()
        self.now = now
        self.monotonic = monotonic
        self.worker = worker or BoundedWorker(profile.runtime.queue_capacity, profile.runtime.stuck_worker_seconds,
                                              profile.runtime.max_worker_restarts)
        self._load_errors = list(load_errors)
        self.encode_calls = 0

    # ------------------------------------------------------------------ readiness

    def readiness(self) -> Readiness:
        p = self.profile
        reasons = list(self._load_errors) + p.readiness_problems()
        if self.encoder is None:
            reasons.append("encoder_missing")
        if self.reranker is None:
            reasons.append("reranker_missing")
        if p.operating_mode != "fixture":
            if self.encoder is not None and getattr(self.encoder, "is_fixture", True):
                reasons.append("fixture_encoder_not_allowed")
            if self.reranker is not None and getattr(self.reranker, "is_fixture", True):
                reasons.append("fixture_reranker_not_allowed")
        if self.reranker is not None:
            try:
                if self.reranker.tokenizer.pair_special_tokens() > p.limits.pair_reserve_tokens:
                    reasons.append("reranker_pair_reserve_exceeded")
            except Exception:
                reasons.append("reranker_tokenizer_invalid")
        if not self.worker.healthy:
            reasons.append("worker_unhealthy")
        for course in self.courses:
            try:
                snap = self.snapshots.pin(course)
                if p.operating_mode != "fixture" and snap.is_fixture:
                    reasons.append("fixture_snapshot_not_allowed")
            except SnapshotError as exc:
                reasons.append(f"snapshot:{exc.reason}")
        reasons = sorted(set(reasons))
        if reasons:
            return Readiness(False, "not_ready", p.operating_mode, reasons)
        return Readiness(True, "fixture_only" if p.operating_mode == "fixture" else "ready", p.operating_mode, [])

    def _require_ready(self) -> None:
        r = self.readiness()
        if not r.ready:
            index = any(x.startswith("snapshot:") for x in r.reasons)
            raise RetrievalError(ErrorCode.INDEX_UNAVAILABLE if index else ErrorCode.MODEL_UNAVAILABLE)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _request_id(raw) -> str:
        rid = raw.get("request_id") if isinstance(raw, dict) else getattr(raw, "request_id", None)
        try:
            if isinstance(rid, str) and str(uuid.UUID(rid)) == rid.lower():
                return str(uuid.UUID(rid))
        except ValueError:
            pass
        return str(uuid.uuid4())

    def _check_context(self, ctx: RetrievalContext, course_id: str) -> None:
        if not isinstance(ctx, RetrievalContext) or ctx.service_id not in self.profile.service_ids:
            raise RetrievalError(ErrorCode.UNAUTHORIZED)
        if not ctx.gate_permitted or ctx.course_id != course_id:
            raise RetrievalError(ErrorCode.FORBIDDEN)
        course = self.registry.courses.get(course_id)
        if course is None or course.tenant_id != ctx.tenant_id:
            raise RetrievalError(ErrorCode.FORBIDDEN)

    def active_authorized(self, ctx: RetrievalContext) -> frozenset[str]:
        return frozenset(self.registry.active_libraries(ctx.course_id)) & frozenset(ctx.authorized_libraries)

    def _check_scope(self, req: RetrievalRequest, ctx: RetrievalContext) -> frozenset[str]:
        allowed = list(req.allowed_libraries)
        if not allowed:
            raise RetrievalError(ErrorCode.INVALID_SCOPE)  # never "search everything"
        if len(set(allowed)) != len(allowed) or any(lib not in LIBRARY_IDS for lib in allowed):
            raise RetrievalError(ErrorCode.INVALID_SCOPE)
        enabled = set(self.registry.enabled_labels(ctx.course_id))
        if any(lib not in enabled for lib in allowed):
            raise RetrievalError(ErrorCode.INVALID_SCOPE)
        if any(lib not in ctx.authorized_libraries for lib in allowed):
            raise RetrievalError(ErrorCode.FORBIDDEN)
        active = self.active_authorized(ctx)
        if set(allowed) != set(active):
            raise RetrievalError(ErrorCode.SCOPE_MISMATCH)
        pref = list(req.preferred_libraries)
        if len(set(pref)) != len(pref) or not set(pref) <= set(allowed):
            raise RetrievalError(ErrorCode.INVALID_SCOPE)
        return active

    def _check_query(self, text: str) -> None:
        lim = self.profile.limits
        if len(text) > lim.max_query_chars:
            raise RetrievalError(ErrorCode.INPUT_TOO_LONG)
        try:
            counts = [self.encoder.tokenizer.count(text, False), self.reranker.tokenizer.count(text, False)]
        except Exception:
            raise RetrievalError(ErrorCode.MODEL_UNAVAILABLE) from None
        if max(counts) > lim.max_query_tokens:
            raise RetrievalError(ErrorCode.INPUT_TOO_LONG)

    def _pin(self, ctx: RetrievalContext, kb_version: str) -> Snapshot:
        try:
            snap = self.snapshots.pin(ctx.course_id)
        except SnapshotError:
            raise RetrievalError(ErrorCode.INDEX_UNAVAILABLE) from None
        if snap.tenant_id != ctx.tenant_id or snap.course_id != ctx.course_id:
            raise RetrievalError(ErrorCode.INDEX_UNAVAILABLE)
        if snap.kb_version != kb_version:
            raise RetrievalError(ErrorCode.VERSION_MISMATCH)  # never silently search the latest corpus
        return snap

    def _usable(self, snap: Snapshot, p: PassageRow, active: frozenset[str], ctx: RetrievalContext,
                now: datetime) -> bool:
        if p.library_id not in active:
            return False
        rec = snap.sources.get(p.source_key)
        if rec is None or not rec.eligible(now, ctx.course_id, ctx.tenant_id):
            return False
        return not self.revocations.is_revoked(p.source_id, p.source_version)

    def _passage_out(self, snap: Snapshot, p: PassageRow, score: Optional[float]) -> EvidencePassage:
        if sha256_text(p.text) != p.text_sha256:
            raise RetrievalError(ErrorCode.INDEX_UNAVAILABLE)
        return EvidencePassage(
            passage_id=p.passage_id, source_id=p.source_id, source_version=p.source_version, title=p.title,
            library_id=p.library_id, locator=Locator.model_validate(p.locator), section_path=p.section_path,
            edition=p.edition, publication_year=p.publication_year, text=p.text, text_sha256=p.text_sha256,
            review_status=snap.sources[p.source_key].status, rights_reference=p.rights_reference,
            relevance_score=score, score_type="cross_encoder_logit" if score is not None else None,
            evidence_uri=evidence_uri(p.passage_id, snap.kb_version))

    def _versions(self, snap: Optional[Snapshot], epoch: Optional[int]) -> dict:
        return {"kb_version": snap.kb_version if snap else None, "index_version": snap.index_version if snap else None,
                "profile_version": self.profile.profile_version, "revocation_epoch": epoch}

    def _final_revalidation(self, snap, selected: list[PassageRow], active, ctx, epoch0: int) -> None:
        self.revocations.refresh()
        if self.revocations.epoch() != epoch0:
            raise RetrievalError(ErrorCode.SOURCE_STATE_CHANGED)
        now = self.now()
        for p in selected:
            if not self._usable(snap, p, active, ctx, now):
                raise RetrievalError(ErrorCode.SOURCE_STATE_CHANGED)

    def _record(self, result: RetrievalResult, op: str, ctx: Optional[RetrievalContext], **extra) -> RetrievalResult:
        rec = RetrievalAuditRecord(
            request_id=result.request_id, operation=op, service_id=getattr(ctx, "service_id", None),
            session_ref=getattr(ctx, "session_ref", None), course_id=getattr(ctx, "course_id", None),
            kb_version=result.kb_version, index_version=result.index_version, profile_version=result.profile_version,
            revocation_epoch=result.revocation_epoch, libraries_searched=result.libraries_searched,
            passage_ids=tuple(p.passage_id for p in result.passages),
            scores=tuple(p.relevance_score for p in result.passages if p.relevance_score is not None),
            status=result.status.value, reason=result.reason.value,
            error_code=result.error_code.value if result.error_code else None, **extra)
        try:
            self.audit.write(rec)
        except Exception:
            if result.status == Status.error:
                return result
            return RetrievalResult.error(result.request_id, ErrorCode.AUDIT_UNAVAILABLE,
                                         **{k: getattr(result, k) for k in ("kb_version", "index_version",
                                                                            "profile_version", "revocation_epoch")})
        return result

    # ------------------------------------------------------------------ search

    def retrieve(self, raw, ctx: RetrievalContext) -> RetrievalResult:
        start = self.monotonic()
        deadline = start + self.profile.runtime.deadline_seconds
        request_id = self._request_id(raw)
        snap = None
        epoch = None
        diag: dict = {}
        try:
            try:
                req = raw if isinstance(raw, RetrievalRequest) else RetrievalRequest.model_validate(raw)
            except (ValidationError, ValueError, TypeError):
                raise RetrievalError(ErrorCode.INVALID_REQUEST) from None
            self._check_context(ctx, req.course_id)
            active = self._check_scope(req, ctx)
            self._require_ready()
            self._check_query(req.query_text)
            snap = self._pin(ctx, req.kb_version)  # pinned for the whole request
            self.revocations.refresh()
            epoch = self.revocations.epoch()

            def check_deadline():
                if self.monotonic() > deadline:
                    raise RetrievalError(ErrorCode.DEADLINE_EXCEEDED)

            result = self.worker.run(lambda: self._search(req, ctx, snap, active, epoch, check_deadline, diag),
                                     deadline - self.monotonic())
            if self.monotonic() > deadline:
                raise RetrievalError(ErrorCode.DEADLINE_EXCEEDED)
        except RetrievalError as exc:
            result = RetrievalResult.error(request_id, exc.code, **self._versions(snap, epoch))
        except Exception:  # never leak exception payloads
            result = RetrievalResult.error(request_id, ErrorCode.SEARCH_FAILED, **self._versions(snap, epoch))
        diag["stage_ms"] = {"total": round((self.monotonic() - start) * 1000, 3)}
        return self._record(result, "search", ctx, **{k: v for k, v in diag.items() if k in (
            "dense_count", "lexical_count", "candidate_count", "embedding_reused", "stage_ms")})

    def _rank(self, req: RetrievalRequest, ctx: RetrievalContext, snap: Snapshot, active: frozenset[str],
              check_deadline, diag: dict):
        """Eligibility filter, dense + lexical search, RRF and raw-logit reranking.

        Returns (reason, searched, fused, logits); reason is a no-evidence Reason or None.
        """
        p = self.profile
        now = self.now()
        eligible = [row for row in snap.rows if self._usable(snap, row, active, ctx, now)]
        if not eligible:
            return Reason.NO_ELIGIBLE_PASSAGES, (), [], {}
        searched = tuple(sorted(active))
        check_deadline()
        vec = reusable_vector(ctx.embedding_ref, req.query_text, self.encoder, p)
        diag["embedding_reused"] = vec is not None
        if vec is None:
            self.encode_calls += 1
            vec = encode_query(self.encoder, req.query_text, p)
        check_deadline()
        rows = np.array([r.vector_row for r in eligible], dtype=np.int64)
        dense = dense_search(snap.matrix, snap.ids_by_row, rows, vec, p.search.dense_k)
        with snap.lock:
            lexical = lexical_search(snap.conn, req.query_text, [r.rowid for r in eligible], p.search.lexical_k)
        diag["dense_count"], diag["lexical_count"] = len(dense), len(lexical)
        diag["dense_ids"], diag["lexical_ids"] = [d[0] for d in dense], [x[0] for x in lexical]
        if not dense and not lexical:
            return Reason.NO_MATCHES, searched, [], {}
        check_deadline()
        fused = rrf_ids([diag["dense_ids"], diag["lexical_ids"]], p.search.rrf_constant, p.search.rerank_k)
        diag["candidate_count"] = len(fused)
        docs = {pid: pair_document(snap.by_id[pid].heading, snap.by_id[pid].text) for pid, _ in fused}
        logits = rerank(req.query_text, [pid for pid, _ in fused], docs, self.reranker, p.reranker.max_pair_tokens,
                        p.reranker.batch_size, check_deadline)
        return None, searched, fused, logits

    def _search(self, req: RetrievalRequest, ctx: RetrievalContext, snap: Snapshot, active: frozenset[str],
                epoch0: int, check_deadline, diag: dict) -> RetrievalResult:
        p = self.profile
        versions = self._versions(snap, epoch0)
        reason, searched, fused, logits = self._rank(req, ctx, snap, active, check_deadline, diag)
        if reason is not None:
            return RetrievalResult.no_evidence(req.request_id, reason, searched, **versions)
        rrf = dict(fused)
        cands = [Candidate(pid, logits[pid], rrf[pid], snap.by_id[pid].source_key, snap.by_id[pid].section_path,
                           snap.by_id[pid].text_sha256, snap.by_id[pid].spans) for pid, _ in fused]
        selection = select(cands, p.threshold.t_rerank, p.search.max_passages, p.search.overlap_limit, snap.conflicts)
        if selection.empty:
            return RetrievalResult.no_evidence(req.request_id, Reason.BELOW_THRESHOLD, searched, **versions)
        chosen = [snap.by_id[c.passage_id] for c in selection.passages]
        self._final_revalidation(snap, chosen, active, ctx, epoch0)
        passages = tuple(self._passage_out(snap, row, logits[row.passage_id]) for row in chosen)
        chosen_ids = {r.passage_id for r in chosen}
        conflicts = tuple(ConflictOut(conflict_id=c.conflict_id, topic_id=c.topic_id,
                                      passage_ids=tuple(x for x in c.passage_ids if x in chosen_ids),
                                      description=c.description, policy_reference=c.policy_reference)
                          for c in selection.conflicts)
        return RetrievalResult(request_id=req.request_id, status=Status.ok, reason=Reason.QUALIFYING_PASSAGES,
                               error_code=None, passages=passages, coverage="unknown", conflicts=conflicts,
                               libraries_searched=searched, **versions)

    def ranking_diagnostics(self, raw, ctx: RetrievalContext) -> dict:
        """Offline evaluation/tuning helper: candidates and raw logits before thresholding.

        Not used on the student path. Raises RetrievalError on any validation/operational failure.
        """
        req = raw if isinstance(raw, RetrievalRequest) else RetrievalRequest.model_validate(raw)
        self._check_context(ctx, req.course_id)
        active = self._check_scope(req, ctx)
        self._require_ready()
        self._check_query(req.query_text)
        snap = self._pin(ctx, req.kb_version)
        diag: dict = {}
        reason, searched, fused, logits = self._rank(req, ctx, snap, active, lambda: None, diag)
        return {"reason": reason, "fused": fused, "logits": logits, "dense_ids": diag.get("dense_ids", []),
                "lexical_ids": diag.get("lexical_ids", []), "snapshot": snap}

    # ------------------------------------------------------------------ item fetch

    def fetch_item_evidence(self, raw, ctx: RetrievalContext) -> RetrievalResult:
        request_id = self._request_id(raw)
        snap = None
        epoch = None
        try:
            try:
                req = raw if isinstance(raw, ItemFetchRequest) else ItemFetchRequest.model_validate(raw)
            except (ValidationError, ValueError, TypeError):
                raise RetrievalError(ErrorCode.INVALID_REQUEST) from None
            self._check_context(ctx, req.course_id)
            self._require_ready()
            snap = self._pin(ctx, req.kb_version)
            self.revocations.refresh()
            epoch = self.revocations.epoch()
            mapping = snap.items.get(req.item_id)
            if mapping is None or not mapping.approved:
                raise RetrievalError(ErrorCode.ITEM_EVIDENCE_UNAVAILABLE)
            if mapping.item_version != req.item_version:
                raise RetrievalError(ErrorCode.VERSION_MISMATCH)
            active = self.active_authorized(ctx)
            now = self.now()
            chosen = None
            for evidence_set in mapping.evidence_sets:  # alternatives; first complete valid set wins
                rows = [snap.by_id.get(pid) for pid in evidence_set.passage_ids]
                if all(r is not None and self._usable(snap, r, active, ctx, now) for r in rows):
                    chosen = (evidence_set, rows)
                    break
            if chosen is None:
                raise RetrievalError(ErrorCode.ITEM_EVIDENCE_UNAVAILABLE)  # never a partial answer key
            evidence_set, rows = chosen
            self._final_revalidation(snap, rows, active, ctx, epoch)
            result = RetrievalResult(
                request_id=req.request_id, status=Status.ok, reason=Reason.APPROVED_ITEM_EVIDENCE, error_code=None,
                passages=tuple(self._passage_out(snap, r, None) for r in rows), coverage=evidence_set.coverage,
                conflicts=(), libraries_searched=(), **self._versions(snap, epoch))
        except RetrievalError as exc:
            result = RetrievalResult.error(request_id, exc.code, **self._versions(snap, epoch))
        except Exception:
            result = RetrievalResult.error(request_id, ErrorCode.ITEM_EVIDENCE_UNAVAILABLE, **self._versions(snap, epoch))
        return self._record(result, "item_fetch", ctx)

    # ------------------------------------------------------------------ orchestrator checks

    def revalidate(self, result: RetrievalResult, ctx: RetrievalContext) -> Optional[ErrorCode]:
        """Re-check revocation epoch and eligibility immediately before Brain invocation."""
        if result.status != Status.ok:
            return None
        self.revocations.refresh()
        if self.revocations.epoch() != result.revocation_epoch:
            return ErrorCode.SOURCE_STATE_CHANGED
        try:
            snap = self._pin(ctx, result.kb_version)
        except RetrievalError as exc:
            return exc.code
        active = self.active_authorized(ctx)
        now = self.now()
        for p in result.passages:
            row = snap.by_id.get(p.passage_id)
            if row is None or row.text_sha256 != p.text_sha256 or not self._usable(snap, row, active, ctx, now):
                return ErrorCode.SOURCE_STATE_CHANGED
        return None

    def approved_item_for(self, ctx: RetrievalContext, kb_version: str, passage_id: str):
        """(item_id, item_version) of an approved item whose first evidence set contains this passage."""
        try:
            snap = self._pin(ctx, kb_version)
        except RetrievalError:
            return None
        for item_id in sorted(snap.items):
            m = snap.items[item_id]
            if m.approved and passage_id in m.evidence_sets[0].passage_ids:
                return (m.item_id, m.item_version)
        return None

    def get_evidence(self, passage_id: str, kb_version: str, ctx: RetrievalContext) -> EvidencePassage:
        """Citation endpoint: access-checked lookup of one stored passage. Raises RetrievalError."""
        self._check_context(ctx, ctx.course_id)
        snap = self._pin(ctx, kb_version)
        row = snap.by_id.get(passage_id)
        if row is None or not self._usable(snap, row, self.active_authorized(ctx), ctx, self.now()):
            raise RetrievalError(ErrorCode.FORBIDDEN)  # no distinction between unknown and revoked
        return self._passage_out(snap, row, None)


def build_service(profile: RetrieverProfile, library_registry, encoder, reranker, *, courses=(), audit=None,
                  revocations: Optional[RevocationRegistry] = None, snapshots_root: Optional[str] = None,
                  tokenizers=None, **kw) -> RetrieverService:
    """Wire a service with a snapshot manager whose fingerprints come from the loaded models."""
    pre_fp = preprocessing_fingerprint(encoder, profile) if encoder is not None else "missing"
    manager = SnapshotManager(snapshots_root or profile.snapshots_root, profile, pre_fp, chunking_fingerprint(profile))
    return RetrieverService(profile, manager, revocations or RevocationRegistry(profile.revocations_path),
                            library_registry, encoder, reranker, courses=courses, audit=audit, **kw)
