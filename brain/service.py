"""Brain service: admission, deadline, cancellation and supervised recovery (Brain plan v0.3, sections 7-11, appendix E).

One service owner manages admission, one active generation and its runtime process. The 30-second Brain
ceiling is measured from admission, including queue time, and capped by the whole-request deadline.
Cancellation is checked before invocation, after transport, before parsing and before returning.
On timeout, cancellation or transport loss during generation, admission closes, the owned slot must be
proven idle within the configured bound, otherwise the owned server is restarted and re-probed before any
further work is admitted. No retries, grammar relaxation, sampling change, model switch or remote call.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Optional

from contracts.json_codec import InvalidJSON, validate_wire
from contracts.evidence import freeze_validated_bundle

from .audit import InMemoryBrainAudit, result_record
from .budget import check_budget
from .capabilities import payload_for, run_probes
from .config import BrainConfig
from .manifest import Manifest, ManifestError, verify_artifacts
from .parse import CompletionFailure, SentencePolicy, observed_metrics, parse_completion
from .grammar import schema_hash
from .prompt import build_messages, data_record, load_prompts, prompt_hashes
from .request_schema import BrainRequest
from .schema import RESULT_SCHEMA, BrainContext, BrainError, BrainMetrics, BrainResult, ErrorCode, error_result
from .tokenizer import RuntimeTokenizer
from .transport import LocalTransport

MAX_REQUEST_BYTES = 64 * 1024


class BrainService:
    def __init__(self, cfg: BrainConfig, *, transport: LocalTransport, manifest: Optional[Manifest],
                 sentence_policy, supervisor=None, audit=None, api_key: Optional[str] = None):
        self.cfg = cfg
        self.transport = transport
        self.tokenizer = RuntimeTokenizer(transport)
        self.manifest = manifest
        self.policy = sentence_policy
        self.supervisor = supervisor
        self.audit = audit if audit is not None else InMemoryBrainAudit()
        self._api_key = api_key
        self.prompts = load_prompts()
        self.ready = False
        self.admission_open = False
        self.last_probe: Optional[dict] = None
        self._slot = asyncio.Lock()
        self._waiting = 0
        self._recovery: Optional[asyncio.Task] = None
        self.identity = {
            "model_version": f"{manifest.candidate_id}.{manifest.gguf_sha256[:12]}" if manifest else None,
            "runtime_version": manifest.runtime_release if manifest else None,
            "prompt_version": cfg.prompt_version,
            "sampling_profile_version": cfg.sampling.profile_version,
        }

    # ------------------------------------------------------------------ lifecycle

    async def startup(self, *, verify_bytes: bool = True) -> dict:
        """Real-model readiness: verify artifacts, start the owned server, run every capability probe."""
        if self.manifest is None or self.supervisor is None or self._api_key is None:
            raise BrainError(ErrorCode.MODEL_UNAVAILABLE)
        if (self.manifest.candidate_id != self.cfg.candidate_id
                or self.manifest.prompt_hashes != prompt_hashes()
                or self.manifest.schema_hash != schema_hash()
                or self.cfg.sampling.profile_version not in self.manifest.sampling_profiles):
            raise BrainError(ErrorCode.ARTIFACT_MISMATCH)  # profile/prompt/schema drift needs a new manifest
        if verify_bytes:
            try:
                await asyncio.to_thread(verify_artifacts, self.manifest,
                                        model_path=self.cfg.path(self.cfg.artifacts.model_path),
                                        server_binary=self.cfg.path(self.cfg.artifacts.server_binary))
            except ManifestError:
                raise BrainError(ErrorCode.ARTIFACT_MISMATCH) from None
        await asyncio.to_thread(self.supervisor.start, self._api_key)
        return await self._probe()

    async def _probe(self) -> dict:
        report = await run_probes(self.cfg, self.manifest, self.transport, api_key=self._api_key or "",
                                  sentence_policy=self.policy)
        self.last_probe = report
        self.ready = self.admission_open = bool(report["passed"])
        self.audit.write({"event": "readiness", "status": "ready" if self.ready else "not_ready",
                          "detail": {k: v["pass"] for k, v in report["probes"].items()}})
        return report

    def mark_fixture_ready(self) -> None:
        """Explicit fixture profile for isolated tests with an injected HTTP transport. Never real-model."""
        if self.cfg.profile != "fixture":
            raise BrainError(ErrorCode.RUNTIME_INCOMPATIBLE)
        self.ready = self.admission_open = True

    async def shutdown(self) -> None:
        self.admission_open = False
        if self._recovery is not None:
            with contextlib.suppress(Exception):
                await self._recovery
        if self.supervisor is not None:
            await asyncio.to_thread(self.supervisor.stop)
        await self.transport.close()
        self.ready = False

    def readiness(self) -> dict:
        return {"ready": self.ready and self.admission_open, "profile": self.cfg.profile,
                "recovering": self._recovery is not None and not self._recovery.done(), **self.identity}

    # ------------------------------------------------------------------ request handling

    async def generate(self, request: BrainRequest | bytes, context: BrainContext, *,
                       cancel: Optional[asyncio.Event] = None) -> BrainResult:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        if isinstance(request, bytes):
            try:
                request = validate_wire(BrainRequest, request, max_bytes=MAX_REQUEST_BYTES)
            except Exception:  # noqa: BLE001 - never log validation detail (student text)
                raise BrainError(ErrorCode.INVALID_REQUEST) from None
        if not isinstance(request, BrainRequest):
            raise BrainError(ErrorCode.INVALID_REQUEST)
        digest = context.evidence.evidence_digest if context and context.evidence else None
        metrics = {"queue_ms": None, "prompt_ms": None, "generation_ms": None, "prompt_tokens": None,
                   "completion_tokens": None, "finish_reason": None}

        def finish(status_or_code, draft=None) -> BrainResult:
            m = BrainMetrics(total_ms=round((loop.time() - t0) * 1000, 3), **metrics)
            if isinstance(status_or_code, ErrorCode):
                result = error_result(request.request_id, status_or_code, identity=self.identity,
                                      evidence_digest=digest, metrics=m)
            else:
                result = BrainResult(schema_version=RESULT_SCHEMA, request_id=request.request_id,
                                     status=status_or_code, draft=draft, error_code=None, evidence_digest=digest,
                                     metrics=m, **self.identity)
            self.audit.write(result_record(result, tuple(request.passage_ids)))
            return result

        try:
            self._check_request(request, context)
        except BrainError as exc:
            return finish(exc.code)

        # deadline: earlier of the whole-request deadline and 30 s from Brain admission (includes queue)
        deadline = min(t0 + self.cfg.limits.brain_deadline_seconds, t0 + (context.deadline - time.monotonic()))
        if deadline <= loop.time():
            return finish(ErrorCode.DEADLINE_EXCEEDED)
        if not (self.ready and self.admission_open):
            return finish(ErrorCode.MODEL_UNAVAILABLE)
        if _cancelled(cancel):
            return finish(ErrorCode.CANCELLED)

        # admission: bounded queue, bounded wait, one active generation
        if self._slot.locked() and self._waiting >= self.cfg.limits.queue_capacity:
            return finish(ErrorCode.QUEUE_FULL)
        wait_limit = min(t0 + self.cfg.limits.queue_wait_seconds, deadline)
        self._waiting += 1
        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=max(0.0, wait_limit - loop.time()))
        except TimeoutError:
            return finish(ErrorCode.DEADLINE_EXCEEDED if wait_limit >= deadline else ErrorCode.QUEUE_FULL)
        finally:
            self._waiting -= 1
        metrics["queue_ms"] = round((loop.time() - t0) * 1000, 3)

        release = True
        inflight = False
        try:
            if not (self.ready and self.admission_open):
                return finish(ErrorCode.MODEL_UNAVAILABLE)
            if _cancelled(cancel):
                return finish(ErrorCode.CANCELLED)
            if context.source_check is not None and not await context.source_check():
                return finish(ErrorCode.CONTEXT_MISMATCH)  # source changed: abort, never re-retrieve

            record = data_record(request, context)
            messages = build_messages(self.prompts[request.task], record)
            await self._reject_control_tokens(record, deadline)
            count = await self.tokenizer.count_messages(messages, deadline=deadline)
            try:
                check_budget(count, prompt_limit=self.cfg.limits.prompt_tokens,
                             output_limit=self.cfg.limits.output_tokens, reserve=self.cfg.limits.reserve_tokens,
                             context=self.cfg.runtime.context_tokens)
            except ValueError as exc:
                return finish(ErrorCode(str(exc)))
            if _cancelled(cancel):
                return finish(ErrorCode.CANCELLED)

            max_visible = context.presentation.max_visible_sentences
            payload = payload_for(self.cfg, messages, tuple(request.passage_ids), max_visible)
            inflight = True
            raw = await _cancellable(self.transport.complete(payload, deadline=deadline), cancel)
            inflight = False

            observed = observed_metrics(raw)
            metrics.update({k: observed[k] for k in ("prompt_ms", "generation_ms", "prompt_tokens",
                                                     "completion_tokens", "finish_reason")})
            if _cancelled(cancel):
                return finish(ErrorCode.CANCELLED)  # late content discarded unlogged
            if metrics["prompt_tokens"] is not None and metrics["prompt_tokens"] != count:
                return finish(ErrorCode.RUNTIME_INCOMPATIBLE)
            try:
                draft = parse_completion(raw, expected_alias=self.cfg.runtime.alias,
                                         supplied_ids=tuple(request.passage_ids), sentence_policy=self.policy,
                                         max_visible_sentences=max_visible)
            except CompletionFailure as exc:
                return finish(ErrorCode(exc.code))
            if _cancelled(cancel):
                return finish(ErrorCode.CANCELLED)
            if draft.status == "no_evidence":
                return finish("no_evidence", draft)
            return finish("ok", draft)
        except BrainError as exc:
            if inflight:
                release = False
                self._start_recovery()
            return finish(exc.code)
        except CompletionFailure as exc:
            if inflight:
                release = False
                self._start_recovery()
            return finish(ErrorCode(exc.code))
        except asyncio.CancelledError:
            if inflight:
                release = False
                self._start_recovery()
            raise
        finally:
            if release:
                self._slot.release()

    # ------------------------------------------------------------------ checks

    def _check_request(self, request: BrainRequest, context: BrainContext) -> None:
        if not isinstance(context, BrainContext) or context.evidence is None:
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)
        if request.task == "tutor_clue" and not self.cfg.features.tutor_clue_enabled:
            raise BrainError(ErrorCode.INVALID_REQUEST)
        if (request.prompt_version != self.cfg.prompt_version
                or request.sampling_profile_version != self.cfg.sampling.profile_version):
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)
        if context.expected_evidence_digest != context.evidence.evidence_digest:
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)
        try:
            value = context.evidence.fresh_value()
            refrozen = freeze_validated_bundle(value)
        except (ValueError, InvalidJSON):
            raise BrainError(ErrorCode.INVALID_EVIDENCE) from None
        if refrozen.evidence_digest != context.evidence.evidence_digest:
            raise BrainError(ErrorCode.INVALID_EVIDENCE)
        if value["request_id"] != str(request.request_id):
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)
        if tuple(request.passage_ids) != refrozen.passage_ids:
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)

    async def _reject_control_tokens(self, record: dict, deadline: float) -> None:
        if await self.tokenizer.contains_control_tokens(record["question"], deadline=deadline):
            raise BrainError(ErrorCode.INVALID_REQUEST)
        item = record.get("approved_item_question")
        if item and await self.tokenizer.contains_control_tokens(item, deadline=deadline):
            raise BrainError(ErrorCode.INVALID_REQUEST)
        for p in record["passages"]:
            if await self.tokenizer.contains_control_tokens(p["text"], deadline=deadline):
                raise BrainError(ErrorCode.INVALID_EVIDENCE)

    # ------------------------------------------------------------------ recovery

    def _start_recovery(self) -> None:
        """Called while holding the slot: the recovery task owns the slot until native work is resolved."""
        self.admission_open = False
        self._recovery = asyncio.get_running_loop().create_task(self._recover())

    async def _recover(self) -> None:
        try:
            idle = await self._slot_idle_within(self.cfg.runtime.cancel_idle_check_seconds)
            self.audit.write({"event": "recovery", "status": "slot_idle" if idle else "restart"})
            if idle:
                self.admission_open = self.ready
                return
            self.ready = False
            if self.supervisor is None or self._api_key is None:
                return  # fail closed until an operator restarts
            await asyncio.to_thread(self.supervisor.stop)
            await asyncio.to_thread(self.supervisor.start, self._api_key)
            await self._probe()
        except Exception:  # noqa: BLE001 - recovery failure leaves the service not ready
            self.ready = self.admission_open = False
            self.audit.write({"event": "recovery", "status": "failed"})
        finally:
            self._slot.release()

    async def _slot_idle_within(self, seconds: float) -> bool:
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while loop.time() < end:
            try:
                slots = await self.transport.json("GET", "/slots", deadline=min(end, loop.time() + 0.5))
                if len(slots) == 1 and slots[0].get("is_processing") is False:
                    return True
            except CompletionFailure:
                pass
            await asyncio.sleep(0.1)
        return False


def _cancelled(cancel: Optional[asyncio.Event]) -> bool:
    return cancel is not None and cancel.is_set()


async def _cancellable(coro, cancel: Optional[asyncio.Event]):
    task = asyncio.ensure_future(coro)
    if cancel is None:
        return await task
    waiter = asyncio.ensure_future(cancel.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        waiter.cancel()
        raise
    if task in done:
        waiter.cancel()
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, CompletionFailure):
        await task
    raise BrainError(ErrorCode.CANCELLED)


def build_service(cfg: BrainConfig, *, audit=None, http_transport=None) -> BrainService:
    """Real-model service wiring from validated configuration (artifact bytes verified at startup)."""
    from .manifest import load_manifest  # noqa: PLC0415
    from .supervisor import Supervisor  # noqa: PLC0415

    key = read_api_key(cfg.path(cfg.runtime.api_key_file))
    manifest = load_manifest(cfg.path(cfg.manifest_path)) if cfg.profile == "real_model" else None
    transport = LocalTransport(f"http://{cfg.runtime.host}:{cfg.runtime.port}", key,
                               max_body=cfg.limits.response_bytes, http_transport=http_transport)
    supervisor = Supervisor(cfg) if cfg.profile == "real_model" else None
    return BrainService(cfg, transport=transport, manifest=manifest, sentence_policy=SentencePolicy(),
                        supervisor=supervisor, audit=audit, api_key=key)


def read_api_key(path: Path) -> str:
    try:
        lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        raise BrainError(ErrorCode.MODEL_UNAVAILABLE) from None
    if len(lines) != 1 or len(lines[0]) < 24:
        raise BrainError(ErrorCode.MODEL_UNAVAILABLE)
    return lines[0]
