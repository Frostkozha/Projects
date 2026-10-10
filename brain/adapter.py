"""Coordinator adapter: the local Brain behind the orchestrator's ``Brain`` protocol (Brain plan v0.3, phase 7).

The orchestrator calls synchronously; the Brain service runs on one owned event loop thread. The adapter
freezes exactly the shown passages from the validated ``RetrievalResult``, builds the strict
``BrainRequest``/``BrainContext`` and maps the typed result:

* ``ok`` / ``no_evidence`` -> the shared ``DraftAnswer`` (the independent verifier still decides delivery);
* ``INVALID_OUTPUT`` -> ``None`` (the orchestrator renders fixed A5, never infers onto prose);
* every other error -> ``AdapterError`` (service unavailable; never labelled no_evidence).
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Optional

from gate_classifier.adapters import AdapterError, BrainRequest as CoordinatorBrainRequest
from gate_classifier.schema import Mode

from .context import freeze_passages
from .request_schema import BrainRequest
from .schema import BrainContext, ErrorCode, PresentationHint

TASK_FOR_MODE = {Mode.answer: "answer_draft", Mode.quiz: "answer_draft", Mode.tutor: "tutor_clue"}


class LocalBrainAdapter:
    def __init__(self, service, *, presentation: Optional[PresentationHint] = None):
        self.service = service
        cfg = service.cfg
        self.context_limit = cfg.runtime.context_tokens
        self.reserved_output_tokens = cfg.limits.output_tokens + cfg.limits.reserve_tokens
        self.deadline_seconds = cfg.limits.brain_deadline_seconds
        self.presentation = presentation or PresentationHint()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="brain-loop", daemon=True)
        self._thread.start()

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def startup(self) -> dict:
        return self._run(self.service.startup(), timeout=self.service.cfg.runtime.startup_timeout_seconds + 120)

    def close(self) -> None:
        try:
            self._run(self.service.shutdown(), timeout=60)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def count_tokens(self, text: str) -> int:
        """Coordinator-side fitting estimate via the loaded GGUF tokenizer (the Brain re-checks exactly)."""
        async def count():
            deadline = asyncio.get_running_loop().time() + 5
            return len(await self.service.tokenizer.tokens(text, parse_special=False, deadline=deadline))
        try:
            return self._run(count(), timeout=10)
        except Exception:  # noqa: BLE001
            raise AdapterError("brain_budget_unavailable") from None

    def draft(self, request: CoordinatorBrainRequest):
        if (request.request_id is None or request.retrieval is None or request.tenant_id is None
                or request.course_id is None or request.registry_version is None):
            raise AdapterError("brain_context_mismatch")
        r = request.retrieval
        try:
            frozen = freeze_passages(
                request_id=request.request_id, tenant_id=request.tenant_id, course_id=request.course_id,
                passages=list(request.evidence.passages), kb_version=r.kb_version, index_version=r.index_version,
                profile_version=r.profile_version, retrieval_policy_version=r.retrieval_policy_version,
                revocation_epoch=r.revocation_epoch, registry_version=request.registry_version)
            breq = BrainRequest(schema_version="brain-request-0.2", request_id=uuid.UUID(request.request_id),
                                task=TASK_FOR_MODE[request.mode], question_redacted=request.question_text,
                                passage_ids=list(frozen.passage_ids), prompt_version=self.service.cfg.prompt_version,
                                sampling_profile_version=self.service.cfg.sampling.profile_version)
        except Exception:  # noqa: BLE001 - never surface validation detail (student text)
            raise AdapterError("brain_invalid_evidence") from None
        ctx = BrainContext(evidence=frozen, expected_evidence_digest=frozen.evidence_digest,
                           deadline=time.monotonic() + self.deadline_seconds, presentation=self.presentation,
                           approved_item_question=request.item_context if breq.task == "tutor_clue" else None)
        result = self._run(self.service.generate(breq, ctx), timeout=self.deadline_seconds + 5)
        if result.status in ("ok", "no_evidence"):
            return result.draft
        if result.error_code == ErrorCode.INVALID_OUTPUT:
            return None  # content rejection -> fixed A5 in the orchestrator
        raise AdapterError(f"brain_{result.error_code.value.lower()}")
