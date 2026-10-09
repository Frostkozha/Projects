"""Metadata audit sink, restricted text store adapter and local alert queue (spec sections 12, 14).

Operational audit records contain metadata only: no prompt text, vectors, entity values or
generated reasoning. Alerts carry a pseudonymous session reference and category only.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict

from .schema import utcnow


class AuditError(RuntimeError):
    pass


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["gate_decision", "final_response", "incident", "alert_retry", "alarm"]
    request_id: str
    session_ref: Optional[str] = None
    route: Optional[str] = None
    reason: Optional[str] = None
    response_code: Optional[str] = None
    inference_status: Optional[str] = None
    versions: Optional[dict] = None
    redaction_categories: tuple[str, ...] = ()
    library_ids: tuple[str, ...] = ()
    latency_ms: Optional[float] = None
    service_status: Optional[str] = None
    detail_code: Optional[str] = None
    timestamp: str = ""

    def stamped(self) -> "AuditRecord":
        return self if self.timestamp else self.model_copy(update={"timestamp": utcnow().isoformat()})


class AuditSink(Protocol):
    def write(self, record: AuditRecord) -> None: ...


class InMemoryAuditSink:
    def __init__(self):
        self.records: list[AuditRecord] = []
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        with self._lock:
            self.records.append(record.stamped())


class JsonlAuditSink:
    """Development JSONL sink. Production needs access control and protected retention."""

    def __init__(self, path: str | Path, retention_days: Optional[int] = None):
        self.path = Path(path)
        self.retention_days = retention_days
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        line = record.stamped().model_dump_json()
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            raise AuditError("audit write failed") from None

    def purge_expired(self, now: Optional[datetime] = None) -> int:
        """Delete records older than the configured retention. Returns removed count."""
        if self.retention_days is None or not self.path.exists():
            return 0
        cutoff = (now or utcnow()) - timedelta(days=self.retention_days)
        with self._lock:
            kept, removed = [], 0
            for line in self.path.read_text(encoding="utf-8").splitlines():
                ts = datetime.fromisoformat(json.loads(line)["timestamp"])
                if ts < cutoff:
                    removed += 1
                else:
                    kept.append(line)
            self.path.write_text("".join(k + "\n" for k in kept), encoding="utf-8")
        return removed


class RestrictedTextStore:
    """Redacted question + verified answer store. Disabled by default (PRO-03 governs enablement)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.items: list[dict] = []

    def maybe_store(self, request_id: str, redacted_question: str, verified_answer: str) -> bool:
        if not self.enabled:
            return False
        self.items.append({"request_id": request_id, "question": redacted_question, "answer": verified_answer,
                           "timestamp": utcnow().isoformat()})
        return True


# ----------------------------------------------------------------------------- alerts


class AlertEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    request_id: str
    session_ref: Optional[str]
    category: Literal["emergency", "self_harm"]
    timestamp_utc: str

    @classmethod
    def new(cls, request_id: str, session_ref: Optional[str], category: str) -> "AlertEvent":
        # deterministic per request+category so a retried request cannot create a duplicate event
        event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gate-alert:{request_id}:{category}"))
        return cls(event_id=event_id, request_id=request_id, session_ref=session_ref, category=category,
                   timestamp_utc=datetime.now(timezone.utc).isoformat())


class AlertSink(Protocol):
    def enqueue(self, event: AlertEvent) -> None: ...


class InMemoryAlertQueue:
    """Restricted local queue, idempotent by event_id. No external messaging."""

    def __init__(self):
        self.events: dict[str, AlertEvent] = {}
        self._lock = threading.Lock()

    def enqueue(self, event: AlertEvent) -> None:
        with self._lock:
            self.events.setdefault(event.event_id, event)


class JsonlAlertQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def enqueue(self, event: AlertEvent) -> None:
        with self._lock:
            seen = set()
            if self.path.exists():
                seen = {json.loads(line)["event_id"] for line in self.path.read_text(encoding="utf-8").splitlines()}
            if event.event_id in seen:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")


@dataclass
class OperationalAlarms:
    """Local operational alarm channel (metadata only)."""

    alarms: list[dict] = field(default_factory=list)

    def raise_alarm(self, code: str, request_id: str) -> None:
        self.alarms.append({"code": code, "request_id": request_id, "timestamp": utcnow().isoformat()})


class AlertDispatcher:
    """Writes A7 alerts without delaying the user's fixed message.

    The first attempt is synchronous; failures retry with bounded backoff on a background
    thread and raise an operational alarm. Retries are logged as metadata only.
    """

    def __init__(self, sink: AlertSink, alarms: OperationalAlarms, audit: Optional[AuditSink] = None,
                 max_attempts: int = 4, base_backoff: float = 0.05, sleep: Callable[[float], None] = time.sleep):
        self.sink = sink
        self.alarms = alarms
        self.audit = audit
        self.max_attempts = max(1, max_attempts)
        self.base_backoff = base_backoff
        self._sleep = sleep
        self._threads: list[threading.Thread] = []
        self.failed_events: list[str] = []

    def dispatch(self, event: AlertEvent) -> bool:
        try:
            self.sink.enqueue(event)
            return True
        except Exception:
            self.alarms.raise_alarm("alert_write_failed", event.request_id)
        t = threading.Thread(target=self._retry, args=(event,), daemon=True)
        self._threads.append(t)
        t.start()
        return False

    def _retry(self, event: AlertEvent) -> None:
        for attempt in range(1, self.max_attempts):
            self._sleep(self.base_backoff * (2 ** (attempt - 1)))
            self._log_retry(event, attempt)
            try:
                self.sink.enqueue(event)
                return
            except Exception:
                continue
        self.failed_events.append(event.event_id)
        self.alarms.raise_alarm("alert_delivery_exhausted", event.request_id)

    def _log_retry(self, event: AlertEvent, attempt: int) -> None:
        if self.audit is None:
            return
        try:
            self.audit.write(AuditRecord(event="alert_retry", request_id=event.request_id,
                                         detail_code=f"attempt_{attempt}"))
        except Exception:
            pass

    def wait_idle(self, timeout: float = 5.0) -> None:
        for t in list(self._threads):
            t.join(timeout)
