"""Admission, deadlines, cancellation and circuit breakers (Integration plan v0.3, section 10).

One active generated interaction, a bounded waiting queue with an absolute wait ceiling, per-session and
per-identity pending caps, monotonic absolute deadlines handed to every stage, and a breaker per mandatory
adapter that opens on operational faults only (content A5/A6 are not faults).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional


class CapacityRejected(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code  # CAPACITY_EXCEEDED | RATE_LIMITED | SERVICE_UNAVAILABLE


class StageTimeout(Exception):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


class Deadline:
    """Absolute monotonic deadline for the whole interaction (60 s from HTTP receipt)."""

    def __init__(self, seconds: float, start: Optional[float] = None):
        self.start = start if start is not None else time.monotonic()
        self.at = self.start + seconds

    def remaining(self) -> float:
        return self.at - time.monotonic()

    def stage(self, ceiling: float) -> float:
        """Absolute deadline for a stage: its own ceiling capped by the shared deadline."""
        return min(time.monotonic() + ceiling, self.at)


class GeneratedSlot:
    def __init__(self, waiting: int, wait_seconds: float):
        self._lock = asyncio.Lock()
        self._waiting = 0
        self.max_waiting = waiting
        self.wait_seconds = wait_seconds

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @asynccontextmanager
    async def hold(self, deadline: Deadline):
        if self._lock.locked() and self._waiting >= self.max_waiting:
            raise CapacityRejected("CAPACITY_EXCEEDED")
        wait = min(self.wait_seconds, deadline.remaining())
        if wait <= 0:
            raise CapacityRejected("SERVICE_UNAVAILABLE")
        self._waiting += 1
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=wait)
        except TimeoutError:
            raise CapacityRejected("CAPACITY_EXCEEDED" if wait >= self.wait_seconds else "SERVICE_UNAVAILABLE") \
                from None
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._lock.release()


class PendingLimiter:
    """At most one pending request per owned session and two per identity."""

    def __init__(self, per_session: int, per_identity: int):
        self.per_session = per_session
        self.per_identity = per_identity
        self._sessions: dict[str, int] = {}
        self._owners: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, owner: str, session_id: Optional[str]):
        if self._owners.get(owner, 0) >= self.per_identity or (
                session_id and self._sessions.get(session_id, 0) >= self.per_session):
            raise CapacityRejected("RATE_LIMITED")
        self._owners[owner] = self._owners.get(owner, 0) + 1
        if session_id:
            self._sessions[session_id] = self._sessions.get(session_id, 0) + 1
        try:
            yield
        finally:
            self._owners[owner] -= 1
            if session_id:
                self._sessions[session_id] -= 1


class CircuitBreaker:
    """Open after N consecutive operational failures within the window; one half-open probe after cooldown."""

    def __init__(self, name: str, failures: int = 3, window: float = 60.0, cooldown: float = 30.0,
                 clock=time.monotonic):
        self.name = name
        self.failures = failures
        self.window = window
        self.cooldown = cooldown
        self.clock = clock
        self._recent: deque[float] = deque()
        self.state = "closed"
        self._opened_at = 0.0
        self._probe_out = False

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and self.clock() - self._opened_at >= self.cooldown:
            self.state = "half_open"
            self._probe_out = False
        if self.state == "half_open" and not self._probe_out:
            self._probe_out = True
            return True
        return False

    def success(self) -> None:
        self._recent.clear()
        self.state = "closed"
        self._probe_out = False

    def failure(self) -> None:
        now = self.clock()
        if self.state == "half_open":
            self.state, self._opened_at, self._probe_out = "open", now, False
            return
        self._recent.append(now)
        while self._recent and now - self._recent[0] > self.window:
            self._recent.popleft()
        if len(self._recent) >= self.failures:
            self.state, self._opened_at = "open", now
            self._recent.clear()


async def run_stage(name: str, ceiling: float, deadline: Deadline, coro_fn):
    """Run one stage bounded by its ceiling and the shared deadline. Never restarts the full budget."""
    until = deadline.stage(ceiling)
    remaining = until - time.monotonic()
    if remaining <= 0:
        raise StageTimeout(name)
    try:
        return await asyncio.wait_for(coro_fn(until), timeout=remaining)
    except TimeoutError:
        raise StageTimeout(name) from None
