"""Bounded inference workers, deadlines, cancellation and late-result discard (Verifier spec v0.2, section 11).

``InProcessWorker`` runs a backend on one dedicated thread (fixture/unit tests). ``SubprocessWorker`` owns
model execution in a spawn-started child process: a timeout terminates and restarts the child (bounded),
so native CPU inference is actually stopped, and late results are discarded by attempt identity.
At most ``capacity`` requests wait; one runs at a time; pairs are processed in batches of up to 8.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import threading
import time
from typing import Optional, Sequence

import numpy as np

from contracts.models import OperationalError

from .evidence import VerifierFault
from .nli import NLIOutputInvalid


class _Gate:
    """Bounded admission: at most ``capacity`` waiting plus one active request."""

    def __init__(self, capacity: int):
        self._sem = threading.BoundedSemaphore(capacity + 1)
        self._active = threading.Lock()

    def __enter__(self):
        if not self._sem.acquire(blocking=False):
            raise VerifierFault(OperationalError.QUEUE_FULL)
        return self

    def acquire_active(self, timeout: float) -> bool:
        return self._active.acquire(timeout=max(0.0, timeout))

    def release_active(self):
        self._active.release()

    def __exit__(self, *exc):
        self._sem.release()


class InProcessWorker:
    def __init__(self, backend, capacity: int = 16, batch_size: int = 8):
        self.backend = backend
        self.batch_size = batch_size
        self._gate = _Gate(capacity)
        self.healthy = True

    def infer(self, pairs: Sequence[tuple[str, str]], deadline: float) -> np.ndarray:
        with self._gate:
            if not self._gate.acquire_active(deadline - time.monotonic()):
                raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)
            try:
                out = []
                for i in range(0, len(pairs), self.batch_size):
                    if time.monotonic() > deadline:
                        raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)
                    out.append(np.asarray(self.backend.logits(list(pairs[i:i + self.batch_size])), dtype=np.float64))
                result = np.vstack(out) if out else np.zeros((0, 3))
                if time.monotonic() > deadline:
                    raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)  # late result discarded
                return result
            except VerifierFault:
                raise
            except NLIOutputInvalid:
                raise VerifierFault(OperationalError.WORKER_FAILED) from None
            except Exception:
                raise VerifierFault(OperationalError.WORKER_FAILED) from None
            finally:
                self._gate.release_active()


# ----------------------------------------------------------------------------- subprocess


def _child_main(conn, backend_spec: dict):
    """Child process: load the pinned model once, then serve (attempt_id, pairs) requests."""
    try:
        from verifier.nli import TransformersNLI  # noqa: PLC0415

        backend = TransformersNLI(**backend_spec)
        conn.send(("ready", backend.label_index, backend.version))
    except Exception as exc:  # noqa: BLE001
        conn.send(("failed", type(exc).__name__, None))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        attempt, pairs, batch = msg
        try:
            out = [np.asarray(backend.logits(pairs[i:i + batch]), dtype=np.float64) for i in range(0, len(pairs), batch)]
            conn.send(("ok", attempt, np.vstack(out) if out else np.zeros((0, 3))))
        except Exception:  # noqa: BLE001
            conn.send(("error", attempt, None))


class SubprocessWorker:
    """Supervised spawn worker. Requires ``if __name__ == '__main__':`` guards in entry scripts on Windows."""

    def __init__(self, backend_spec: dict, capacity: int = 16, batch_size: int = 8, start_timeout: float = 120.0,
                 max_restarts: int = 3):
        self.backend_spec = backend_spec
        self.batch_size = batch_size
        self.start_timeout = start_timeout
        self.max_restarts = max_restarts
        self.restarts = 0
        self._gate = _Gate(capacity)
        self._attempts = itertools.count(1)
        self._ctx = mp.get_context("spawn")
        self._proc: Optional[mp.Process] = None
        self._conn = None
        self.label_index: Optional[dict] = None
        self.version: Optional[str] = None
        self._start()

    @property
    def healthy(self) -> bool:
        return self._proc is not None and self._proc.is_alive() and self.restarts <= self.max_restarts

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_child_main, args=(child, self.backend_spec), daemon=True)
        proc.start()
        if not parent.poll(self.start_timeout):
            proc.terminate()
            raise VerifierFault(OperationalError.MODEL_UNAVAILABLE)
        status, a, b = parent.recv()
        if status != "ready":
            proc.join(5)
            raise VerifierFault(OperationalError.MODEL_UNAVAILABLE)
        self._proc, self._conn, self.label_index, self.version = proc, parent, a, b

    def _restart(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(5)
        self.restarts += 1
        self._proc = None
        if self.restarts <= self.max_restarts:
            try:
                self._start()
            except VerifierFault:
                self._proc = None

    def infer(self, pairs: Sequence[tuple[str, str]], deadline: float) -> np.ndarray:
        with self._gate:
            if not self._gate.acquire_active(deadline - time.monotonic()):
                raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)
            try:
                if not self.healthy:
                    raise VerifierFault(OperationalError.WORKER_FAILED)
                attempt = next(self._attempts)
                self._conn.send((attempt, list(pairs), self.batch_size))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._conn.poll(remaining):
                        self._restart()  # stops native inference; late result can never be read
                        raise VerifierFault(OperationalError.DEADLINE_EXCEEDED)
                    status, got, data = self._conn.recv()
                    if got != attempt:
                        continue  # stale result of an earlier attempt: discard
                    if status != "ok":
                        raise VerifierFault(OperationalError.WORKER_FAILED)
                    return data
            except (EOFError, OSError, BrokenPipeError):
                self._restart()
                raise VerifierFault(OperationalError.WORKER_FAILED) from None
            finally:
                self._gate.release_active()

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: BLE001
                pass
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.terminate()
