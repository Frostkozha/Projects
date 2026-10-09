"""Internal gate API (spec sections 5.1, 5.3, 13).

``POST /v1/gate`` is reachable only by the authenticated local backend (service token). The
backend's trusted principal headers identify the pseudonymous owner, tenant and course; the
course registry, deployment state and session are then resolved server-side.
Every non-200 body is exactly ``{"error_code": ..., "request_id": ...}``.
"""

from __future__ import annotations

import hmac
import json
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .schema import ErrorCode, GateContext, GateError, GateRequest, Reason, Route, new_request_id
from .service import GateService

OWNER_HEADER = "x-gate-owner"
TENANT_HEADER = "x-gate-tenant"
COURSE_HEADER = "x-gate-course"

ContextResolver = Callable[[str, str, str, Optional[str]], GateContext]
"""(owner_code, tenant_id, course_id, session_id) -> trusted GateContext; raises GateError 403/409."""


def _error(code: ErrorCode, request_id: str) -> JSONResponse:
    from .schema import ERROR_HTTP_STATUS  # noqa: PLC0415

    return JSONResponse(status_code=ERROR_HTTP_STATUS[code], content={"error_code": code.value, "request_id": request_id})


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


_UNAVAILABLE_CODES = {Reason.MAINTENANCE: ErrorCode.MAINTENANCE, Reason.EXAM_DISABLED: ErrorCode.EXAM_DISABLED}


def create_app(service: GateService, resolve_context: ContextResolver, service_tokens: list[str],
               expected_mode: Optional[str] = None,
               apply_session_update: Optional[Callable[[object], None]] = None) -> FastAPI:
    """Build the internal app. ``service_tokens`` must be non-empty secrets from the environment."""
    if not service_tokens or any(not t for t in service_tokens):
        raise ValueError("at least one non-empty service token is required")
    app = FastAPI(title="Gate Classifier (internal)", docs_url=None, redoc_url=None, openapi_url=None)
    max_body = service.config.limits.max_body_bytes
    expected_mode = expected_mode or service.config.operating_mode

    def authenticated(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return False
        presented = header[7:].encode()
        return any(hmac.compare_digest(presented, t.encode()) for t in service_tokens)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        r = service.readiness()
        ok = r.ready and (r.status == "ready" or expected_mode == "fixture")
        if expected_mode == "production" and r.status != "ready":
            ok = False
        return JSONResponse(status_code=200 if ok else 503, content={"status": r.status if ok else "not_ready"})

    @app.post("/v1/gate")
    async def gate(request: Request):
        request_id = new_request_id()
        try:
            if not authenticated(request):
                return _error(ErrorCode.UNAUTHORIZED, request_id)
            declared = request.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > max_body):
                return _error(ErrorCode.INPUT_TOO_LONG, request_id)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > max_body:
                    return _error(ErrorCode.INPUT_TOO_LONG, request_id)
            try:
                data = json.loads(bytes(body).decode("utf-8"), object_pairs_hook=_no_duplicates)
                if not isinstance(data, dict):
                    raise ValueError
                gate_request = GateRequest.model_validate(data)
            except (ValueError, ValidationError, UnicodeDecodeError):
                return _error(ErrorCode.INVALID_REQUEST, request_id)

            owner = request.headers.get(OWNER_HEADER)
            tenant = request.headers.get(TENANT_HEADER)
            course = request.headers.get(COURSE_HEADER)
            if not owner or not tenant or not course:
                return _error(ErrorCode.FORBIDDEN, request_id)
            ctx = resolve_context(owner, tenant, course, gate_request.session_id)
            result = service.evaluate(gate_request, ctx, request_id)
            decision = result.decision
            if decision.route == Route.unavailable:
                return _error(_UNAVAILABLE_CODES.get(decision.reason, ErrorCode.SERVICE_UNAVAILABLE), request_id)
            return JSONResponse(status_code=200, content=decision.model_dump(mode="json"))
        except GateError as exc:
            if exc.session_update is not None and apply_session_update is not None:
                try:
                    apply_session_update(exc.session_update)
                except Exception:
                    pass
            return _error(exc.code, request_id)
        except Exception:  # never echo text, stack traces or entity matches
            return _error(ErrorCode.SERVICE_UNAVAILABLE, request_id)

    return app
