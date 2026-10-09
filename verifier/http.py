"""Optional internal authenticated verifier wrapper (Verifier spec v0.2, sections 4.1, 11).

POST /v1/verify accepts ``{"request": VerifyRequest, "evidence_ref": "<opaque>"}`` from an authenticated
internal service only. The server resolves ``evidence_ref`` to the trusted VerifyContext (request, scope
and expiry checked); callers can never submit evidence text. There is no public student endpoint.

Status mapping: 401/403 authentication, 413 body over the limit (checked before JSON parsing),
422 malformed transport or schema, 200 domain decisions (approved/rejected), 503 operational error.
Health endpoints expose no source text or account details.
"""

from __future__ import annotations

import asyncio
import hmac
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from contracts.models import ContractError, VerifyRequest, strict_json_loads

from .service import Verifier, VerifyContext


class VerifyEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request: VerifyRequest
    evidence_ref: StrictStr


class EvidenceRefError(Exception):
    """Unknown, expired or out-of-scope evidence reference (403)."""


ContextResolver = Callable[[str, VerifyRequest, str], VerifyContext]
"""(evidence_ref, request, caller_service_id) -> trusted VerifyContext; raises EvidenceRefError."""


def create_app(verifier: Verifier, resolve: ContextResolver, service_tokens: dict[str, str]) -> FastAPI:
    """``service_tokens`` maps service id -> secret bearer token (from the environment, never committed)."""
    if not service_tokens or any(not t for t in service_tokens.values()):
        raise ValueError("at least one non-empty service token is required")
    app = FastAPI(title="Verifier (internal)", docs_url=None, redoc_url=None, openapi_url=None)
    max_body = verifier.profile.runtime.max_body_bytes
    lock = asyncio.Lock()  # one scheduling point per process: no duplicate request scheduling

    def caller(request: Request) -> Optional[str]:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return None
        presented = header[7:].encode()
        for sid, token in service_tokens.items():
            if hmac.compare_digest(presented, token.encode()):
                return sid
        return None

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        r = verifier.readiness()
        return JSONResponse(status_code=200 if r["ready"] else 503,
                            content={"status": "ready" if r["ready"] else "not_ready",
                                     "operating_mode": r["operating_mode"],
                                     "student_release_ready": r["student_release_ready"]})

    @app.post("/v1/verify")
    async def verify(request: Request):
        if request.headers.get("authorization") is None:
            return JSONResponse(status_code=401, content={"error_code": "UNAUTHORIZED"})
        service_id = caller(request)
        if service_id is None:
            return JSONResponse(status_code=403, content={"error_code": "FORBIDDEN"})
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > max_body):
            return JSONResponse(status_code=413, content={"error_code": "BODY_TOO_LARGE"})
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_body:
                return JSONResponse(status_code=413, content={"error_code": "BODY_TOO_LARGE"})
        try:
            envelope = VerifyEnvelope.model_validate(strict_json_loads(body))
        except (ContractError, ValidationError):
            return JSONResponse(status_code=422, content={"error_code": "INVALID_REQUEST"})
        try:
            ctx = resolve(envelope.evidence_ref, envelope.request, service_id)
        except EvidenceRefError:
            return JSONResponse(status_code=403, content={"error_code": "EVIDENCE_REF_REJECTED"})
        async with lock:
            result = await asyncio.to_thread(verifier.verify, envelope.request, ctx)
        status = 503 if result.status == "error" else 200
        return JSONResponse(status_code=status, content=result.model_dump(mode="json"))

    return app
