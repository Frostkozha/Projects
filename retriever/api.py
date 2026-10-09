"""Optional authenticated internal HTTP wrapper (Retriever spec v0.2, sections 6, 7.2, 14).

POST /v1/retrieve, POST /v1/retrieve-item, GET /v1/evidence/{passage_id}, health endpoints. The
trusted RetrievalContext is resolved server-side from the authenticated service identity; it is never
read from a JSON body. Every non-200 body is exactly {"error_code", "request_id"}.
"""

from __future__ import annotations

import hmac
import json
import uuid
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schema import ErrorCode, RetrievalContext, RetrievalError, RetrievalResult, Status, http_status_for
from .service import RetrieverService

ContextResolver = Callable[[str, Request], RetrievalContext]
"""(service_id, request) -> trusted context; raises RetrievalError(FORBIDDEN) when not resolvable."""


def _err(code: ErrorCode, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=http_status_for(code), content={"error_code": code.value, "request_id": request_id})


def _valid_uuid(v) -> Optional[str]:
    try:
        if isinstance(v, str) and str(uuid.UUID(v)) == v.lower():
            return v.lower()
    except ValueError:
        pass
    return None


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def create_app(service: RetrieverService, resolve_context: ContextResolver, service_tokens: dict[str, str]) -> FastAPI:
    """``service_tokens`` maps a secret bearer token to its service_id (from the environment, never code)."""
    if not service_tokens:
        raise ValueError("service tokens required")
    app = FastAPI(title="Retriever (internal)", docs_url=None, redoc_url=None, openapi_url=None)
    max_body = service.profile.limits.max_body_bytes

    def identify(request: Request) -> Optional[str]:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return None
        presented = header[7:].encode()
        for token, sid in service_tokens.items():
            if hmac.compare_digest(presented, token.encode()):
                return sid
        return None

    async def body_json(request: Request):
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > max_body):
            raise RetrievalError(ErrorCode.INPUT_TOO_LONG)
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
            if len(buf) > max_body:
                raise RetrievalError(ErrorCode.INPUT_TOO_LONG)
        try:
            data = json.loads(bytes(buf).decode("utf-8"), object_pairs_hook=_no_dupes)
        except (ValueError, UnicodeDecodeError):
            raise RetrievalError(ErrorCode.INVALID_REQUEST) from None
        if not isinstance(data, dict):
            raise RetrievalError(ErrorCode.INVALID_REQUEST)
        return data

    def respond(result: RetrievalResult) -> JSONResponse:
        if result.status == Status.error:
            return _err(result.error_code, result.request_id)
        return JSONResponse(status_code=200, content=result.model_dump(mode="json"))

    async def handle(request: Request, op: str):
        request_id = str(uuid.uuid4())
        try:
            sid = identify(request)
            if sid is None:
                return _err(ErrorCode.UNAUTHORIZED, request_id)
            data = await body_json(request)
            request_id = _valid_uuid(data.get("request_id")) or request_id
            ctx = resolve_context(sid, request)
            fn = service.retrieve if op == "search" else service.fetch_item_evidence
            return respond(fn(data, ctx))
        except RetrievalError as exc:
            return _err(exc.code, request_id)
        except Exception:
            return _err(ErrorCode.SEARCH_FAILED, request_id)

    @app.post("/v1/retrieve")
    async def retrieve(request: Request):
        return await handle(request, "search")

    @app.post("/v1/retrieve-item")
    async def retrieve_item(request: Request):
        return await handle(request, "item")

    @app.get("/v1/evidence/{passage_id}")
    async def evidence(passage_id: str, kb_version: str, request: Request):
        request_id = str(uuid.uuid4())
        try:
            sid = identify(request)
            if sid is None:
                return _err(ErrorCode.UNAUTHORIZED, request_id)
            ctx = resolve_context(sid, request)
            p = service.get_evidence(passage_id, kb_version, ctx)
            return JSONResponse(status_code=200, content=p.model_dump(mode="json"))
        except RetrievalError as exc:
            return _err(exc.code, request_id)
        except Exception:
            return _err(ErrorCode.INDEX_UNAVAILABLE, request_id)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        r = service.readiness()
        return JSONResponse(status_code=200 if r.ready else 503, content={"status": r.status if r.ready else "not_ready"})

    return app
