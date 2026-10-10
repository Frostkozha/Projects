"""Public routes and response envelopes (Integration plan v0.3, sections 5, 10, appendix A, G).

``POST /v1/courses/{course_id}/interactions`` is the only teaching entry point. Bodies are read
incrementally and stopped at 32 KiB, JSON is parsed with duplicate-key rejection, identity comes only from
the server-side adapter and every non-200 body is exactly ``{"error_code", "request_id"}``. The minimal
student page is served from the backend: no CDN, fonts, analytics or build framework.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from contracts.json_codec import validate_wire

from .app import TutorApp
from .auth import COOKIE, Identity, IdentityError
from .input_schema import CloseSessionInput, DevLoginInput, NoticeAcceptInput, ReportInput
from .public import ERROR_STATUS, ApiError, NoticeView

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 32 * 1024
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}


def _error(code: str, request_id: Optional[str] = None) -> JSONResponse:
    return JSONResponse(status_code=ERROR_STATUS[code], content={"error_code": code, "request_id": request_id},
                        headers=SECURITY_HEADERS)


def _ok(model) -> JSONResponse:
    return JSONResponse(status_code=200, content=model.model_dump(mode="json") if hasattr(model, "model_dump")
                        else model, headers=SECURITY_HEADERS)


async def _read_body(request: Request) -> bytes:
    ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if ctype != "application/json":
        raise ApiError("INVALID_REQUEST")
    buf = bytearray()
    async for chunk in request.stream():
        if len(buf) + len(chunk) > MAX_BODY:
            raise ApiError("INPUT_TOO_LONG")  # stop without joining an oversized buffer
        buf.extend(chunk)
    return bytes(buf)


def create_app(tutor: TutorApp, *, start_components: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if start_components:
            await tutor.start()
        yield
        if start_components:
            await tutor.stop()

    app = FastAPI(title="Histology study tutor (local)", lifespan=lifespan, docs_url=None, redoc_url=None)
    coord = tutor.coordinator
    cfg = tutor.cfg

    def identify(request: Request) -> Identity:
        try:
            return tutor.identity.identify({k.lower(): v for k, v in request.headers.items()},
                                           dict(request.cookies))
        except IdentityError as exc:
            raise ApiError(exc.code) from None

    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):
        return _error(exc.code, exc.request_id)

    @app.exception_handler(Exception)
    async def _unexpected(request, exc):  # never leak detail
        return _error("SERVICE_UNAVAILABLE")

    from fastapi.exceptions import RequestValidationError  # noqa: PLC0415

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc):
        return _error("INVALID_REQUEST")

    # ---------------------------------------------------------------- page and health

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html", headers=SECURITY_HEADERS)

    @app.get("/static/{name}", include_in_schema=False)
    async def static(name: str):
        if name not in ("app.js", "app.css"):
            return _error("CONTENT_UNAVAILABLE")
        return FileResponse(STATIC / name, headers=SECURITY_HEADERS)

    @app.get("/v1/health/live")
    async def live():
        return _ok({"live": True})

    @app.get("/v1/health/ready")
    async def ready():
        r = tutor.readiness()
        return JSONResponse(status_code=200 if r["ready"] else 503,
                            content={"ready": r["ready"], "status": r["status"], "profile": r["profile"]},
                            headers=SECURITY_HEADERS)

    # ---------------------------------------------------------------- development identity

    @app.post("/v1/dev/login")
    async def dev_login(request: Request):
        if cfg.profile == "student_release" or not hasattr(tutor.identity, "issue"):
            return _error("FORBIDDEN")
        body = validate_or_422(DevLoginInput, await _read_body(request))
        resp = JSONResponse({"user": body.user, "synthetic": True, "course_id": cfg.course_id},
                            headers=SECURITY_HEADERS)
        resp.set_cookie(COOKIE, tutor.identity.issue(body.user), httponly=True, samesite="strict", secure=False,
                        path="/")
        return resp

    @app.post("/v1/dev/logout")
    async def dev_logout():
        resp = JSONResponse({"ok": True}, headers=SECURITY_HEADERS)
        resp.delete_cookie(COOKIE, path="/")
        return resp

    @app.get("/v1/me")
    async def me(request: Request):
        ident = identify(request)
        return _ok({"course_id": cfg.course_id, "profile": cfg.profile, "synthetic_identity": ident.synthetic,
                    "notice_accepted": await coord.notice_accepted(ident)})

    # ---------------------------------------------------------------- notices

    @app.get("/v1/notices/current", response_model=NoticeView)
    async def notice_current(request: Request):
        ident = identify(request)
        return _ok(NoticeView(notice_version=cfg.notice.version, text=cfg.notice.text,
                              accepted=await coord.notice_accepted(ident)))

    @app.post("/v1/notices/accept")
    async def notice_accept(request: Request):
        ident = identify(request)
        body = validate_or_422(NoticeAcceptInput, await _read_body(request))
        await coord.accept_notice(ident, body.notice_version)
        return _ok({"notice_version": body.notice_version, "accepted": True})

    # ---------------------------------------------------------------- teaching entry point

    @app.post("/v1/courses/{course_id}/interactions")
    async def interactions(course_id: str, request: Request):
        ident = identify(request)
        key = request.headers.get("idempotency-key", "")
        raw = await _read_body(request)
        reply = await coord.interact(ident, course_id, key, raw)
        return _ok(reply)

    @app.post("/v1/courses/{course_id}/interactions/cancel")
    async def cancel(course_id: str, request: Request):
        ident = identify(request)
        key = request.headers.get("idempotency-key", "")
        return _ok({"cancelled": coord.cancel(ident, key)})

    # ---------------------------------------------------------------- sessions, reports, evidence

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        return _ok(await coord.get_session(identify(request), session_id))

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        ident = identify(request)
        body = validate_or_422(CloseSessionInput, await _read_body(request))
        return _ok(await coord.close_session(ident, session_id, body.expected_session_revision))

    @app.post("/v1/interactions/{request_id}/reports")
    async def report(request_id: str, request: Request):
        ident = identify(request)
        body = validate_or_422(ReportInput, await _read_body(request))
        await coord.report(ident, request_id, body.category, body.message)
        return _ok({"received": True})

    @app.get("/v1/evidence/{passage_id}")
    async def evidence(passage_id: str, request: Request, kb_version: str = ""):
        return _ok(await coord.evidence(identify(request), passage_id, kb_version))

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response: Response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    return app


def validate_or_422(model, raw: bytes):
    try:
        return validate_wire(model, raw, max_bytes=MAX_BODY)
    except Exception:  # noqa: BLE001
        raise ApiError("INVALID_REQUEST") from None
