"""Bounded loopback HTTP transport (Brain plan v0.3, section 11 and appendix D).

Trusted operator configuration only: fixed loopback base URL, redirects disabled, no proxy/env, identity
encoding, 256 KiB private buffer and an absolute event-loop deadline. Headers, payloads and bodies are
never printed or logged. ``http_transport`` exists for isolated HTTPX mock tests only.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from urllib.parse import urlsplit

import httpx

from contracts.json_codec import InvalidJSON, canonical_json, load_object

from .parse import CompletionFailure

MAX_BODY = 256 * 1024


class LocalTransport:
    def __init__(self, base_url: str, api_key: str, *, max_body: int = MAX_BODY, http_transport=None):
        url = urlsplit(base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"}
                or url.username or url.password or url.path not in {"", "/"}
                or url.query or url.fragment or url.port is None):
            raise ValueError("INVALID_REQUEST")
        if not api_key or any(c in api_key for c in "\r\n"):
            raise ValueError("INVALID_REQUEST")
        self.max_body = max_body
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), trust_env=False, follow_redirects=False,
            headers={"Authorization": "Bearer " + api_key, "Accept-Encoding": "identity"},
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            transport=http_transport,
        )

    def __repr__(self) -> str:  # never expose headers
        return "LocalTransport(<redacted>)"

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(self, method: str, path: str, payload: Optional[dict], *, deadline: float) -> bytes:
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise CompletionFailure("DEADLINE_EXCEEDED")
        try:
            async with asyncio.timeout_at(deadline):
                timeout = httpx.Timeout(remaining, connect=min(2.0, remaining))
                kwargs = {"timeout": timeout}
                if payload is not None:
                    kwargs["content"] = canonical_json(payload)
                    kwargs["headers"] = {"Content-Type": "application/json"}
                async with self.client.stream(method, path, **kwargs) as response:
                    if response.status_code != 200:
                        raise CompletionFailure("TRANSPORT_FAILURE")
                    mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if mime != "application/json":
                        raise CompletionFailure("TRANSPORT_FAILURE")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise CompletionFailure("TRANSPORT_FAILURE")
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(body) + len(chunk) > self.max_body:
                            raise CompletionFailure("OUTPUT_LIMIT")
                        body.extend(chunk)
                    return bytes(body)
        except (TimeoutError, httpx.TimeoutException):
            raise CompletionFailure("DEADLINE_EXCEEDED") from None
        except httpx.HTTPError:
            raise CompletionFailure("TRANSPORT_FAILURE") from None

    async def complete(self, payload: dict, *, deadline: float) -> bytes:
        return await self._request("POST", "/v1/chat/completions", payload, deadline=deadline)

    async def json(self, method: str, path: str, payload: Optional[dict] = None, *, deadline: float):
        """Auxiliary runtime endpoints (tokenize, apply-template, props, slots, health)."""
        if path not in {"/tokenize", "/apply-template", "/props", "/slots", "/health", "/v1/models"}:
            raise ValueError("INVALID_REQUEST")
        raw = await self._request(method, path, payload, deadline=deadline)
        if path == "/slots":
            import json  # noqa: PLC0415 - top-level array, still duplicate-key checked per object below

            from contracts.json_codec import _unique_object  # noqa: PLC0415
            try:
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError):
                raise CompletionFailure("RUNTIME_INCOMPATIBLE") from None
            if not isinstance(value, list):
                raise CompletionFailure("RUNTIME_INCOMPATIBLE")
            return value
        try:
            return load_object(raw, max_bytes=self.max_body)
        except InvalidJSON:
            raise CompletionFailure("RUNTIME_INCOMPATIBLE") from None
