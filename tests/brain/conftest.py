"""Brain fixture runtime: an in-process fake of the pinned llama-server HTTP surface (synthetic only).

It is a structural test double, not a model. It tokenizes on whitespace, parses ``<|...|>`` as one special
token only when ``parse_special`` is set, renders a ChatML-like template with the reasoning-off scaffold and
returns scripted completion bodies. Real-model behaviour is covered by ``tests/brain/real_model``.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest

from brain.capabilities import REASONING_OFF_SUFFIX
from brain.config import load_config
from brain.context import synthetic_context
from brain.parse import SentencePolicy
from brain.request_schema import BrainRequest
from brain.service import BrainService
from brain.transport import LocalTransport

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CONFIG = ROOT / "config/brain-fixture.yaml"
KEY = "fixture-only-key-not-a-secret-000000000000"
BASE = "http://127.0.0.1:8080"
P1 = ("fixture-passage-001", "Simple squamous epithelium has one layer of flattened cells.")
P2 = ("fixture-passage-002", "Goblet cells secrete mucus.")
_SPECIAL = re.compile(r"<\|[^<>|]{1,64}\|>")


def draft_body(sentences=None, used=None, status="draft") -> dict:
    sentences = sentences if sentences is not None else [
        {"sentence_id": "s1", "text": P1[1], "kind_hint": "factual", "visibility": "student",
         "cites": [P1[0]], "depends_on": []}]
    if used is None:
        used = sorted({c for s in sentences for c in s["cites"]})
    return {"schema_version": "brain-draft-0.2", "status": status, "sentences": sentences, "used_passage_ids": used}


def envelope(content, *, finish="stop", model="brain-local-v02", prompt_tokens=None, message_extra=None,
             choices=None) -> dict:
    msg = {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)}
    msg.update(message_extra or {})
    env = {"choices": choices if choices is not None else [{"finish_reason": finish, "index": 0, "message": msg}],
           "model": model, "object": "chat.completion",
           "usage": {"completion_tokens": 20, "prompt_tokens": prompt_tokens, "total_tokens": 0},
           "timings": {"prompt_n": 1, "prompt_ms": 12.5, "predicted_n": 20, "predicted_ms": 100.0}}
    if prompt_tokens is None:
        env["usage"].pop("prompt_tokens")
    return env


class FakeRuntime:
    def __init__(self):
        self.completions: list[dict] = []
        self.paths: list[str] = []
        self.completion_script: Optional[Callable[[dict], object]] = None
        self.completion_delay = 0.0
        self.processing = False
        self.stuck_processing = False
        self.accept_bad_schema = False
        self.props = {
            "default_generation_settings": {"n_ctx": 4096}, "total_slots": 1, "model_alias": "brain-local-v02",
            "model_path": "D:\\models\\Qwen3.5-9B-Q4_K_M.gguf", "modalities": {"vision": False, "video": False,
                                                                                "audio": False},
            "endpoint_props": False, "ui": False, "build_info": "b11435-43fe9c642", "chat_template": "fixture-template"}

    # tokenizer/template --------------------------------------------------------------
    @staticmethod
    def tokenize(content: str, parse_special: bool) -> list[int]:
        pieces: list[str] = []
        for p in re.findall(r"<\|[^<>|]{1,64}\|>|[^\s<]+|<", content):
            pieces.extend([p] if parse_special or not _SPECIAL.fullmatch(p) else list(p))
        return [hash(p) % 100000 for p in pieces]

    @staticmethod
    def render(messages: list[dict]) -> str:
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + REASONING_OFF_SUFFIX

    def count(self, messages) -> int:
        return len(self.tokenize(self.render(messages), True))

    # http ----------------------------------------------------------------------------
    async def handler(self, request: httpx.Request) -> httpx.Response:
        return _unread(await self._handle(request))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "unauthorized"})
        body = json.loads(request.content) if request.content else {}
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": self.tokenize(body["content"], body["parse_special"])})
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": self.render(body["messages"])})
        if path == "/props":
            return httpx.Response(200, json=self.props)
        if path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "is_processing": self.processing or self.stuck_processing}])
        if path == "/v1/chat/completions":
            schema = (body.get("response_format") or {}).get("schema") or {}
            if "$ref" in json.dumps(schema) and not self.accept_bad_schema:
                return httpx.Response(500, json={"error": {"message": "unsupported $ref"}})
            self.completions.append(body)
            self.processing = True
            try:
                if self.completion_delay:
                    await asyncio.sleep(self.completion_delay)
            finally:
                self.processing = False
            result = self.completion_script(body) if self.completion_script else None
            if isinstance(result, httpx.Response):
                return result
            if result is None:
                ids = schema["properties"]["used_passage_ids"]["items"]["enum"]
                text = json.loads(body["messages"][1]["content"])["passages"][0]["text"] if ids else ""
                result = envelope(draft_body([{"sentence_id": "s1", "text": text.split(". ")[0].rstrip(".") + ".",
                                               "kind_hint": "factual", "visibility": "student", "cites": [ids[0]],
                                               "depends_on": []}]))
            if isinstance(result, dict):
                result = dict(result)
                result.setdefault("usage", {})
                if "prompt_tokens" not in result["usage"]:
                    result["usage"] = {**result["usage"], "prompt_tokens": self.count(body["messages"])}
                return httpx.Response(200, json=result)
            return httpx.Response(200, content=result, headers={"content-type": "application/json"})
        return httpx.Response(404)


def _unread(resp: httpx.Response) -> httpx.Response:
    """Give MockTransport an unconsumed byte stream so the bounded transport can read raw chunks."""
    if not resp.is_stream_consumed and not hasattr(resp, "_content"):
        return resp
    return httpx.Response(resp.status_code, headers=resp.headers, stream=httpx.ByteStream(resp.content))


@pytest.fixture
def runtime():
    return FakeRuntime()


@pytest.fixture
def fixture_cfg():
    return load_config(FIXTURE_CONFIG)


def make_service(runtime: FakeRuntime, cfg=None, *, ready=True, audit=None) -> BrainService:
    cfg = cfg or load_config(FIXTURE_CONFIG)
    transport = LocalTransport(BASE, KEY, http_transport=httpx.MockTransport(runtime.handler))
    svc = BrainService(cfg, transport=transport, manifest=None, sentence_policy=SentencePolicy(), audit=audit,
                       api_key=KEY)
    if ready:
        svc.mark_fixture_ready()
    return svc


def make_request(ctx_passages=(P1,), *, question="Describe simple squamous epithelium.", task="answer_draft",
                 rid=None, passage_ids=None, deadline_seconds=60.0, **ctx_kw):
    rid, ctx = synthetic_context(list(ctx_passages), request_id=rid, deadline_seconds=deadline_seconds, **ctx_kw)
    req = BrainRequest(schema_version="brain-request-0.2", request_id=uuid.UUID(rid), task=task,
                       question_redacted=question,
                       passage_ids=list(passage_ids or [p[0] for p in ctx_passages]),
                       prompt_version="brain-prompt-0.2", sampling_profile_version="grounded-dev-0.2")
    return req, ctx


def run(coro):
    return asyncio.run(coro)
