"""Coordinator fixture harness: the real API, store, gate, verifier and delivery with explicit fixture adapters.

All content is synthetic. Forbidden-call assertions use the fixture adapters' call logs.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tutor_app.api import create_app
from tutor_app.app import build_app
from tutor_app.config import load_tutor_config
from tutor_app.fixtures import FixtureBrain

ROOT = Path(__file__).resolve().parents[2]
NOTICE = "fixture-notice-0.2"
COURSE = "histology-dev"
Q = "Explain the structure of simple squamous epithelium."


class SpyVerifier:
    def __init__(self, inner, before_return=None, mutate=None):
        self.inner = inner
        self.calls = []
        self.before_return = before_return
        self.mutate = mutate
        self.rules = inner.rules

    def verify(self, request, ctx):
        self.calls.append((request, ctx))
        result = self.inner.verify(request, ctx)
        if self.before_return:
            self.before_return(result)
        return self.mutate(result) if self.mutate else result

    def readiness(self):
        return self.inner.readiness()


class UserClient:
    """Sends one synthetic identity's cookie explicitly on the shared client."""

    def __init__(self, client, cookie):
        self._c = client
        self.cookie = cookie

    def _h(self, headers):
        h = dict(headers or {})
        h["Cookie"] = f"tutor_dev_identity={self.cookie}"
        return h

    def get(self, url, **kw):
        return self._c.get(url, headers=self._h(kw.pop("headers", None)), **kw)

    def post(self, url, **kw):
        return self._c.post(url, headers=self._h(kw.pop("headers", None)), **kw)

    def delete(self, url, **kw):
        return self._c.request("DELETE", url, headers=self._h(kw.pop("headers", None)), **kw)


class Harness:
    def __init__(self, tmp_path, *, brain=None, cfg_update=None, scorer=None):
        cfg = load_tutor_config(ROOT / "config/fixture.yaml")
        update = {"db_path": str(tmp_path / "app.sqlite"), "approved_roots": [".", str(tmp_path)]}
        update.update(cfg_update or {})
        self.cfg = cfg.model_copy(update=update)
        self.brain = brain or FixtureBrain()
        self.tutor = build_app(self.cfg, brain=self.brain)
        self.comps = self.tutor.components
        self.retriever = self.comps.retriever
        self.verifier = SpyVerifier(self.comps.verifier)
        self.comps.verifier = self.verifier
        if scorer is not None:
            self.comps.gate.fixture_scorer = scorer
        self.store = self.tutor.store
        self.coord = self.tutor.coordinator
        self.app = create_app(self.tutor)
        self._client = TestClient(self.app)
        self._client.__enter__()  # one client = one event loop, as under uvicorn
        self.cookies: dict[str, str] = {}
        self.users: dict[str, UserClient] = {}
        self.login("student-a")
        self.client = self.users["student-a"]

    def close(self):
        self._client.__exit__(None, None, None)  # lifespan shutdown closes the store

    def login(self, user: str, *, accept=True) -> "UserClient":
        r = self._client.post("/v1/dev/login", json={"user": user})
        assert r.status_code == 200
        self.cookies[user] = r.cookies.get("tutor_dev_identity")
        self._client.cookies.clear()
        c = self.users[user] = UserClient(self._client, self.cookies[user])
        if accept:
            assert c.post("/v1/notices/accept", json={"notice_version": NOTICE, "accepted": True}).status_code == 200
        return c

    def ask(self, text, *, mode=None, session=None, key=None, user="student-a", raw=None, course=COURSE,
            headers=None):
        c = self.users[user]
        body = raw if raw is not None else json.dumps(
            {"text": text, "requested_mode": mode, "session_id": session["session_id"] if session else None,
             "expected_session_revision": session["revision"] if session else None, "notice_version": NOTICE})
        h = {"Idempotency-Key": key or str(uuid.uuid4()), "Content-Type": "application/json"}
        h.update(headers or {})
        r = c.post(f"/v1/courses/{course}/interactions", content=body, headers=h)
        return r.status_code, r.json()

    def q(self, sql, *args):
        return self.store.run_sync(lambda c: [dict(r) for r in c.execute(sql, args)])

    @property
    def calls(self):
        return {"retrieve": len(self.retriever.calls), "brain": len(self.brain.calls),
                "verify": len(self.verifier.calls)}


@pytest.fixture
def make(tmp_path):
    made = []

    def factory(**kw):
        h = Harness(tmp_path / f"h{len(made)}", **kw)
        made.append(h)
        return h

    (tmp_path).mkdir(exist_ok=True)
    yield factory
    for h in made:
        h.close()


@pytest.fixture
def h(make):
    return make()


def scores(*, topic=None, risks=None, mode=None):
    s = {"mode": mode or {"answer": 0.9, "tutor": 0.05, "quiz": 0.05},
         "topic_scope": topic or {"course_related": 0.95, "outside_course": 0.02, "nonmedical": 0.01, "unclear": 0.02},
         "risks": {k: 0.01 for k in ("real_person_advice", "imminent_emergency", "self_harm_crisis", "assessed_work",
                                     "instruction_override", "private_data_request")},
         "libraries": {"lib1": 0.9, "lib2": 0.6, "lib3": None, "lib4": None, "lib5": 0.4, "lib6": 0.2}}
    s["risks"].update(risks or {})
    return s
