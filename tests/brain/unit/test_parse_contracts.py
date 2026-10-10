"""B01, B12, B14-B22: request schema, inline grammar, strict parsing and metrics (synthetic fixtures)."""

from __future__ import annotations

import json
import uuid

import pytest

from brain.grammar import inline_schema
from brain.parse import CompletionFailure, SentencePolicy, observed_metrics, parse_completion
from brain.request_schema import BrainRequest
from brain.schema import BrainMetrics, BrainResult
from contracts.draft import DraftAnswer
from contracts.json_codec import InvalidJSON, canonical_json, load_object, validate_wire

from ..conftest import P1, P2, draft_body, envelope

POLICY = SentencePolicy()
REQ = {"schema_version": "brain-request-0.2", "request_id": "00000000-0000-4000-8000-000000000004",
       "task": "answer_draft", "question_redacted": "Describe simple squamous epithelium.",
       "passage_ids": ["fixture-passage-001"], "prompt_version": "brain-prompt-0.2",
       "sampling_profile_version": "grounded-dev-0.2"}


def parse(env, ids=(P1[0],), policy=POLICY, max_visible=4):
    raw = env if isinstance(env, bytes) else canonical_json(env)
    return parse_completion(raw, expected_alias="brain-local-v02", supplied_ids=ids, sentence_policy=policy,
                            max_visible_sentences=max_visible)


def code(env, **kw):
    with pytest.raises(CompletionFailure) as e:
        parse(env, **kw)
    return e.value.code


def sent(text, sid="s1", cites=(P1[0],), deps=(), kind="factual", vis="student"):
    return {"sentence_id": sid, "text": text, "kind_hint": kind, "visibility": vis, "cites": list(cites),
            "depends_on": list(deps)}


# ----------------------------------------------------------------------------- B01 request schema

def test_B01_valid_request_and_uuid_wire():
    r = validate_wire(BrainRequest, canonical_json(REQ), max_bytes=65536)
    assert isinstance(r.request_id, uuid.UUID)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra="x"),
    lambda d: d.update(task="quiz_grade"),
    lambda d: d.update(task="image_caption"),
    lambda d: d.update(passage_ids="fixture-passage-001"),
    lambda d: d.update(passage_ids=[]),
    lambda d: d.update(passage_ids=["a", "a"]),
    lambda d: d.update(passage_ids=["a", "b", "c", "d", "e", "f"]),
    lambda d: d.update(question_redacted="   "),
    lambda d: d.update(question_redacted="x" * 4001),
    lambda d: d.update(request_id="not-a-uuid"),
    lambda d: d.update(schema_version="brain-request-0.3"),
    lambda d: d.update(prompt_version="bad version!"),
    lambda d: d.pop("sampling_profile_version"),
])
def test_B01_request_rejections(mutate):
    d = dict(REQ)
    mutate(d)
    with pytest.raises(ValueError):
        validate_wire(BrainRequest, canonical_json(d), max_bytes=65536)


def test_B01_duplicate_keys_nonfinite_and_bad_utf8():
    raw = canonical_json(REQ)[:-1] + b',"task":"tutor_clue"}'
    with pytest.raises(InvalidJSON):
        validate_wire(BrainRequest, raw, max_bytes=65536)
    with pytest.raises(InvalidJSON):
        load_object(b'{"a": NaN}', max_bytes=100)
    with pytest.raises(InvalidJSON):
        load_object(b'{"a": "\xff"}', max_bytes=100)
    with pytest.raises(InvalidJSON):
        load_object(b'{"a": "\\ud800"}', max_bytes=100)
    with pytest.raises(InvalidJSON):
        load_object(b'{"a": 1} {"b": 2}', max_bytes=100)


# ----------------------------------------------------------------------------- B12 inline grammar

def test_B12_inline_schema_is_closed_and_bound_to_current_ids():
    s = inline_schema((P1[0], P2[0]), 4)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "$ref" not in node and "$defs" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    item = s["properties"]["sentences"]["items"]["properties"]
    assert item["cites"]["items"]["enum"] == [P1[0], P2[0]]
    assert item["kind_hint"]["enum"] == ["factual"] and item["visibility"]["enum"] == ["student"]
    assert s["properties"]["sentences"]["maxItems"] == 4
    with pytest.raises(ValueError):
        inline_schema((), 4)
    with pytest.raises(ValueError):
        inline_schema(("a", "a"), 4)


# ----------------------------------------------------------------------------- B14 whole-object parsing

def test_B14_valid_draft_parses():
    d = parse(envelope(draft_body()))
    assert isinstance(d, DraftAnswer) and d.status == "draft"


@pytest.mark.parametrize("content", [
    "```json\n" + json.dumps(draft_body()) + "\n```",
    "Here is the answer: " + json.dumps(draft_body()),
    json.dumps(draft_body()) + json.dumps(draft_body()),
    json.dumps(draft_body()) + " trailing prose",
    json.dumps(draft_body())[:-1] + ',"status":"draft"}',
    "<think>plan</think>" + json.dumps(draft_body()),
    "// comment\n" + json.dumps(draft_body()),
    json.dumps({**draft_body(), "notes": "extra"}),
    "",
])
def test_B14_noncanonical_content_rejected(content):
    assert code(envelope(content)) == "INVALID_OUTPUT"


def test_B14_envelope_duplicate_key_rejected():
    raw = canonical_json(envelope(draft_body()))[:-1] + b',"model":"brain-local-v02"}'
    assert code(raw) == "INVALID_OUTPUT"


def test_B14_alias_mismatch_is_runtime_incompatible():
    assert code(envelope(draft_body(), model="other-model")) == "RUNTIME_INCOMPATIBLE"


# ----------------------------------------------------------------------------- B15 structure invariants

@pytest.mark.parametrize("sentences,used", [
    ([sent("Cells are flat."), sent("Cells are thin.", sid="s1")], None),                    # duplicate id
    ([sent("Cells are flat.", deps=("s2",)), sent("Cells are thin.", sid="s2")], None),       # forward dependency
    ([sent("Cells are flat.", deps=("s1",))], None),                                          # self dependency
    ([sent("Cells are flat.")], ["fixture-passage-002", "fixture-passage-001"]),               # wrong union
    ([sent("Cells are flat.", cites=(P1[0], P2[0]))], [P2[0], P1[0]]),                        # unsorted union
    ([sent("Cells are flat.", cites=(P1[0], P1[0]))], [P1[0]]),                               # duplicate cites
])
def test_B15_structural_negatives(sentences, used):
    assert code(envelope(draft_body(sentences, used)), ids=(P1[0], P2[0])) == "INVALID_OUTPUT"


def test_B15_positive_dependencies_and_union():
    body = draft_body([sent("Cells are flat."), sent("They line vessels.", sid="s2", cites=(P2[0], P1[0]),
                                                      deps=("s1",))])
    d = parse(envelope(body), ids=(P1[0], P2[0]))
    assert d.used_passage_ids == (P1[0], P2[0])


def test_B15_unshown_citation_rejected():
    assert code(envelope(draft_body([sent("Cells are flat.", cites=("other-passage",))]))) == "INVALID_OUTPUT"


# ----------------------------------------------------------------------------- B16 one sentence per object

@pytest.mark.parametrize("text", [
    "Cells are flat. They line vessels.",
    "- Cells are flat.",
    "Cells are flat.\nThey line vessels.",
    "Cells are **flat**.",
    "See `cells` for detail.",
    "# Cells are flat.",
    "Cells are flat",
    " Cells are flat.",
    "Cells are <b>flat</b>.",
    "Read https://example.org for detail.",
])
def test_B16_hidden_sentences_and_markup_rejected(text):
    assert code(envelope(draft_body([sent(text)]))) == "INVALID_OUTPUT"


# ----------------------------------------------------------------------------- B17/B18 visibility and kind

def test_B17_no_evidence_accepted_structurally():
    d = parse(envelope(draft_body([], [], status="no_evidence")))
    assert d.status == "no_evidence" and not d.sentences


def test_B17_nonempty_abstention_and_empty_draft_rejected():
    assert code(envelope(draft_body([sent("Cells are flat.")], status="no_evidence"))) == "INVALID_OUTPUT"
    assert code(envelope(draft_body([], []))) == "INVALID_OUTPUT"


@pytest.mark.parametrize("kind,vis", [("question", "student"), ("advice", "student"), ("factual", "internal"),
                                      ("transition", "student")])
def test_B18_pilot_requires_factual_student(kind, vis):
    assert code(envelope(draft_body([sent("Cells are flat.", kind=kind, vis=vis)]))) == "INVALID_OUTPUT"


def test_B18_visible_sentence_cap():
    body = draft_body([sent(f"Cells are flat number {i}.", sid=f"s{i}") for i in range(1, 4)])
    assert code(envelope(body), max_visible=2) == "INVALID_OUTPUT"


# ----------------------------------------------------------------------------- B19 trusted metadata

@pytest.mark.parametrize("text", ["The response code is A5.", "Cells are flat per fixture-passage-001.",
                                  "Use reply_key no_source here.", "See www.example.org for cells."])
def test_B19_trusted_metadata_rejected(text):
    assert code(envelope(draft_body([sent(text)]))) == "INVALID_OUTPUT"


# ----------------------------------------------------------------------------- B10/B20 finish reasons, reasoning

def test_B20_finish_reasons():
    assert code(envelope(draft_body(), finish="length")) == "OUTPUT_LIMIT"
    assert code(envelope(draft_body(), finish="tool_calls")) == "INVALID_OUTPUT"
    assert code(envelope(draft_body(), finish=None)) == "INVALID_OUTPUT"
    assert code(envelope(draft_body(), finish="content_filter")) == "INVALID_OUTPUT"


@pytest.mark.parametrize("extra", [{"reasoning_content": "thinking..."},
                                   {"tool_calls": [{"id": "1", "type": "function"}]},
                                   {"function_call": {"name": "x"}}, {"audio": None}])
def test_B10_reasoning_tools_and_unknown_message_keys_rejected(extra):
    assert code(envelope(draft_body(), message_extra=extra)) == "INVALID_OUTPUT"


def test_B10_empty_reasoning_metadata_allowed():
    assert parse(envelope(draft_body(), message_extra={"reasoning_content": ""})).status == "draft"


# ----------------------------------------------------------------------------- B21 metrics

def test_B21_metrics_observed_or_null():
    m = observed_metrics(canonical_json(envelope(draft_body(), prompt_tokens=33)))
    assert m == {"prompt_ms": 12.5, "generation_ms": 100.0, "prompt_tokens": 33, "completion_tokens": 20,
                 "finish_reason": "stop"}
    bare = {"model": "brain-local-v02", "choices": [{"finish_reason": "stop", "message": {}}]}
    assert observed_metrics(canonical_json(bare)) == {"prompt_ms": None, "generation_ms": None,
                                                      "prompt_tokens": None, "completion_tokens": None,
                                                      "finish_reason": "stop"}
    bad = envelope(draft_body(), prompt_tokens=True)
    bad["usage"]["completion_tokens"] = -1
    m2 = observed_metrics(canonical_json(bad))
    assert m2["prompt_tokens"] is None and m2["completion_tokens"] is None
    with pytest.raises(ValueError):
        BrainMetrics(queue_ms=float("inf"), prompt_ms=None, generation_ms=None, total_ms=None, prompt_tokens=None,
                     completion_tokens=None, finish_reason=None)
    with pytest.raises(ValueError):
        BrainMetrics(queue_ms=None, prompt_ms=None, generation_ms=None, total_ms=None, prompt_tokens=True,
                     completion_tokens=None, finish_reason=None)


# ----------------------------------------------------------------------------- B22 envelope shape

def test_B22_choice_count_and_bytes():
    assert code(envelope(draft_body(), choices=[])) == "INVALID_OUTPUT"
    two = envelope(draft_body())
    two["choices"] = two["choices"] * 2
    assert code(two) == "INVALID_OUTPUT"
    assert code(b"not json") == "INVALID_OUTPUT"
    assert code(b"[" + b"1," * 140000 + b"1]") == "INVALID_OUTPUT"


# ----------------------------------------------------------------------------- BrainResult invariants

def test_brain_result_invariants():
    m = BrainMetrics(queue_ms=0.0, prompt_ms=None, generation_ms=None, total_ms=1.0, prompt_tokens=None,
                     completion_tokens=None, finish_reason="stop")
    base = dict(schema_version="brain-result-0.2", request_id=uuid.uuid4(), model_version=None,
                runtime_version=None, prompt_version="brain-prompt-0.2", sampling_profile_version="grounded-dev-0.2",
                evidence_digest="a" * 64, metrics=m)
    draft = DraftAnswer.model_validate(draft_body())
    BrainResult(status="ok", draft=draft, error_code=None, **base)
    with pytest.raises(ValueError):
        BrainResult(status="ok", draft=None, error_code=None, **base)
    with pytest.raises(ValueError):
        BrainResult(status="error", draft=draft, error_code="INVALID_OUTPUT", **base)
    with pytest.raises(ValueError):
        BrainResult(status="error", draft=None, error_code=None, **base)
    with pytest.raises(ValueError):
        BrainResult(status="no_evidence", draft=draft, error_code=None, **base)
    with pytest.raises(ValueError):
        BrainResult(status="error", draft=None, error_code="NOT_A_CODE", **base)
