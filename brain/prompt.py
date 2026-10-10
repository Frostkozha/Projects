"""Versioned fixed prompts and role-safe data encoding (Brain plan v0.3, section 8 and appendix G).

One fixed system message plus one JSON-encoded user data record. No request ID, session, tenant,
credential, history or hidden item key is ever placed in the messages.
"""

from __future__ import annotations

import re
from pathlib import Path

from contracts.json_codec import canonical_json

from .manifest import sha256_text
from .schema import BrainContext, BrainError, ErrorCode
from .request_schema import BrainRequest

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_VERSION = "brain-prompt-0.2"
TASK_FILES = {"answer_draft": "answer_draft.txt", "tutor_clue": "tutor_clue.txt"}

# Chat-control syntax that must never reach the template as data. The runtime tokenizer check
# (special-token parse comparison) is authoritative; this static screen is defence in depth.
_CONTROL = re.compile(r"<\|[^<>|]{1,64}\|>|</?think>|</?tool_call>|</?tool_response>", re.IGNORECASE)


def load_prompts() -> dict[str, str]:
    return {task: (PROMPT_DIR / name).read_text(encoding="utf-8") for task, name in TASK_FILES.items()}


def prompt_hashes() -> dict[str, str]:
    return {task: sha256_text(text) for task, text in load_prompts().items()}


def has_control_syntax(text: str) -> bool:
    return bool(_CONTROL.search(text))


def data_record(request: BrainRequest, context: BrainContext) -> dict:
    """The only student-derived content the model sees: question, passages, trusted presentation hint."""
    value = context.evidence.fresh_value()
    by_id = {p["passage_id"]: p for p in value["passages"]}
    record = {
        "task": request.task,
        "question": request.question_redacted,
        "presentation": {"level": context.presentation.level,
                         "max_visible_sentences": context.presentation.max_visible_sentences},
        "passages": [{"passage_id": pid, "text": by_id[pid]["text"]} for pid in request.passage_ids],
    }
    if request.task == "tutor_clue":
        if not context.approved_item_question:
            raise BrainError(ErrorCode.CONTEXT_MISMATCH)
        record["approved_item_question"] = context.approved_item_question
    return record


def build_messages(system_prompt: str, record: dict) -> list[dict]:
    if has_control_syntax(record["question"]) or has_control_syntax(record.get("approved_item_question") or ""):
        raise BrainError(ErrorCode.INVALID_REQUEST)
    for text in _strings(record["passages"]):
        if has_control_syntax(text):
            raise BrainError(ErrorCode.INVALID_EVIDENCE)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": canonical_json(record).decode("utf-8")},
    ]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
