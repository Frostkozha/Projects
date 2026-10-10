"""Exact rendered-prompt token counting through the loaded runtime (Brain plan v0.3, section 8).

The prompt is rendered by the runtime's own loaded chat template (``/apply-template``) and counted by the
matching GGUF tokenizer (``/tokenize`` with special-token parsing, as the server does for chat prompts).
Probes establish that this count equals the runtime's reported ``usage.prompt_tokens``; character counts,
E5 token counts and guessed overhead are never used.
"""

from __future__ import annotations

from .parse import CompletionFailure
from .transport import LocalTransport


class RuntimeTokenizer:
    def __init__(self, transport: LocalTransport):
        self.transport = transport

    async def render(self, messages: list[dict], *, deadline: float) -> str:
        value = await self.transport.json("POST", "/apply-template", {"messages": messages}, deadline=deadline)
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise CompletionFailure("RUNTIME_INCOMPATIBLE")
        return prompt

    async def tokens(self, text: str, *, parse_special: bool, deadline: float) -> list[int]:
        value = await self.transport.json(
            "POST", "/tokenize", {"content": text, "add_special": True, "parse_special": parse_special},
            deadline=deadline)
        toks = value.get("tokens")
        if not isinstance(toks, list) or not all(type(t) is int for t in toks):
            raise CompletionFailure("RUNTIME_INCOMPATIBLE")
        return toks

    async def count_messages(self, messages: list[dict], *, deadline: float) -> int:
        return len(await self.tokens(await self.render(messages, deadline=deadline), parse_special=True,
                                     deadline=deadline))

    async def contains_control_tokens(self, text: str, *, deadline: float) -> bool:
        """True when the tokenizer would parse part of ``text`` as a special/control token."""
        if not text:
            return False
        special = await self.tokens(text, parse_special=True, deadline=deadline)
        plain = await self.tokens(text, parse_special=False, deadline=deadline)
        return special != plain
