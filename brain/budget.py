"""Context ceiling primitive (Brain plan v0.3, section 3 and appendix B).

``prompt_tokens`` must come from the matching GGUF tokenizer applied to the fully rendered chat request.
An oversize count fails before model work; evidence is never clipped.
"""

from __future__ import annotations

CONTEXT_TOKENS = 4096
PROMPT_LIMIT = 2816
OUTPUT_LIMIT = 1024
RESERVE_TOKENS = 256


def check_budget(prompt_tokens: int, *, prompt_limit: int = PROMPT_LIMIT, output_limit: int = OUTPUT_LIMIT,
                 reserve: int = RESERVE_TOKENS, context: int = CONTEXT_TOKENS) -> None:
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("INVALID_REQUEST")
    if prompt_tokens > prompt_limit or prompt_tokens + output_limit + reserve > context:
        raise ValueError("INPUT_TOO_LONG")
