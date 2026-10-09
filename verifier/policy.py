"""Layer 0b: contextual output policy (Verifier spec v0.2, section 5.1).

Inspects the whole redacted request, trusted teaching context and draft before evidence checking.
Regex is one layer, not a complete safety classifier. The gate's own rule engine is reused for the
*request* context; the draft is never fed to the input gate as if it were a new student request.
A generated fictional emergency in the draft never creates a welfare alert: A7 is re-evaluated only
against the current request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from gate_classifier.policy_rules import RuleEngine
from gate_classifier.privacy import Redactor

POLICY_VERSION = "verifier-output-policy-0.2"

_ADVICE = [
    r"\byou should (?:take|stop taking|start taking|increase|decrease|double|skip|use|apply|inject|get tested"
    r"|see a doctor)\b",
    r"\b(?:i|we) (?:recommend|suggest|advise) (?:that )?you\b",
    r"\byour (?:results?|test results?|biopsy|scan|diagnosis|symptoms?|dose|dosage|medication"
    r"|blood (?:test|pressure|sugar))\b",
    r"\b(?:take|give him|give her|give them) \d+(?:\.\d+)? ?(?:mg|mcg|µg|g|ml|units?)\b",
    r"\b(?:he|she|they|your (?:mother|father|friend|patient)) should (?:take|stop|start|be given)\b",
]
_UNCERTAIN = [
    r"\bif you (?:have|feel|notice|experience|are experiencing)\b",
    r"\byour (?:body|heart|lungs?|skin|pain|condition|patient)\b",
    r"\bin your case\b",
]
_PRIVACY = [
    r"\b(?:another|other) students?'?s? (?:records?|grades?|answers?|logs?|data)\b",
    r"\b(?:student|patient) (?:id|number|record)\b",
]
_OVERRIDE = [
    r"\b(?:ignore|disregard) (?:all |any |the |previous |prior )*(?:instructions|rules|sources)\b",
    r"\bwithout (?:using |citing )?(?:the )?(?:approved )?sources\b",
    r"\b(?:developer|jailbreak|unrestricted) mode\b",
    r"\bi (?:have no|am not bound by) (?:rules|restrictions)\b",
]
_INTERNAL = [
    r"\b(?:hidden|expected) answer\b", r"\bsystem prompt\b", r"\binternal (?:note|instruction|reasoning)s?\b",
    r"<\s*/?\s*think", r"\breasoning\s*:", r"^\s*(?:system|assistant|developer)\s*:", r"\bpassage[_ ]id\b",
    r"\bpsg-[0-9a-f]{8,}\b",
]
_ASSESSED = [r"\bthe answer to (?:question|q) ?\d+ (?:of|on|in) (?:the|your) (?:exam|test|assignment)\b",
             r"\bhere (?:is|are) (?:the )?(?:answers?|solutions?) (?:to|for) your (?:exam|test|assignment|homework)\b"]


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class PolicyUnavailable(RuntimeError):
    pass


@dataclass
class PolicyOutcome:
    violations: list[str] = field(default_factory=list)
    uncertain: bool = False
    crisis: Optional[str] = None  # "emergency" | "self_harm" from the current request only

    @property
    def clean(self) -> bool:
        return not self.violations and not self.uncertain and self.crisis is None


class OutputPolicyAdapter:
    version = POLICY_VERSION

    def __init__(self, rules: Optional[RuleEngine] = None, redactor: Optional[Redactor] = None):
        self.rules = rules or RuleEngine()
        self.redactor = redactor or Redactor()

    def evaluate(self, redacted_request: str, draft_texts: list[str], real_person_context: bool = False,
                 internal_texts: tuple[str, ...] = ()) -> PolicyOutcome:
        """``draft_texts`` are student-visible sentences; ``internal_texts`` are hidden teaching sentences.

        Hidden sentences get every check except internal-content leakage, which concerns visible text only.
        """
        try:
            req = self.rules.evaluate(redacted_request) if redacted_request.strip() else None
            out = PolicyOutcome()
            if req is not None and req.imminent_emergency:
                out.crisis = "emergency"
            elif req is not None and req.self_harm_crisis:
                out.crisis = "self_harm"
            all_texts = list(draft_texts) + list(internal_texts)
            joined = " ".join(all_texts)
            real_person = real_person_context or (req is not None and req.real_person_advice)
            if _any(_ADVICE, joined) or (real_person and _any(_UNCERTAIN, joined)):
                out.violations.append("real_person_advice")
            identifiers = {c for t in all_texts for c in self.redactor.redact(t).categories}
            if identifiers & {"EMAIL", "PHONE", "ID", "ADDRESS"} or _any(_PRIVACY, joined):
                out.violations.append("privacy_disclosure")
            if _any(_ASSESSED, joined) or (req is not None and req.assessed_work):
                out.violations.append("assessed_work")
            if _any(_OVERRIDE, joined) or (req is not None and req.instruction_override):
                out.violations.append("instruction_override")
            if any(_any(_INTERNAL, t) for t in draft_texts):
                out.violations.append("internal_content")
            if not out.violations and _any(_UNCERTAIN, joined):
                out.uncertain = True
            return out
        except Exception:
            raise PolicyUnavailable("output policy adapter failed") from None
