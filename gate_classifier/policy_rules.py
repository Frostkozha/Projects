"""Versioned semantic pattern rules (spec section 6).

Rules are additional signals for explicit first-person/current situations. They are not a
substitute for trained risk heads, and their absence never establishes that a request is safe.
A single keyword (``suicide``, ``exam``, ``patient``, ``ignore``, ``phone``) is never a hard block.
Quoted segments are removed before hard-rule matching so quoted hostile or sensitive text
submitted for explanation is not treated as the user's own intent.
"""

from __future__ import annotations

import re

from .normalize import casefold_view
from .schema import Mode, RuleFlags

RULES_VERSION = "gate-rules-0.2"

_RELATIVE = (
    r"(?:mother|father|mom|mum|dad|parents?|brother|sister|son|daughter|wife|husband|friend|grandmother|"
    r"grandfather|grandma|grandpa|granny|aunt|uncle|cousin|roommate|flatmate|neighbou?r|partner|boyfriend|"
    r"girlfriend|child|baby|kid|niece|nephew|colleague|classmate|relative|patient|baby's)"
)
_HEALTH = (
    r"(?:diagnos\w*|symptom\w*|pain|ache\w*|fever|rash|lump|cancer|tumou?r|biops\w*|(?:test |lab |blood )?results|"
    r"blood (?:test|pressure|sugar|work)|medicat\w*|medicine|pills?|tablets?|dos(?:e|age)\w*|treatment|infect\w*|disease|sick|"
    r"pregnan\w*|swell\w*|bleed\w*|cough\w*|vomit\w*|dizz\w*|faint\w*|antibiotic\w*|insulin|surgery|"
    r"diabet\w*|asthma|seizures?|injur\w*|wound|scan|x-ray|mri|ecg|prescri\w*|condition|disorder|mole|ulcer)"
)
_SENT = r"[^.?!\n]"

_EMERGENCY = [
    rf"\b(?:my|our) {_RELATIVE}\b{_SENT}{{0,40}}?\b(?:is|has|isn't|is not|just|has just|'s|keeps)\b{_SENT}{{0,20}}?"
    r"\b(?:not breathing|stopped breathing|can't breathe|cannot breathe|unconscious|unresponsive|collapsed|"
    r"passed out|having a seizure|seizing|choking|bleeding (?:heavily|a lot|badly|profusely)|having a heart attack|"
    r"having a stroke|turning blue|overdosed|taken an overdose|took an overdose)",
    r"\b(?:someone|somebody|a (?:man|woman|person|child|student|guy|girl)|he|she|they)\b (?:here |next to me |in front of me |near me )?"
    r"(?:is|has|just|'s|has just) (?:just )?(?:collapsed|stopped breathing|not breathing|unconscious|unresponsive|choking|"
    r"having a seizure|bleeding heavily|overdosed)",
    r"\bi(?:'m| am) (?:having|bleeding|choking|losing consciousness)\b[^.?!\n]{0,30}?"
    r"(?:heart attack|stroke|seizure|heavily|badly|right now|a lot|severe chest pain|can't breathe)",
    r"\bi (?:can't|cannot|can not) breathe\b",
    r"(?:^|[.!?]\s*)(?:please\s+)?call (?:an? )?ambulance\b",
    r"\b(?:i|we) need an ambulance\b",
]

_SELF_HARM = [
    r"\bi (?:want|wanna|am going|'m going|plan|intend|am planning|'m planning|have decided|'ve decided|need|feel like i want) to (?:kill|hurt|harm|cut) myself\b",
    r"\bi(?:'m| am) (?:going to|gonna) (?:kill myself|end (?:it|it all|my life)|take my (?:own )?life)\b",
    r"\bi (?:want|wanna) to (?:die|end my life|end it all)\b",
    r"\bi (?:don't|do not) want to (?:live|be alive) any ?more\b",
    r"\bi(?:'m| am| feel| have been feeling| keep feeling| have been) (?:really |very |so |kind of )?suicidal\b",
    r"\bi (?:have|'ve) (?:just )?(?:taken|swallowed) (?:all|a lot|too many|a whole|an overdose)",
    r"\bi(?:'m| am| have been| keep) (?:cutting|hurting|harming) myself\b",
    r"\bi(?:'m| am| keep| have been) (?:thinking|having thoughts) (?:about|of) (?:killing|hurting|harming) myself\b",
]

_REAL_PERSON = [
    rf"\bmy {_RELATIVE}(?:'s|\s+(?:has|had|have|is|was|got|gets|takes|took|keeps|just|been|became|developed|needs|"
    rf"started|feels|felt|complains|complained|woke|came|seems))\b{_SENT}{{0,60}}?\b{_HEALTH}\b",
    r"\bmy (?:own )?(?:lab|blood|test|biopsy|pathology|scan|x-ray|mri|ct|ultrasound|urine|pap smear|smear|thyroid|liver|kidney)"
    r"(?: test| panel| work)? (?:results?|report|findings|values|levels|numbers)\b",
    r"\b(?:interpret|explain|read|look at|check) my (?:\w+ ){0,2}(?:results?|report|biopsy|scan|ecg|labs|bloodwork)\b",
    r"\b(?:should|can|could|may) i (?:take|stop taking|increase|double|lower|give (?:him|her|my \w+)|skip)\b"
    rf"{_SENT}{{0,30}}?(?:mg|pill|tablet|medic|dose|antibiotic|drug|paracetamol|ibuprofen|insulin|aspirin|steroid)",
    r"\b(?:what|how much)(?: dose| dosage)?(?: of \w+)? (?:should|can) (?:i|my \w+|he|she) (?:take|use)\b",
    r"\b(?:what|how much)(?: dose| dosage)?(?: of \w+)? (?:should|can) i give (?:my \w+|him|her)\b",
    r"\bi (?:have|'ve|have been|'ve been|was|got|am|'m) (?:just )?(?:diagnosed|having|getting|experiencing|feeling)?"
    r" ?(?:a lump|a rash|a fever|chest pain|abdominal pain|bleeding|these symptoms|symptoms|pregnant)\b",
    r"\bdo i have\b[^.?!\n]{0,40}(?:cancer|disease|infection|syndrome|tumou?r|diabetes|carcinoma|condition|disorder)",
]

_PRIVATE = [
    r"\b(?:show|give|reveal|print|tell|display|send|share|dump|leak|repeat|output|list)\b (?:me |us )?(?:your|the) "
    r"(?:full |hidden |secret |internal |original |exact |complete )*(?:system prompt|system instructions?|hidden instructions?|"
    r"internal instructions?|developer (?:prompt|instructions?)|api keys?|passwords?|credentials|secrets?|access tokens?|"
    r"admin (?:password|credentials))\b",
    r"\b(?:what is|what's|what are) (?:your|the) (?:system prompt|hidden instructions|api keys?|admin password|credentials)\b",
    r"\b(?:show|give|send|get|see|access|read|view|download|list|export|open|find|pull up)\b[^.?!\n]{0,40}?"
    r"\b(?:another student'?s?|other students'?|a different student'?s?|someone else'?s|classmates?'?s?|"
    r"student [a-z]+'s|other users?'?s?|another user'?s?)\b[^.?!\n]{0,25}?"
    r"\b(?:logs?|records?|grades?|marks|data|history|chats?|conversations?|messages|scores|results|answers|sessions?)\b",
]

_OVERRIDE = [
    r"\b(?:ignore|disregard|forget|override|bypass|skip)\b (?:all |any |every |of )?(?:the |your |my |previous |prior |above |"
    r"earlier |preceding |these |those |existing |current |safety |source )*(?:instructions?|rules|guidelines|constraints|"
    r"restrictions|guardrails|policies|safety (?:rules|filters?|checks?)|filters?|system prompt|citation requirements?)\b",
    r"\banswer\b[^.?!\n]{0,30}\bwithout (?:using |citing |any |the )*(?:approved )?(?:sources|citations|evidence|references)\b",
    r"\byou are now\b[^.?!\n]{0,30}\b(?:unrestricted|jailbroken|dan|in developer mode|without (?:rules|restrictions))",
    r"\b(?:pretend|act as if|imagine) (?:that )?(?:you have no|there are no|you're not bound by|you are not bound by) (?:rules|restrictions|guidelines|limits)",
    r"\b(?:enable|enter|switch to|activate|turn on) (?:developer|god|admin|debug|jailbreak|unrestricted) mode\b",
    r"^\s*(?:system|developer|assistant)\s*(?:prompt)?\s*:",
]

_ASSESSED = [
    r"\b(?:do|complete|write|finish|solve|answer|take)\b (?:my|this|the|these|today's)\b[^.?!\n]{0,20}?"
    r"\b(?:graded|assessed|take-home|live|ongoing|summative|online)\b[^.?!\n]{0,15}?"
    r"\b(?:exam|test|quiz|assignment|homework|coursework|essay|paper|questions?|assessment)\b",
    r"\b(?:do|complete|write|finish) my (?:homework|assignment|coursework|essay|lab report|graded \w+)\b",
    r"\bi(?:'m| am) (?:currently |right now )?(?:in|taking|sitting|writing|doing) (?:an?|the|my) (?:\w+ )?"
    r"(?:exam|test|midterm|quiz|osce|assessment)\b[^.?!\n]{0,25}?\b(?:right now|now|at the moment|currently)\b",
    r"\bi(?:'m| am) (?:currently|right now) (?:in|taking|sitting|writing|doing) (?:an?|the|my) (?:\w+ )?"
    r"(?:exam|test|midterm|quiz|osce|assessment)\b",
    r"\b(?:answers?|answer key|solutions?) (?:to|for) (?:the |my |this |today's )?(?:live |current |ongoing |graded |"
    r"upcoming |final |real )?(?:exam|test|midterm|assessment)\b",
    r"\bfor (?:my|a|the) (?:graded|assessed|summative|take-home) (?:exam|test|quiz|assignment|homework|coursework|essay)\b",
]

_AMBIGUOUS_CLINICAL = [
    r"\b(?:a|the|this) (?:\d+[- ]year[- ]old )?patient\b[^?]{0,200}?\b(?:what (?:should|do|would) (?:i|we|you) "
    r"(?:give|prescribe|do|start|administer)|how much [\w ]{0,20}?(?:should|to) (?:give|administer)|which "
    r"(?:drug|medication|antibiotic|treatment) (?:should|to)|what dose|should (?:i|we) (?:give|start|prescribe|treat|administer))",
    r"\bwhat (?:dose|dosage)(?: of [\w-]+)? (?:should|to|do) (?:i |we |you )?(?:give|use|prescribe|administer)\b",
]

_UNSUPPORTED = [
    r"\b(?:analy[sz]e|look at|read|interpret|identify|describe|check|grade|mark|see)\b (?:this|my|the attached|the uploaded|"
    r"the following|attached|uploaded|the) (?:image|photo|picture|photograph|micrograph|slide image|scan|x-ray|video|"
    r"screenshot|file|pdf|histology image)s?\b(?! (?:of|in) (?:the|our) (?:lecture|atlas|textbook))",
    r"\b(?:i(?:'ve| have)? )?(?:attached|uploaded|uploading|attaching|sending you|here is|here's) (?:an? |the |this |my )?"
    r"(?:image|photo|picture|micrograph|video|screenshot|file|pdf)\b",
    r"\b(?:calculate|compute|work out) (?:the |a |an )?(?:dose|dosage|infusion rate|drip rate|creatinine clearance)\b",
    r"\b(?:simulate|run|role-?play|play out) (?:a |an )?(?:clinical )?(?:case|patient (?:case|encounter|scenario)|osce)\b",
    r"\b(?:grade|mark|score) my (?:essay|answer|answers|work|report|assignment|submission)\b",
]

_BARE_REPLY = re.compile(
    r"^(?:[a-e]|\(?[a-e]\)|option [a-e]|answer [a-e]|-?\d+(?:\.\d+)?|yes|no|yep|yeah|nope|i don't know|i do not know|"
    r"idk|not sure|continue|next|ok|okay|sure)[.!?]*$"
)
_MODE_SWITCH = [
    (re.compile(r"^(?:please\s+)?(?:switch|change|go|move)\s+(?:back\s+)?to\s+(answer|tutor|quiz)\s+mode\b"), None),
    (re.compile(r"^(?:please\s+)?(?:use|start|enter)\s+(answer|tutor|quiz)\s+mode\b"), None),
    (re.compile(r"^(?:please\s+|can you\s+|could you\s+)?quiz me\b"), "quiz"),
    (re.compile(r"^(?:please\s+|can you\s+|could you\s+)?tutor me\b"), "tutor"),
]
_QUOTED = re.compile(r'"[^"\n]{0,400}"|«[^»\n]{0,400}»|`[^`\n]{0,400}`')


def _compile(patterns):
    return [re.compile(p) for p in patterns]


_COMPILED = {
    "imminent_emergency": _compile(_EMERGENCY),
    "self_harm_crisis": _compile(_SELF_HARM),
    "real_person_advice": _compile(_REAL_PERSON),
    "private_data_request": _compile(_PRIVATE),
    "instruction_override": _compile(_OVERRIDE),
    "assessed_work": _compile(_ASSESSED),
    "ambiguous_clinical": _compile(_AMBIGUOUS_CLINICAL),
    "unsupported_modality": _compile(_UNSUPPORTED),
}


def strip_quoted(view: str) -> str:
    return _QUOTED.sub(" [quoted] ", view)


class RuleEngine:
    version = RULES_VERSION

    def evaluate(self, normalized_text: str) -> RuleFlags:
        """Evaluate on the normalized, pre-redaction text. Returns categories only."""
        view = casefold_view(normalized_text)
        unquoted = strip_quoted(view)
        hits = {name: any(p.search(unquoted) for p in pats) for name, pats in _COMPILED.items()}
        # an explicit real-person signal removes the "ambiguous" status: it is real-person advice
        if hits["real_person_advice"]:
            hits["ambiguous_clinical"] = False
        switch = None
        stripped = view.strip()
        for pat, fixed in _MODE_SWITCH:
            m = pat.search(stripped)
            if m:
                switch = Mode(fixed or m.group(1))
                break
        bare = bool(_BARE_REPLY.match(stripped))
        return RuleFlags(**hits, bare_reply=bare, text_mode_switch=switch, rules_version=self.version)
