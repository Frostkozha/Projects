"""Layer 2: relational hard-fact constraints (Verifier spec v0.2, section 5.3).

Compares local relationships, not bags of tokens: each quantity is bound to its entity anchors within the
same clause, units must match exactly (reviewed conversions only), comparators/ranges must match, and
negation, exceptions, quantifiers, causal/comparative claims and antonym/laterality terms are checked
against the best-aligned premise clause. A clear match only allows NLI to run; it never approves alone.
High-risk facts whose scope cannot be resolved reject with HARD_FACT_UNRESOLVED.

Limitation (recorded per sentence): pattern coverage is deterministic and narrow; ordinary prose without a
supported pattern continues to NLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

RULES_VERSION = "hard-facts-0.2"

_STOP = set("""a an the of in on at to for from by with and or but is are was were be been being has have had do does
did it its this that these those as into onto than then there their they them which who whom whose what when where
why how also such can may will would should could very more most less least each every all some any no not""".split())
_COMPARATORS = [
    ("less than or equal to", "<="), ("greater than or equal to", ">="), ("no more than", "<="), ("at most", "<="),
    ("up to", "<="), ("no less than", ">="), ("at least", ">="), ("less than", "<"), ("fewer than", "<"),
    ("under", "<"), ("below", "<"), ("more than", ">"), ("greater than", ">"), ("over", ">"), ("above", ">"),
    ("approximately", "~"), ("about", "~"), ("around", "~"), ("roughly", "~"), ("nearly", "~"),
    ("<=", "<="), (">=", ">="), ("≤", "<="), ("≥", ">="), ("<", "<"), (">", ">"), ("~", "~"),
]
_CLAUSE_SPLIT = re.compile(r"[,;:()]|\b(?:and|or|but|whereas|while|which|although)\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+(?:\.\d+)?", re.UNICODE)


@dataclass(frozen=True)
class Quantity:
    low: Decimal
    high: Decimal
    comparator: str  # "=", "<", "<=", ">", ">=", "~", "range"
    unit: Optional[str]
    start: int
    end: int

    def key(self):
        return (self.low, self.high, self.comparator, self.unit)


@dataclass
class HardFactResult:
    status: str  # pass | fail | unresolved | not_applicable
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Cues:
    number: bool
    unit: bool
    negation: bool
    exception: bool
    condition: bool
    laterality: bool
    high_risk_term: bool
    sequence: bool
    comparative: bool
    causal: bool
    question: bool

    @property
    def high_risk(self) -> bool:
        return any((self.number, self.unit, self.negation, self.exception, self.laterality, self.high_risk_term,
                    self.sequence))


def _norm_token(t: str) -> str:
    t = t.lower()
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def content_tokens(text: str, rules) -> set[str]:
    units = {u.lower() for u in rules.units.get("aliases", {})}
    numwords = set(rules.cues.get("number_words", {}))
    out = set()
    for raw in _WORD.findall(text):
        if raw[0].isdigit():
            continue
        if len(raw) == 1 and raw.isupper():
            out.add(raw)  # single-letter entity labels such as "drug A"
            continue
        low = raw.lower()
        if low in _STOP or low in units or low in numwords or len(low) < 2:
            continue
        out.add(_norm_token(low))
    return out


def _phrase_in(phrase: str, text_low: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text_low) is not None


class HardFactChecker:
    version = RULES_VERSION

    def __init__(self, rules):
        self.rules = rules
        aliases = rules.units.get("aliases", {})
        self._units = sorted(aliases.items(), key=lambda kv: -len(kv[0]))
        self._numwords = rules.cues.get("number_words", {})
        num = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        words = "|".join(sorted((re.escape(w) for w in self._numwords), key=len, reverse=True))
        comp = "|".join(re.escape(p) for p, _ in _COMPARATORS)
        unit = "|".join(re.escape(a) for a, _ in self._units)
        self._qty = re.compile(
            rf"(?:(?P<comp>{comp})\s*)?(?<![\w.])(?P<a>{num}|{words})(?![\w.])"
            rf"(?:\s*(?:-|–|to)\s*(?P<b>{num}))?(?:\s*(?P<unit>{unit})(?![\w/]))?",
            re.IGNORECASE)
        self._antonyms = [tuple(p) for p in rules.antonyms]
        self._laterality = {"left", "right", "anterior", "posterior", "superior", "inferior", "medial", "lateral",
                            "proximal", "distal", "afferent", "efferent", "dorsal", "ventral", "apical", "basal"}

    # ------------------------------------------------------------------ extraction

    def _decimal(self, s: str) -> Decimal:
        s = s.lower()
        if s in self._numwords:
            return Decimal(self._numwords[s])
        try:
            return Decimal(s.replace(",", ""))
        except InvalidOperation:
            raise ValueError("bad number") from None

    def quantities(self, text: str) -> list[Quantity]:
        out = []
        for m in self._qty.finditer(text):
            a = m.group("a")
            if a.lower() in self._numwords and not m.group("unit"):
                continue  # bare number words ("one of the...") are not quantities without a unit
            comp_raw = (m.group("comp") or "").lower()
            comp = next((c for p, c in _COMPARATORS if p == comp_raw), "=") if comp_raw else "="
            low = self._decimal(a)
            high = low
            if m.group("b"):
                high = self._decimal(m.group("b"))
                comp = "range" if comp == "=" else comp + "range"
            unit = None
            if m.group("unit"):
                raw = m.group("unit")
                unit = next((c for alias, c in self._units if alias.lower() == raw.lower()), raw.lower())
            out.append(Quantity(low, high, comp, unit, m.start(), m.end()))
        return out

    def cues(self, text: str) -> Cues:
        low = text.lower()
        c = self.rules.cues
        qs = self.quantities(text)
        anton_terms = {t for pair in self._antonyms for t in pair}
        words = set(re.findall(r"[a-z']+", low))
        return Cues(
            number=any(True for _ in re.finditer(r"\d", text)) or any(q.unit for q in qs),
            unit=any(q.unit for q in qs),
            negation=any(_phrase_in(n, low) for n in c["negation"]) or "n't" in low,
            exception=any(_phrase_in(e, low) for e in c["exception"]),
            condition=any(_phrase_in(e, low) for e in c["condition"]),
            laterality=bool(words & self._laterality) or bool(words & anton_terms & {"increase", "decrease"}),
            high_risk_term=any(_phrase_in(t, low) for t in c["high_risk_terms"]),
            sequence=any(_phrase_in(t, low) for t in c["sequence"]),
            comparative=any(_phrase_in(t, low) for t in c["comparative"]),
            causal=any(_phrase_in(t, low) for t in c["causal"]),
            question=text.rstrip().endswith("?"),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def clauses(text: str) -> list[tuple[int, int]]:
        spans, start = [], 0
        for m in _CLAUSE_SPLIT.finditer(text):
            if text[start:m.start()].strip():
                spans.append((start, m.start()))
            start = m.end()
        if text[start:].strip():
            spans.append((start, len(text)))
        return spans

    def _negated(self, text: str) -> bool:
        low = text.lower()
        return any(_phrase_in(n, low) for n in self.rules.cues["negation"]) or "n't" in low

    def _not_all_none(self, text: str) -> Optional[str]:
        low = text.lower()
        if re.search(r"\bnot (?:all|every|always|each)\b", low):
            return "not_all"
        if re.search(r"\b(?:none|never|no)\b", low):
            return "none"
        return None

    def _strength(self, text: str) -> int:
        low = text.lower()
        scale = self.rules.cues["quantifier_scale"]
        return max([v for w, v in scale.items() if _phrase_in(w, low)] or [0])

    def _best_set(self, tokens: set[str], candidates: list[str]) -> tuple[list[int], float]:
        """All candidates tied for the best overlap (polarity must match at least one of them)."""
        scores = [(len(tokens & content_tokens(c, self.rules)) / len(tokens)) if tokens else 0.0 for c in candidates]
        if not scores:
            return [], 0.0
        best = max(scores)
        return [i for i, v in enumerate(scores) if v == best], best

    def _best(self, tokens: set[str], candidates: list[str]) -> tuple[int, float]:
        best_i, best = -1, -1.0
        for i, c in enumerate(candidates):
            ct = content_tokens(c, self.rules)
            if not tokens:
                score = 0.0
            else:
                score = len(tokens & ct) / len(tokens)
            if score > best or (score == best and best_i >= 0 and len(c) < len(candidates[best_i])):
                best_i, best = i, score
        return best_i, best

    # ------------------------------------------------------------------ main

    def check(self, hypothesis: str, premise: str) -> HardFactResult:
        reasons_fail: list[str] = []
        reasons_unres: list[str] = []
        cues = self.cues(hypothesis)
        p_low = premise.lower()
        p_sentences = [s for s in _SENT_SPLIT.split(premise) if s.strip()]
        p_clauses = [premise[a:b] for a, b in self.clauses(premise)]
        p_tokens = content_tokens(premise, self.rules)
        applicable = False

        # quantities bound to entities in the same clause
        h_qty = self.quantities(hypothesis)
        p_qty = self.quantities(premise)
        h_clause_spans = self.clauses(hypothesis)
        for q in h_qty:
            applicable = True
            equal = [pq for pq in p_qty if pq.key() == q.key()]
            if not equal:
                reasons_fail.append("quantity_mismatch")
                continue
            span = next(((a, b) for a, b in h_clause_spans if a <= q.start < b), (0, len(hypothesis)))
            anchors = content_tokens(hypothesis[span[0]:span[1]], self.rules) & p_tokens
            # discriminative anchors: tokens the premise binds to a *different* value of the same unit
            p_spans = self.clauses(premise)
            discriminative = set()
            for pq in p_qty:
                if pq.unit == q.unit and pq.key() != q.key():
                    for a, b in p_spans:
                        if a <= pq.start < b:
                            discriminative |= anchors & content_tokens(premise[a:b], self.rules)
            h_all = content_tokens(hypothesis, self.rules)
            bound = False
            for pq in equal:
                for a, b in p_spans:
                    if not a <= pq.start < b:
                        continue
                    e_tok = content_tokens(premise[a:b], self.rules)
                    foreign = e_tok - h_all          # entity named with this value but not in the sentence
                    missing = discriminative - e_tok  # sentence entity that the premise binds elsewhere
                    if not (foreign and missing):
                        bound = True
            if not bound:
                reasons_fail.append("entity_binding")

        # clause-level polarity, antonyms, quantifiers
        for a, b in h_clause_spans:
            h_clause = hypothesis[a:b]
            h_tok = content_tokens(h_clause, self.rules)
            if not h_tok:
                continue
            tied, overlap = self._best_set(h_tok, p_clauses)
            if not tied:
                continue
            # among tied clauses, judge against the one whose polarity matches (a flip must hold for all)
            h_neg_c = self._negated(h_clause)
            matching = [i for i in tied if self._negated(p_clauses[i]) == h_neg_c]
            p_clause = p_clauses[(matching or tied)[0]]
            sent_idx, _ = self._best(h_tok, p_sentences)
            p_sent = p_sentences[sent_idx] if sent_idx >= 0 else premise
            h_neg, p_neg = self._negated(h_clause), self._negated(p_clause)
            if h_neg or p_neg:
                applicable = True
                if h_neg != p_neg:
                    reasons_fail.append("negation_added" if h_neg else "negation_missing")
                na_h, na_p = self._not_all_none(h_clause), self._not_all_none(p_clause)
                if na_h and na_p and na_h != na_p:
                    reasons_fail.append("not_all_vs_none")
            h_words = set(re.findall(r"[a-z]+", h_clause.lower()))
            pc_words = set(re.findall(r"[a-z]+", p_clause.lower()))
            all_p_words = set(re.findall(r"[a-z]+", p_low))
            for x, y in self._antonyms:
                for term, other in ((x, y), (y, x)):
                    if term in h_words:
                        applicable = True
                        if other in pc_words and term not in pc_words:
                            reasons_fail.append("antonym_flip")
                        elif term in self._laterality and term not in all_p_words:
                            reasons_unres.append("laterality_unresolved")
            hs, ps = self._strength(h_clause), self._strength(p_clause)
            if hs:
                applicable = True
                if ps and hs > ps:
                    reasons_fail.append("quantifier_strengthened")
                elif not ps and hs >= 3:
                    reasons_unres.append("unsupported_universal")
            # exceptions and conditions are judged against the whole aligned premise sentence
            p_sent_low = p_sent.lower()
            h_low = hypothesis.lower()
            p_exc = [e for e in self.rules.cues["exception"] if _phrase_in(e, p_sent_low)]
            h_exc = [e for e in self.rules.cues["exception"] if _phrase_in(e, h_low)]
            if p_exc or h_exc:
                applicable = True
                if p_exc and not h_exc:
                    reasons_fail.append("exception_dropped")
                elif h_exc and not set(h_exc) <= set(p_exc):
                    reasons_fail.append("exception_added")
            p_cond = [e for e in self.rules.cues["condition"] if _phrase_in(e, p_sent_low)]
            h_cond = [e for e in self.rules.cues["condition"] if _phrase_in(e, h_low)]
            if p_cond and not h_cond and cues.high_risk:
                applicable = True
                reasons_fail.append("condition_dropped")
            if cues.causal and not any(_phrase_in(t, p_sent_low) for t in self.rules.cues["causal"]):
                applicable = True
                reasons_unres.append("unsupported_causal")
            if cues.comparative and not any(_phrase_in(t, p_sent_low) for t in self.rules.cues["comparative"]):
                applicable = True
                reasons_unres.append("unsupported_comparative")
            if cues.high_risk and overlap < 0.5:
                applicable = True
                reasons_unres.append("low_alignment")

        if reasons_fail:
            return HardFactResult("fail", sorted(set(reasons_fail)))
        if reasons_unres:
            return HardFactResult("unresolved", sorted(set(reasons_unres)))
        return HardFactResult("pass" if applicable else "not_applicable", [])
