"""Pure ordered routing policy (spec section 8).

``decide`` takes validated inputs and returns a GateDecision. It performs no I/O, model calls,
logging, randomness, clock or environment reads. The first applicable row wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import GateConfig
from .schema import (
    RISK_LABELS,
    DecisionFlags,
    GateContext,
    GateDecision,
    InferenceStatus,
    Mode,
    Prediction,
    Reason,
    ResponseCode,
    RetrievalPlan,
    Route,
    RuleFlags,
    SessionUpdate,
    Versions,
)
from .session import SessionContextView, resolve_mode

# rows 5-8: (risk, reason, reply_key)
_REFUSALS = (
    ("real_person_advice", Reason.REAL_PERSON_REQUEST, "real_person"),
    ("private_data_request", Reason.PRIVATE_DATA_REQUEST, "private_data"),
    ("instruction_override", Reason.INSTRUCTION_OVERRIDE, "instruction_override"),
    ("assessed_work", Reason.ASSESSED_WORK, "assessed_work"),
)


@dataclass(frozen=True)
class DecisionInputs:
    request_id: str
    redacted_text: Optional[str]
    rules: Optional[RuleFlags]
    prediction: Optional[Prediction]
    inference_status: InferenceStatus
    context: GateContext
    session_view: SessionContextView
    requested_mode: Optional[Mode]
    flags: DecisionFlags
    versions: Versions
    # "preprocess": normalization/privacy/rules/tokenizer failure (never bypassed)
    # "inference": mandatory model failure, deadline or invalid output
    failure: Optional[str] = None


def _blocked(i: DecisionInputs, route: Route, reason: Reason, code: Optional[ResponseCode], reply_key: str,
             prediction: Optional[Prediction] = None, status: Optional[InferenceStatus] = None) -> GateDecision:
    return GateDecision(
        request_id=i.request_id,
        route=route,
        reason=reason,
        response_code=code,
        reply_key=reply_key,
        effective_mode=None,
        mode_source=None,
        retrieval_plan=None,
        prediction=prediction,
        flags=i.flags,
        inference_status=status or i.inference_status,
        versions=i.versions,
    )


def _unavailable(i: DecisionInputs, reason: Reason, reply_key: str) -> GateDecision:
    status = i.inference_status
    if status == InferenceStatus.completed:
        status = InferenceStatus.failed
    return _blocked(i, Route.unavailable, reason, None, reply_key, prediction=None, status=status)


def preferred_libraries(prediction: Optional[Prediction], allowed: tuple[str, ...], cfg: GateConfig) -> tuple[str, ...]:
    if prediction is None:
        return ()
    scored = [
        (lib, s) for lib, s in prediction.libraries.items()
        if lib in allowed and s is not None and s >= cfg.thresholds.library_preference
    ]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return tuple(lib for lib, _ in scored[: cfg.thresholds.max_preferred_libraries])


def decide(i: DecisionInputs, cfg: GateConfig) -> GateDecision:
    th = cfg.thresholds
    rules = i.rules or RuleFlags()
    p = i.prediction
    ok_status = i.inference_status in (InferenceStatus.completed, InferenceStatus.skipped_rule)

    # 1. deployment state
    if i.context.deployment_state == "exam_shutdown":
        return _unavailable(i, Reason.EXAM_DISABLED, "exam_disabled")
    if i.context.deployment_state != "active":
        return _unavailable(i, Reason.MAINTENANCE, "maintenance")

    # 2. failures. Exception: an explicit emergency/crisis hard rule that was successfully
    #    evaluated (privacy and rules succeeded) survives a later model failure.
    crisis_hard_ok = i.failure != "preprocess" and i.rules is not None and rules.crisis_hard
    if (i.failure is not None or not ok_status) and not crisis_hard_ok:
        return _unavailable(i, Reason.GATE_FAILURE, "service_unavailable")
    if i.inference_status == InferenceStatus.completed and p is None and not crisis_hard_ok:
        return _unavailable(i, Reason.GATE_FAILURE, "service_unavailable")

    pred_out = p if i.inference_status == InferenceStatus.completed else None

    def risk(name: str) -> float:
        return p.risks[name] if p is not None else 0.0

    def blocks(name: str) -> bool:
        return getattr(rules, name) or risk(name) >= th.risks[name].block

    # 3-4. escalation
    if blocks("imminent_emergency"):
        return _blocked(i, Route.escalate, Reason.EMERGENCY, ResponseCode.A7, "emergency", pred_out)
    if blocks("self_harm_crisis"):
        return _blocked(i, Route.escalate, Reason.SELF_HARM, ResponseCode.A7, "self_harm", pred_out)

    # A failure that survived only because of the crisis exception may not go further.
    if i.failure is not None:
        return _unavailable(i, Reason.GATE_FAILURE, "service_unavailable")

    # 5-8. refusals
    for name, reason, key in _REFUSALS:
        if blocks(name):
            return _blocked(i, Route.reply, reason, ResponseCode.A6, key, pred_out)

    # Below this point a model prediction is mandatory (hard rules never reach here).
    if p is None or i.inference_status != InferenceStatus.completed:
        return _unavailable(i, Reason.GATE_FAILURE, "service_unavailable")

    # 9. safety uncertainty
    if rules.ambiguous_clinical or any(p.risks[r] >= th.risks[r].uncertainty for r in RISK_LABELS):
        return _blocked(i, Route.clarify, Reason.SAFETY_UNCERTAIN, ResponseCode.A3, "safety_clarify", p)

    # 10. unsupported modality/feature
    if rules.unsupported_modality:
        return _blocked(i, Route.reply, Reason.UNSUPPORTED_MODALITY, ResponseCode.A5, "unsupported_feature", p)

    topic = p.topic_scope
    course_ok = topic["course_related"] >= th.topic
    # 11-12. outside course / nonmedical
    if topic["outside_course"] >= th.topic and not course_ok:
        return _blocked(i, Route.reply, Reason.OUTSIDE_COURSE, ResponseCode.A5, "outside_course", p)
    if topic["nonmedical"] >= th.topic and not course_ok:
        return _blocked(i, Route.reply, Reason.NONMEDICAL, ResponseCode.A5, "nonmedical", p)

    # 13. ambiguous / weak topic / bare reply without valid pending context
    bare_without_context = rules.bare_reply and not i.session_view.valid_pending
    if bare_without_context or not course_ok:
        return _blocked(i, Route.clarify, Reason.AMBIGUOUS_REQUEST if bare_without_context else Reason.LOW_CONFIDENCE,
                        ResponseCode.A3, "clarify_topic", p)

    # 14. no authorized active sources
    allowed = tuple(sorted(i.context.authorized_active_libraries))
    if not allowed:
        return _blocked(i, Route.reply, Reason.NO_ACTIVE_SOURCES, ResponseCode.A5, "no_source", p)

    # 15. retrieve across every authorized active library; predictions are hints only
    mode, source = resolve_mode(i.requested_mode, rules, i.session_view, p, th.mode)
    plan = RetrievalPlan(
        allowed_libraries=allowed,
        preferred_libraries=preferred_libraries(p, allowed, cfg),
        query_text=i.redacted_text or "",
        strategy="all_active",
    )
    return GateDecision(
        request_id=i.request_id,
        route=Route.retrieve,
        reason=Reason.COURSE_REQUEST,
        response_code=None,
        reply_key=None,
        effective_mode=mode,
        mode_source=source,
        retrieval_plan=plan,
        prediction=p,
        flags=i.flags,
        inference_status=i.inference_status,
        versions=i.versions,
    )


def propose_session_update(decision: GateDecision, ctx: GateContext) -> Optional[SessionUpdate]:
    """Gate proposals: close on A7, pause on A6, otherwise preserve pending state."""
    s = ctx.session
    if s is None:
        return None
    if decision.route == Route.escalate:
        op = "close"
    elif decision.route == Route.reply and decision.response_code == ResponseCode.A6:
        op = "pause"
    else:
        return None
    return SessionUpdate(session_id=s.session_id, expected_revision=s.revision, operation=op)
