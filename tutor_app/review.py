"""Deterministic review primitives (Integration plan v0.3, section 7 and appendix D).

A bounded heuristic, not FSRS: incorrect -> due in 10 minutes and streak reset; correct streaks
1, 2, 3+ -> due in 1, 3, 7 days. The practice indicator is a descriptive study indicator, not a grade.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def next_review(now: datetime, *, correct: bool, previous_streak: int) -> tuple[int, datetime]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("aware_authorization_time_required")
    if type(correct) is not bool or type(previous_streak) is not int or previous_streak < 0:
        raise ValueError("invalid_review_state")
    now = now.astimezone(timezone.utc)
    if not correct:
        return 0, now + timedelta(minutes=10)
    streak = min(previous_streak + 1, 3)
    return streak, now + timedelta(days={1: 1, 2: 3, 3: 7}[streak])


def practice_indicator(*, correct: int, attempts: int) -> float:
    if type(correct) is not int or type(attempts) is not int or not 0 <= correct <= attempts:
        raise ValueError("invalid_counts")
    return (1 + correct) / (2 + attempts)
