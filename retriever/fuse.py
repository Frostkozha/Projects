"""Deterministic reciprocal rank fusion (Retriever spec v0.2, section 10.2)."""

from __future__ import annotations


def rrf_ids(branches, constant=60, limit=30):
    """Reference implementation from the spec: dedupe each branch, rank from 1, sort by score then ID."""
    if constant <= 0 or limit <= 0:
        raise ValueError("Invalid RRF configuration")
    scores = {}
    for branch in branches:
        unique = list(dict.fromkeys(branch))
        for rank, passage_id in enumerate(unique, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda passage_id: (-scores[passage_id], passage_id))
    return [(passage_id, scores[passage_id]) for passage_id in ordered[:limit]]
