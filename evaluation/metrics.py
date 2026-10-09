"""Interval metrics for verifier reports (Verifier spec v0.2, section 10).

Point estimates are descriptive; related injected cases are not independent trials, so a group-aware
bootstrap is reported as a companion. Software tests passing is not an evaluation result.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval (95% by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def zero_miss_upper_bound(n: int, alpha: float = 0.05) -> float:
    """One-sided exact upper bound on the miss rate after zero misses in n independent items."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - alpha ** (1.0 / n)


def unsupported_recall(rejected_flags: Sequence[bool]) -> float:
    """Share of labeled partial/unsupported/contradictory sentences that were rejected."""
    if not rejected_flags:
        raise ValueError("no labeled unsupported sentences")
    return sum(rejected_flags) / len(rejected_flags)


def group_bootstrap(groups: dict[str, list[int]], iterations: int = 2000, seed: int = 7,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap of a proportion resampling whole groups (lists of 0/1 outcomes)."""
    keys = sorted(groups)
    if not keys:
        raise ValueError("no groups")
    rng = random.Random(seed)
    stats = []
    for _ in range(iterations):
        sample = [groups[rng.choice(keys)] for _ in keys]
        total = sum(len(g) for g in sample)
        stats.append(sum(sum(g) for g in sample) / total if total else 0.0)
    stats.sort()
    lo = stats[int(alpha / 2 * iterations)]
    hi = stats[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return lo, hi
