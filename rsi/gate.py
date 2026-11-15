"""Smoke and development promotion gates and the Wilson interval used in reports."""

from __future__ import annotations

# Shared promotion gates: smoke screens pass on >= SMOKE_GATE/10,
# development promotions require >= DEV_GATE/100.
SCREEN_PASS: int = 8
SMOKE_GATE: int = SCREEN_PASS
DEV_GATE: int = 77


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    """Wilson 95% score interval for a binomial success count."""
    if trials <= 0:
        return [0.0, 1.0]
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * ((p * (1 - p) + z * z / (4 * trials)) / trials) ** 0.5 / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]
