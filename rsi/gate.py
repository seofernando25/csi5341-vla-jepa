"""Sequential multi-fidelity smoke gate with exact binomial operating characteristics.

The legacy single-screen gate (pass on >= 8/10 smoke trials) kills a good branch
(true p = 0.87) with probability 0.1308. This module adds two stages around it:

- C0: a 6-trial fast-reject probe on hard LIBERO-Spatial tasks 0, 5 and 8, where
  B16 development failures cluster (4/10, 5/10, 3/10 vs 10/10 elsewhere in
  ``studies/evaluation/development/summary.json``). Two trials per hard task;
  reject the branch on <= 3/6 without spending a full screen.
- Near-miss band: exactly 7/10 on the standard screen triggers a disjoint
  confirm screen (default 10 trials, pass on >= 8) instead of instant rejection.

All rates are exact binomial sums over ``math.comb``; no simulation. Phase
separation: screen and confirm seed sets must be disjoint (checked by
:func:`assert_disjoint_seeds`), and neither may be pooled into dev-100 counts;
reserved-500 trials never gate selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import comb

HARD_TASKS: tuple[int, ...] = (0, 5, 8)
C0_TRIALS: int = 6
C0_REJECT_MAX: int = 3
SCREEN_TRIALS: int = 10
SCREEN_PASS: int = 8
NEAR_MISS_SCORE: int = 7
CONFIRM_TRIALS: int = 10
CONFIRM_PASS: int = 8
REFERENCE_P: float = 0.87


def _check_prob(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    return p


def _check_trials(n: int, k: int) -> None:
    if n < 1:
        raise ValueError(f"trials n must be >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"threshold k must satisfy 0 <= k <= n, got k={k}, n={n}")


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Exact P(X == k) for X ~ Binomial(n, p)."""
    _check_trials(n, k)
    _check_prob(p)
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def prob_at_most(n: int, k: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p)."""
    _check_trials(n, k)
    _check_prob(p)
    return sum(binomial_pmf(n, i, p) for i in range(k + 1))


def prob_at_least(n: int, k: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p)."""
    _check_trials(n, k)
    _check_prob(p)
    return sum(binomial_pmf(n, i, p) for i in range(k, n + 1))


def c0_decision(successes: int) -> str:
    """Fast-reject verdict after the C0 hard-task probe: ``reject`` or ``continue``.

    C0 never passes a branch; it only rejects (<= 3/6) or lets it reach screening.
    """
    if not 0 <= successes <= C0_TRIALS:
        raise ValueError(f"C0 successes must be in [0, {C0_TRIALS}], got {successes}")
    return "reject" if successes <= C0_REJECT_MAX else "continue"


def screen_decision(successes: int) -> str:
    """Standard screen verdict: ``pass`` (>= 8/10), ``confirm`` (7/10), ``reject``."""
    if not 0 <= successes <= SCREEN_TRIALS:
        raise ValueError(f"screen successes must be in [0, {SCREEN_TRIALS}], got {successes}")
    if successes >= SCREEN_PASS:
        return "pass"
    if successes == NEAR_MISS_SCORE:
        return "confirm"
    return "reject"


def confirm_decision(successes: int) -> str:
    """Disjoint confirm-screen verdict: ``pass`` (>= 8/10) or ``reject``."""
    if not 0 <= successes <= CONFIRM_TRIALS:
        raise ValueError(f"confirm successes must be in [0, {CONFIRM_TRIALS}], got {successes}")
    return "pass" if successes >= CONFIRM_PASS else "reject"


def assert_disjoint_seeds(*labeled: tuple[str, Iterable[int]]) -> None:
    """Raise ValueError if any trial seed appears in more than one phase.

    Pass ``(phase_name, seeds)`` pairs, e.g. ``("screen", s1, ...)``. Screen and
    confirm seeds must be disjoint, and neither may overlap dev-100 seeds, so no
    pooled count can leak selection trials into development evidence.
    """
    seen: dict[int, str] = {}
    for label, seeds in labeled:
        for s in seeds:
            if s in seen:
                raise ValueError(
                    f"seed {s} appears in both {seen[s]!r} and {label!r}: "
                    "phases must stay disjoint and never pool into dev-100"
                )
            seen[s] = label


def sequential_oc(
    true_p: float,
    n_screen: int = SCREEN_TRIALS,
    k_screen: int = SCREEN_PASS,
    near_miss: Sequence[int] = (NEAR_MISS_SCORE,),
    n_confirm: int = CONFIRM_TRIALS,
    k_confirm: int = CONFIRM_PASS,
) -> dict[str, float]:
    """Exact operating characteristics of the screen (+ near-miss confirm) gate.

    A branch passes outright on >= ``k_screen`` screen successes, dies outright
    below the near-miss band, and on a near-miss score takes a disjoint confirm
    screen (pass on >= ``k_confirm``). Returns pass/reject probabilities at
    ``true_p`` plus the expected trial spend per gated branch.
    """
    _check_prob(true_p)
    _check_trials(n_screen, k_screen)
    _check_trials(n_confirm, k_confirm)
    band = tuple(sorted(set(near_miss)))
    if any(not 0 <= s < k_screen for s in band):
        raise ValueError(f"near-miss scores must lie below k_screen={k_screen}, got {band}")
    p_pass = prob_at_least(n_screen, k_screen, true_p)
    p_band = sum(binomial_pmf(n_screen, s, true_p) for s in band)
    p_confirm_pass = prob_at_least(n_confirm, k_confirm, true_p)
    p_pass_total = p_pass + p_band * p_confirm_pass
    return {
        "true_p": true_p,
        "p_screen_pass": p_pass,
        "p_near_miss": p_band,
        "p_screen_reject": 1.0 - p_pass - p_band,
        "p_confirm_pass_given_near_miss": p_confirm_pass,
        "p_pass_overall": p_pass_total,
        "p_reject_overall": 1.0 - p_pass_total,
        "expected_trials": float(n_screen) + p_band * float(n_confirm),
    }


def c0_oc(true_p: float) -> dict[str, float]:
    """Exact operating characteristics of the C0 fast-reject probe at ``true_p``."""
    _check_prob(true_p)
    p_reject = prob_at_most(C0_TRIALS, C0_REJECT_MAX, true_p)
    return {"true_p": true_p, "p_reject": p_reject, "p_continue": 1.0 - p_reject}


def combined_oc(true_p: float) -> dict[str, float]:
    """Exact operating characteristics of the full C0 -> screen -> confirm gate.

    C0 rejects first; survivors face the sequential screen. Expected trials count
    the C0 probe plus downstream screens for C0 survivors only.
    """
    c0 = c0_oc(true_p)
    seq = sequential_oc(true_p)
    p_reject = c0["p_reject"] + c0["p_continue"] * seq["p_reject_overall"]
    return {
        "true_p": true_p,
        "p_c0_reject": c0["p_reject"],
        "p_screen_pass": c0["p_continue"] * seq["p_screen_pass"],
        "p_near_miss": c0["p_continue"] * seq["p_near_miss"],
        "p_pass_overall": c0["p_continue"] * seq["p_pass_overall"],
        "p_reject_overall": p_reject,
        "expected_trials": float(C0_TRIALS) + c0["p_continue"] * seq["expected_trials"],
    }


def legacy_oc(true_p: float) -> dict[str, float]:
    """Exact operating characteristics of the legacy single 8/10 screen (no C0/retry)."""
    _check_prob(true_p)
    p_pass = prob_at_least(SCREEN_TRIALS, SCREEN_PASS, true_p)
    return {
        "true_p": true_p,
        "p_pass_overall": p_pass,
        "p_reject_overall": 1.0 - p_pass,
        "expected_trials": float(SCREEN_TRIALS),
    }
