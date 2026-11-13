"""Shared decision interface, observation signals, and planning types (Dream-RSI Appendix D.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GATE_NEAR_MISS = "gate_near_miss"
GATE_FAILED = "gate_failed"
# One literal per fail class; policies compare against these constants, never raw strings.


def gate_fail_class(successes: int, gate: int, near_miss: int = 1) -> str:
    """Classify a success count against a gate; `near_miss` successes short is a near miss."""
    if successes >= gate:
        return "ok"
    return GATE_NEAR_MISS if gate - successes <= near_miss else GATE_FAILED


@dataclass(frozen=True)
class Observation:
    """Prefix-only observation of a revealed discovery node."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str | None
    score: float | None
    evaluated: bool
    valid: bool
    fail_class: str
    error: str | None
    delta_vs_baseline: float | None
    delta_vs_parent: float | None
    n_valid: int | None = None
    n_total: int | None = None
    mechanism_family: str = ""
    summary: str = ""
    cycle: int = 0


@dataclass(frozen=True)
class CellMeta:
    """Structural metadata for a cell in the discovery grid/tree."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str | None
    seq: int
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GridPlanningContext:
    """Cross-cycle context passed to plan_grid before creating a new live grid."""

    iteration: int
    max_parallelism: int
    recent_live_manifests: tuple[dict[str, Any], ...] = ()
    recent_beta_sweeps: tuple[dict[str, Any], ...] = ()
    history_summaries: tuple[dict[str, Any], ...] = ()
    default_budget: int | None = None


@dataclass(frozen=True)
class GridPlan:
    """Deterministic next-cycle grid configuration."""

    max_branches: int
    max_attempts_per_branch: int
    max_parallelism: int
    rounds_cap: int
    rationale: str = ""


@dataclass
class SimResult:
    """Bookkeeping result for one replay or live episode."""

    curve: list[tuple[int, float]] = field(default_factory=list)
    best_score: float | None = None
    total_probes: int = 0
    decision_rounds: int = 0
    effective_sequential_rounds: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def _budget_done(question: Any, budget: int | None) -> bool:
    if budget is not None and question.budget_spent >= budget:
        return True
    return not question.legal_actions()


def _record_curve(res: SimResult, question: Any) -> None:
    res.total_probes = question.budget_spent
    res.best_score = question.best_so_far
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    res.curve.append((res.total_probes, res.best_score))


def finalize_result(question: Any, res: SimResult) -> SimResult:
    res.total_probes = question.budget_spent
    res.best_score = question.best_so_far
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    return res


def branch_promising(obs: Observation, baseline_score: float) -> bool:
    """Return True if an evaluated observation is valid and improves over baseline."""
    if not obs.evaluated or obs.error is not None or obs.fail_class != "ok":
        return False
    if obs.score is None:
        return False
    return obs.score > baseline_score


def branch_failed_hard(obs: Observation) -> bool:
    """Return True if an observation failed evaluation (requires fail_class inspection)."""
    if not obs.evaluated:
        return False
    return obs.error is not None or obs.fail_class != "ok"


def probe_improved_vs_parent(obs: Observation) -> bool:
    return obs.delta_vs_parent is not None and obs.delta_vs_parent > 0.0


def probe_improved_vs_baseline(obs: Observation) -> bool:
    return obs.delta_vs_baseline is not None and obs.delta_vs_baseline > 0.0


class LLMDesignedMethod:
    """Base class for Dream-RSI exploration policies."""

    NAME = "BasePolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        raise NotImplementedError

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        raise NotImplementedError
