"""Additional replay candidate policies for Dream-RSI selection scale-up.

All policies are prefix-only and CPU-replayable. Stochastic policies take `seed`;
deterministic ones ignore it.
"""

from __future__ import annotations

import random
from typing import Any

from rsi.api import (
    GATE_NEAR_MISS,
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)
from rsi.policies.baselines import _PatiencePolicy


class EpsilonGreedyPolicy(_PatiencePolicy):
    """With prob `epsilon` probe a uniform random batch, else the best-parent leaf.

    Config: `epsilon` in [0, 1] (default 0.2), `seed` for the draw (default 0).
    """

    NAME = "EpsilonGreedyPolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.epsilon = max(0.0, min(1.0, float(self.config.get("epsilon", 0.2))))

    def _begin(self) -> None:
        self._rng = random.Random(int(self.config.get("seed", 0)))

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        if self._rng.random() < self.epsilon:
            eligible = [*continuations, *roots]
            return self._rng.sample(eligible, min(width, len(eligible)))
        prefix = question.observed()
        scored = []
        for cid in continuations:
            parent = prefix.get(question.meta(cid).parent_id)
            if parent is not None and parent.fail_class == "ok" and parent.score is not None:
                scored.append((-parent.score, cid))
        if scored:
            scored.sort()
            return [c for _, c in scored[:width]]
        return roots[:width]


class ShallowFirstPolicy(_PatiencePolicy):
    """Size-penalized argmax proxy: probe the mildest (lowest-attempt) cells first.

    Distilled compression chains order attempts mild -> aggressive, so preferring
    small attempt indices approximates penalizing aggressive (small-byte) cuts.
    Full parallel width; same patience stopping rule as the other baselines.
    """

    NAME = "ShallowFirstPolicy"

    def _begin(self) -> None:
        pass

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        ordered = sorted(continuations, key=lambda c: (question.meta(c).attempt, c))
        return [*roots[:1], *ordered][:width]


class TwoStageGatePolicy(LLMDesignedMethod):
    """Screen roots in stage 1, refine only passing roots in stage 2 with one retry.

    Stage 1 probes every root (batched to `max_parallelism`). Stage 2 extends only
    children of `ok` roots, plus one child of a `gate_near_miss` root (single retry).
    Stops after `patience` refinement rounds without a new best.
    """

    NAME = "TwoStageGatePolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.5))

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        question.reset()
        res = SimResult()
        # Stage 1: screen all roots.
        while question.legal_roots() and not _budget_done(question, budget):
            roots = question.legal_roots()
            width = question.max_parallelism
            if budget is not None:
                width = min(width, budget - question.budget_spent)
            question.probe_batch(roots[:width], on_reveal=lambda _: _record_curve(res, question))
        # Stage 2: refine passing roots, one near-miss retry.
        patience = max(1, round(1 + 3 * max(0.0, min(1.0, self.beta))))
        stagnant, best_seen, retried = 0, question.best_so_far, False
        while stagnant < patience and not _budget_done(question, budget):
            prefix = question.observed()
            ok_roots = {
                o.cell_id for o in prefix.values() if o.parent_id == "root" and o.fail_class == "ok"
            }
            near = [
                o.cell_id
                for o in prefix.values()
                if o.parent_id == "root" and o.fail_class == GATE_NEAR_MISS
            ]
            batch = [
                c
                for c in question.legal_actions()
                if question.meta(c).parent_id in ok_roots
                or (not retried and question.meta(c).parent_id in near[:1])
            ]
            if not batch:
                break
            width = question.max_parallelism
            if budget is not None:
                width = min(width, budget - question.budget_spent)
            batch = batch[:width]
            if any(question.meta(c).parent_id in near[:1] for c in batch):
                retried = True
            revealed = question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
            ok = [o.score for o in revealed if o.fail_class == "ok" and o.score is not None]
            if ok and max(ok) > best_seen:
                best_seen = max(ok)
                stagnant = 0
            else:
                stagnant += 1
        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        return GridPlan(
            max_branches=6,
            max_attempts_per_branch=4,
            max_parallelism=max(1, context.max_parallelism),
            rounds_cap=12,
            rationale="Static two-stage gate grid plan.",
        )


class IncumbentWithExceptions(LLMDesignedMethod):
    """Serial incumbent that grants one extra probe to a gate near-miss leaf.

    Identical to `InitialPolicy` except a revealed `gate_near_miss` observation
    resets stagnation once per episode instead of counting against patience.
    """

    NAME = "IncumbentWithExceptions"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.5))

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        question.reset()
        res = SimResult()
        sched_target = max(1, round(1 + 3 * max(0.0, min(1.0, self.beta))))
        stagnant, best_seen, excused = 0, question.baseline_score, False
        while not _budget_done(question, budget):
            prefix = question.observed()
            legal = question.legal_actions()
            if not legal:
                break
            roots = question.legal_roots()
            opened = question.opened_branches()
            if roots and len(opened) < sched_target:
                batch = [roots[0]]
            else:
                frontiers = [c for c in legal if c not in roots]
                scored = []
                for cid in frontiers:
                    parent = prefix.get(question.meta(cid).parent_id)
                    if (
                        parent is not None
                        and parent.fail_class == "ok"
                        and parent.score is not None
                    ):
                        scored.append((-parent.score, cid))
                if scored:
                    scored.sort()
                    batch = [scored[0][1]]
                elif roots and stagnant < sched_target:
                    batch = [roots[0]]
                else:
                    break
            revealed = question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
            o = revealed[0]
            if o.fail_class == "ok" and o.score is not None and o.score > best_seen:
                best_seen = o.score
                stagnant = 0
            elif o.fail_class == GATE_NEAR_MISS and not excused:
                excused = True
            else:
                stagnant += 1
            if stagnant >= sched_target and len(opened) >= sched_target:
                break
        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        return GridPlan(
            max_branches=6,
            max_attempts_per_branch=4,
            max_parallelism=1,
            rounds_cap=12,
            rationale="Static serial grid plan.",
        )
