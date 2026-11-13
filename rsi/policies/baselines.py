"""Trivial exploration baselines for the Dream-RSI candidate pool.

Each baseline maps beta to the same stopping rule: stop after `round(1 + 3 * beta)` consecutive
rounds without a new best score. Roots open in recorded order, as in the paper's `Child(r)`.
"""

from __future__ import annotations

import random
from typing import Any

from rsi.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)


class _PatiencePolicy(LLMDesignedMethod):
    """Shared loop; subclasses order the eligible cells and set the batch width."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.5))

    def _width(self, question: Any) -> int:
        return question.max_parallelism

    def _begin(self) -> None:
        """Reset per-episode state."""

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        raise NotImplementedError

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        question.reset()
        self._begin()
        res = SimResult()
        patience = max(1, round(1 + 3 * max(0.0, min(1.0, self.beta))))
        stagnant = 0
        best_seen = question.baseline_score

        while stagnant < patience and not _budget_done(question, budget):
            roots = question.legal_roots()
            root_set = set(roots)
            continuations = [c for c in question.legal_actions() if c not in root_set]
            width = self._width(question)
            if budget is not None:
                width = min(width, budget - question.budget_spent)
            batch = self._batch(question, continuations, roots, width)
            if not batch:
                break
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
            rationale="Static baseline grid plan.",
        )


class RandomPolicy(_PatiencePolicy):
    """Uniform random batch of eligible cells; `seed` in the config fixes the draw."""

    NAME = "RandomPolicy"

    def _begin(self) -> None:
        self._rng = random.Random(int(self.config.get("seed", 0)))

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        slots = [*continuations, *([None] * len(roots))]
        picked = self._rng.sample(slots, min(width, len(slots)))
        opened = roots[: sum(1 for c in picked if c is None)]
        return [c for c in picked if c is not None] + opened


class RoundRobinPolicy(_PatiencePolicy):
    """Breadth-first: open the next root, then advance the least recently probed branches."""

    NAME = "RoundRobinPolicy"

    def _begin(self) -> None:
        self._last_probed: dict[int, int] = {}
        self._round = 0

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        ordered = sorted(
            continuations,
            key=lambda c: (self._last_probed.get(question.meta(c).branch, -1), c),
        )
        batch = [*roots[:1], *ordered][:width]
        self._round += 1
        for cid in batch:
            self._last_probed[question.meta(cid).branch] = self._round
        return batch


class GreedySerialPolicy(_PatiencePolicy):
    """One probe per round: extend the best passing leaf, else open the next root."""

    NAME = "GreedySerialPolicy"

    def _width(self, question: Any) -> int:
        return 1

    def _batch(self, question: Any, continuations: list[str], roots: list[str], width: int):
        prefix = question.observed()
        scored = []
        for cid in continuations:
            parent = prefix.get(question.meta(cid).parent_id)
            if parent is not None and parent.fail_class == "ok" and parent.score is not None:
                scored.append((-parent.score, cid))
        if scored:
            return [min(scored)[1]][:width]
        return roots[:1][:width]
