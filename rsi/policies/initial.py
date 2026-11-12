"""Baseline serial exploration policy (pi_0) for Dream-RSI benchmarking."""

from __future__ import annotations

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


class InitialPolicy(LLMDesignedMethod):
    """Simple serial baseline: open a few roots, greedily refine best leaf, stop on failure."""

    NAME = "InitialPolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.5))

    def _schedule(self, beta: float) -> dict[str, int]:
        b = max(0.0, min(1.0, beta))
        return {
            "target_roots": max(1, round(1 + 3 * b)),
            "patience": max(1, round(1 + 3 * b)),
        }

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        question.reset()
        res = SimResult()
        sched = self._schedule(self.beta)
        stagnant = 0
        best_seen = question.baseline_score

        while not _budget_done(question, budget):
            prefix = question.observed()
            legal = question.legal_actions()
            if not legal:
                break
            roots = question.legal_roots()
            opened = question.opened_branches()

            # Open roots serially until target_roots is reached
            if roots and len(opened) < sched["target_roots"]:
                batch = [roots[0]]
            else:
                # Pick highest-scoring legal leaf whose parent did not fail
                frontiers = [c for c in legal if c not in roots]
                ok_frontiers: list[tuple[float, str]] = []
                for cid in frontiers:
                    meta = question.meta(cid)
                    parent = prefix.get(meta.parent_id) if meta.parent_id else None
                    if (
                        parent is not None
                        and parent.fail_class == "ok"
                        and parent.score is not None
                    ):
                        ok_frontiers.append((parent.score, cid))
                if ok_frontiers:
                    ok_frontiers.sort(key=lambda x: (-x[0], x[1]))
                    batch = [ok_frontiers[0][1]]
                elif roots and stagnant < sched["patience"]:
                    batch = [roots[0]]
                else:
                    break

            revealed = question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
            improved = any(
                o.fail_class == "ok" and o.score is not None and o.score > best_seen
                for o in revealed
            )
            if improved:
                best_seen = max(
                    o.score for o in revealed if o.fail_class == "ok" and o.score is not None
                )
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= sched["patience"] and len(opened) >= sched["target_roots"]:
                break

        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        return GridPlan(
            max_branches=6,
            max_attempts_per_branch=4,
            max_parallelism=max(1, context.max_parallelism),
            rounds_cap=12,
            rationale="Static initial grid plan.",
        )
