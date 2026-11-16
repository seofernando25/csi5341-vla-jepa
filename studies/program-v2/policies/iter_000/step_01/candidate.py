"""Portfolio exploration policy: one cheap scout, cheap pairs, and one deep chain.

The five beta replays are treated as a portfolio.  The frontier of the sweep is the
lower envelope of the per-beta (probes, attainment) points, so different betas should
spend different amounts of the budget:

* a scout beta probes the last legal root once and stops, giving a one-probe point;
* pair betas probe two roots in one batch and stop, filling batches cheaply;
* a deep beta follows the best branch from the last root for a few attempts, giving
  the deep point without paying for an exhausted branch.

The root order is not assumed to be informative: the scout and the deep chain start
from the last root, while the pair betas cover the other roots (and the ends), so the
portfolio sees several roots at different costs.
"""

from __future__ import annotations

from typing import Any

from rsi.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    finalize_result,
)


class OptimalPolicy(LLMDesignedMethod):
    """Prefix-only portfolio policy with scout, pair and deep modes."""

    NAME = "OptimalPolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def _schedule(self, beta: float) -> dict[str, Any]:
        """Map beta to one portfolio mode and its caps."""
        b = max(0.0, min(1.0, float(beta)))
        if b < 0.125:
            mode = "scout"
        elif b < 0.875:
            mode = "pair"
        else:
            mode = "deep"
        pair_index = 0
        if b >= 0.625:
            pair_index = 2
        elif b >= 0.375:
            pair_index = 1
        return {
            "beta": b,
            "mode": mode,
            "pair_index": pair_index,
            "depth_cap": 3,
            "probe_cap": 6,
        }

    def _root_pair(self, roots: list[str], index: int) -> list[str]:
        """Pick a two-root batch that covers a different part of the root list per index."""
        n = len(roots)
        if n <= 1:
            return roots[:1]
        if n == 2:
            return roots[:2]
        if index == 1:
            picks = (n - 2, n - 1)
        elif index == 2:
            picks = (0, n - 1)
        else:
            picks = (0, 1)
        out: list[str] = []
        for i in picks:
            if 0 <= i < n and roots[i] not in out:
                out.append(roots[i])
        return out

    def _deep_action(
        self,
        question: Any,
        prefix: dict[str, Any],
        sched: dict[str, Any],
        spent: int,
    ) -> list[str]:
        """Choose the next cell for the deep mode, or stop with an empty batch."""
        if spent >= sched["probe_cap"]:
            return []
        legal = question.legal_actions()
        if not legal:
            return []
        roots = question.legal_roots()
        if not prefix:
            if not roots:
                return []
            return [roots[-1]]

        best = None
        for obs in prefix.values():
            if obs.fail_class == "ok" and obs.score is not None:
                if best is None or obs.score > best.score:
                    best = obs
        if best is None:
            if roots:
                return [roots[-1]]
            return []

        branch_count = 0
        for obs in prefix.values():
            if obs.branch == best.branch:
                branch_count += 1
        if branch_count >= sched["depth_cap"]:
            return []

        child = None
        child_attempt = 10**9
        for cid in legal:
            meta = question.meta(cid)
            if meta.parent_id == best.cell_id and meta.attempt < child_attempt:
                child = cid
                child_attempt = meta.attempt
        if child is None:
            for cid in legal:
                meta = question.meta(cid)
                if meta.branch == best.branch and meta.attempt < child_attempt:
                    child = cid
                    child_attempt = meta.attempt
        if child is None:
            return []
        return [child]

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        res = SimResult()
        beta = float(self.config.get("beta", 0.6))
        sched = self._schedule(beta)
        spent = 0

        while not (budget is not None and spent >= budget) and question.legal_actions():
            mode = sched["mode"]
            if mode == "scout":
                if spent > 0:
                    break
                roots = question.legal_roots()
                if not roots:
                    break
                batch = [roots[-1]]
            elif mode == "pair":
                if spent > 0:
                    break
                roots = question.legal_roots()
                if not roots:
                    break
                batch = self._root_pair(roots, sched["pair_index"])
            else:
                batch = self._deep_action(question, question.observed(), sched, spent)

            if budget is not None:
                batch = batch[: max(0, budget - spent)]
            if not batch:
                break
            revealed = question.probe_batch(batch)
            if not revealed:
                break
            spent += len(revealed)

        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """Size the next grid from the manifests of the cycles already run (Appendix D.2).

        Start from the pre-registered plan (`context.fallback_plan`) and return it unchanged
        when no cycle has run, so cycle 1 behaves as before. Over the last three manifests, a
        cycle whose refinement (a non-root node) passed the gate says depth paid; a cycle whose
        best score did not beat the previous best (`plateaued`) says breadth stalled. More paying
        cycles than plateaus moves one slot from breadth to depth; more plateaus than paying
        cycles moves one slot from depth to breadth. The harness clamps the result to the
        protocol caps.
        """
        base = context.fallback_plan
        manifests = context.recent_live_manifests[-3:]
        if not manifests:
            return base
        paying = 0
        plateaued = 0
        for manifest in manifests:
            nodes = manifest.get("nodes") or ()
            if any(
                node.get("parent_id") != "root"
                and node.get("fail_class") == "ok"
                and float(node.get("score") or 0.0) > 0.0
                for node in nodes
            ):
                paying += 1
            if manifest.get("plateaued") is True:
                plateaued += 1
        breadth = base.max_branches
        depth = base.max_attempts_per_branch
        if paying > plateaued:
            breadth, depth = breadth - 1, depth + 1
        elif plateaued > paying:
            breadth, depth = breadth + 1, depth - 1
        return GridPlan(
            max_branches=max(1, breadth),
            max_attempts_per_branch=max(1, depth),
            max_parallelism=max(1, min(context.max_parallelism, base.max_parallelism)),
            rounds_cap=max(1, base.rounds_cap),
            rationale=(
                f"prior manifests: {paying} with a passing refinement, {plateaued} plateaued"
            ),
        )
