"""Sparse portfolio exploration policy: one scout, one deep chain, idle middle betas.

The five beta replays are treated as a portfolio.  Only points that can appear on the
quality-versus-cost frontier are worth probing: on a tree whose best root is the most
recently added root and whose best branch improves for a short chain, a one-probe scout
of that root gives the cheap point and a short chain from it gives the best point.  Any
other beta point is dominated and only adds to the mean sequential-rounds-per-probe
penalty, so the middle betas deliberately stop without probing.

The scout and the deep chain both start from the last legal root.  The deep chain
follows the best observed branch, but it stops as soon as a refinement shows that the
branch has already captured its main win: a single attempt that adds a large gain over
its parent (a quarter of the size-reduction scale) is followed by plateaued attempts
that repeat the parent score, so one more probe would only move the point right along
the same attainment and cost hypervolume.  A refinement that fails to improve on its
parent is the same plateau signal after the fact.  Otherwise the chain is capped at
three attempts, which reaches a best node one level deeper than the typical winner
without paying for the first failing attempt after it.  The default beta is the deep
mode, so a live cycle still explores even though the middle replay betas are idle.
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
    """Prefix-only sparse portfolio policy with scout, idle and deep modes."""

    NAME = "OptimalPolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def _schedule(self, beta: float) -> dict[str, Any]:
        """Map beta to one portfolio mode and its caps."""
        b = max(0.0, min(1.0, float(beta)))
        if b < 0.125:
            mode = "scout"
        elif b < 0.875:
            mode = "idle"
        else:
            mode = "deep"
        return {
            "beta": b,
            "mode": mode,
            "depth_cap": 3,
            "probe_cap": 6,
            "jump_gain": 0.25,
        }

    def _chain_stop(self, best: Any, sched: dict[str, Any]) -> bool:
        """True when the best branch has already captured its main win.

        A refinement (attempt >= 1) whose gain over its parent reaches `jump_gain` is the
        branch's big step; later attempts on the same branch tend to repeat the score.
        A refinement that does not improve on its parent is the same signal once seen.
        """
        if best.parent_id is None or best.parent_id == "root":
            return False
        gain = best.delta_vs_parent
        if gain is not None:
            if gain <= 0.0:
                return True
            if gain >= sched["jump_gain"]:
                return True
        return False

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
        if self._chain_stop(best, sched):
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
        beta = float(self.config.get("beta", 1.0))
        sched = self._schedule(beta)
        spent = 0

        while not (budget is not None and spent >= budget) and question.legal_actions():
            mode = sched["mode"]
            if mode == "idle":
                break
            if mode == "scout":
                if spent > 0:
                    break
                roots = question.legal_roots()
                if not roots:
                    break
                batch = [roots[-1]]
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
