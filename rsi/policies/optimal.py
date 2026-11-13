"""Dynamic portfolio exploration policy (Dream-RSI Appendix D.2 Listing 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rsi.api import (
    GATE_NEAR_MISS,
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    Observation,
    SimResult,
    _budget_done,
    _record_curve,
    branch_failed_hard,
    branch_promising,
    finalize_result,
    probe_improved_vs_parent,
)

REPAIRABLE_FAIL_CLASSES = {"implementation_failure", "runtime_failure", "compile_other"}


@dataclass(frozen=True)
class BranchTrajectory:
    """Prefix-reconstructed trajectory for one opened branch."""

    branch: int
    observations: tuple[Observation, ...]
    successful_anchor: float | None
    latest: Observation
    ok_count: int
    consecutive_failures: int
    regressions: int
    last_gain: float
    classification: str


class OptimalPolicy(LLMDesignedMethod):
    """Prefix-only dynamic portfolio policy with trajectory & failure classification."""

    NAME = "OptimalPolicy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.6))
        self.gate_retry = bool(self.config.get("gate_retry", True))

    def _schedule(self, beta: float) -> dict[str, Any]:
        b = max(0.0, min(1.0, float(beta)))
        return {
            "beta": b,
            "target_roots": max(1, round(1 + 4 * b)),
            "patience_rounds": max(1, round(1 + 3 * b)),
            "max_regressions": 1 + round(2 * b),
            "max_repairs": 1 if b < 0.4 else 2,
            "explore_cap": 0 if b < 0.2 else (1 if b < 0.6 else 2),
            "allow_recovery": b >= 0.25,
            "prune_ratio": 0.25 + 0.65 * b,
            "width_scale": 0.25 if b < 0.2 else (0.75 if b < 0.5 else 1.0),
        }

    def _reconstruct_branches(
        self,
        prefix: dict[str, Observation],
        baseline_score: float,
        sched: dict[str, Any],
    ) -> dict[int, BranchTrajectory]:
        by_branch: dict[int, list[Observation]] = {}
        for obs in prefix.values():
            by_branch.setdefault(obs.branch, []).append(obs)

        trajectories: dict[int, BranchTrajectory] = {}
        global_best = max(
            (o.score for o in prefix.values() if o.fail_class == "ok" and o.score is not None),
            default=baseline_score,
        )
        global_gain = max(1e-9, global_best - baseline_score)

        for branch, items in by_branch.items():
            items.sort(key=lambda o: o.attempt)
            ok_scores = [o.score for o in items if o.fail_class == "ok" and o.score is not None]
            anchor = max(ok_scores) if ok_scores else None
            latest = items[-1]

            consec_fail = 0
            for o in reversed(items):
                if branch_failed_hard(o):
                    consec_fail += 1
                else:
                    break

            regressions = 0
            prev_ok: float | None = None
            for o in items:
                if o.fail_class == "ok" and o.score is not None:
                    if prev_ok is not None and o.score < prev_ok:
                        regressions += 1
                    prev_ok = o.score

            last_gain = latest.delta_vs_parent if latest.delta_vs_parent is not None else 0.0

            if latest.fail_class == "ok":
                if (
                    len(ok_scores) >= 2
                    and regressions >= sched["max_regressions"]
                    and anchor is not None
                    and (anchor - baseline_score) < (1.0 - sched["prune_ratio"]) * global_gain
                ):
                    cls = "repeatedly_unpromising"
                elif len(ok_scores) == 1 and not branch_promising(latest, baseline_score):
                    cls = "weak_underexplored"
                else:
                    cls = "promising"
            elif (
                latest.fail_class in REPAIRABLE_FAIL_CLASSES
                and consec_fail <= sched["max_repairs"]
                and (anchor is not None or len(items) <= 2)
            ):
                cls = "repairable_failure"
            elif self.gate_retry and latest.fail_class == GATE_NEAR_MISS and consec_fail == 1:
                # One success short of a 10-trial gate is noise-compatible
                # (P(<8/10 | p=0.87) = 0.13), so the branch gets one more probe before closing.
                cls = "repairable_failure"
            else:
                cls = "hard_unrecoverable"

            trajectories[branch] = BranchTrajectory(
                branch=branch,
                observations=tuple(items),
                successful_anchor=anchor,
                latest=latest,
                ok_count=len(ok_scores),
                consecutive_failures=consec_fail,
                regressions=regressions,
                last_gain=last_gain,
                classification=cls,
            )
        return trajectories

    def _update_closed(
        self,
        closed: set[int],
        trajectories: dict[int, BranchTrajectory],
    ) -> None:
        for branch, traj in trajectories.items():
            if traj.classification in {"hard_unrecoverable", "repeatedly_unpromising"}:
                closed.add(branch)
            elif branch in closed and traj.latest.fail_class == "ok":
                closed.discard(branch)

    def _select_batch(
        self,
        prefix: dict[str, Observation],
        question: Any,
        closed: set[int],
        sched: dict[str, Any],
        stagnant_rounds: int,
        budget_left: int | None,
    ) -> list[str]:
        legal = question.legal_actions()
        if not legal:
            return []
        max_w = max(1, round(question.max_parallelism * sched["width_scale"]))
        if budget_left is not None:
            max_w = min(max_w, budget_left)
        if max_w <= 0:
            return []

        roots = question.legal_roots()
        trajectories = self._reconstruct_branches(prefix, question.baseline_score, sched)
        self._update_closed(closed, trajectories)

        exploit_q: list[tuple[float, str]] = []
        explore_q: list[tuple[float, str]] = []
        recovery_q: list[tuple[float, str]] = []

        root_set = set(roots)
        for cid in legal:
            if cid in root_set:
                continue
            meta = question.meta(cid)
            if meta.branch in closed:
                continue
            traj = trajectories.get(meta.branch)
            if traj is None:
                continue
            anchor = traj.successful_anchor if traj.successful_anchor is not None else -1e9
            gain_bonus = 0.05 if probe_improved_vs_parent(traj.latest) else 0.0
            if traj.classification == "promising":
                exploit_q.append((anchor + gain_bonus - 0.005 * meta.attempt, cid))
            elif traj.classification == "weak_underexplored":
                explore_q.append((anchor - 0.005 * meta.attempt, cid))
            elif traj.classification == "repairable_failure" and sched["allow_recovery"]:
                recovery_q.append((anchor - 0.01 * traj.consecutive_failures, cid))

        exploit_q.sort(key=lambda x: (-x[0], x[1]))
        explore_q.sort(key=lambda x: (-x[0], x[1]))
        recovery_q.sort(key=lambda x: (-x[0], x[1]))

        opened_count = len(question.opened_branches())
        if (
            stagnant_rounds >= sched["patience_rounds"]
            and opened_count >= sched["target_roots"]
            and not exploit_q
        ):
            return []

        batch: list[str] = []
        used: set[str] = set()

        def add_cell(cid: str) -> None:
            if len(batch) < max_w and cid not in used:
                batch.append(cid)
                used.add(cid)

        # 1. Seed initial roots up to target_roots (batched in parallel!)
        need_roots = max(0, sched["target_roots"] - opened_count)
        root_budget = min(need_roots, max_w if not exploit_q else sched["explore_cap"])
        for r_cid in roots[:root_budget]:
            add_cell(r_cid)

        # 2. Fill exploitation slots with strong refinements
        for _, cid in exploit_q:
            if len(batch) >= max_w:
                break
            add_cell(cid)

        # 3. Include at most one justified recovery if capacity remains
        if recovery_q and len(batch) < max_w:
            add_cell(recovery_q[0][1])

        # 4. Fill remaining slots with underexplored branches or remaining target roots
        for _, cid in explore_q:
            if len(batch) >= max_w:
                break
            add_cell(cid)
        roots_planned = sum(1 for cid in batch if cid in root_set)
        if opened_count + roots_planned < sched["target_roots"]:
            for r_cid in roots:
                if len(batch) >= max_w or opened_count + roots_planned >= sched["target_roots"]:
                    break
                if r_cid not in used:
                    add_cell(r_cid)
                    roots_planned += 1

        return batch

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        question.reset()
        res = SimResult()
        closed: set[int] = set()
        sched = self._schedule(self.beta)
        best_seen = question.baseline_score
        stagnant_rounds = 0

        while not _budget_done(question, budget):
            prefix = question.observed()
            budget_left = None if budget is None else max(0, budget - question.budget_spent)
            batch = self._select_batch(
                prefix=prefix,
                question=question,
                closed=closed,
                sched=sched,
                stagnant_rounds=stagnant_rounds,
                budget_left=budget_left,
            )
            if not batch:
                break
            revealed = question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
            ok_scores = [o.score for o in revealed if o.fail_class == "ok" and o.score is not None]
            if ok_scores and max(ok_scores) > best_seen:
                best_seen = max(ok_scores)
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            if (
                stagnant_rounds >= sched["patience_rounds"]
                and len(question.opened_branches()) >= sched["target_roots"]
            ):
                break

        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """Deterministic cross-cycle grid planner based on prior manifests and sweeps."""
        base_branches = 6
        base_depth = 4
        if context.recent_live_manifests:
            last = context.recent_live_manifests[-1]
            if last.get("plateaued", False):
                base_branches += 2
            if last.get("deepest_improvement", 0) >= 3:
                base_depth += 1
        w = max(1, context.max_parallelism)
        rounds_cap = max(4, (base_branches * base_depth + w - 1) // w)
        return GridPlan(
            max_branches=base_branches,
            max_attempts_per_branch=base_depth,
            max_parallelism=w,
            rounds_cap=rounds_cap,
            rationale=(
                f"Iter {context.iteration}: branches={base_branches}, depth={base_depth}, W={w}."
            ),
        )
