"""Grid plan read from a policy's explicit pre-registered config."""

from __future__ import annotations

from typing import Any

from rsi.api import GridPlan


def explicit_grid_plan(config: dict[str, Any]) -> GridPlan:
    """Return the `grid_plan` from config; the clean protocol defines no default plan."""
    plan = config.get("grid_plan")
    if not isinstance(plan, dict):
        raise TypeError(
            "Clean protocol requires an explicit grid_plan in the policy config; "
            "no adaptive planner is defined."
        )
    return GridPlan(
        max_branches=int(plan["max_branches"]),
        max_attempts_per_branch=int(plan["max_attempts_per_branch"]),
        max_parallelism=int(plan["max_parallelism"]),
        rounds_cap=int(plan["rounds_cap"]),
        rationale=str(plan.get("rationale", "pre-registered static grid plan")),
    )
