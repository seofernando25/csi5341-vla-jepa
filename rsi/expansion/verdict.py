"""One evaluator result to a node verdict: fail class, score and measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rsi.api import gate_fail_class
from rsi.expansion.states import EvalStatus, FailClass, eval_fail_class

EVAL_ERROR_TAIL = 4000


@dataclass(frozen=True)
class Verdict:
    """Fail class, score and measurements derived from one evaluator result."""

    fail_class: FailClass
    score: float
    successes: int | None
    episodes: int | None
    size_reduction: float | None
    error: str | None
    sizes: dict[str, Any] = field(default_factory=dict)


def classify(raw: dict[str, Any], gate: int) -> Verdict:
    """Map one evaluator result (solution.py `result.json` fields) to a fail class and score."""
    status = raw.get("status")
    if status == EvalStatus.OK:
        try:
            succ = int(raw["successes"])
            episodes = int(raw["episodes"])
            measured = raw["sizes"]
            score = raw.get("score")
            if score is None:
                score = measured["size_reduction"]
            score = float(score)
        except (KeyError, TypeError, ValueError):
            return Verdict(
                FailClass.RUNTIME_FAILURE,
                0.0,
                None,
                None,
                None,
                "ok result lacks successes, episodes or a score",
            )
        reduction = measured.get("size_reduction")
        reduction = float(reduction) if reduction is not None else None
        sizes = {
            "resident_bytes": measured.get("resident_bytes"),
            "baseline_resident_bytes": measured.get("baseline_resident_bytes"),
            "size_reduction": reduction,
        }
        fail_class = FailClass(gate_fail_class(succ, gate))
        if fail_class == FailClass.OK:
            return Verdict(FailClass.OK, score, succ, episodes, reduction, None, sizes)
        return Verdict(
            fail_class, 0.0, succ, episodes, reduction, f"smoke={succ}/{episodes}<{gate}", sizes
        )
    error = str(raw.get("error") or status or "no status")[-EVAL_ERROR_TAIL:]
    return Verdict(eval_fail_class(status), 0.0, None, None, None, error)


def per_task_counts(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per-task successes and episodes from the evaluator's rollout block, if it reported one."""
    rollout = raw.get("rollout")
    per_task = rollout.get("per_task") if isinstance(rollout, dict) else None
    out: dict[str, dict[str, int]] = {}
    for task_id, counts in (per_task or {}).items():
        if not isinstance(counts, dict):
            continue
        successes = counts.get("successes")
        episodes = counts.get("episodes")
        if isinstance(successes, int) and isinstance(episodes, int):
            out[str(task_id)] = {"successes": successes, "episodes": episodes}
    return out
