"""Node lifecycle, evaluator statuses and fail classes of one node expansion.

A node is never given a status field: its status is what its directory holds.

    pending    -> agent | done         nothing yet; `done` when the parent has no snapshot
    agent      -> evaluating | done    `logs/stdout.log` exists; `done` on an agent failure
    evaluating -> done                 `eval/` exists
    done                               `score.json` exists; it carries the fail class

A viewer that finds no loop process holding the study lock demotes `agent` and `evaluating` to
`stopped`. The fail class of a done node follows from the evaluator status: `ok` is gated on
smoke successes, a solution-side status is an implementation failure, anything else a runtime
failure. Agent failures and timeouts are decided before evaluation.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from evaluation.fsm import StateMachine
from rsi.api import (
    AGENT_FAILURE,
    AGENT_TIMEOUT,
    GATE_FAILED,
    GATE_NEAR_MISS,
    IMPLEMENTATION_FAILURE,
    OK,
    RUNTIME_FAILURE,
)


class NodeStatus(StrEnum):
    """Where one node is in its expansion, as read from its directory."""

    PENDING = "pending"
    AGENT = "agent"
    EVALUATING = "evaluating"
    DONE = "done"
    STOPPED = "stopped"


class FailClass(StrEnum):
    """`score.json` `fail_class`; the values are the `rsi.api` constants policies compare to."""

    OK = OK
    GATE_NEAR_MISS = GATE_NEAR_MISS
    GATE_FAILED = GATE_FAILED
    RUNTIME_FAILURE = RUNTIME_FAILURE
    IMPLEMENTATION_FAILURE = IMPLEMENTATION_FAILURE
    AGENT_TIMEOUT = AGENT_TIMEOUT
    AGENT_FAILURE = AGENT_FAILURE


class EvalStatus(StrEnum):
    """`status` of the fixed evaluator's `result.json`, and of the sandbox's own failure."""

    OK = "ok"
    SOLUTION_ERROR = "solution_error"
    SIZE_VIOLATION = "size_violation"
    GUARD_VIOLATION = "guard_violation"
    EVALUATOR_FAILURE = "evaluator_failure"


_N = NodeStatus
NODE = StateMachine(
    "node",
    NodeStatus,
    {
        _N.PENDING: {_N.AGENT, _N.DONE},
        _N.AGENT: {_N.EVALUATING, _N.DONE, _N.STOPPED},
        _N.EVALUATING: {_N.DONE, _N.STOPPED},
    },
    initial={_N.PENDING},
)
IN_FLIGHT = frozenset({_N.AGENT, _N.EVALUATING})
# Evaluator statuses caused by the submitted solution rather than by the run.
SOLUTION_STATUSES = frozenset(
    {EvalStatus.SOLUTION_ERROR, EvalStatus.SIZE_VIOLATION, EvalStatus.GUARD_VIOLATION}
)


def eval_fail_class(status: object) -> FailClass:
    """Fail class of an evaluator status that is not `ok`."""
    return (
        FailClass.IMPLEMENTATION_FAILURE
        if status in SOLUTION_STATUSES
        else FailClass.RUNTIME_FAILURE
    )


def observed_node_status(node_dir: Path, *, scored: bool) -> NodeStatus:
    """A node's status from the files its expansion has written so far.

    `scored` says whether `score.json` was read; a half-written one does not count yet.
    """
    if scored:
        return NodeStatus.DONE
    if (node_dir / "eval").is_dir():
        return NodeStatus.EVALUATING
    if (node_dir / "logs" / "stdout.log").is_file():
        return NodeStatus.AGENT
    return NodeStatus.PENDING
