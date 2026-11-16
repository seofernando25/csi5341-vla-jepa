"""Status machines of a loop run: `loop_state.json`, its stages, what a viewer sees, and `stop`.

The loop is `running` while a process holds the study lock and ends `stopped` (a stop rule),
`done` or `error`; any end state may resume to `running`, and so may `running` itself when the
previous process died without writing an end state. Each iteration runs a `cycle` stage, then
a `policy_dev` stage; a stage resumes from `running` after a crash.

A detached run is launched as a `systemd` unit or a `detached` process group (`service.json`
`mode`). `vla-rsi stop` requests a stop, waits for a clean exit, and only then escalates to the
launcher's own stop; its `method` is the state it ended in.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from evaluation.fsm import StateMachine


class LoopStatus(StrEnum):
    """`loop_state.json` `status`."""

    RUNNING = "running"
    STOPPED = "stopped"
    DONE = "done"
    ERROR = "error"


class StageStatus(StrEnum):
    """`status` of an iteration's `cycle` or `policy_dev` block."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"


class Stage(StrEnum):
    """The stage an unfinished iteration is in."""

    CYCLE = "cycle"
    POLICY_DEV = "policy_dev"


class LoopView(StrEnum):
    """Loop status as a viewer sees it: the recorded status corrected by the lock."""

    RUNNING = "running"
    STOPPED = "stopped"
    DONE = "done"
    ERROR = "error"
    IDLE = "idle"


class LaunchMode(StrEnum):
    """`service.json` `mode`: how `vla-rsi start` launched the loop."""

    SYSTEMD = "systemd"
    DETACHED = "detached"


class StopStep(StrEnum):
    """Progress of `vla-rsi stop`; the final state is the reported `method`."""

    REQUESTED = "requested"
    WAITING = "waiting"
    NOT_RUNNING = "not-running"
    CLEAN = "clean"
    UNIT_STOP = "unit-stop"
    SIGNAL = "signal"
    UNKNOWN_HOLDER = "unknown-holder"


_L, _S, _T = LoopStatus, StageStatus, StopStep
LOOP = StateMachine(
    "loop",
    LoopStatus,
    {
        _L.RUNNING: {_L.RUNNING, _L.STOPPED, _L.DONE, _L.ERROR},
        _L.STOPPED: {_L.RUNNING},
        _L.DONE: {_L.RUNNING},
        _L.ERROR: {_L.RUNNING},
    },
    initial={_L.RUNNING},
)
# A finished cycle may be found already written (its manifest exists) and marked done directly.
CYCLE_STAGE = StateMachine(
    "cycle stage",
    StageStatus,
    {_S.PENDING: {_S.RUNNING, _S.DONE}, _S.RUNNING: {_S.RUNNING, _S.DONE}},
    initial={_S.PENDING},
)
# Policy development is skipped when no recorded tree has a node to replay.
POLICY_DEV_STAGE = StateMachine(
    "policy_dev stage",
    StageStatus,
    {
        _S.PENDING: {_S.RUNNING, _S.DONE, _S.SKIPPED},
        _S.RUNNING: {_S.RUNNING, _S.DONE, _S.SKIPPED},
    },
    initial={_S.PENDING},
)
STAGE_FINISHED = frozenset({_S.DONE, _S.SKIPPED})
STOP = StateMachine(
    "stop",
    StopStep,
    {
        _T.REQUESTED: {_T.NOT_RUNNING, _T.WAITING},
        _T.WAITING: {_T.CLEAN, _T.UNIT_STOP, _T.SIGNAL, _T.UNKNOWN_HOLDER},
    },
    initial={_T.REQUESTED},
)
# The launcher's own stop, used once the grace period has passed.
ESCALATION = {LaunchMode.SYSTEMD: _T.UNIT_STOP, LaunchMode.DETACHED: _T.SIGNAL}


def launch_mode(service: dict[str, Any]) -> LaunchMode | None:
    """The launch mode `service.json` records, or None when it records none."""
    try:
        return LaunchMode(service.get("mode"))
    except ValueError:
        return None


def _block(it: dict[str, Any], key: str) -> dict[str, Any]:
    value = it.get(key)
    return value if isinstance(value, dict) else {}


def iteration_done(it: dict[str, Any]) -> bool:
    """True when the iteration's cycle is done and its policy development done or skipped."""
    return (
        _block(it, "cycle").get("status") == _S.DONE
        and _block(it, "policy_dev").get("status") in STAGE_FINISHED
    )


def current_stage(it: dict[str, Any]) -> Stage:
    """The stage an unfinished iteration is in: its cycle until that is done, then policy_dev."""
    return Stage.CYCLE if _block(it, "cycle").get("status") != _S.DONE else Stage.POLICY_DEV


def observed_status(recorded: object, running: bool) -> LoopView:
    """What a viewer should see: `running` only while the lock is held.

    A recorded `running` without a lock holder is a loop that died; it shows as `stopped`.
    A missing or unknown recorded status shows as `idle`.
    """
    if running:
        return LoopView.RUNNING
    if recorded == _L.RUNNING:
        return LoopView.STOPPED
    if recorded in (_L.STOPPED, _L.DONE, _L.ERROR):
        return LoopView(recorded)
    return LoopView.IDLE
