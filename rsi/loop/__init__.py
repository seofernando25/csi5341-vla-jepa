"""The durable Dream-RSI outer loop and its detached launcher.

`run` iterates cycle and policy-development stages, `records` and `store` keep
`loop_state.json`, `states` is the status machine those records follow, and `launcher` and
`status` start, stop and report a detached run.
"""

from rsi.loop.launcher import start, stop, systemd_argv, unit_name
from rsi.loop.records import PolicyMismatchError, recorded_policy_path
from rsi.loop.run import run_dream_rsi_loop
from rsi.loop.status import status
from rsi.loop.store import (
    LoopLockedError,
    StudyLock,
    load_loop_state,
    lock_holder,
    save_loop_state,
    stop_requested,
    write_json_atomic,
)

__all__ = [
    "LoopLockedError",
    "PolicyMismatchError",
    "StudyLock",
    "load_loop_state",
    "lock_holder",
    "recorded_policy_path",
    "run_dream_rsi_loop",
    "save_loop_state",
    "start",
    "status",
    "stop",
    "stop_requested",
    "systemd_argv",
    "unit_name",
    "write_json_atomic",
]
