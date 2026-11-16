"""The rsi state machines: a legal path, a rejected move, and the values already on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.fsm import IllegalTransition, StateMachine, UnknownState
from rsi.expansion.states import (
    NODE,
    EvalStatus,
    FailClass,
    NodeStatus,
    eval_fail_class,
    observed_node_status,
)
from rsi.loop.states import (
    CYCLE_STAGE,
    LOOP,
    POLICY_DEV_STAGE,
    STOP,
    LaunchMode,
    LoopStatus,
    LoopView,
    StageStatus,
    StopStep,
    current_stage,
    iteration_done,
    launch_mode,
    observed_status,
)
from rsi.policy_dev.states import RECORDED, STEP, StepStatus

STUDIES = Path(__file__).resolve().parents[1] / "studies"


def _walk(machine: StateMachine, path: list) -> None:
    """Drive a record through `path` from the start; every move must be legal."""
    record: dict = {}
    for state in path:
        machine.advance(record, state)
        assert record["status"] == state and type(record["status"]) is str


def _json_files(pattern: str) -> list[dict]:
    files = sorted(STUDIES.rglob(pattern))
    assert files, f"no {pattern} under studies/"
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


# Generic machine


def test_machine_errors_name_the_machine_and_the_options() -> None:
    with pytest.raises(UnknownState, match=r"loop: unknown state 'paused'; expected one of"):
        LOOP.parse("paused")
    with pytest.raises(
        IllegalTransition, match=r"loop: done -> error is not allowed; from done: running"
    ):
        LOOP.check("done", "error")
    with pytest.raises(IllegalTransition, match=r"start -> done .* from start: running"):
        LOOP.check(None, "done")
    with pytest.raises(IllegalTransition, match=r"from accepted: none \(terminal\)"):
        STEP.check("accepted", "rejected")


def test_states_serialize_exactly_like_the_old_strings() -> None:
    for member in [*LoopStatus, *StageStatus, *StepStatus, *NodeStatus, *FailClass]:
        assert json.dumps({"status": member}) == json.dumps({"status": member.value})
        assert f"{member}" == member.value and member == member.value


# Loop, stages and the viewer's loop status


def test_loop_resumes_from_every_end_state() -> None:
    _walk(LOOP, ["running", "stopped", "running", "running", "error", "running", "done"])


def test_loop_cannot_end_twice() -> None:
    with pytest.raises(IllegalTransition):
        LOOP.check("stopped", "done")


def test_stages_follow_their_tables() -> None:
    _walk(CYCLE_STAGE, ["pending", "running", "running", "done"])
    _walk(CYCLE_STAGE, ["pending", "done"])
    _walk(POLICY_DEV_STAGE, ["pending", "running", "skipped"])
    with pytest.raises(IllegalTransition):
        CYCLE_STAGE.check("pending", "skipped")
    with pytest.raises(IllegalTransition):
        POLICY_DEV_STAGE.check("done", "running")


def test_recorded_loop_states_load() -> None:
    for state in _json_files("loop_state.json"):
        LOOP.parse(state["status"])
        for it in state["iterations"]:
            CYCLE_STAGE.parse(it["cycle"]["status"])
            POLICY_DEV_STAGE.parse(it["policy_dev"]["status"])
            assert iteration_done(it) or current_stage(it) in ("cycle", "policy_dev")


def test_viewer_sees_running_only_while_the_lock_is_held() -> None:
    assert observed_status("done", running=True) is LoopView.RUNNING
    assert observed_status("running", running=False) is LoopView.STOPPED
    assert observed_status("error", running=False) is LoopView.ERROR
    assert observed_status(None, running=False) is LoopView.IDLE
    assert observed_status("weird", running=False) is LoopView.IDLE


def test_current_stage_follows_the_cycle() -> None:
    it = {"cycle": {"status": "running"}, "policy_dev": {"status": "pending"}}
    assert current_stage(it) == "cycle" and not iteration_done(it)
    it["cycle"]["status"] = "done"
    assert current_stage(it) == "policy_dev" and not iteration_done(it)
    it["policy_dev"]["status"] = "skipped"
    assert iteration_done(it)


# Detached launcher and stop


def test_stop_escalates_only_after_waiting() -> None:
    _walk(STOP, ["requested", "waiting", "unit-stop"])
    _walk(STOP, ["requested", "not-running"])
    with pytest.raises(IllegalTransition):
        STOP.check("requested", "signal")
    with pytest.raises(IllegalTransition):
        STOP.check("clean", "signal")


def test_recorded_launch_modes_load() -> None:
    for service in _json_files("service.json"):
        assert launch_mode(service) in tuple(LaunchMode)
    assert launch_mode({}) is None and launch_mode({"mode": "cron"}) is None
    assert {step.value for step in STOP.terminal()} == {
        "not-running",
        "clean",
        "unit-stop",
        "signal",
        "unknown-holder",
    }
    assert StopStep.WAITING not in STOP.terminal()


# Node lifecycle, evaluator status and fail class


def test_node_lifecycle(tmp_path: Path) -> None:
    _walk(NODE, ["pending", "agent", "evaluating", "done"])
    _walk(NODE, ["pending", "agent", "stopped"])
    with pytest.raises(IllegalTransition):
        NODE.check("done", "stopped")
    node = tmp_path / "node"
    assert observed_node_status(node, scored=False) is NodeStatus.PENDING
    (node / "logs").mkdir(parents=True)
    (node / "logs" / "stdout.log").write_text("", encoding="utf-8")
    assert observed_node_status(node, scored=False) is NodeStatus.AGENT
    (node / "eval").mkdir()
    assert observed_node_status(node, scored=False) is NodeStatus.EVALUATING
    assert observed_node_status(node, scored=True) is NodeStatus.DONE


def test_evaluator_status_decides_the_fail_class() -> None:
    assert eval_fail_class("guard_violation") is FailClass.IMPLEMENTATION_FAILURE
    assert eval_fail_class("evaluator_failure") is FailClass.RUNTIME_FAILURE
    assert eval_fail_class(None) is FailClass.RUNTIME_FAILURE


def test_recorded_node_outcomes_load() -> None:
    scores = _json_files("score.json")
    for score in scores:
        if "fail_class" in score:
            FailClass(score["fail_class"])
        if score.get("status") is not None:
            EvalStatus(score["status"])
    for result in _json_files("result.json"):
        if isinstance(result, dict) and result.get("status") is not None:
            EvalStatus(result["status"])


# Policy-development steps


def test_step_paths() -> None:
    _walk(STEP, ["rejected", "not_better", "valid", "accepted"])
    _walk(STEP, ["rejected", "rejected", "timeout_salvaged"])
    _walk(STEP, ["incumbent"])
    with pytest.raises(IllegalTransition):
        STEP.check("not_better", "not_better")
    with pytest.raises(IllegalTransition):
        STEP.check("timeout", "timeout_salvaged")


def test_recorded_step_statuses_load() -> None:
    for verdict in _json_files("verdict.json"):
        assert STEP.parse(verdict["status"]) in RECORDED
    for selection in _json_files("selection.json"):
        for variant in selection.get("variants") or []:
            assert STEP.parse(variant["status"]) in RECORDED
