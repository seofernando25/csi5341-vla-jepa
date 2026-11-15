"""Durable Dream-RSI loop: resume, stop rules, locking, atomic state, and the detached-run service."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rsi import durable_loop
from rsi.__main__ import main
from rsi.agent_backends import AgentSpec
from rsi.durable_loop import PolicyMismatchError, run_dream_rsi_loop
from rsi.service import (
    LoopLockedError,
    StudyLock,
    load_loop_state,
    lock_holder,
    start,
    status,
    stop,
    systemd_argv,
    unit_name,
    write_json_atomic,
)

AGENT = AgentSpec(backend="claude", model="test-model", effort=None, tools="")
REPO = Path(__file__).resolve().parents[1]


class Calls:
    """Records stage invocations by kind and cycle id."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def count(self, kind: str, cid: str) -> int:
        return self.log.count((kind, cid))


def fake_cycle(study: Path, calls: Calls, fail_on: str | None = None) -> Any:
    """Cycle stand-in: writes the manifest the real cycle writes last; raises once on `fail_on`."""
    state = {"failed": False}

    def run(**kw: Any) -> dict[str, Any]:
        cid = kw["cycle_id"]
        calls.log.append(("cycle", cid))
        if cid == fail_on and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("simulated crash")
        manifest = {"best_score": 0.5, "agent": {"cost_usd": 0.01}}
        write_json_atomic(study / "cycles" / cid / "live_cycle_manifest.json", manifest)
        return manifest

    return run


def fake_dev(calls: Calls) -> Any:
    """Policy-development stand-in: writes a distinct policy file and selection.json."""

    def run(trees: list[object], **kw: Any) -> dict[str, Any]:
        cid = kw["cycle_id"]
        calls.log.append(("dev", cid))
        out = kw["output_dir"]
        policy = out / "selected.py"
        policy.parent.mkdir(parents=True, exist_ok=True)
        policy.write_text(f"# selected for {cid}\n", encoding="utf-8")
        summary = {"selected_policy_path": str(policy), "agent": {"cost_usd": 0.02}}
        write_json_atomic(out / "selection.json", summary)
        return summary

    return run


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Study dir whose policy development always sees one tree."""
    monkeypatch.setattr(durable_loop, "_load_trees", lambda _root: [object()])
    return tmp_path / "study"


def run_loop(study: Path, calls: Calls, *, cycles: int = 3, fail_on: str | None = None, **kw: Any):
    return run_dream_rsi_loop(
        study_dir=study,
        cycles=cycles,
        agent=AGENT,
        cycle_fn=fake_cycle(study, calls, fail_on),
        policy_dev_fn=fake_dev(calls),
        **kw,
    )


def test_crash_then_rerun_resumes_from_unfinished_stage(study: Path) -> None:
    calls = Calls()
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_loop(study, calls, fail_on="iter_002")
    assert load_loop_state(study)["status"] == "error"

    summary = run_loop(study, calls)

    assert calls.count("cycle", "iter_000") == 1
    assert calls.count("dev", "iter_000") == 1
    assert calls.count("cycle", "iter_001") == 1
    assert calls.count("dev", "iter_001") == 1
    assert calls.count("cycle", "iter_002") == 2
    assert calls.count("dev", "iter_002") == 1
    state = load_loop_state(study)
    assert state is not None
    assert state["status"] == "done"
    assert sum(it["cycle"]["status"] == "done" for it in state["iterations"]) == 3
    assert sum(it["policy_dev"]["status"] == "done" for it in state["iterations"]) == 3
    assert summary["status"] == "done"


def test_policy_of_iteration_is_recorded_and_fed_forward(study: Path) -> None:
    calls = Calls()
    run_loop(study, calls, cycles=2)
    state = load_loop_state(study)
    assert state is not None
    first_out = state["iterations"][0]["policy_dev"]
    assert state["iterations"][1]["policy_in"]["path"] == first_out["selected_policy_path"]
    assert state["iterations"][1]["policy_in"]["sha256"] == first_out["sha256"]


def test_repeating_finished_loop_calls_nothing(study: Path) -> None:
    calls = Calls()
    run_loop(study, calls)
    before = list(calls.log)
    summary = run_loop(study, calls)
    assert calls.log == before
    assert summary["status"] == "done"


def test_policy_sha_mismatch_stops_naming_the_file(study: Path) -> None:
    calls = Calls()
    with pytest.raises(RuntimeError):
        run_loop(study, calls, cycles=2, fail_on="iter_001")
    state = load_loop_state(study)
    assert state is not None
    policy = Path(state["iterations"][0]["policy_dev"]["selected_policy_path"])
    policy.write_text("# tampered\n", encoding="utf-8")

    with pytest.raises(PolicyMismatchError, match=re.escape(str(policy))):
        run_loop(study, calls, cycles=2)
    assert calls.count("cycle", "iter_001") == 1


def test_second_loop_on_same_study_is_refused(study: Path) -> None:
    calls = Calls()
    with StudyLock(study), pytest.raises(LoopLockedError, match=str(os.getpid())):
        run_loop(study, calls)
    assert calls.log == []


@pytest.mark.parametrize(
    ("kwargs", "touch_stop"),
    [({}, True), ({"max_hours": 0.0}, False), ({"max_agent_usd": 0.0}, False)],
    ids=["stop-file", "max-hours-0", "max-agent-usd-0"],
)
def test_stop_rules_stop_cleanly_before_next_stage(
    study: Path, kwargs: dict[str, Any], touch_stop: bool
) -> None:
    study.mkdir(parents=True)
    if touch_stop:
        (study / "STOP").touch()
    calls = Calls()
    summary = run_loop(study, calls, **kwargs)
    assert calls.log == []
    state = load_loop_state(study)
    assert state is not None
    assert state["status"] == "stopped"
    assert state["stop_reason"]
    assert summary["stop_reason"] == state["stop_reason"]
    assert lock_holder(study) is None


def test_atomic_write_keeps_previous_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"a": 1})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_loop_state_records_run_args_and_provenance(study: Path) -> None:
    calls = Calls()
    run_loop(
        study,
        calls,
        cycles=1,
        probe_budget=3,
        max_parallelism=2,
        promote_top_k=0,
        max_hours=1.0,
        max_agent_usd=2.0,
    )
    state = load_loop_state(study)
    assert state is not None
    assert state["run"] == {
        "probe_budget": 3,
        "max_parallelism": 2,
        "max_hours": 1.0,
        "max_agent_usd": 2.0,
        "promote_top_k": 0,
    }
    assert state["first_run"] == state["run"]
    prov = state["provenance"]
    assert set(prov) >= {
        "code_revision",
        "code_dirty",
        "protocol_sha256",
        "checkpoint_revision",
        "suite",
        "seed",
        "gpu",
        "environment",
    }
    head = _git(REPO, "rev-parse", "HEAD")
    dirty = bool(_git(REPO, "status", "--porcelain"))
    assert prov["code_dirty"] is dirty
    assert prov["code_revision"] == (None if dirty else head)
    assert state["iterations"][0]["cycle"]["agent_cost_usd"] == 0.01
    assert state["iterations"][0]["cycle"]["agent_turns"] is None


def test_loop_stops_when_a_cycle_reports_a_stop_reason(study: Path) -> None:
    def cycle(**kw: Any) -> dict[str, Any]:
        cid = kw["cycle_id"]
        manifest = {
            "best_score": 0.0,
            "agent": {"cost_usd": 0.01},
            "stop_reason": "agent_output_invalid",
        }
        write_json_atomic(study / "cycles" / cid / "live_cycle_manifest.json", manifest)
        return manifest

    summary = run_dream_rsi_loop(
        study_dir=study,
        cycles=3,
        agent=AGENT,
        cycle_fn=cycle,
        policy_dev_fn=fake_dev(Calls()),
    )
    assert summary["status"] == "stopped"
    assert "agent_output_invalid" in summary["stop_reason"]


def test_loop_provenance_is_taken_before_any_study_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(durable_loop, "_load_trees", lambda _root: [object()])
    repo = tmp_path / "repo"
    study = repo / "study"
    study.mkdir(parents=True)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "one")
    head = _git(repo, "rev-parse", "HEAD")

    calls = Calls()
    run_dream_rsi_loop(
        study_dir=study,
        cycles=1,
        agent=AGENT,
        cycle_fn=fake_cycle(study, calls),
        policy_dev_fn=fake_dev(calls),
    )
    state = load_loop_state(study)
    assert state is not None
    assert state["provenance"]["code_revision"] == head
    assert state["provenance"]["code_dirty"] is False
    # The loop wrote the initial policy, the lock and the cycle manifest after that reading.
    assert _git(repo, "status", "--porcelain")


def test_budget_guard_stops_when_agent_cost_is_unknown(study: Path) -> None:
    def cycle(**kw: Any) -> dict[str, Any]:
        cid = kw["cycle_id"]
        manifest = {"best_score": 0.0, "agent": {"cost_usd": None, "num_turns": 4}}
        write_json_atomic(study / "cycles" / cid / "live_cycle_manifest.json", manifest)
        return manifest

    summary = run_dream_rsi_loop(
        study_dir=study,
        cycles=2,
        agent=AGENT,
        cycle_fn=cycle,
        policy_dev_fn=fake_dev(Calls()),
        max_agent_usd=2.0,
    )
    assert summary["status"] == "stopped"
    assert "unknown" in summary["stop_reason"]
    state = load_loop_state(study)
    assert state is not None
    assert state["iterations"][0]["cycle"]["agent_turns"] == 4


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_lock_released_on_exit(tmp_path: Path) -> None:
    with StudyLock(tmp_path):
        assert lock_holder(tmp_path) == str(os.getpid())
    assert lock_holder(tmp_path) is None


def test_start_argv_uses_unit_restart_log_and_minimal_env(tmp_path: Path) -> None:
    study = tmp_path / "study"
    env = {"PATH": "/usr/bin", "HOME": "/home/u", "SECRET": "x"}
    argv = systemd_argv(
        study, ["--cycles", "3"], repo=Path("/repo"), python="/venv/bin/python", env=env
    )
    assert argv[:3] == ["systemd-run", "--user", "--unit"]
    assert argv[3] == unit_name(study)
    assert argv[3].startswith("rsi-") and argv[3].endswith("study")
    assert "--property=Restart=on-failure" in argv
    assert "--property=RestartSec=30" in argv
    assert "--property=StartLimitBurst=5" in argv
    assert f"--property=StandardOutput=append:{study / 'loop.log'}" in argv
    setenv = [a for a in argv if a.startswith("--setenv=")]
    assert setenv == ["--setenv=PATH=/usr/bin", "--setenv=HOME=/home/u"]
    tail = argv[argv.index("--") + 1 :]
    assert tail == [
        "/venv/bin/python",
        "-m",
        "rsi",
        "loop",
        "--study-dir",
        str(study),
        "--cycles",
        "3",
    ]


def test_start_refuses_locked_study(tmp_path: Path) -> None:
    with StudyLock(tmp_path), pytest.raises(LoopLockedError):
        start(tmp_path, [])


def _prepared_study(study: Path) -> None:
    state = {
        "protocol": "dream-rsi-loop-state",
        "status": "running",
        "stop_reason": None,
        "cycles_target": 3,
        "start_iteration": 0,
        "initial_policy": {"path": "x", "sha256": "x"},
        "first_started_utc": "2026-01-01T00:00:00+00:00",
        "started_utc": "2026-01-01T00:00:00+00:00",
        "updated_utc": "2026-01-01T01:00:00+00:00",
        "iterations": [
            {
                "iteration": 0,
                "cycle_id": "iter_000",
                "cycle": {"status": "done", "agent_cost_usd": 0.01},
                "policy_dev": {"status": "done", "agent_cost_usd": 0.02},
            },
            {
                "iteration": 1,
                "cycle_id": "iter_001",
                "cycle": {"status": "running", "agent_cost_usd": None},
                "policy_dev": {"status": "pending", "agent_cost_usd": None},
            },
        ],
    }
    write_json_atomic(study / "loop_state.json", state)
    nodes = study / "cycles" / "iter_001" / "nodes"
    (nodes / "a").mkdir(parents=True)
    (nodes / "a" / "score.json").write_text(json.dumps({"score": 0.7}), encoding="utf-8")
    (nodes / "b").mkdir(parents=True)
    (study / "loop.log").write_text("\n".join(f"line {i}" for i in range(12)) + "\n")


def test_status_json_reports_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study = tmp_path / "study"
    _prepared_study(study)
    assert main(["status", "--study-dir", str(study), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is False
    assert payload["iteration"] == 1
    assert payload["stage"] == "cycle"
    assert payload["cycle_id"] == "iter_001"
    assert payload["nodes_done"] == 1
    assert payload["nodes_total"] == 2
    assert payload["best_score"] == 0.7
    assert payload["agent_cost_usd"] == pytest.approx(0.03)
    assert payload["iterations_done"] == 1
    assert payload["log_tail"][-1] == "line 11"
    assert len(payload["log_tail"]) == 10


def test_status_direct_matches_fields(tmp_path: Path) -> None:
    study = tmp_path / "study"
    _prepared_study(study)
    payload = status(study)
    assert {"running", "iteration", "stage", "nodes_done", "nodes_total", "best_score"} <= set(
        payload
    )


def test_stop_touches_stop_file_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study = tmp_path / "study"
    assert main(["stop", "--study-dir", str(study), "--grace", "0"]) == 0
    assert (study / "STOP").exists()
    assert json.loads(capsys.readouterr().out)["method"] == "not-running"
    assert stop(study, grace=0)["stopped"] is True
