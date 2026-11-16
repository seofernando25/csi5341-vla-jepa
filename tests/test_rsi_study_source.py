"""The study directory is the only source of a run's protocol, gates and seed.

A Dream-RSI run names its study; the evaluator, the sandbox binds and the agent's own
`tools/check.py` must all read that study's `protocol.json` and never another study's. These
tests pin the wiring, so a run under a new study cannot silently score against program-v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.compression import solution
from rsi import expansion
from rsi.expansion import sandboxed_evaluator
from rsi.sandbox import (
    DEFAULT_STUDY,
    REPO_ROOT,
    evaluator_argv,
    repo_source_binds,
    study_protocol,
    study_rel,
)

V1 = REPO_ROOT / "studies" / "program-v1"
V2 = REPO_ROOT / "studies" / "program-v2"
# Fields a second clean-room study must share with the first, or its results are not comparable.
SHARED = (
    "checkpoint",
    "checkpoint_revision",
    "suite",
    "task_ids",
    "seed",
    "final_seed_offset",
    "development_states_per_task",
    "reserved_states_per_task",
    "max_steps",
    "action_chunk",
    "flow_steps",
    "dtype",
    "batch_size",
    "phases",
    "gates",
    "statistics",
)


def _protocol(study: Path) -> dict:
    return json.loads((study / "protocol.json").read_text(encoding="utf-8"))


def test_program_v2_shares_the_frozen_inputs_of_program_v1() -> None:
    v1, v2 = _protocol(V1), _protocol(V2)
    for field in SHARED:
        assert v2[field] == v1[field], f"{field} differs between program-v1 and program-v2"
    assert v2["reserved_lock"] is True


def test_program_v2_forbids_pooling_with_program_v1() -> None:
    v2 = _protocol(V2)
    assert "program-v1" in v2["supersedes"]
    assert v2["revisions"] == []
    assert {rule["field"] for rule in v2["starting_rules"]} == {
        "dream_rsi.policy_dev_tools + policy_dev environment/prompt",
        "dream_rsi.policy_timeout_s",
        "dream_rsi.policy_step_timeout",
        "dream_rsi.grid_plan",
        "dream_rsi.evaluator.memory_rule+prompt",
    }


def test_program_v2_starts_without_program_v1_history() -> None:
    assert V2.is_dir()
    assert not (V2 / "cycles").exists()
    assert not (V2 / "policies").exists()
    assert not (V2 / "loop_state.json").exists()


def test_study_rel_defaults_to_program_v1_and_is_none_outside_the_repo(tmp_path: Path) -> None:
    assert study_rel(None) == DEFAULT_STUDY
    assert study_rel(V2) == "studies/program-v2"
    assert study_rel(tmp_path) is None
    assert study_protocol(None) == f"{DEFAULT_STUDY}/protocol.json"
    assert study_protocol(tmp_path) is None


def test_repo_source_binds_only_the_named_studys_protocol() -> None:
    v2 = repo_source_binds(study_dir=V2)
    assert v2[(V2 / "protocol.json").resolve()] == "/tmp/repo/studies/program-v2/protocol.json"
    assert not [dst for dst in v2.values() if "program-v1" in dst]
    v1 = repo_source_binds(study_dir=V1)
    assert v1[(V1 / "protocol.json").resolve()] == "/tmp/repo/studies/program-v1/protocol.json"
    assert not [dst for dst in v1.values() if "program-v2" in dst]


def test_evaluator_argv_names_the_protocol() -> None:
    argv = evaluator_argv("smoke", "/tmp/repo/studies/program-v2/protocol.json")
    assert argv[argv.index("--protocol") + 1] == "/tmp/repo/studies/program-v2/protocol.json"
    assert argv[argv.index("--phase") + 1] == "smoke"


def test_sandboxed_evaluator_forwards_the_study_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(workspace: Path, phase: str, out: Path, **kw: object) -> dict:
        seen.update(workspace=workspace, phase=phase, out=out, **kw)
        return {"status": "ok"}

    monkeypatch.setattr(expansion, "run_sandboxed_evaluator", fake)
    run = sandboxed_evaluator(30.0, Path("/tmp/slots"), 2, study_dir=V2)
    run(Path("/tmp/ws"), "check", Path("/tmp/out"))
    assert seen["study_dir"] == V2
    assert seen["gpu_slot_count"] == 2


def test_solution_protocol_path_honours_argument_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(solution.PROTOCOL_ENV, raising=False)
    assert solution.protocol_path() == V1 / "protocol.json"
    monkeypatch.setenv(solution.PROTOCOL_ENV, str(V2 / "protocol.json"))
    assert solution.protocol_path() == V2 / "protocol.json"
    assert solution.protocol_path(tmp_path / "p.json") == tmp_path / "p.json"
