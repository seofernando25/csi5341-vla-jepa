"""Dream-RSI online cycle on per-node workspaces: scripted agents and a stub evaluator, CPU only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rsi.agent_backends import AgentSpec
from rsi.api import (
    AGENT_FAILURE,
    AGENT_TIMEOUT,
    GATE_FAILED,
    GATE_NEAR_MISS,
    IMPLEMENTATION_FAILURE,
    OK,
    RUNTIME_FAILURE,
)
from rsi.expansion import (
    DigestRegistry,
    NodeContext,
    NodeJob,
    _agent_output_reason,
    evaluate_snapshot,
    expand_node,
)
from rsi.online_cycle import code_revision, run_online_cycle
from rsi.online_evidence import build_discovery_prompt
from rsi.runner import check_phase_lock, load_cycle_config
from rsi.sandbox import AgentRunResult
from rsi.workspace import code_digest, commit_root, init_store

REPO = Path(__file__).resolve().parents[1]
PRELUDE = """import pathlib, re, sys
prompt = sys.stdin.read()
cell = re.search(r"worker `([^`]+)`", prompt).group(1)
ws = pathlib.Path.cwd()
with open(__LOG__, "a") as fh:
    fh.write(cell + "\\n")
"""
EDIT_BODY = """
with (ws / "solution.py").open("a") as fh:
    fh.write(f"\\n# edit by {cell}\\n")
(ws / "proposal.md").write_text(f"proposal {cell}\\n")
"""


def _study(root: Path, *, gpu_slots: int = 2) -> Path:
    study = root / "study"
    study.mkdir(parents=True)
    proto = {
        "reserved_lock": True,
        "phases": {"smoke": {"trials": 10}},
        "gates": {"smoke_min_successes": 8, "development_min_successes": 77},
        "dream_rsi": {
            "grid_plan": {
                "max_branches": 2,
                "max_attempts_per_branch": 1,
                "max_parallelism": 2,
                "rounds_cap": 3,
            },
            "agent": {"backend": "claude", "model": "haiku", "effort": "low"},
            "discovery_tools": "read,write,edit,bash",
            "policy_dev_tools": "read,write,edit",
            "agent_timeout_s": 30,
            "eval_timeout_s": 30,
            "gpu_slots": gpu_slots,
        },
    }
    (study / "protocol.json").write_text(json.dumps(proto), encoding="utf-8")
    return study


def _agent(root: Path, body: str, name: str = "agent.py") -> list[str]:
    """A stdlib-only scripted discovery agent; it logs each cell id it was run for."""
    path = root / name
    log = root / "calls.log"
    path.write_text(PRELUDE.replace("__LOG__", repr(str(log))) + body, encoding="utf-8")
    return [sys.executable, str(path)]


def _stub_evaluator(raw: dict[str, Any] | None = None, calls: list[str] | None = None) -> Any:
    def _run(snapshot: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        if calls is not None:
            calls.append(snapshot.name)
        if raw is not None:
            return dict(raw)
        text = (snapshot / "solution.py").read_text(encoding="utf-8")
        reduction = min(0.99, len(text) / 10000.0)
        return {
            "status": "ok",
            "successes": 9,
            "episodes": 10,
            "sizes": {
                "size_reduction": reduction,
                "resident_bytes": 1000,
                "baseline_resident_bytes": 2000,
            },
        }

    return _run


def _ctx(root: Path, cmd: list[str], evaluator: Any, agent_timeout: float = 30.0) -> NodeContext:
    return NodeContext(
        cycle_id="c",
        agent=AgentSpec(backend="custom", model="none", effort=None, tools="read,write,edit,bash"),
        agent_timeout=agent_timeout,
        eval_timeout=30.0,
        gpu_slots_dir=root / "slots",
        gpu_slots=1,
        smoke_trials=10,
        smoke_gate=8,
        dev_gate=77,
        evaluator=evaluator,
        agent_cmd=cmd,
        use_bwrap=False,
    )


def _read_at(store: Path, commit: str, name: str) -> str:
    return subprocess.run(
        ["git", "--git-dir", str(store), "show", f"{commit}:{name}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _job(store: Path, root: Path, cell: str, parent_commit: str | None) -> NodeJob:
    return NodeJob(
        cell_id=cell,
        parent_id="root",
        store=store,
        parent_commit=parent_commit,
        node_dir=root / "nodes" / cell,
        evidence_nodes=[],
        tree_index={"cycle_id": "c", "nodes": []},
        best_score=0.0,
    )


def _run_one(
    root: Path, cmd: list[str], evaluator: Any, agent_timeout: float = 30.0
) -> dict[str, Any]:
    store = init_store(root / "cycle")
    base = commit_root(store)
    registry = DigestRegistry({code_digest(store, base): "root"})
    job = _job(store, root, "c-b0-a0", base)
    return expand_node(_ctx(root, cmd, evaluator, agent_timeout), job, registry)


def _calls(root: Path) -> int:
    log = root / "calls.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_cycle_end_to_end_inherits_and_records(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assert study.is_relative_to(tmp_path)
    cmd = _agent(tmp_path, EDIT_BODY)
    calls: list[str] = []
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=3,
        max_parallelism=2,
        promote_top_k=0,
        agent_cmd=cmd,
        use_bwrap=False,
        evaluator=_stub_evaluator(calls=calls),
    )
    cdir = study / "cycles" / "iter_000"
    store = cdir / "snapshots.git"
    assert (cdir / "tree.json").is_file()
    assert (cdir / "live_cycle_manifest.json").is_file()
    assert manifest["decision_rounds"] <= 3
    assert manifest["gpu_slots"] == 2
    assert manifest["totals"]["agent_timed_out"] == 0
    nodes = manifest["nodes"]
    assert nodes
    by_cell = {rec["cell_id"]: rec for rec in nodes}
    for rec in nodes:
        ndir = cdir / "nodes" / rec["cell_id"]
        assert (ndir / "score.json").is_file()
        assert (ndir / "proposal.md").is_file()
        assert (ndir / "diff.patch").is_file()
        assert rec["digest"]
        assert rec["commit"]
        assert rec["fail_class"] == OK
        assert rec["parent_commit"]
        if rec["parent_id"] != "root":
            parent = by_cell[rec["parent_id"]]
            assert rec["parent_commit"] == parent["commit"]
            parent_text = _read_at(store, parent["commit"], "solution.py")
            child_text = _read_at(store, rec["commit"], "solution.py")
            assert child_text.startswith(parent_text)
            assert f"# edit by {rec['cell_id']}" in child_text
            assert f"# edit by {rec['cell_id']}" in (ndir / "diff.patch").read_text()
    child_nodes = [r for r in nodes if r["parent_id"] != "root"]
    assert child_nodes, "the cycle should expand at least one child"
    seen_later = 0
    for rec in nodes:
        evidence = cdir / "nodes" / rec["cell_id"] / "evidence"
        assert not list(evidence.rglob("solution.py"))
        assert (evidence / "tree.json").is_file()
        for prior in (evidence / "nodes").glob("*"):
            seen_later += 1
            for name in ("proposal.md", "score.json"):
                src = cdir / "nodes" / prior.name / name
                assert (prior / name).read_text() == src.read_text()
    assert seen_later > 0
    assert not [p for p in (cdir / "nodes").rglob("*") if p.is_dir() and p.name in _LEFTOVER_DIRS]
    assert not list(cdir.rglob("solution.py"))


_LEFTOVER_DIRS = frozenset({"workspace", "export", "snapshot"})


def test_cycle_creates_nothing_under_real_studies(tmp_path: Path) -> None:
    before = sorted(str(x) for x in (REPO / "studies").rglob("*"))
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=_study(tmp_path),
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    assert manifest["probes_used"] == 1
    assert sorted(str(x) for x in (REPO / "studies").rglob("*")) == before


def test_identical_workspace_is_rejected_without_evaluation(tmp_path: Path) -> None:
    cmd = _agent(tmp_path, "(ws / 'proposal.md').write_text('no change\\n')\n")
    calls: list[str] = []
    rec = _run_one(tmp_path, cmd, _stub_evaluator(calls=calls))
    assert rec["fail_class"] == IMPLEMENTATION_FAILURE
    assert "identical" in (tmp_path / "nodes" / "c-b0-a0" / "error.txt").read_text()
    assert calls == []


def test_symlinked_solution_is_rejected_without_evaluation(tmp_path: Path) -> None:
    body = """
(ws / "solution.py").unlink()
(ws / "solution.py").symlink_to("/etc/passwd")
(ws / "proposal.md").write_text("x\\n")
"""
    calls: list[str] = []
    rec = _run_one(tmp_path, _agent(tmp_path, body), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == IMPLEMENTATION_FAILURE
    assert calls == []


NONZERO_AGENT = """
import sys
sys.stderr.write("Error: 401 {\\"detail\\":\\"Invalid API Key\\"}\\n")
sys.exit(1)
"""


def test_nonzero_agent_exit_is_agent_failure_without_evaluation(tmp_path: Path) -> None:
    calls: list[str] = []
    rec = _run_one(tmp_path, _agent(tmp_path, NONZERO_AGENT), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == AGENT_FAILURE
    assert rec["evaluated"] is False
    assert rec["agent_returncode"] == 1
    assert rec["agent_backend"] == "custom"
    assert "exited 1" in rec["error"]
    assert "Invalid API Key" in rec["error"]
    assert calls == []
    assert not (tmp_path / "nodes" / "c-b0-a0" / "workspace").exists()
    assert (tmp_path / "nodes" / "c-b0-a0" / "error.txt").is_file()


def test_nonzero_agent_exit_after_valid_outputs_is_still_agent_failure(tmp_path: Path) -> None:
    calls: list[str] = []
    body = EDIT_BODY + NONZERO_AGENT
    rec = _run_one(tmp_path, _agent(tmp_path, body), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == AGENT_FAILURE
    assert rec["commit"] is None
    assert calls == []


def test_zero_exit_agent_with_stderr_is_evaluated(tmp_path: Path) -> None:
    body = 'import sys\nsys.stderr.write("warning: model renamed\\n")\n' + EDIT_BODY
    calls: list[str] = []
    rec = _run_one(tmp_path, _agent(tmp_path, body), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == OK
    assert calls == ["export"]


def test_cycle_records_agent_failures_without_crashing(tmp_path: Path) -> None:
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=_study(tmp_path),
        probe_budget=2,
        max_parallelism=2,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, NONZERO_AGENT),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    nodes = manifest["nodes"]
    assert nodes
    assert {rec["fail_class"] for rec in nodes} == {AGENT_FAILURE}
    assert all(rec["evaluated"] is False for rec in nodes)
    assert manifest["totals"]["agent_timed_out"] == 0
    assert (tmp_path / "study" / "cycles" / "iter_000" / "live_cycle_manifest.json").is_file()


def test_timeout_after_valid_outputs_is_still_evaluated(tmp_path: Path) -> None:
    body = EDIT_BODY + "import time\ntime.sleep(60)\n"
    calls: list[str] = []
    rec = _run_one(
        tmp_path, _agent(tmp_path, body), _stub_evaluator(calls=calls), agent_timeout=3.0
    )
    assert rec["agent_timed_out"] is True
    assert rec["fail_class"] == OK
    assert calls == ["export"]


def test_timeout_without_valid_outputs_is_agent_timeout(tmp_path: Path) -> None:
    body = "import time\ntime.sleep(60)\n"
    calls: list[str] = []
    rec = _run_one(
        tmp_path, _agent(tmp_path, body), _stub_evaluator(calls=calls), agent_timeout=2.0
    )
    assert rec["fail_class"] == AGENT_TIMEOUT
    assert rec["agent_timed_out"] is True
    assert calls == []


@pytest.mark.parametrize(
    ("raw", "fail_class", "needle"),
    [
        (
            {"status": "solution_error", "error": "boom in solution"},
            IMPLEMENTATION_FAILURE,
            "boom in solution",
        ),
        (
            {"status": "size_violation", "error": "meta tensor"},
            IMPLEMENTATION_FAILURE,
            "meta tensor",
        ),
        ({"status": "evaluator_failure", "error": "timed out"}, RUNTIME_FAILURE, "timed out"),
        (
            {"status": "ok", "successes": 7, "episodes": 10, "sizes": {"size_reduction": 0.5}},
            GATE_NEAR_MISS,
            "smoke=7/10<8",
        ),
        (
            {"status": "ok", "successes": 2, "episodes": 10, "sizes": {"size_reduction": 0.5}},
            GATE_FAILED,
            "smoke=2/10<8",
        ),
    ],
)
def test_evaluator_results_map_to_fail_classes(
    tmp_path: Path, raw: dict[str, Any], fail_class: str, needle: str
) -> None:
    cmd = _agent(tmp_path, EDIT_BODY)
    rec = _run_one(tmp_path, cmd, _stub_evaluator(raw=raw))
    assert rec["fail_class"] == fail_class
    if fail_class != OK:
        assert rec["score"] == 0.0
    text = (tmp_path / "nodes" / "c-b0-a0" / "error.txt").read_text()
    assert needle in text


def test_resume_does_not_rerun_finished_nodes(tmp_path: Path) -> None:
    study = _study(tmp_path)
    cmd = _agent(tmp_path, EDIT_BODY)
    kwargs: dict[str, Any] = {
        "cycle_id": "iter_000",
        "study_dir": study,
        "probe_budget": 2,
        "max_parallelism": 2,
        "promote_top_k": 0,
        "agent_cmd": cmd,
        "use_bwrap": False,
    }
    run_online_cycle(**kwargs, evaluator=_stub_evaluator())
    first = _calls(tmp_path)
    assert first > 0
    calls: list[str] = []
    run_online_cycle(**kwargs, evaluator=_stub_evaluator(calls=calls))
    assert _calls(tmp_path) == first
    assert calls == []


def test_phase_lock_refuses_final_before_any_process(tmp_path: Path) -> None:
    study = _study(tmp_path)
    calls: list[str] = []
    with pytest.raises(ValueError, match="final"):
        evaluate_snapshot(_stub_evaluator(calls=calls), tmp_path, "final", tmp_path / "out")
    with pytest.raises(ValueError, match="final"):
        check_phase_lock("final", study)
    assert calls == []


def test_cycle_config_reads_protocol_constants(tmp_path: Path) -> None:
    cfg = load_cycle_config(_study(tmp_path, gpu_slots=3))
    assert cfg.gpu_slots == 3
    assert cfg.discovery_tools == "read,write,edit,bash"
    assert cfg.policy_dev_tools == "read,write,edit"
    assert cfg.smoke_gate == 8 and cfg.dev_gate == 77


SOLUTION_ONLY = """
with (ws / "solution.py").open("a") as fh:
    fh.write("\\n# edit without a proposal\\n")
"""

PLACEHOLDER_ONLY = """
import sys
sys.stdout.write("[object Object]" * 400 + "\\n")
"""


def test_task_prompt_requires_both_outputs() -> None:
    prompt = build_discovery_prompt(
        cell_id="c-b0-a0",
        cycle_id="c",
        parent_id="root",
        smoke_trials=10,
        smoke_gate=8,
        best_score=0.0,
        gpu_slots=2,
        minutes=30,
    )
    assert "proposal.md" in prompt and "solution.py" in prompt
    flat = " ".join(prompt.split())
    assert "A missing or unreadable `proposal.md` fails the node" in flat
    assert "both files must exist" in flat.lower()


def test_missing_proposal_fails_with_the_contract_message(tmp_path: Path) -> None:
    calls: list[str] = []
    rec = _run_one(tmp_path, _agent(tmp_path, SOLUTION_ONLY), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == IMPLEMENTATION_FAILURE
    assert rec["evaluated"] is False
    assert calls == []
    error = (tmp_path / "nodes" / "c-b0-a0" / "error.txt").read_text(encoding="utf-8")
    assert "proposal.md" in error
    assert "solution.py" in error
    assert "both are required" in error
    assert str(tmp_path) not in error


def test_placeholder_only_agent_output_is_agent_failure_without_evaluation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    rec = _run_one(tmp_path, _agent(tmp_path, PLACEHOLDER_ONLY), _stub_evaluator(calls=calls))
    assert rec["fail_class"] == AGENT_FAILURE
    assert rec["agent_output_invalid"] is True
    assert rec["evaluated"] is False
    assert rec["commit"] is None
    assert "object Object" in rec["error"]
    assert calls == []
    error = (tmp_path / "nodes" / "c-b0-a0" / "error.txt").read_text(encoding="utf-8")
    assert "placeholders" in error


def _agent_result(text: str, stdout: str) -> AgentRunResult:
    return AgentRunResult(
        returncode=0,
        stdout=stdout,
        stderr="",
        elapsed_seconds=1.0,
        sandboxed=False,
        logs_dir="",
        backend="pi",
        model="m",
        effort="max",
        num_turns=5,
        cost_usd=0.01,
        is_error=False,
        text=text,
    )


def test_usable_final_text_with_placeholder_thinking_is_not_flagged() -> None:
    stdout = "[object Object]" * 1000
    text = "I wrote proposal.md and solution.py; check.py reports 0.41 removed."
    assert _agent_output_reason(_agent_result(text, stdout)) is None


def test_missing_or_placeholder_final_text_falls_back_to_the_stream() -> None:
    stdout = "[object Object]" * 1000
    for text in ("", "[", "[object Object]" * 20):
        reason = _agent_output_reason(_agent_result(text, stdout))
        assert reason is not None, text
        assert "object Object" in reason


def test_healthy_stream_is_not_flagged() -> None:
    res = _agent_result("I wrote proposal.md and solution.py.", "step 1\nstep 2\n")
    assert _agent_output_reason(res) is None


def test_cycle_stops_after_a_batch_of_placeholder_outputs(tmp_path: Path) -> None:
    study = _study(tmp_path)
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=4,
        max_parallelism=2,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, PLACEHOLDER_ONLY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    assert manifest["stop_reason"] == "agent_output_invalid"
    assert "placeholder" in (manifest["solve_error"] or "")
    assert 0 < manifest["probes_used"] < 4
    assert {rec["fail_class"] for rec in manifest["nodes"]} == {AGENT_FAILURE}
    assert all(rec["agent_output_invalid"] for rec in manifest["nodes"])


def test_error_paths_are_relative_in_score_and_manifest(tmp_path: Path) -> None:
    study = _study(tmp_path)

    def evaluator(snapshot: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        return {"status": "solution_error", "error": f"boom in {snapshot}/solution.py"}

    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=evaluator,
    )
    error = manifest["nodes"][0]["error"]
    assert "<node>/export/solution.py" in error
    assert str(study) not in error
    score = study / "cycles" / "iter_000" / "nodes" / manifest["nodes"][0]["cell_id"] / "score.json"
    assert str(study) not in score.read_text(encoding="utf-8")


def test_development_record_redacts_host_paths(tmp_path: Path) -> None:
    """`development.json` is committed evidence, so it must carry no host path either."""
    study = _study(tmp_path)
    stub = _stub_evaluator()

    def evaluator(snapshot: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        if phase == "development":
            return {"status": "solution_error", "error": f"boom in {snapshot}/solution.py"}
        return stub(snapshot, phase, out_dir)

    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=1,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=evaluator,
    )
    promoted = manifest["development_promotions"]
    assert len(promoted) == 1
    assert "<node>/export/solution.py" in promoted[0]["error"]
    path = study / "cycles" / "iter_000" / "nodes" / promoted[0]["cell_id"] / "development.json"
    text = path.read_text(encoding="utf-8")
    assert str(study) not in text
    assert "/home/" not in text
    assert "/tmp/opencode/" not in text


def test_solve_error_is_redacted_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver exception can name any file it touched; the manifest must not quote the path."""
    study = _study(tmp_path)

    class ExplodingPolicy:
        beta = None

        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def solve(self, view: Any, budget: int) -> None:
            raise RuntimeError(f"cannot read {study}/protocol.json")

    monkeypatch.setattr("rsi.online_cycle.load_policy_class", lambda *a, **k: ExplodingPolicy)
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    error = manifest["solve_error"] or ""
    assert "<study>/protocol.json" in error
    assert str(study) not in error
    assert manifest["stop_reason"] is None
    text = (study / "cycles" / "iter_000" / "live_cycle_manifest.json").read_text(encoding="utf-8")
    assert str(study) not in text
    assert "/home/" not in text
    assert "/tmp/opencode/" not in text


def test_cycle_records_contain_no_host_paths(tmp_path: Path) -> None:
    study = _study(tmp_path)
    run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=2,
        max_parallelism=2,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, SOLUTION_ONLY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    root = study / "cycles" / "iter_000"
    # Raw agent transcripts are ignored by `.gitignore` and never committed, so they are not
    # scanned here; every other file in the cycle tree can reach the post-run commit.
    kept = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "snapshots.git" not in path.relative_to(root).parts
        and path.name not in {"stdout.log", "stderr.log"}
    ]
    assert kept
    for path in kept:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "/home/" not in text, path
        assert "/tmp/opencode/" not in text, path
        assert str(study) not in text, path


def test_post_run_tree_is_clean_after_the_evidence_commit(tmp_path: Path) -> None:
    """A cycle leaves raw transcripts and scratch; the post-run commit must take only evidence."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text(
        (REPO / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    cycle = repo / "studies" / "program-v1" / "cycles" / "iter_000"
    node = cycle / "nodes" / "c-b0-a0"
    kept = [
        cycle / "tree.json",
        cycle / "live_cycle_manifest.json",
        node / "score.json",
        node / "development.json",
        node / "proposal.md",
        node / "diff.patch",
        node / "logs" / "command.json",
    ]
    raw = [
        cycle / "grid_plan.json",
        cycle / "snapshots.git" / "HEAD",
        cycle / "gpu-slots" / "slot_0",
        node / "workspace" / "solution.py",
        node / "export" / "solution.py",
        node / "evidence" / "TASK.md",
        node / "eval" / "result.json",
        node / "development" / "result.json",
        node / "logs" / "stdout.log",
        node / "logs" / "stderr.log",
        node / "logs" / "agent_result.json",
    ]
    for path in kept + raw:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "post-run")
    assert _git(repo, "status", "--short") == ""
    tracked = set(_git(repo, "ls-files").splitlines())
    for path in kept:
        assert path.relative_to(repo).as_posix() in tracked, path
    for path in raw:
        assert path.relative_to(repo).as_posix() not in tracked, path


def test_development_record_carries_per_task_scores(tmp_path: Path) -> None:
    per_task = {str(task): {"successes": 1 if task < 8 else 0, "episodes": 1} for task in range(10)}
    raw = {
        "status": "ok",
        "successes": 8,
        "episodes": 10,
        "sizes": {"size_reduction": 0.4, "resident_bytes": 1000, "baseline_resident_bytes": 2000},
        "rollout": {"per_task": per_task},
    }
    study = _study(tmp_path)
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=1,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=_stub_evaluator(raw=raw),
    )
    cell = manifest["development_promotions"][0]["cell_id"]
    path = study / "cycles" / "iter_000" / "nodes" / cell / "development.json"
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert set(record["per_task"]) == {str(task) for task in range(10)}
    assert record["per_task"]["0"] == {"successes": 1, "episodes": 1}
    assert record["per_task"]["9"] == {"successes": 0, "episodes": 1}
    assert all(set(counts) == {"successes", "episodes"} for counts in record["per_task"].values())
    assert str(study) not in text
    assert "/home/" not in text and "/tmp/opencode/" not in text


def test_manifest_records_run_args_and_provenance(tmp_path: Path) -> None:
    study = _study(tmp_path)
    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=2,
        max_parallelism=2,
        promote_top_k=0,
        max_hours=1.5,
        max_agent_usd=2.0,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    assert manifest["run"] == {
        "probe_budget": 2,
        "max_parallelism": 2,
        "max_hours": 1.5,
        "max_agent_usd": 2.0,
        "promote_top_k": 0,
    }
    prov = manifest["provenance"]
    assert set(prov) == {
        "code_revision",
        "code_dirty",
        "protocol_sha256",
        "checkpoint",
        "checkpoint_revision",
        "suite",
        "seed",
        "gpu",
        "environment",
    }
    assert (
        prov["protocol_sha256"]
        == hashlib.sha256((study / "protocol.json").read_bytes()).hexdigest()
    )
    assert prov["checkpoint_revision"] is None
    assert prov["suite"] is None and prov["seed"] is None
    assert set(prov["environment"]) == {"python", "torch", "cuda"}
    assert set(prov["gpu"]) == {"name", "count"}
    head = _git(REPO, "rev-parse", "HEAD")
    dirty = bool(_git(REPO, "status", "--porcelain"))
    assert prov["code_dirty"] is dirty
    assert prov["code_revision"] == (None if dirty else head)
    assert len(manifest["policy_sha256"]) == 64
    assert not manifest["policy_path"].startswith("/")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_code_revision_is_head_only_on_a_clean_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "one")
    head = _git(repo, "rev-parse", "HEAD")
    assert code_revision(repo) == {"code_revision": head, "code_dirty": False}
    (repo / "a.txt").write_text("b\n", encoding="utf-8")
    assert code_revision(repo) == {"code_revision": None, "code_dirty": True}


def test_provenance_is_taken_before_the_cycle_writes_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    study = _study(repo)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "one")
    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "status", "--porcelain") == ""

    manifest = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study,
        probe_budget=1,
        max_parallelism=1,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path, EDIT_BODY),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
    )
    prov = manifest["provenance"]
    assert prov["code_revision"] == head
    assert prov["code_dirty"] is False
    # The cycle did write evidence into the repo; the revision above is the pre-write state.
    assert _git(repo, "status", "--porcelain")


def test_cli_help_has_no_vocab_mode() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "rsi", "cycle", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "--vocab-mode" not in out.stdout
    assert "--direction" not in out.stdout


def test_rsi_package_has_no_recipe_menu() -> None:
    hits = [
        f"{p.relative_to(REPO)}:{i}"
        for p in sorted((REPO / "rsi").rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"recipe", line, re.IGNORECASE)
    ]
    assert hits == []
