"""Tests for the policy audit, the sandboxed replay worker and the sequential development loop."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rsi import durable_loop, policy_dev
from rsi.agent_backends import AgentSpec, agent_record
from rsi.objective import DEFAULT_BETA_GRID, evaluate_policy_on_tree
from rsi.policies.optimal import OptimalPolicy
from rsi.policy_audit import (
    PolicyAuditError,
    audit_no_world_literals,
    audit_policy_source,
    load_policy_class,
)
from rsi.policy_dev import _replay_policy, develop_policies_llm
from rsi.replay_worker import save_tree_json
from rsi.sandbox import REPO_ROOT, SandboxTimeoutError, build_bwrap_argv, run_sandboxed_agent
from rsi.synth import scenario_suite
from rsi.tree import ReplayQuestion

INCUMBENT = REPO_ROOT / "rsi" / "policies" / "optimal.py"
TRAIN = scenario_suite("train")
requires_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
    reason="bwrap or repo .venv not available",
)

FAKE_AGENT = """
import os, pathlib, sys, time
mode, calls, extra = sys.argv[1], sys.argv[2], sys.argv[3]
with open(calls, "a") as fh:
    fh.write(mode + "\\n")
with open(calls) as fh:
    n = len(fh.read().splitlines())
method = pathlib.Path("method.py")
if mode == "noop":
    pass
elif mode == "liar":
    pathlib.Path("result.json").write_text('{"mean_v": 99.0, "worlds": {}}')
    pathlib.Path("scaling_curve.json").write_text('{"mean_v": 99.0}')
elif mode == "broken":
    method.write_text("def broken(:\\n")
elif mode == "import_os_once":
    method.write_text(("import os\\n" + pathlib.Path(extra).read_text()) if n == 1 else pathlib.Path(extra).read_text())
elif mode == "better":
    method.write_text(pathlib.Path(extra).read_text())
elif mode == "better_then_log":
    if n == 1:
        method.write_text(pathlib.Path(extra).read_text())
    else:
        fh = open(calls, "a")
        seen = os.path.exists(os.environ["RSI_EVIDENCE_DIR"] + "/history/step_01/scaling_curve.json")
        fh.write(f"seen_history={seen}\\n")
        fh.close()
elif mode == "symlink":
    method.unlink(missing_ok=True)
    os.symlink("/etc/passwd", method)
elif mode == "sleep":
    time.sleep(60)
elif mode == "better_then_sleep":
    method.write_text(pathlib.Path(extra).read_text())
    time.sleep(60)
elif mode == "broken_then_sleep":
    method.write_text("def broken(:\\n")
    time.sleep(60)
elif mode == "delete_then_sleep":
    method.unlink(missing_ok=True)
    time.sleep(60)
"""

WRITE_PROBE = """
import json, pathlib
targets = ["/tmp/work/probe.txt", "/tmp/pkgs/probe.txt", "/tmp/evidence/probe.txt",
           "/tmp/repo/evaluation/probe.txt", "/tmp/venv/probe.txt"]
out = {}
for target in targets:
    try:
        pathlib.Path(target).write_text("x")
        out[target] = "ok"
    except OSError as exc:
        out[target] = type(exc).__name__
pathlib.Path("/tmp/work/writes.json").write_text(json.dumps(out))
"""


def _fake_agent(tmp_path: Path, mode: str, extra: Path | None = None) -> list[str]:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    extra_file = extra or INCUMBENT
    return [sys.executable, str(script), mode, str(tmp_path / "calls.txt"), str(extra_file)]


def _calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def _better_policy(tmp_path: Path) -> Path:
    """The incumbent with patience reduced at beta=0; scores higher on the train suite."""
    src = INCUMBENT.read_text(encoding="utf-8")
    old = '"patience_rounds": max(1, round(1 + 3 * b)),'
    assert src.count(old) == 1
    path = tmp_path / "better.py"
    path.write_text(src.replace(old, '"patience_rounds": max(1, round(3 * b)),'), encoding="utf-8")
    return path


def _policy_with_body(stmt: str) -> str:
    return (
        "from rsi.api import LLMDesignedMethod\n\n"
        "class OptimalPolicy(LLMDesignedMethod):\n"
        '    NAME = "OptimalPolicy"\n\n'
        "    def solve(self, question, budget=None):\n"
        f"        {stmt}\n"
        "        return None\n\n"
        "    def plan_grid(self, context):\n"
        "        raise TypeError('unused')\n"
    )


def _recorded_score(tree) -> float:
    for cid in tree.order:
        score = tree.nodes[cid].score
        if (
            score is not None
            and score not in (0.0, 0.5, 1.0)
            and len(repr(score).split(".")[-1]) >= 3
        ):
            return float(score)
    raise AssertionError("no recorded score with three decimals")


def test_audit_accepts_incumbent() -> None:
    audit_policy_source(INCUMBENT.read_text(encoding="utf-8"))
    audit_no_world_literals(INCUMBENT.read_text(encoding="utf-8"), TRAIN[:4])


@pytest.mark.parametrize(
    "stmt",
    [
        "import os",
        "import typing as t; x = t.sys",
        "x = question.budget_spent",
        "x = question.tree",
        "x = question.events",
        "x = getattr(question, 'tree')",
        "x = type(question)",
        "x = open('f')",
        "x = question._revealed",
        "x = question.__class__",
    ],
)
def test_audit_rejects(stmt: str) -> None:
    with pytest.raises(PolicyAuditError):
        audit_policy_source(_policy_with_body(stmt))


def test_audit_accepts_own_private_and_super_init() -> None:
    src = (
        "from rsi.api import LLMDesignedMethod\n\n"
        "class OptimalPolicy(LLMDesignedMethod):\n"
        '    NAME = "OptimalPolicy"\n\n'
        "    def __init__(self, config=None):\n"
        "        super().__init__(config)\n"
        "        self._helper()\n\n"
        "    def _helper(self):\n"
        "        return 1\n\n"
        "    def solve(self, question, budget=None):\n"
        "        return None\n\n"
        "    def plan_grid(self, context):\n"
        "        raise TypeError('unused')\n"
    )
    audit_policy_source(src)


def test_world_literals_rejected() -> None:
    tree = TRAIN[0]
    cell = tree.order[0]
    score = _recorded_score(tree)
    with pytest.raises(PolicyAuditError, match="cell or world id"):
        audit_no_world_literals(_policy_with_body(f"x = {cell!r}"), [tree])
    with pytest.raises(PolicyAuditError, match="node score"):
        audit_no_world_literals(_policy_with_body(f"x = {score!r}"), [tree])


def test_optimal_probe_count_matches_observed_and_budget() -> None:
    for tree in TRAIN:
        for beta in DEFAULT_BETA_GRID:
            for budget in (None, 5):
                question = ReplayQuestion(tree, max_parallelism=4)
                res = OptimalPolicy({"beta": beta}).solve(question, budget=budget)
                assert res.total_probes == len(question.observed())
                assert budget is None or res.total_probes <= budget


def test_worker_matches_host_objective_and_is_deterministic(tmp_path: Path) -> None:
    trees = TRAIN[:4]
    evidence = tmp_path / "evidence"
    paths = [save_tree_json(t, evidence / f"{i}.json") for i, t in enumerate(trees)]
    host = [
        evaluate_policy_on_tree(load_policy_class(INCUMBENT), t, lambda_parallel=0.05).reward
        for t in trees
    ]
    result, traces = _replay_policy(
        INCUMBENT,
        paths,
        ws_dir=tmp_path / "r1",
        evidence_dir=evidence,
        max_parallelism=4,
        lambda_parallel=0.05,
        use_bwrap=False,
    )
    again, traces_again = _replay_policy(
        INCUMBENT,
        paths,
        ws_dir=tmp_path / "r2",
        evidence_dir=evidence,
        max_parallelism=4,
        lambda_parallel=0.05,
        use_bwrap=False,
    )
    assert abs(result["mean_v"] - sum(host) / len(host)) < 1e-9
    assert result == again
    assert traces == traces_again
    assert set(result["worlds"]) == {t.world_id for t in trees}


@requires_bwrap
def test_worker_under_bwrap_matches_host(tmp_path: Path) -> None:
    trees = TRAIN[:2]
    evidence = tmp_path / "evidence"
    paths = [save_tree_json(t, evidence / f"{i}.json") for i, t in enumerate(trees)]
    result, _ = _replay_policy(
        INCUMBENT,
        paths,
        ws_dir=tmp_path / "ws",
        evidence_dir=evidence,
        max_parallelism=4,
        lambda_parallel=0.05,
        use_bwrap=True,
    )
    host = [
        evaluate_policy_on_tree(load_policy_class(INCUMBENT), t, lambda_parallel=0.05).reward
        for t in trees
    ]
    assert abs(result["mean_v"] - sum(host) / len(host)) < 1e-9


def _develop(tmp_path: Path, agent: list[str], *, m: int, timeout: float = 60.0, history=None):
    return develop_policies_llm(
        history or TRAIN[:2],
        cycle_id="t",
        output_dir=tmp_path / "out",
        m=m,
        agent_cmd=agent,
        timeout=timeout,
        use_bwrap=False,
    )


def test_loop_unchanged_agent_keeps_incumbent(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "noop"), m=3)
    assert summary["selected_slot"] == "m0_incumbent"
    statuses = [v["status"] for v in summary["variants"]]
    assert statuses == ["incumbent", "not_better", "not_better"]
    # Each flat step gets one repair attempt, as a failed validation does.
    assert len(_calls(tmp_path)) == 4


def test_loop_broken_source_rejected_without_exception(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "broken"), m=3)
    assert summary["selected_slot"] == "m0_incumbent"
    assert [v["status"] for v in summary["variants"][1:]] == ["rejected", "rejected"]
    assert all(v["attempts"] == 2 for v in summary["variants"][1:])


def test_loop_forbidden_import_gets_one_repair(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "import_os_once"), m=2)
    variant = summary["variants"][1]
    assert len(_calls(tmp_path)) == 2
    assert variant["attempts"] == 2
    assert variant["status"] == "not_better"


def test_loop_timeout_scores_the_file_left_behind(tmp_path: Path) -> None:
    """A timed-out step whose file is untouched is scored and loses to the incumbent."""
    summary = _develop(tmp_path, _fake_agent(tmp_path, "sleep"), m=2, timeout=2.0)
    variant = summary["variants"][1]
    assert variant["status"] == "timeout_salvaged"
    assert variant["attempts"] == 1
    assert variant["mean_v"] == pytest.approx(summary["incumbent_in_sample_V"])
    verdict = json.loads(
        (tmp_path / "out" / "evidence" / "history" / "step_01" / "verdict.json").read_text()
    )
    assert verdict["status"] == "timeout_salvaged"
    assert "timed out" in verdict["error"]
    assert summary["selected_slot"] == "m0_incumbent"
    assert len(_calls(tmp_path)) == 1


def test_timeout_salvage_is_accepted_when_it_beats_the_incumbent(tmp_path: Path) -> None:
    better = _better_policy(tmp_path)
    summary = _develop(
        tmp_path, _fake_agent(tmp_path, "better_then_sleep", extra=better), m=2, timeout=2.0
    )
    variant = summary["variants"][1]
    assert variant["status"] == "timeout_salvaged"
    assert variant["mean_v"] > summary["incumbent_in_sample_V"]
    assert summary["selected_slot"] == "m1_llm"
    assert (tmp_path / "out" / "selected_policy.py").read_text() == better.read_text()
    # The killed run is still counted: the step cost money and ran turns.
    assert summary["agent"]["calls"] == 1


def test_timeout_salvage_rejected_by_audit_is_a_plain_timeout(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "broken_then_sleep"), m=2, timeout=2.0)
    variant = summary["variants"][1]
    assert variant["status"] == "timeout"
    assert variant["mean_v"] is None
    assert "timed out" in variant["error"]
    assert "salvage failed" in variant["error"]
    assert summary["selected_slot"] == "m0_incumbent"


def test_timeout_without_a_candidate_is_a_plain_timeout(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "delete_then_sleep"), m=2, timeout=2.0)
    variant = summary["variants"][1]
    assert variant["status"] == "timeout"
    assert variant["mean_v"] is None
    assert summary["selected_slot"] == "m0_incumbent"


def test_loop_symlinked_method_rejected(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "symlink"), m=2)
    assert summary["variants"][1]["status"] == "rejected"
    assert summary["selected_slot"] == "m0_incumbent"


def test_loop_selects_better_policy_and_feeds_history_forward(tmp_path: Path) -> None:
    better = _better_policy(tmp_path)
    agent = _fake_agent(tmp_path, "better", extra=better)
    summary = _develop(tmp_path, agent, m=3)
    assert summary["variants"][1]["status"] == "accepted"
    assert summary["selected_slot"] == "m1_llm"
    assert summary["selected_in_sample_V"] > summary["incumbent_in_sample_V"]
    assert (tmp_path / "out" / "evidence" / "history" / "step_01" / "scaling_curve.json").exists()
    assert (tmp_path / "out" / "selected_policy.py").read_text() == better.read_text()


def test_next_step_sees_previous_scaling_curve(tmp_path: Path) -> None:
    better = _better_policy(tmp_path)
    _develop(tmp_path, _fake_agent(tmp_path, "better_then_log", extra=better), m=3)
    lines = _calls(tmp_path)
    assert lines[0] == "better_then_log"
    assert lines[-1] == "seen_history=True"


def test_agent_numbers_are_not_trusted(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "liar"), m=2)
    variant = summary["variants"][1]
    assert variant["status"] == "not_better"
    assert variant["mean_v"] == pytest.approx(summary["incumbent_in_sample_V"])
    assert summary["selected_slot"] == "m0_incumbent"
    assert summary["selected_in_sample_V"] < 99.0


def test_selected_policy_path_is_relative_to_record_root(tmp_path: Path) -> None:
    study = tmp_path / "study"
    summary = develop_policies_llm(
        TRAIN[:1],
        cycle_id="iter_000",
        output_dir=study / "policies" / "iter_000",
        m=2,
        agent_cmd=_fake_agent(tmp_path, "noop"),
        timeout=60.0,
        use_bwrap=False,
        record_root=study,
    )
    assert summary["selected_policy_path"] == "policies/iter_000/selected_policy.py"
    paths = [v["policy_path"] for v in summary["variants"] if v["policy_path"]]
    assert paths and all(not Path(path).is_absolute() for path in paths)


def test_timed_out_step_keeps_its_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(**kw: object) -> None:
        logs = Path(str(kw["logs_dir"]))
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "agent_result.json").write_text(
            json.dumps(
                {
                    "backend": "pi",
                    "model": "m",
                    "effort": "max",
                    "returncode": -1,
                    "cost_usd": 0.632185,
                    "num_turns": 27,
                    "is_error": None,
                    "text": "",
                }
            ),
            encoding="utf-8",
        )
        raise SandboxTimeoutError("Sandboxed process timed out after 1.0s")

    monkeypatch.setattr(policy_dev, "run_sandboxed_agent", timed_out)
    out = tmp_path / "out"
    evidence = out / "evidence"
    evidence.mkdir(parents=True)
    world_paths = [save_tree_json(TRAIN[0], evidence / "worlds" / "00.json")]
    run = policy_dev._Run(
        cycle_id="t",
        history=TRAIN[:1],
        world_paths=world_paths,
        evidence=evidence,
        out_root=out,
        steps=1,
        max_parallelism=2,
        lambda_parallel=0.05,
        prompt_path=policy_dev.POLICY_DEV_PROMPT,
        agent_cmd=["true"],
        agent=AgentSpec(backend="pi", model="m", effort="max", tools="read,write,edit,bash"),
        timeout=1.0,
        use_bwrap=False,
    )
    rec = policy_dev._run_agent_step(run, 1, INCUMBENT.read_text(encoding="utf-8"), 0.5)
    # The seed file is still there, so the killed step is scored: it does not beat 0.5.
    assert rec["status"] == "timeout_salvaged"
    assert "timed out" in rec["error"]
    assert [r.cost_usd for r in run.results] == [0.632185]
    assert agent_record(None, run.results)["cost_usd"] == 0.632185
    assert (out / "step_01" / "TASK.md").read_text(encoding="utf-8").startswith("# Task")
    assert (out / "step_01" / "workspace" / "replay.py").is_file()


def test_selection_counts_a_timed_out_step_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timed_out(**kw: object) -> None:
        logs = Path(str(kw["logs_dir"]))
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "agent_result.json").write_text(
            json.dumps(
                {
                    "backend": "pi",
                    "model": "m",
                    "effort": "max",
                    "returncode": -1,
                    "cost_usd": 0.632185,
                    "num_turns": 27,
                    "is_error": None,
                    "text": "",
                }
            ),
            encoding="utf-8",
        )
        raise SandboxTimeoutError("Sandboxed process timed out after 1.0s")

    monkeypatch.setattr(policy_dev, "run_sandboxed_agent", timed_out)
    summary = develop_policies_llm(
        TRAIN[:1],
        cycle_id="t",
        output_dir=tmp_path / "out",
        m=2,
        agent_cmd=["true"],
        timeout=1.0,
        use_bwrap=False,
    )
    assert summary["variants"][1]["status"] == "timeout_salvaged"
    assert summary["agent"]["cost_usd"] == 0.632185
    recorded = json.loads((tmp_path / "out" / "selection.json").read_text(encoding="utf-8"))
    assert recorded["agent"]["cost_usd"] == 0.632185


def test_prompt_states_the_real_limit_and_the_closed_lines() -> None:
    text = policy_dev._build_policy_dev_prompt(
        cycle_id="c",
        step=1,
        steps=3,
        max_parallelism=2,
        lambda_parallel=0.05,
        minutes=20.0,
        prompt_path=policy_dev.POLICY_DEV_PROMPT,
    )
    assert "20 minutes" in text
    assert "240" not in text
    assert "at most half" in text
    assert "/tmp/work/replay.py" in text
    for phrase in ("read-only", "held-out", "never used for selection", "No writes outside"):
        assert phrase in text


def test_policy_dev_source_hides_the_heldout_generator(tmp_path: Path) -> None:
    sandbox = policy_dev._agent_sandbox(tmp_path / "step")
    sources = {str(path) for path in sandbox["source_ro"]}
    assert str(REPO_ROOT / "rsi" / "synth.py") not in sources
    assert str(REPO_ROOT / "rsi") not in sources
    assert str(REPO_ROOT / "rsi" / "api.py") in sources


def test_policy_dev_agent_mounts_keep_evidence_read_only(tmp_path: Path) -> None:
    ws, ev = tmp_path / "ws", tmp_path / "evidence"
    ws.mkdir()
    ev.mkdir()
    argv = build_bwrap_argv(
        ["python", "-c", "pass"],
        workspace_dir=ws,
        evidence_dir=ev,
        gpu=True,
        **policy_dev._agent_sandbox(tmp_path / "step"),
    )
    mounts = [
        (token, argv[i + 1], argv[i + 2])
        for i, token in enumerate(argv)
        if token in ("--bind", "--ro-bind", "--dev-bind", "--ro-bind-try") and i + 2 < len(argv)
    ]
    writable = {dest for mode, _, dest in mounts if mode == "--bind"}
    assert writable == {"/tmp/work", "/tmp/pkgs", "/tmp/gpu-slots"}
    assert any(mode == "--ro-bind" and dest == "/tmp/evidence" for mode, _, dest in mounts)
    assert not any(src.endswith("synth.py") for _, src, _ in mounts)
    assert any(
        src == str(REPO_ROOT / "rsi" / "api.py") and dest == "/tmp/repo/rsi/api.py"
        for _, src, dest in mounts
    )
    assert "/tmp/pkgs:/tmp/repo" in argv


@requires_bwrap
def test_policy_dev_agent_cannot_write_outside_its_dirs(tmp_path: Path) -> None:
    summary = develop_policies_llm(
        TRAIN[:1],
        cycle_id="t",
        output_dir=tmp_path / "out",
        m=2,
        agent_cmd=[sys.executable, "-c", WRITE_PROBE],
        timeout=60.0,
        use_bwrap=True,
    )
    assert summary["variants"][1]["status"] == "not_better"
    writes = json.loads(
        (tmp_path / "out" / "step_01" / "workspace" / "writes.json").read_text(encoding="utf-8")
    )
    assert writes["/tmp/work/probe.txt"] == "ok"
    assert writes["/tmp/pkgs/probe.txt"] == "ok"
    for outside in (
        "/tmp/evidence/probe.txt",
        "/tmp/repo/evaluation/probe.txt",
        "/tmp/venv/probe.txt",
    ):
        assert writes[outside] != "ok"


@requires_bwrap
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_policy_dev_agent_can_run_uv(tmp_path: Path) -> None:
    step = tmp_path / "step_01"
    (step / "workspace").mkdir(parents=True)
    (step / "evidence").mkdir()
    res = run_sandboxed_agent(
        prompt_text="check uv",
        workspace_dir=step / "workspace",
        evidence_dir=step / "evidence",
        logs_dir=step / "logs",
        agent_cmd=["uv", "--version"],
        timeout=60.0,
        use_bwrap=True,
        gpu=True,
        **policy_dev._agent_sandbox(step),
    )
    assert res.returncode == 0
    assert res.stdout.strip().startswith("uv ")


def test_policy_timeout_defaults_to_twenty_minutes() -> None:
    assert policy_dev.DEFAULT_POLICY_TIMEOUT_S == 1200.0
    assert (
        inspect.signature(policy_dev.develop_policies_llm).parameters["timeout"].default == 1200.0
    )
    assert (
        inspect.signature(durable_loop.run_dream_rsi_loop).parameters["policy_timeout"].default
        == 1200.0
    )


def test_replay_script_scores_like_the_harness(tmp_path: Path) -> None:
    trees = TRAIN[:2]
    evidence = tmp_path / "evidence"
    paths = [save_tree_json(t, evidence / "worlds" / f"{i:02d}.json") for i, t in enumerate(trees)]
    script = tmp_path / "replay.py"
    script.write_text(policy_dev._replay_script(2, 0.05), encoding="utf-8")
    env = {
        **os.environ,
        "RSI_EVIDENCE_DIR": str(evidence),
        "RSI_WORKSPACE_DIR": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
    }
    proc = subprocess.run(
        [sys.executable, str(script), str(INCUMBENT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    printed = float(proc.stdout.strip().splitlines()[-1].split("=")[1].split()[0])
    result, _ = _replay_policy(
        INCUMBENT,
        paths,
        ws_dir=tmp_path / "r",
        evidence_dir=evidence,
        max_parallelism=2,
        lambda_parallel=0.05,
        use_bwrap=False,
    )
    assert abs(printed - result["mean_v"]) < 1e-6
