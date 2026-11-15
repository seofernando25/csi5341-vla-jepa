"""Tests for the policy audit, the sandboxed replay worker and the sequential development loop."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

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
from rsi.sandbox import REPO_ROOT
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
    assert len(_calls(tmp_path)) == 2


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


def test_loop_timeout_rejects_step(tmp_path: Path) -> None:
    summary = _develop(tmp_path, _fake_agent(tmp_path, "sleep"), m=2, timeout=2.0)
    assert summary["variants"][1]["status"] == "rejected"
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
