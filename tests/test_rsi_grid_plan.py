"""Cross-cycle grid planning: the policy's `plan_grid`, caps, fallback and resume (CPU only)."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from rsi.api import GridPlan, GridPlanningContext
from rsi.grid_plan import (
    GridCaps,
    PlanError,
    applied_plan,
    call_plan_grid,
    clamp_plan,
    grid_caps,
    protocol_plan,
    read_plan_record,
    validate_plan,
)
from rsi.online_cycle import run_online_cycle
from rsi.policy_dev import develop_policies_llm
from rsi.synth import scenario_suite

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_GRID = {
    "max_branches": 2,
    "max_attempts_per_branch": 1,
    "max_parallelism": 2,
    "rounds_cap": 3,
}
PROTOCOL_PLAN = GridPlan(
    max_branches=2,
    max_attempts_per_branch=1,
    max_parallelism=2,
    rounds_cap=3,
    rationale="pre-registered static grid plan",
)
CAPS = GridCaps(max_branches=2, max_attempts_per_branch=4, max_parallelism=2, rounds_cap=3)
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


def _study(root: Path) -> Path:
    study = root / "study"
    study.mkdir(parents=True)
    proto = {
        "reserved_lock": True,
        "phases": {"smoke": {"trials": 10}},
        "gates": {"smoke_min_successes": 8, "development_min_successes": 77},
        "dream_rsi": {
            "grid_plan": dict(PROTOCOL_GRID),
            "agent": {"backend": "claude", "model": "haiku", "effort": "low"},
            "discovery_tools": "read,write,edit,bash",
            "policy_dev_tools": "read,write,edit",
            "agent_timeout_s": 30,
            "eval_timeout_s": 30,
            "gpu_slots": 2,
        },
    }
    (study / "protocol.json").write_text(json.dumps(proto), encoding="utf-8")
    return study


def _agent(root: Path) -> list[str]:
    path = root / "agent.py"
    log = root / "calls.log"
    path.write_text(PRELUDE.replace("__LOG__", repr(str(log))) + EDIT_BODY, encoding="utf-8")
    return [sys.executable, str(path)]


def _stub_evaluator() -> Any:
    def _run(snapshot: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        text = (snapshot / "solution.py").read_text(encoding="utf-8")
        return {
            "status": "ok",
            "successes": 9,
            "episodes": 10,
            "sizes": {
                "size_reduction": min(0.99, len(text) / 10000.0),
                "resident_bytes": 1000,
                "baseline_resident_bytes": 2000,
            },
        }

    return _run


def _policy(root: Path, name: str, plan_body: str) -> Path:
    """A minimal auditable policy whose `plan_grid` body is supplied by the test."""
    path = root / name
    path.write_text(
        "from __future__ import annotations\n\n"
        "from rsi.api import GridPlan, LLMDesignedMethod\n\n\n"
        "class OptimalPolicy(LLMDesignedMethod):\n"
        '    NAME = "OptimalPolicy"\n\n'
        "    def solve(self, question, budget=None):\n"
        "        return None\n\n"
        "    def plan_grid(self, context):\n" + plan_body,
        encoding="utf-8",
    )
    return path


def _cycle(
    tmp_path: Path,
    study: Path,
    policy: Path,
    cycle_id: str = "iter_000",
    iteration: int = 0,
    grid_plan: GridPlan | None = None,
) -> dict[str, Any]:
    return run_online_cycle(
        cycle_id=cycle_id,
        study_dir=study,
        policy_path=policy,
        iteration=iteration,
        probe_budget=2,
        max_parallelism=2,
        promote_top_k=0,
        agent_cmd=_agent(tmp_path),
        use_bwrap=False,
        evaluator=_stub_evaluator(),
        grid_plan=grid_plan,
    )


def _context(manifests: tuple[dict[str, Any], ...] = (), iteration: int = 0):
    return GridPlanningContext(
        iteration=iteration,
        max_parallelism=2,
        fallback_plan=PROTOCOL_PLAN,
        recent_live_manifests=manifests,
    )


def _manifest(applied: dict[str, Any], *, plateaued: bool, refinement: bool) -> dict[str, Any]:
    nodes = [
        {"cell_id": "c-b0-a0", "parent_id": "root", "fail_class": "ok", "score": 0.4},
        {
            "cell_id": "c-b0-a1",
            "parent_id": "c-b0-a0",
            "fail_class": "ok" if refinement else "gate_failed",
            "score": 0.5 if refinement else 0.0,
        },
    ]
    return {
        "cycle_id": "c",
        "plateaued": plateaued,
        "nodes": nodes,
        "grid_plan": {"applied": applied},
    }


def _applied(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["grid_plan"]["applied"]


# --- the harness calls the selected policy ------------------------------------------------


def test_cycle_one_uses_the_protocol_plan_and_cycle_two_sees_one_manifest(tmp_path: Path) -> None:
    study = _study(tmp_path)
    body = (
        "        base = context.fallback_plan\n"
        "        return GridPlan(\n"
        "            max_branches=1 + len(context.recent_live_manifests),\n"
        "            max_attempts_per_branch=base.max_attempts_per_branch,\n"
        "            max_parallelism=base.max_parallelism,\n"
        "            rounds_cap=base.rounds_cap,\n"
        "            rationale=f'iter={context.iteration} "
        "manifests={len(context.recent_live_manifests)}',\n"
        "        )\n"
    )
    policy = _policy(tmp_path, "history.py", body)
    first = _cycle(tmp_path, study, policy, "iter_000")
    second = _cycle(tmp_path, study, policy, "iter_001", iteration=1)

    assert first["grid_plan"]["source"] == "policy"
    assert first["grid_plan"]["requested"]["max_branches"] == 1
    assert first["grid_plan"]["requested"]["rationale"] == "iter=0 manifests=0"
    assert second["grid_plan"]["requested"]["max_branches"] == 2
    assert second["grid_plan"]["requested"]["rationale"] == "iter=1 manifests=1"
    assert second["grid_plan"]["requested"]["max_parallelism"] == PROTOCOL_PLAN.max_parallelism


def test_plan_grid_is_called_once_per_cycle_and_not_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi import grid_plan

    calls: list[int] = []
    real = grid_plan.call_plan_grid

    def counting(policy_path: Path, context: GridPlanningContext, **kwargs: Any) -> GridPlan:
        calls.append(context.iteration)
        return real(policy_path, context, **kwargs)

    monkeypatch.setattr(grid_plan, "call_plan_grid", counting)
    study = _study(tmp_path)
    policy = _policy(
        tmp_path,
        "plain.py",
        "        return context.fallback_plan\n",
    )
    _cycle(tmp_path, study, policy, "iter_000")
    assert calls == [0]
    # A resumed cycle (its manifest was never written) reuses the recorded plan.
    (study / "cycles" / "iter_000" / "live_cycle_manifest.json").unlink()
    _cycle(tmp_path, study, policy, "iter_000")
    assert calls == [0]


def test_requested_plan_is_clamped_to_the_protocol_caps(tmp_path: Path) -> None:
    study = _study(tmp_path)
    body = (
        "        return GridPlan(\n"
        "            max_branches=99,\n"
        "            max_attempts_per_branch=99,\n"
        "            max_parallelism=99,\n"
        "            rounds_cap=99,\n"
        "            rationale='too big',\n"
        "        )\n"
    )
    manifest = _cycle(tmp_path, study, _policy(tmp_path, "big.py", body))
    record = manifest["grid_plan"]
    assert record["source"] == "policy"
    assert record["requested"]["max_branches"] == 99
    assert record["applied"] == {
        "max_branches": CAPS.max_branches,
        "max_attempts_per_branch": CAPS.max_attempts_per_branch,
        "max_parallelism": CAPS.max_parallelism,
        "rounds_cap": CAPS.rounds_cap,
        "rationale": "too big",
    }
    assert record["caps"] == {
        "max_branches": 2,
        "max_attempts_per_branch": 4,
        "max_parallelism": 2,
        "rounds_cap": 3,
    }
    assert manifest["rounds_cap"] == 3


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        (
            "        raise RuntimeError('planner exploded')\n",
            "planner exploded",
        ),
        (
            "        return GridPlan(0, 1, 1, 1, rationale='empty')\n",
            "max_branches must be positive",
        ),
        (
            "        return {'max_branches': 2}\n",
            "not GridPlan",
        ),
    ],
)
def test_unusable_plan_falls_back_to_the_protocol_plan(
    tmp_path: Path, body: str, needle: str
) -> None:
    study = _study(tmp_path)
    manifest = _cycle(tmp_path, study, _policy(tmp_path, "bad.py", body))
    record = manifest["grid_plan"]
    assert record["source"] == "protocol_fallback"
    assert record["requested"] is None
    assert needle in (record["fallback_reason"] or "")
    assert record["applied"]["max_branches"] == PROTOCOL_PLAN.max_branches
    assert record["applied"]["max_attempts_per_branch"] == PROTOCOL_PLAN.max_attempts_per_branch
    assert manifest["rounds_cap"] == PROTOCOL_PLAN.rounds_cap


def test_resume_reuses_the_recorded_plan_instead_of_replanning(tmp_path: Path) -> None:
    study = _study(tmp_path)
    raising = _policy(tmp_path, "raising.py", "        raise RuntimeError('planner exploded')\n")
    first = _cycle(tmp_path, study, raising)
    assert first["grid_plan"]["source"] == "protocol_fallback"
    (study / "cycles" / "iter_000" / "live_cycle_manifest.json").unlink()

    other = _policy(
        tmp_path,
        "other.py",
        "        return GridPlan(1, 0, 1, 1, rationale='different')\n",
    )
    second = _cycle(tmp_path, study, other)
    assert second["grid_plan"] == first["grid_plan"]
    assert second["grid_plan"]["applied"]["max_branches"] == PROTOCOL_PLAN.max_branches


def test_harness_override_is_recorded_and_clamped(tmp_path: Path) -> None:
    study = _study(tmp_path)
    policy = _policy(tmp_path, "plain.py", "        return context.fallback_plan\n")
    manifest = _cycle(tmp_path, study, policy, grid_plan=GridPlan(9, 9, 9, 9, rationale="override"))
    record = manifest["grid_plan"]
    assert record["source"] == "override"
    assert record["requested"]["max_branches"] == 9
    assert record["applied"]["max_branches"] == CAPS.max_branches


def test_resume_from_a_legacy_grid_plan_file(tmp_path: Path) -> None:
    cycle = tmp_path / "study" / "cycles" / "iter_000"
    cycle.mkdir(parents=True)
    (cycle / "grid_plan.json").write_text(json.dumps(asdict(PROTOCOL_PLAN)), encoding="utf-8")
    record = read_plan_record(cycle)
    assert record is not None
    assert record["source"] == "resume"
    assert record["requested"] is None
    assert record["applied"] == asdict(PROTOCOL_PLAN)


def test_plan_worker_reaudits_the_policy(tmp_path: Path) -> None:
    policy = _policy(
        tmp_path, "importing.py", "        import os\n        return context.fallback_plan\n"
    )
    with pytest.raises(PlanError, match="Forbidden import"):
        call_plan_grid(policy, _context(), use_bwrap=False)


def test_plan_worker_timeout_is_a_fallback(tmp_path: Path) -> None:
    body = "        while True:\n            pass\n"
    policy = _policy(tmp_path, "slow.py", body)
    with pytest.raises(PlanError, match="timed out"):
        call_plan_grid(policy, _context(), use_bwrap=False, timeout=0.3)
    # The cycle itself is not run here: the fallback path is covered above.


requires_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None or not (REPO / ".venv" / "bin" / "python").exists(),
    reason="bwrap or repo .venv not available",
)


@requires_bwrap
def test_plan_grid_runs_under_bwrap(tmp_path: Path) -> None:
    policy = _policy(
        tmp_path,
        "plain.py",
        "        return GridPlan(3, 2, 1, 4, rationale=f'iter={context.iteration}')\n",
    )
    plan = call_plan_grid(policy, _context(iteration=7), use_bwrap=True)
    assert plan == GridPlan(3, 2, 1, 4, rationale="iter=7")


# --- the interface: validate, clamp, read ------------------------------------------------


def test_validate_plan_rejects_wrong_types_and_non_positive_values() -> None:
    with pytest.raises(PlanError, match="not GridPlan"):
        validate_plan({"max_branches": 1})
    with pytest.raises(PlanError, match="must be an int"):
        validate_plan(GridPlan(True, 1, 1, 1))
    with pytest.raises(PlanError, match="max_branches must be positive"):
        validate_plan(GridPlan(0, 1, 1, 1))
    with pytest.raises(PlanError, match="max_attempts_per_branch must be non-negative"):
        validate_plan(GridPlan(1, -1, 1, 1))
    with pytest.raises(PlanError, match="max_parallelism must be positive"):
        validate_plan(GridPlan(1, 1, 0, 1))
    with pytest.raises(PlanError, match="rounds_cap must be positive"):
        validate_plan(GridPlan(1, 1, 1, 0))
    assert validate_plan(GridPlan(1, 0, 1, 1)) == GridPlan(1, 0, 1, 1)


def test_clamp_plan_bounds_every_field() -> None:
    clamped = clamp_plan(GridPlan(99, 99, 99, 99, rationale="big"), CAPS)
    assert clamped == GridPlan(2, 4, 2, 3, rationale="big")
    assert clamp_plan(GridPlan(1, 0, 1, 1), CAPS) == GridPlan(1, 0, 1, 1)


def test_caps_and_protocol_plan_come_from_the_protocol() -> None:
    caps = grid_caps(
        PROTOCOL_GRID,
        gpu_slots=1,
        probe_budget=3,
        max_parallelism=4,
        hard_max_branches=8,
        hard_max_depth=4,
    )
    assert caps == GridCaps(
        max_branches=3, max_attempts_per_branch=4, max_parallelism=1, rounds_cap=3
    )
    assert protocol_plan(PROTOCOL_GRID, caps).max_branches == 2
    assert protocol_plan({}, caps) == GridPlan(
        3, 4, 1, 3, rationale="caps default: the protocol records no grid_plan"
    )


def test_applied_plan_reads_new_and_legacy_manifests() -> None:
    plan = {"max_branches": 2, "max_attempts_per_branch": 1, "max_parallelism": 2, "rounds_cap": 3}
    assert applied_plan({"grid_plan": {"applied": plan}}).max_branches == 2
    assert applied_plan({"grid_plan": plan}).max_branches == 2
    assert applied_plan({"grid_plan": None}) is None
    assert applied_plan({}) is None


# --- the incumbent heuristic --------------------------------------------------------------


def test_incumbent_shifts_one_slot_toward_depth_when_refinements_pay() -> None:
    from rsi.policies.optimal import OptimalPolicy

    manifest = _manifest(asdict(PROTOCOL_PLAN), plateaued=False, refinement=True)
    plan = OptimalPolicy({}).plan_grid(_context((manifest,), iteration=1))
    assert (plan.max_branches, plan.max_attempts_per_branch) == (1, 2)
    assert "1 with a passing refinement" in plan.rationale


def test_incumbent_shifts_one_slot_toward_breadth_on_plateaus() -> None:
    from rsi.policies.optimal import OptimalPolicy

    manifest = _manifest(asdict(PROTOCOL_PLAN), plateaued=True, refinement=False)
    plan = OptimalPolicy({}).plan_grid(_context((manifest,), iteration=2))
    assert (plan.max_branches, plan.max_attempts_per_branch) == (3, 0)
    assert "1 plateaued" in plan.rationale


def test_incumbent_returns_the_protocol_plan_without_history() -> None:
    from rsi.policies.optimal import OptimalPolicy

    assert OptimalPolicy({}).plan_grid(_context()) == PROTOCOL_PLAN


# --- policy development exercises the candidate's plan -----------------------------------

DEV_AGENT = """
import pathlib, sys
pathlib.Path("method.py").write_text(pathlib.Path(sys.argv[1]).read_text())
"""
BROKEN_PLAN = '''"""Policy whose plan_grid always fails."""
from __future__ import annotations

from typing import Any

from rsi.api import LLMDesignedMethod, SimResult


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question: Any, budget: int | None = None) -> SimResult:
        return SimResult()

    def plan_grid(self, context: Any) -> Any:
        raise ValueError("no plan")
'''


def _dev(tmp_path: Path, candidate: Path) -> dict[str, Any]:
    agent = tmp_path / "dev_agent.py"
    agent.write_text(DEV_AGENT, encoding="utf-8")
    manifest = _manifest(PROTOCOL_GRID, plateaued=False, refinement=True)
    return develop_policies_llm(
        scenario_suite("train")[:2],
        cycle_id="t",
        output_dir=tmp_path / "out",
        live_manifests=[manifest],
        m=2,
        agent_cmd=[sys.executable, str(agent), str(candidate)],
        timeout=60.0,
        use_bwrap=False,
    )


def test_policy_dev_rejects_a_broken_plan_grid(tmp_path: Path) -> None:
    candidate = tmp_path / "broken.py"
    candidate.write_text(BROKEN_PLAN, encoding="utf-8")
    summary = _dev(tmp_path, candidate)
    variant = summary["variants"][1]
    assert variant["status"] == "rejected"
    assert "plan_grid" in variant["error"]
    assert variant["attempts"] == 2
    assert summary["selected_slot"] == "m0_incumbent"


def test_policy_dev_records_the_plan_a_candidate_would_request(tmp_path: Path) -> None:
    summary = _dev(tmp_path, REPO / "rsi" / "policies" / "optimal.py")
    variant = summary["variants"][1]
    assert variant["status"] == "not_better"
    assert variant["grid_plan"]["max_branches"] == 1
    assert variant["grid_plan"]["max_attempts_per_branch"] == 2
