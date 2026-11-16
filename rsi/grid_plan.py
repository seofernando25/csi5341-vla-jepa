"""Cross-cycle grid planning: request, validate and clamp a policy's `plan_grid` (Appendix D.2).

The protocol's grid values are ceilings, not the plan: the selected policy decides the next
cycle's capacity from the manifests of the cycles already run, and the harness clamps that
request to the ceilings. A request that fails, hangs or is unusable falls back to the
pre-registered plan and the manifest records why.
"""

from __future__ import annotations

import functools
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsi.api import GridPlan, GridPlanningContext
from rsi.sandbox import REPO_ROOT, SANDBOX_WORK, SandboxTimeoutError, run_sandboxed_python

# The replay worker's limits (rsi.policy_dev): a plan_grid call gets the same CPU and wall clock.
PLAN_TIMEOUT_S = 120.0
PLAN_CPU_S = 120
PLAN_FIELDS = ("max_branches", "max_attempts_per_branch", "max_parallelism", "rounds_cap")
# Runtime sidecars: the flat plan the watch page reads, and the full request/apply record.
PLAN_FILE = "grid_plan.json"
REQUEST_FILE = "grid_plan_request.json"
RSI_DIR = Path(__file__).resolve().parent
RSI_LIB_ROOT = "/tmp/rsi_lib"


class PlanError(RuntimeError):
    """A policy's `plan_grid` failed, timed out or returned a plan the cycle cannot use."""


@dataclass(frozen=True)
class GridCaps:
    """Protocol ceilings a requested plan is clamped to."""

    max_branches: int
    max_attempts_per_branch: int
    max_parallelism: int
    rounds_cap: int


def grid_caps(
    protocol_grid: dict[str, Any],
    *,
    gpu_slots: int,
    probe_budget: int,
    max_parallelism: int,
    hard_max_branches: int,
    hard_max_depth: int,
) -> GridCaps:
    """Ceilings from the protocol and the run arguments: probes, workers, GPU slots, rounds."""
    proto_parallel = int(protocol_grid.get("max_parallelism", max_parallelism))
    proto_rounds = int(protocol_grid.get("rounds_cap", probe_budget))
    return GridCaps(
        # A branch costs one probe, so the cycle's probe budget also bounds the branch count.
        max_branches=max(1, min(hard_max_branches, probe_budget)),
        max_attempts_per_branch=max(0, hard_max_depth),
        max_parallelism=max(1, min(max_parallelism, proto_parallel, gpu_slots)),
        rounds_cap=max(1, proto_rounds),
    )


def protocol_plan(protocol_grid: dict[str, Any], caps: GridCaps) -> GridPlan:
    """The pre-registered plan: the protocol's grid values, or the caps when it records none."""
    if not protocol_grid:
        return GridPlan(
            max_branches=caps.max_branches,
            max_attempts_per_branch=caps.max_attempts_per_branch,
            max_parallelism=caps.max_parallelism,
            rounds_cap=caps.rounds_cap,
            rationale="caps default: the protocol records no grid_plan",
        )
    return GridPlan(
        max_branches=int(protocol_grid["max_branches"]),
        max_attempts_per_branch=int(protocol_grid["max_attempts_per_branch"]),
        max_parallelism=int(protocol_grid["max_parallelism"]),
        rounds_cap=int(protocol_grid["rounds_cap"]),
        rationale=str(protocol_grid.get("rationale", "pre-registered static grid plan")),
    )


def applied_plan(manifest: dict[str, Any]) -> GridPlan | None:
    """The plan a recorded cycle applied, from a new or a legacy manifest; None if unreadable."""
    grid = manifest.get("grid_plan")
    if not isinstance(grid, dict):
        return None
    applied = grid.get("applied") if isinstance(grid.get("applied"), dict) else grid
    try:
        return GridPlan(
            max_branches=int(applied["max_branches"]),
            max_attempts_per_branch=int(applied["max_attempts_per_branch"]),
            max_parallelism=int(applied["max_parallelism"]),
            rounds_cap=int(applied["rounds_cap"]),
            rationale=str(applied.get("rationale", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def validate_plan(plan: Any) -> GridPlan:
    """Normalise a returned plan; raise `PlanError` on a wrong type or an unusable value."""
    if not isinstance(plan, GridPlan):
        raise PlanError(f"plan_grid returned {type(plan).__name__}, not GridPlan")
    values: dict[str, int] = {}
    for name in PLAN_FIELDS:
        value = getattr(plan, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlanError(f"{name} must be an int, got {type(value).__name__}")
        values[name] = value
    if values["max_branches"] < 1:
        raise PlanError(f"max_branches must be positive, got {values['max_branches']}")
    if values["max_attempts_per_branch"] < 0:
        raise PlanError(
            f"max_attempts_per_branch must be non-negative, got {values['max_attempts_per_branch']}"
        )
    if values["max_parallelism"] < 1:
        raise PlanError(f"max_parallelism must be positive, got {values['max_parallelism']}")
    if values["rounds_cap"] < 1:
        raise PlanError(f"rounds_cap must be positive, got {values['rounds_cap']}")
    if not isinstance(plan.rationale, str):
        raise PlanError(f"rationale must be a str, got {type(plan.rationale).__name__}")
    return GridPlan(**values, rationale=plan.rationale)


def clamp_plan(plan: GridPlan, caps: GridCaps) -> GridPlan:
    """Bound a plan by the protocol ceilings; the requested plan is kept beside it in the record."""
    return GridPlan(
        max_branches=max(1, min(caps.max_branches, plan.max_branches)),
        max_attempts_per_branch=max(
            0, min(caps.max_attempts_per_branch, plan.max_attempts_per_branch)
        ),
        max_parallelism=max(1, min(caps.max_parallelism, plan.max_parallelism)),
        rounds_cap=max(1, min(caps.rounds_cap, plan.rounds_cap)),
        rationale=plan.rationale,
    )


def read_plan_record(cycle_dir: Path) -> dict[str, Any] | None:
    """The plan record already written for this cycle, so a resumed run never re-plans."""
    directory = Path(cycle_dir)
    request = directory / REQUEST_FILE
    if request.is_file():
        try:
            record = json.loads(request.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if isinstance(record, dict) and isinstance(record.get("applied"), dict):
            return record
    legacy = directory / PLAN_FILE
    if legacy.is_file():
        try:
            applied = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(applied, dict) and all(name in applied for name in PLAN_FIELDS):
            return {
                "requested": None,
                "applied": applied,
                "caps": None,
                "source": "resume",
                "fallback_reason": None,
            }
    return None


def request_plan(
    policy_path: Path,
    context: GridPlanningContext,
    caps: GridCaps,
    *,
    cycle_dir: Path,
    use_bwrap: bool,
    override: GridPlan | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Plan one cycle and write its record; a resumed cycle returns the record it started with.

    `override` is a harness-supplied plan (tests, calibration) and skips the policy call.
    """
    existing = read_plan_record(cycle_dir)
    if existing is not None:
        return existing
    requested: GridPlan | None = None
    reason: str | None = None
    source = "policy"
    if override is not None:
        requested, source = override, "override"
    else:
        try:
            requested = call_plan_grid(policy_path, context, use_bwrap=use_bwrap, timeout=timeout)
        except PlanError as exc:
            reason, source = str(exc), "protocol_fallback"
    applied = clamp_plan(requested if requested is not None else context.fallback_plan, caps)
    record = {
        "requested": asdict(requested) if requested is not None else None,
        "applied": asdict(applied),
        "caps": asdict(caps),
        "source": source,
        "fallback_reason": reason,
    }
    directory = Path(cycle_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLAN_FILE).write_text(
        json.dumps(asdict(applied), indent=2) + "\n", encoding="utf-8"
    )
    (directory / REQUEST_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def call_plan_grid(
    policy_path: Path,
    context: GridPlanningContext,
    *,
    use_bwrap: bool = True,
    timeout: float | None = None,
) -> GridPlan:
    """Run one policy's `plan_grid` on `context` in a subprocess; raise `PlanError` on failure.

    The worker sees only the context, never a question object, and runs under the replay
    worker's CPU and wall-clock limits. The protocol plan is passed as the policy config, so a
    policy that reads `config["grid_plan"]` keeps working.
    """
    limit = PLAN_TIMEOUT_S if timeout is None else float(timeout)
    ws = Path(tempfile.mkdtemp(prefix="rsi-plan-"))
    try:
        shutil.copyfile(policy_path, ws / "policy.py")
        (ws / "context.json").write_text(json.dumps(asdict(context)), encoding="utf-8")
        config = {"grid_plan": asdict(context.fallback_plan)}
        (ws / "config.json").write_text(json.dumps(config), encoding="utf-8")
        root = Path(SANDBOX_WORK) if use_bwrap else ws
        args = [
            "-m",
            "rsi.plan_worker",
            str(root / "policy.py"),
            str(root / "context.json"),
            str(root / "plan.json"),
            "--config",
            str(root / "config.json"),
        ]
        proc = _run_worker(args, ws, use_bwrap=use_bwrap, timeout=limit)
        if proc.returncode != 0:
            raise PlanError(_failure(proc))
        try:
            payload = json.loads((ws / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlanError(f"plan_grid wrote no readable plan: {exc}") from exc
        return _from_payload(payload)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _from_payload(payload: Any) -> GridPlan:
    """Rebuild the worker's JSON plan and validate it like a returned object."""
    if not isinstance(payload, dict):
        raise PlanError("plan_grid did not write a plan object")
    missing = [name for name in PLAN_FIELDS if name not in payload]
    if missing:
        raise PlanError(f"plan_grid wrote no {', '.join(missing)}")
    plan = GridPlan(
        **{name: payload[name] for name in PLAN_FIELDS},
        rationale=payload.get("rationale", ""),
    )
    return validate_plan(plan)


def _run_worker(
    args: list[str], ws: Path, *, use_bwrap: bool, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run the plan worker with the replay worker's limits, inside bwrap when it is enabled."""
    if use_bwrap:
        try:
            return run_sandboxed_python(
                args,
                workspace_dir=ws,
                extra_ro_binds={RSI_DIR: f"{RSI_LIB_ROOT}/rsi"},
                env_overrides={"PYTHONPATH": RSI_LIB_ROOT},
                timeout=timeout,
                cpu_seconds=PLAN_CPU_S,
            )
        except SandboxTimeoutError as exc:
            raise PlanError(f"plan_grid timed out after {timeout:.0f}s") from exc
    try:
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            env={**os.environ},
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            # The child only calls setrlimit before exec; no thread pool is running yet.
            preexec_fn=functools.partial(_limit_cpu, PLAN_CPU_S),
        )
    except subprocess.TimeoutExpired as exc:
        raise PlanError(f"plan_grid timed out after {timeout:.0f}s") from exc


def _limit_cpu(cpu_seconds: int) -> None:
    """Cap CPU time in the child, matching the replay worker's RLIMIT_CPU."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def _failure(proc: subprocess.CompletedProcess[str]) -> str:
    """One-line reason from a worker that exited non-zero."""
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"plan worker exited {proc.returncode}"
