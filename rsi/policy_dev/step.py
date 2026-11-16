"""One policy-development step: the agent rewrites the best policy so far, the harness audits and replays it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rsi.agents import (
    AgentRunResult,
    AgentSpec,
    run_sandboxed_agent,
    timed_out_result,
)
from rsi.api import GridPlanningContext
from rsi.grid_plan import PlanError, applied_plan, call_plan_grid
from rsi.policy_audit import (
    PolicyAuditError,
    audit_no_world_literals,
    audit_policy_source,
)
from rsi.policy_dev.agent_env import (
    NOT_BETTER_SUFFIX,
    REPAIR_SUFFIX,
    REPLAY_SCRIPT_NAME,
    agent_sandbox,
    build_prompt,
    replay_script,
    workspace_bytes,
)
from rsi.policy_dev.replay import PolicyReplayError, replay_policy
from rsi.policy_dev.states import STEP, StepStatus
from rsi.sandbox import (
    SandboxOutputError,
    SandboxTimeoutError,
    read_agent_output,
)
from rsi.tree import DiscoveryTree
from rsi.workspace import MAX_WORKSPACE_BYTES

MAX_METHOD_BYTES = 200_000


@dataclass
class StepRun:
    """Fixed inputs of one policy-development cycle."""

    cycle_id: str
    history: list[DiscoveryTree]
    world_paths: list[Path]
    evidence: Path
    out_root: Path
    steps: int
    max_parallelism: int
    lambda_parallel: float
    prompt_path: Path
    agent_cmd: str | list[str] | None
    agent: AgentSpec
    timeout: float
    use_bwrap: bool
    plan_context: GridPlanningContext | None = None
    study_dir: Path | None = None
    results: list[AgentRunResult] = field(default_factory=list)

    def replay(
        self, src: Path, ws_dir: Path, worlds: list[Path], evidence: Path
    ) -> tuple[dict[str, Any], str]:
        return replay_policy(
            src,
            worlds,
            ws_dir=ws_dir,
            evidence_dir=evidence,
            max_parallelism=self.max_parallelism,
            lambda_parallel=self.lambda_parallel,
            use_bwrap=self.use_bwrap,
        )


def plan_context(
    manifests: list[dict[str, Any]], max_parallelism: int
) -> GridPlanningContext | None:
    """Planning context for candidate validation, or None when no cycle recorded a plan.

    The fallback is the plan the last recorded cycle applied. Without one there is nothing to
    plan from, and the next live cycle validates the candidate before it runs.
    """
    for manifest in reversed(manifests):
        fallback = applied_plan(manifest)
        if fallback is not None:
            return GridPlanningContext(
                iteration=len(manifests),
                max_parallelism=max_parallelism,
                fallback_plan=fallback,
                recent_live_manifests=tuple(manifests[-3:]),
            )
    return None


def check_plan_grid(run: StepRun, candidate: Path) -> dict[str, Any] | None:
    """Exercise a candidate's `plan_grid` on the recorded history; a broken plan rejects it.

    Replay V scores `solve` only, so this validates the plan's shape and range, never its
    quality: a plan that would fail at the next live cycle fails here instead.
    """
    if run.plan_context is None:
        return None
    try:
        return asdict(call_plan_grid(candidate, run.plan_context, use_bwrap=run.use_bwrap))
    except PlanError as exc:
        raise PolicyReplayError(f"plan_grid: {exc}") from exc


def _check_text(
    run: StepRun, step_dir: Path, text: str
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """Audit the agent's source, exercise its plan, then replay it; raises on any violation."""
    audit_policy_source(text)
    audit_no_world_literals(text, run.history)
    candidate = step_dir / "candidate.py"
    candidate.write_text(text, encoding="utf-8")
    plan = check_plan_grid(run, candidate)
    result, traces = run.replay(candidate, step_dir / "replay", run.world_paths, run.evidence)
    return result, traces, plan


def _reject(
    rec: dict[str, Any], error: str, status: StepStatus = StepStatus.REJECTED
) -> dict[str, Any]:
    """Record a failed step; a measurement from an earlier attempt no longer applies."""
    STEP.advance(rec, status)
    rec.update(
        error=error,
        result=None,
        traces=None,
        plan=None,
        path=None,
        mean_v=None,
        delta_vs_best=None,
    )
    return rec


def _salvage_timeout(
    run: StepRun,
    ws: Path,
    step_dir: Path,
    rec: dict[str, Any],
    best_mean_v: float,
    timeout_error: str,
) -> dict[str, Any]:
    """Score the file a timed-out step left behind, so the step is not wasted.

    The step is anytime: the agent is told to keep a valid `method.py` from early on, and the
    harness audits and replays whatever is there when the clock runs out. A scored file is
    `timeout_salvaged` and competes like any other candidate; a missing or unusable file leaves the
    step a plain `timeout`, with the reason recorded.
    """
    if workspace_bytes(ws) > MAX_WORKSPACE_BYTES:
        return _reject(
            rec, f"workspace exceeds {MAX_WORKSPACE_BYTES} bytes", status=StepStatus.TIMEOUT
        )
    try:
        text = read_agent_output(ws, "method.py", max_bytes=MAX_METHOD_BYTES)
    except SandboxOutputError:
        return _reject(rec, timeout_error, status=StepStatus.TIMEOUT)
    rec["text"] = text
    try:
        result, traces, plan = _check_text(run, step_dir, text)
    except (SandboxTimeoutError, SandboxOutputError) as exc:
        return _reject(rec, f"{timeout_error}; salvage failed: {exc}", status=StepStatus.TIMEOUT)
    except (PolicyAuditError, PolicyReplayError) as exc:
        return _reject(rec, f"{timeout_error}; salvage failed: {exc}", status=StepStatus.TIMEOUT)
    mean_v = float(result["mean_v"])
    STEP.advance(rec, StepStatus.TIMEOUT_SALVAGED)
    rec.update(
        result=result,
        traces=traces,
        plan=plan,
        path=step_dir / "candidate.py",
        mean_v=mean_v,
        delta_vs_best=mean_v - best_mean_v,
        error=timeout_error,
    )
    return rec


def run_agent_step(run: StepRun, step: int, seed_text: str, best_mean_v: float) -> dict[str, Any]:
    """Rewrite the best policy so far; one repair attempt after any failed or flat step."""
    step_dir = run.out_root / f"step_{step:02d}"
    ws = step_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "method.py").write_text(seed_text, encoding="utf-8")
    (ws / REPLAY_SCRIPT_NAME).write_text(
        replay_script(run.max_parallelism, run.lambda_parallel), encoding="utf-8"
    )
    base_prompt = build_prompt(
        cycle_id=run.cycle_id,
        step=step,
        steps=run.steps,
        max_parallelism=run.max_parallelism,
        lambda_parallel=run.lambda_parallel,
        minutes=run.timeout / 60.0,
        prompt_path=run.prompt_path,
    )
    # `run_sandboxed_agent` rewrites evidence/TASK.md on every call; keep the step's own copy.
    (step_dir / "TASK.md").write_text(base_prompt, encoding="utf-8")
    sandbox = agent_sandbox(step_dir, run.study_dir) if run.use_bwrap else {}
    rec: dict[str, Any] = {
        "step": step,
        "slot": f"m{step}_llm",
        "status": StepStatus.REJECTED.value,
        "error": None,
        "attempts": 0,
        "text": None,
        "result": None,
        "traces": None,
        "plan": None,
        "path": None,
        "mean_v": None,
        "delta_vs_best": None,
    }
    prompt = base_prompt
    for attempt in (1, 2):
        rec["attempts"] = attempt
        logs = step_dir / "logs" / f"attempt_{attempt}"
        if attempt > 1:
            (step_dir / f"TASK_attempt_{attempt:02d}.md").write_text(prompt, encoding="utf-8")
        try:
            agent = run_sandboxed_agent(
                prompt_text=prompt,
                workspace_dir=ws,
                evidence_dir=run.evidence,
                logs_dir=logs,
                agent_cmd=run.agent_cmd,
                agent=run.agent,
                timeout=run.timeout,
                use_bwrap=run.use_bwrap,
                gpu=True,
                **sandbox,
            )
        except SandboxTimeoutError as exc:
            # The killed run already spent money: keep its cost and turns for the budget guard.
            spent = timed_out_result(logs, sandboxed=run.use_bwrap)
            if spent is not None:
                run.results.append(spent)
            return _salvage_timeout(run, ws, step_dir, rec, best_mean_v, str(exc))
        run.results.append(agent)
        if agent.returncode != 0:
            return _reject(rec, f"Agent exited {agent.returncode}: {agent.stderr[-300:]}")
        if workspace_bytes(ws) > MAX_WORKSPACE_BYTES:
            return _reject(rec, f"workspace exceeds {MAX_WORKSPACE_BYTES} bytes")
        try:
            text = read_agent_output(ws, "method.py", max_bytes=MAX_METHOD_BYTES)
        except SandboxOutputError as exc:
            return _reject(rec, str(exc))
        rec["text"] = text
        try:
            result, traces, plan = _check_text(run, step_dir, text)
        except (SandboxTimeoutError, SandboxOutputError) as exc:
            return _reject(rec, str(exc))
        except (PolicyAuditError, PolicyReplayError) as exc:
            _reject(rec, str(exc))
            prompt = base_prompt + REPAIR_SUFFIX.format(error=rec["error"])
            continue
        mean_v = float(result["mean_v"])
        rec.update(
            result=result,
            traces=traces,
            plan=plan,
            path=step_dir / "candidate.py",
            mean_v=mean_v,
            delta_vs_best=mean_v - best_mean_v,
            error=None,
        )
        if mean_v <= best_mean_v and attempt == 1:
            STEP.advance(rec, StepStatus.NOT_BETTER)
            prompt = base_prompt + NOT_BETTER_SUFFIX.format(
                mean_v=mean_v, best=best_mean_v, delta=mean_v - best_mean_v
            )
            continue
        STEP.advance(rec, StepStatus.VALID)
        return rec
    return rec
