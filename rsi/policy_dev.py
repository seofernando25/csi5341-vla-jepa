"""Sequential LLM development of `OptimalPolicy` with replay-gated, monotonic selection (Dream-RSI App. D.2)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.agent_backends import (
    SANDBOX_EVIDENCE,
    SANDBOX_PKGS,
    SANDBOX_REPO,
    SANDBOX_WORK,
    STUDY_DIR,
    AgentSpec,
    agent_record,
    load_agent_spec,
)
from rsi.api import GridPlanningContext
from rsi.dream import leave_one_world_out
from rsi.grid_plan import PlanError, applied_plan, call_plan_grid
from rsi.objective import DEFAULT_LAMBDA_PARALLEL
from rsi.policy_audit import (
    PolicyAuditError,
    audit_no_world_literals,
    audit_policy_source,
    load_policy_class,
)
from rsi.replay_worker import load_tree_json, save_tree_json
from rsi.sandbox import (
    REPO_ROOT,
    SANDBOX_VENV,
    AgentRunResult,
    SandboxOutputError,
    SandboxTimeoutError,
    read_agent_output,
    repo_source_binds,
    run_sandboxed_agent,
    run_sandboxed_python,
    timed_out_result,
)
from rsi.synth import scenario_suite
from rsi.tree import DiscoveryTree
from rsi.workspace import MAX_WORKSPACE_BYTES

__all__ = [
    "INCUMBENT_TEMPLATE",
    "PolicyAuditError",
    "develop_policies_llm",
    "load_policy_class",
    "load_tree_json",
    "save_tree_json",
]

RSI_DIR = Path(__file__).resolve().parent
INCUMBENT_TEMPLATE = RSI_DIR / "policies" / "optimal.py"
POLICY_DEV_PROMPT = RSI_DIR / "prompts" / "policy_dev.md"
RSI_LIB_ROOT = "/tmp/rsi_lib"
RSI_LIB_FILES = (
    "__init__.py",
    "api.py",
    "tree.py",
    "objective.py",
    "policy_audit.py",
    "replay_worker.py",
)
# Repo source the policy agent may read, minus the held-out world generator: the held-out
# worlds are reported next to V and must not be reproducible from inside the workspace.
POLICY_DEV_SOURCE_EXCLUDE = ("rsi/synth.py",)
PKG_DIR_NAME = "pkgs"
REPLAY_SCRIPT_NAME = "replay.py"
UV_BIN = "/tmp/bin/uv"
DEFAULT_POLICY_TIMEOUT_S = 1200.0
MAX_METHOD_BYTES = 200_000
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_TRACE_BYTES = 32 * 1024 * 1024
REPLAY_TIMEOUT_S = 120.0
REPLAY_CPU_S = 120
REPAIR_SUFFIX = (
    "\n\n### Previous attempt validation failure\n{error}\n\n"
    "Fix every violation in `/tmp/work/method.py` cleanly."
)
NOT_BETTER_SUFFIX = (
    "\n\n### Previous attempt did not improve V\n"
    "Its mean V {mean_v:.6f} is not above the incumbent's {best:.6f} "
    "(delta {delta:+.6f}).\n\n"
    "Make one different, larger change in `/tmp/work/method.py`. Run "
    "`python /tmp/work/replay.py` and keep the change only if mean V rises above {best:.6f}."
)
REPLAY_SCRIPT = '''"""Score a candidate policy on this step's recorded trees, exactly as the harness does.

Usage: python replay.py [POLICY]   (default: method.py in this workspace)

Prints V per world and the mean V the harness would record. The harness reruns the replay
itself after you finish; only its result counts.
"""

import os
import sys
from pathlib import Path

from rsi.replay_worker import run_worker

MAX_PARALLELISM = __MAX_PARALLELISM__
LAMBDA_PARALLEL = __LAMBDA_PARALLEL__


def main() -> int:
    evidence = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
    workspace = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
    worlds = sorted((evidence / "worlds").glob("*.json"))
    policy = Path(sys.argv[1]) if len(sys.argv) > 1 else workspace / "method.py"
    try:
        result, _ = run_worker(
            policy, worlds, max_parallelism=MAX_PARALLELISM, lambda_parallel=LAMBDA_PARALLEL
        )
    except Exception as exc:
        print(f"replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for world_id, world in result["worlds"].items():
        print(
            f"{world_id}: V={world['v']:.6f} hypervolume={world['hypervolume']:.6f} "
            f"penalty={world['penalty']:.6f}"
        )
    print(f"mean V = {result['mean_v']:.6f} over {len(result['worlds'])} world(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


class PolicyReplayError(RuntimeError):
    """Raised when the sandboxed replay worker rejects a policy."""


def _rsi_ro_binds() -> dict[Path, str]:
    return {RSI_DIR / name: f"{RSI_LIB_ROOT}/rsi/{name}" for name in RSI_LIB_FILES}


def _replay_script(max_parallelism: int, lambda_parallel: float) -> str:
    return REPLAY_SCRIPT.replace("__MAX_PARALLELISM__", str(int(max_parallelism))).replace(
        "__LAMBDA_PARALLEL__", repr(float(lambda_parallel))
    )


def _build_policy_dev_prompt(
    *,
    cycle_id: str,
    step: int,
    steps: int,
    max_parallelism: int,
    lambda_parallel: float,
    minutes: float,
    prompt_path: Path,
) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8")
    return template.format(
        step=step,
        steps=steps,
        cycle_id=cycle_id,
        lambda_parallel=lambda_parallel,
        max_parallelism=max_parallelism,
        minutes=f"{minutes:.0f}",
    )


def _agent_sandbox(step_dir: Path, study_dir: Path | None = None) -> dict[str, Any]:
    """Extra mounts and env for the policy agent: repo source, GPU, and a writable package dir."""
    pkgs = Path(step_dir) / PKG_DIR_NAME
    pkgs.mkdir(parents=True, exist_ok=True)
    binds: dict[Path, str] = {}
    path = [f"{SANDBOX_VENV}/bin", "/usr/bin"]
    uv = shutil.which("uv")
    if uv:
        # `uv` is a host tool, not a venv entry point; the agent needs it to install packages.
        binds[Path(uv)] = UV_BIN
        path.insert(0, str(Path(UV_BIN).parent))
    return {
        "source_ro": repo_source_binds(study_dir=study_dir, exclude=POLICY_DEV_SOURCE_EXCLUDE),
        "writable_dirs": {pkgs: SANDBOX_PKGS},
        "extra_ro_binds": binds,
        "env_overrides": {
            "PATH": ":".join(path),
            "PYTHONPATH": f"{SANDBOX_PKGS}:{SANDBOX_REPO}",
            "UV_CACHE_DIR": "/tmp/.cache/uv",
            "UV_LINK_MODE": "copy",
        },
    }


def _workspace_bytes(root: Path) -> int:
    """Total size of the regular files in one workspace; symlinks are not followed."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _recorded_path(path: Path | None, record_root: Path | None) -> str | None:
    """A path as recorded in evidence: relative to `record_root` when it is under it."""
    if path is None:
        return None
    if record_root is None:
        return str(path)
    try:
        return Path(path).resolve().relative_to(Path(record_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _worker_failure(proc: subprocess.CompletedProcess[str]) -> str:
    """One-line reason for a worker that exited non-zero (signals name the limit hit)."""
    if proc.returncode < 0:
        return f"replay worker killed by {signal.Signals(-proc.returncode).name}"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"worker exited {proc.returncode}"


def _run_worker_once(
    *,
    ws: Path,
    worker_args: list[str],
    evidence: Path,
    use_bwrap: bool,
    hash_seed: str,
) -> tuple[dict[str, Any], str]:
    """Run the worker under one PYTHONHASHSEED; returns (result json, traces text)."""
    if use_bwrap:
        proc = run_sandboxed_python(
            worker_args,
            workspace_dir=ws,
            evidence_dir=evidence,
            extra_ro_binds=_rsi_ro_binds(),
            env_overrides={"PYTHONPATH": RSI_LIB_ROOT, "PYTHONHASHSEED": hash_seed},
            timeout=REPLAY_TIMEOUT_S,
            cpu_seconds=REPLAY_CPU_S,
        )
    else:
        try:
            proc = subprocess.run(
                [sys.executable, *worker_args],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=False,
                timeout=REPLAY_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(f"replay timed out after {REPLAY_TIMEOUT_S:.0f}s") from exc
    if proc.returncode != 0:
        raise PolicyReplayError(_worker_failure(proc))
    result = json.loads(read_agent_output(ws, "result.json", max_bytes=MAX_RESULT_BYTES))
    traces = read_agent_output(ws, "traces.jsonl", max_bytes=MAX_TRACE_BYTES)
    return result, traces


def _replay_policy(
    policy_src: Path,
    world_paths: list[Path],
    *,
    ws_dir: Path,
    evidence_dir: Path,
    max_parallelism: int,
    lambda_parallel: float,
    use_bwrap: bool,
) -> tuple[dict[str, Any], str]:
    """Replay one policy file under two hash seeds; reject it if the two runs disagree."""
    ws = Path(ws_dir)
    ws.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", "traces.jsonl", "policy.py"):
        (ws / stale).unlink(missing_ok=True)
    shutil.copyfile(policy_src, ws / "policy.py")
    ev = Path(evidence_dir).resolve()
    root = Path(SANDBOX_WORK) if use_bwrap else ws
    ev_root = Path(SANDBOX_EVIDENCE) if use_bwrap else ev
    worlds = [str(ev_root / Path(w).resolve().relative_to(ev).as_posix()) for w in world_paths]
    worker_args = [
        "-m",
        "rsi.replay_worker",
        str(root / "policy.py"),
        str(root / "result.json"),
        *worlds,
        "--max-parallelism",
        str(max_parallelism),
        "--lambda",
        repr(float(lambda_parallel)),
        "--traces",
        str(root / "traces.jsonl"),
    ]
    run = {"ws": ws, "worker_args": worker_args, "evidence": ev, "use_bwrap": use_bwrap}
    first = _run_worker_once(**run, hash_seed="0")
    second = _run_worker_once(**run, hash_seed="1")
    if first != second:
        raise PolicyReplayError("non-deterministic across hash seeds")
    return first


@dataclass
class _Run:
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
        return _replay_policy(
            src,
            worlds,
            ws_dir=ws_dir,
            evidence_dir=evidence,
            max_parallelism=self.max_parallelism,
            lambda_parallel=self.lambda_parallel,
            use_bwrap=self.use_bwrap,
        )


def _plan_context(
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


def _check_plan_grid(run: _Run, candidate: Path) -> dict[str, Any] | None:
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
    run: _Run, step_dir: Path, text: str
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """Audit the agent's source, exercise its plan, then replay it; raises on any violation."""
    audit_policy_source(text)
    audit_no_world_literals(text, run.history)
    candidate = step_dir / "candidate.py"
    candidate.write_text(text, encoding="utf-8")
    plan = _check_plan_grid(run, candidate)
    result, traces = run.replay(candidate, step_dir / "replay", run.world_paths, run.evidence)
    return result, traces, plan


def _reject(rec: dict[str, Any], error: str, status: str = "rejected") -> dict[str, Any]:
    """Record a failed step; a measurement from an earlier attempt no longer applies."""
    rec.update(
        status=status,
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
    run: _Run,
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
    if _workspace_bytes(ws) > MAX_WORKSPACE_BYTES:
        return _reject(rec, f"workspace exceeds {MAX_WORKSPACE_BYTES} bytes", status="timeout")
    try:
        text = read_agent_output(ws, "method.py", max_bytes=MAX_METHOD_BYTES)
    except SandboxOutputError:
        return _reject(rec, timeout_error, status="timeout")
    rec["text"] = text
    try:
        result, traces, plan = _check_text(run, step_dir, text)
    except (SandboxTimeoutError, SandboxOutputError) as exc:
        return _reject(rec, f"{timeout_error}; salvage failed: {exc}", status="timeout")
    except (PolicyAuditError, PolicyReplayError) as exc:
        return _reject(rec, f"{timeout_error}; salvage failed: {exc}", status="timeout")
    mean_v = float(result["mean_v"])
    rec.update(
        status="timeout_salvaged",
        result=result,
        traces=traces,
        plan=plan,
        path=step_dir / "candidate.py",
        mean_v=mean_v,
        delta_vs_best=mean_v - best_mean_v,
        error=timeout_error,
    )
    return rec


def _run_agent_step(run: _Run, step: int, seed_text: str, best_mean_v: float) -> dict[str, Any]:
    """Rewrite the best policy so far; one repair attempt after any failed or flat step."""
    step_dir = run.out_root / f"step_{step:02d}"
    ws = step_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "method.py").write_text(seed_text, encoding="utf-8")
    (ws / REPLAY_SCRIPT_NAME).write_text(
        _replay_script(run.max_parallelism, run.lambda_parallel), encoding="utf-8"
    )
    base_prompt = _build_policy_dev_prompt(
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
    sandbox = _agent_sandbox(step_dir, run.study_dir) if run.use_bwrap else {}
    rec: dict[str, Any] = {
        "step": step,
        "slot": f"m{step}_llm",
        "status": "rejected",
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
        if _workspace_bytes(ws) > MAX_WORKSPACE_BYTES:
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
            rec["status"] = "not_better"
            prompt = base_prompt + NOT_BETTER_SUFFIX.format(
                mean_v=mean_v, best=best_mean_v, delta=mean_v - best_mean_v
            )
            continue
        rec["status"] = "valid"
        return rec
    return rec


def _write_history(evidence: Path, rec: dict[str, Any], verdict: dict[str, Any]) -> None:
    d = evidence / "history" / f"step_{rec['step']:02d}"
    d.mkdir(parents=True, exist_ok=True)
    if rec["text"] is not None:
        (d / "method.py").write_text(rec["text"], encoding="utf-8")
    if rec["result"] is not None:
        (d / "scaling_curve.json").write_text(
            json.dumps(rec["result"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (d / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")


def _write_best_evidence(evidence: Path, best: dict[str, Any]) -> None:
    (evidence / "beta_sweep.json").write_text(
        json.dumps(best["result"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "policy_execution_traces.jsonl").write_text(best["traces"], encoding="utf-8")


def _verdict(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": rec["step"],
        "slot": rec["slot"],
        "status": rec["status"],
        "mean_v": rec.get("mean_v"),
        "delta_vs_best": rec.get("delta_vs_best"),
        "error": rec["error"],
        "attempts": rec["attempts"],
        "grid_plan": rec.get("plan"),
    }


def _summary_variant(rec: dict[str, Any], record_root: Path | None = None) -> dict[str, Any]:
    return {
        "slot": rec["slot"],
        "step": rec["step"],
        "status": rec["status"],
        "mean_v": rec.get("mean_v"),
        "delta_vs_best": rec.get("delta_vs_best"),
        "error": rec["error"],
        "attempts": rec["attempts"],
        "grid_plan": rec.get("plan"),
        "policy_path": _recorded_path(rec["path"], record_root),
        "policy_sha256": _sha256(rec["path"]) if rec["path"] is not None else None,
    }


def develop_policies_llm(
    history: list[DiscoveryTree],
    *,
    cycle_id: str,
    incumbent_path: Path | None = None,
    output_dir: Path,
    live_manifests: list[dict[str, Any]] | None = None,
    m: int = 4,
    max_parallelism: int = 4,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    timeout: float = DEFAULT_POLICY_TIMEOUT_S,
    use_bwrap: bool = True,
    prompt_path: Path | None = None,
    record_root: Path | None = None,
) -> dict[str, Any]:
    """Develop m-1 sequential rewrites of the incumbent; select argmax mean V on history.

    `record_root` is the directory recorded policy paths are made relative to (the study dir);
    without it they stay absolute, as calibration output does.
    """
    if not history:
        raise ValueError("Policy development requires at least 1 frozen DiscoveryTree in history")
    if m < 2:
        raise ValueError("m must be >= 2 (incumbent + at least 1 rewritten candidate)")

    inc_src = Path(incumbent_path) if incumbent_path is not None else INCUMBENT_TEMPLATE
    out_root = Path(output_dir)
    evidence = out_root / "evidence"
    manifests = list(live_manifests or ())
    world_paths = [
        save_tree_json(t, evidence / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(history)
    ]
    (evidence / "live_cycle_manifests.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "reference").mkdir(parents=True, exist_ok=True)
    for name in ("api.py", "tree.py", "objective.py"):
        shutil.copyfile(RSI_DIR / name, evidence / "reference" / name)

    spec = agent or load_agent_spec(record_root or STUDY_DIR)
    run = _Run(
        cycle_id=cycle_id,
        history=history,
        world_paths=world_paths,
        evidence=evidence,
        out_root=out_root,
        steps=m - 1,
        max_parallelism=max_parallelism,
        lambda_parallel=lambda_parallel,
        prompt_path=Path(prompt_path) if prompt_path is not None else POLICY_DEV_PROMPT,
        agent_cmd=agent_cmd,
        agent=spec,
        timeout=timeout,
        use_bwrap=use_bwrap,
        plan_context=_plan_context(manifests, max_parallelism),
        study_dir=Path(record_root) if record_root is not None else None,
    )

    inc_text = inc_src.read_text(encoding="utf-8")
    inc_dir = out_root / "step_00"
    inc_dir.mkdir(parents=True, exist_ok=True)
    (inc_dir / "method.py").write_text(inc_text, encoding="utf-8")
    shutil.copyfile(inc_src, inc_dir / "candidate.py")
    inc_plan = _check_plan_grid(run, inc_dir / "candidate.py")
    inc_result, inc_traces = run.replay(
        inc_dir / "candidate.py", inc_dir / "replay", world_paths, evidence
    )
    best: dict[str, Any] = {
        "step": 0,
        "slot": "m0_incumbent",
        "status": "incumbent",
        "error": None,
        "attempts": 0,
        "text": inc_text,
        "result": inc_result,
        "traces": inc_traces,
        "plan": inc_plan,
        "path": inc_dir / "candidate.py",
        "mean_v": inc_result["mean_v"],
        "delta_vs_best": 0.0,
    }
    records: list[dict[str, Any]] = [best]
    _write_history(evidence, best, _verdict(best))

    for step in range(1, m):
        _write_best_evidence(evidence, best)
        rec = _run_agent_step(run, step, best["text"], best["mean_v"])
        if rec["status"] == "valid":
            rec["status"] = "accepted" if rec["mean_v"] > best["mean_v"] else "not_better"
            if rec["status"] == "accepted":
                best = rec
        elif rec["status"] == "timeout_salvaged" and rec["mean_v"] > best["mean_v"]:
            best = rec
        _write_history(evidence, rec, _verdict(rec))
        records.append(rec)

    valid = [
        r
        for r in records
        if r["status"] in ("incumbent", "accepted", "not_better", "timeout_salvaged")
    ]
    world_ids = [t.world_id for t in history]
    mean_by_slot = {r["slot"]: r["mean_v"] for r in valid}
    selected = max(valid, key=lambda r: (r["mean_v"], -r["step"]))
    winner_path = out_root / "selected_policy.py"
    shutil.copyfile(selected["path"], winner_path)

    terms = {
        r["slot"]: {
            wid: (w["hypervolume"], w["penalty"]) for wid, w in r["result"]["worlds"].items()
        }
        for r in valid
    }
    lowo_report = (
        leave_one_world_out(terms, world_ids, lambda_parallel) if len(history) >= 2 else None
    )

    heldout_dir = out_root / "heldout"
    heldout_paths = [
        save_tree_json(t, heldout_dir / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(scenario_suite("heldout"))
    ]
    heldout_mean: dict[str, float] = {}
    for r in valid:
        res, _ = _replay_policy(
            r["path"],
            heldout_paths,
            ws_dir=heldout_dir / "replay" / r["slot"],
            evidence_dir=heldout_dir,
            max_parallelism=max_parallelism,
            lambda_parallel=lambda_parallel,
            use_bwrap=use_bwrap,
        )
        heldout_mean[r["slot"]] = round(float(res["mean_v"]), 6)
    incumbent_heldout = heldout_mean["m0_incumbent"]

    summary = {
        "protocol": "dream-rsi-policy-dev",
        "cycle_id": cycle_id,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "num_worlds": len(history),
        "world_ids": world_ids,
        "max_parallelism": max_parallelism,
        "lambda_parallel": lambda_parallel,
        "selected_slot": selected["slot"],
        "selected_policy_path": _recorded_path(winner_path, record_root),
        "incumbent_in_sample_V": round(float(mean_by_slot["m0_incumbent"]), 6),
        "selected_in_sample_V": round(float(selected["mean_v"]), 6),
        "in_sample_V_by_slot": {k: round(float(v), 6) for k, v in mean_by_slot.items()},
        "heldout": {
            "num_worlds": len(heldout_paths),
            "mean_V_by_slot": heldout_mean,
        },
        "heldout_delta_vs_incumbent": round(heldout_mean[selected["slot"]] - incumbent_heldout, 6),
        "lowo": (
            {
                "mean_selected_reward": round(float(lowo_report["mean_selected_reward"]), 6),
                "mean_incumbent_reward": round(float(lowo_report["mean_incumbent_reward"]), 6),
                "folds_selected_beats_incumbent": int(
                    lowo_report["folds_selected_beats_incumbent"]
                ),
                "folds_selected_below_incumbent": int(
                    lowo_report["folds_selected_below_incumbent"]
                ),
                "mean_held_out_reward_by_policy": {
                    k: round(float(v), 6)
                    for k, v in lowo_report["mean_held_out_reward_by_policy"].items()
                },
            }
            if lowo_report is not None
            else None
        ),
        "variants": [_summary_variant(r, record_root) for r in records],
        "agent": agent_record(None if agent_cmd is not None else spec, run.results),
    }
    (out_root / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
