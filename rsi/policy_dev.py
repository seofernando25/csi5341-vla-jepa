"""Sequential LLM development of `OptimalPolicy` with replay-gated, monotonic selection (Dream-RSI App. D.2)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.agent_backends import (
    SANDBOX_EVIDENCE,
    SANDBOX_WORK,
    AgentSpec,
    agent_record,
    default_agent_spec,
)
from rsi.dream import leave_one_world_out
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
    AgentRunResult,
    SandboxOutputError,
    SandboxTimeoutError,
    read_agent_output,
    run_sandboxed_agent,
    run_sandboxed_python,
)
from rsi.synth import scenario_suite
from rsi.tree import DiscoveryTree

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
MAX_METHOD_BYTES = 200_000
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_TRACE_BYTES = 32 * 1024 * 1024
REPLAY_TIMEOUT_S = 120.0
REPLAY_CPU_S = 120
REPAIR_SUFFIX = (
    "\n\n### Previous attempt validation failure\n{error}\n\n"
    "Fix every violation in `/tmp/work/method.py` cleanly."
)


class PolicyReplayError(RuntimeError):
    """Raised when the sandboxed replay worker rejects a policy."""


def _rsi_ro_binds() -> dict[Path, str]:
    return {RSI_DIR / name: f"{RSI_LIB_ROOT}/rsi/{name}" for name in RSI_LIB_FILES}


def _build_policy_dev_prompt(
    *,
    cycle_id: str,
    step: int,
    steps: int,
    max_parallelism: int,
    lambda_parallel: float,
    prompt_path: Path,
) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8")
    return template.format(
        step=step,
        steps=steps,
        cycle_id=cycle_id,
        lambda_parallel=lambda_parallel,
        max_parallelism=max_parallelism,
    )


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


def _check_text(run: _Run, step_dir: Path, text: str) -> tuple[dict[str, Any], str]:
    """Audit the agent's source, then replay it; raises on any violation."""
    audit_policy_source(text)
    audit_no_world_literals(text, run.history)
    candidate = step_dir / "candidate.py"
    candidate.write_text(text, encoding="utf-8")
    return run.replay(candidate, step_dir / "replay", run.world_paths, run.evidence)


def _run_agent_step(run: _Run, step: int, seed_text: str) -> dict[str, Any]:
    """Let the agent rewrite the best policy so far; one repair attempt on validation failure."""
    step_dir = run.out_root / f"step_{step:02d}"
    ws = step_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "method.py").write_text(seed_text, encoding="utf-8")
    base_prompt = _build_policy_dev_prompt(
        cycle_id=run.cycle_id,
        step=step,
        steps=run.steps,
        max_parallelism=run.max_parallelism,
        lambda_parallel=run.lambda_parallel,
        prompt_path=run.prompt_path,
    )
    rec: dict[str, Any] = {
        "step": step,
        "slot": f"m{step}_llm",
        "status": "rejected",
        "error": None,
        "attempts": 0,
        "text": None,
        "result": None,
        "traces": None,
        "path": None,
    }
    prompt = base_prompt
    for attempt in (1, 2):
        rec["attempts"] = attempt
        try:
            agent = run_sandboxed_agent(
                prompt_text=prompt,
                workspace_dir=ws,
                evidence_dir=run.evidence,
                logs_dir=step_dir / "logs" / f"attempt_{attempt}",
                agent_cmd=run.agent_cmd,
                agent=run.agent,
                timeout=run.timeout,
                use_bwrap=run.use_bwrap,
                extra_ro_binds=_rsi_ro_binds() if run.use_bwrap else None,
                env_overrides={"PYTHONPATH": RSI_LIB_ROOT} if run.use_bwrap else None,
            )
        except SandboxTimeoutError as exc:
            rec["error"] = str(exc)
            return rec
        run.results.append(agent)
        if agent.returncode != 0:
            rec["error"] = f"Agent exited {agent.returncode}: {agent.stderr[-300:]}"
            return rec
        try:
            text = read_agent_output(ws, "method.py", max_bytes=MAX_METHOD_BYTES)
        except SandboxOutputError as exc:
            rec["error"] = str(exc)
            return rec
        rec["text"] = text
        try:
            rec["result"], rec["traces"] = _check_text(run, step_dir, text)
        except (SandboxTimeoutError, SandboxOutputError) as exc:
            rec["error"] = str(exc)
            return rec
        except (PolicyAuditError, PolicyReplayError) as exc:
            rec["error"] = str(exc)
            prompt = base_prompt + REPAIR_SUFFIX.format(error=rec["error"])
            continue
        rec["path"] = step_dir / "candidate.py"
        rec["status"] = "valid"
        rec["error"] = None
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
    }


def _summary_variant(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot": rec["slot"],
        "step": rec["step"],
        "status": rec["status"],
        "mean_v": rec.get("mean_v"),
        "delta_vs_best": rec.get("delta_vs_best"),
        "error": rec["error"],
        "attempts": rec["attempts"],
        "policy_path": str(rec["path"]) if rec["path"] is not None else None,
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
    timeout: float = 240.0,
    use_bwrap: bool = True,
    prompt_path: Path | None = None,
) -> dict[str, Any]:
    """Develop m-1 sequential rewrites of the incumbent; select argmax mean V on history."""
    if not history:
        raise ValueError("Policy development requires at least 1 frozen DiscoveryTree in history")
    if m < 2:
        raise ValueError("m must be >= 2 (incumbent + at least 1 rewritten candidate)")

    inc_src = Path(incumbent_path) if incumbent_path is not None else INCUMBENT_TEMPLATE
    out_root = Path(output_dir)
    evidence = out_root / "evidence"
    world_paths = [
        save_tree_json(t, evidence / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(history)
    ]
    (evidence / "live_cycle_manifests.json").write_text(
        json.dumps(live_manifests or [], indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "reference").mkdir(parents=True, exist_ok=True)
    for name in ("api.py", "tree.py", "objective.py"):
        shutil.copyfile(RSI_DIR / name, evidence / "reference" / name)

    spec = agent or default_agent_spec()
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
    )

    inc_text = inc_src.read_text(encoding="utf-8")
    inc_dir = out_root / "step_00"
    inc_dir.mkdir(parents=True, exist_ok=True)
    (inc_dir / "method.py").write_text(inc_text, encoding="utf-8")
    shutil.copyfile(inc_src, inc_dir / "candidate.py")
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
        "path": inc_dir / "candidate.py",
        "mean_v": inc_result["mean_v"],
        "delta_vs_best": 0.0,
    }
    records: list[dict[str, Any]] = [best]
    _write_history(evidence, best, _verdict(best))

    for step in range(1, m):
        _write_best_evidence(evidence, best)
        rec = _run_agent_step(run, step, best["text"])
        if rec["status"] == "valid":
            rec["mean_v"] = rec["result"]["mean_v"]
            rec["delta_vs_best"] = rec["mean_v"] - best["mean_v"]
            rec["status"] = "accepted" if rec["mean_v"] > best["mean_v"] else "not_better"
            if rec["status"] == "accepted":
                best = rec
        _write_history(evidence, rec, _verdict(rec))
        records.append(rec)

    valid = [r for r in records if r["status"] in ("incumbent", "accepted", "not_better")]
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
        "selected_policy_path": str(winner_path),
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
        "variants": [_summary_variant(r) for r in records],
        "agent": agent_record(None if agent_cmd is not None else spec, run.results),
    }
    (out_root / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
