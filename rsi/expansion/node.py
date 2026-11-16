"""One Dream-RSI node expansion: fork the parent's snapshot, run the agent, freeze, evaluate."""

from __future__ import annotations

import json
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsi.agents import AgentRunResult, run_sandboxed_agent, timed_out_result
from rsi.expansion.context import DigestRegistry, NodeContext, NodeJob, evaluate_snapshot
from rsi.expansion.outputs import agent_output_reason, redact_node_paths
from rsi.expansion.states import EvalStatus, FailClass
from rsi.expansion.verdict import classify, per_task_counts
from rsi.online_evidence import build_discovery_prompt, discovery_prompt_path
from rsi.sandbox import SandboxTimeoutError, discovery_visibility, repo_source_binds
from rsi.workspace import (
    STORE_NAME,
    WorkspaceError,
    code_digest,
    diff_vs_parent,
    export,
    fork,
    snapshot,
    validate_outputs,
    write_evidence,
)

AGENT_STDERR_TAIL = 600


def _rm_tree(path: Path) -> None:
    """Remove a tree, restoring owner write permission on anything the agent made read-only."""

    def _onerror(func: Callable[..., Any], target: str, _exc: object) -> None:
        Path(target).parent.chmod(stat.S_IRWXU)
        Path(target).chmod(stat.S_IRWXU)
        func(target)

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, onerror=_onerror)


def _fresh(path: Path) -> Path:
    _rm_tree(path)
    return path


def _run_agent(
    ctx: NodeContext, job: NodeJob, work: Path, evidence: Path, prompt: str
) -> tuple[AgentRunResult | None, bool]:
    """Run the discovery agent once; return (result or None, timed_out)."""
    visibility = discovery_visibility(ctx.study_dir)
    try:
        res = run_sandboxed_agent(
            prompt_text=prompt,
            workspace_dir=work,
            evidence_dir=evidence,
            logs_dir=job.node_dir / "logs",
            agent_cmd=ctx.agent_cmd,
            agent=ctx.agent,
            timeout=ctx.agent_timeout,
            use_bwrap=ctx.use_bwrap,
            gpu=True,
            source_ro=repo_source_binds(study_dir=ctx.study_dir, exclude=visibility.repo_exclude),
            gpu_slots_dir=ctx.gpu_slots_dir,
            study_dir=ctx.study_dir,
            env_overrides={"RSI_GPU_SLOTS": str(ctx.gpu_slots)},
        )
    except SandboxTimeoutError:
        # Recover the cost and turns the killed run already paid for, so the budget guard sees them.
        return timed_out_result(job.node_dir / "logs", sandboxed=ctx.use_bwrap), True
    return res, False


def _record(job: NodeJob, fail_class: str, stats: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Assemble one node's record: identity, outcome, measurements and agent accounting."""
    return {
        "cell_id": job.cell_id,
        "parent_id": job.parent_id,
        "parent_commit": job.parent_commit,
        "commit": None,
        "fail_class": fail_class,
        "evaluated": False,
        "score": 0.0,
        "successes": None,
        "episodes": None,
        "sizes": {},
        "digest": None,
        "workspace_sha256": None,
        "status": None,
        "error": None,
        "eval_seconds": None,
        "agent_output_invalid": False,
        **stats,
        **fields,
    }


def _write_score(job: NodeJob, record: dict[str, Any]) -> dict[str, Any]:
    """Write `score.json` and, when the node failed, `error.txt`; both carry no host paths."""
    if record.get("error"):
        record["error"] = redact_node_paths(str(record["error"]), job.node_dir)
        (job.node_dir / "error.txt").write_text(record["error"].rstrip() + "\n", encoding="utf-8")
    (job.node_dir / "score.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _agent_stats(res: AgentRunResult | None, timed_out: bool) -> dict[str, Any]:
    return {
        "agent_timed_out": timed_out,
        "agent_seconds": res.elapsed_seconds if res is not None else None,
        "agent_cost_usd": res.cost_usd if res is not None else None,
        "agent_turns": res.num_turns if res is not None else None,
    }


def _agent_failure_error(res: AgentRunResult) -> str:
    """Describe a nonzero agent exit: backend, model, exit code and the stderr tail."""
    tail = " ".join(res.stderr.split())[-AGENT_STDERR_TAIL:]
    return (
        f"agent {res.backend}/{res.model or 'default'} exited {res.returncode}: "
        f"{tail or 'no stderr'}"
    )


def expand_node(ctx: NodeContext, job: NodeJob, registry: DigestRegistry) -> dict[str, Any]:
    """Expand one slot and return its record; a node with `score.json` is never re-run."""
    score_path = job.node_dir / "score.json"
    if score_path.is_file():
        return json.loads(score_path.read_text(encoding="utf-8"))
    job.node_dir.mkdir(parents=True, exist_ok=True)
    if job.parent_commit is None:
        error = "parent node has no committed snapshot"
        return _write_score(
            job,
            _record(job, FailClass.IMPLEMENTATION_FAILURE, _agent_stats(None, False), error=error),
        )
    work = fork(job.store, job.parent_commit, _fresh(job.node_dir / "workspace"))
    evidence = _fresh(job.node_dir / "evidence")
    write_evidence(evidence, job.evidence_nodes, job.tree_index)
    prompt = build_discovery_prompt(
        cell_id=job.cell_id,
        cycle_id=ctx.cycle_id,
        parent_id=job.parent_id,
        smoke_trials=ctx.smoke_trials,
        smoke_gate=ctx.smoke_gate,
        best_score=job.best_score,
        gpu_slots=ctx.gpu_slots,
        minutes=ctx.agent_timeout / 60.0,
        prompt_path=discovery_prompt_path(ctx.study_dir),
    )
    res, timed_out = _run_agent(ctx, job, work, evidence, prompt)
    stats = _agent_stats(res, timed_out)
    try:
        # A nonzero agent exit (expired key, renamed model, missing auth file) is an agent
        # failure, not a model-quality result: the parent fork is untouched, so validating it
        # would report "proposal.md missing" or "identical to parent" as
        # `implementation_failure`. Record `agent_failure` and continue the cycle, so healthy
        # siblings still run and the tree keeps the distinction instead of aborting the round.
        # A timed-out run is not an exit: its sidecar records returncode -1, and the node is
        # classified by what its workspace holds.
        if res is not None and res.returncode != 0 and not timed_out:
            return _write_score(
                job,
                _record(
                    job,
                    FailClass.AGENT_FAILURE,
                    stats,
                    error=_agent_failure_error(res),
                    agent_returncode=res.returncode,
                    agent_backend=res.backend,
                    agent_model=res.model,
                ),
            )
        # A placeholder-only transcript is a backend/model-output fault, not a proposal-quality
        # result: the workspace is not evaluated and the node is not an `implementation_failure`.
        invalid = agent_output_reason(res)
        if invalid is not None:
            return _write_score(
                job,
                _record(
                    job,
                    FailClass.AGENT_FAILURE,
                    stats,
                    error=invalid,
                    agent_output_invalid=True,
                    agent_returncode=res.returncode if res is not None else None,
                    agent_backend=res.backend if res is not None else None,
                    agent_model=res.model if res is not None else None,
                ),
            )
        validate_outputs(work)
        commit = snapshot(job.store, work, job.parent_commit, job.cell_id)
        shutil.copyfile(work / "proposal.md", job.node_dir / "proposal.md")
        (job.node_dir / "diff.patch").write_text(
            diff_vs_parent(job.store, commit), encoding="utf-8"
        )
        digest = code_digest(job.store, commit)
    except (WorkspaceError, OSError, UnicodeDecodeError) as exc:
        fail = FailClass.AGENT_TIMEOUT if timed_out else FailClass.IMPLEMENTATION_FAILURE
        return _write_score(job, _record(job, fail, stats, error=str(exc)))
    finally:
        _rm_tree(work)

    owner = registry.claim(digest, job.cell_id)
    if owner is not None and owner != job.cell_id:
        error = f"workspace identical to {owner}"
        return _write_score(
            job,
            _record(
                job,
                FailClass.IMPLEMENTATION_FAILURE,
                stats,
                commit=commit,
                digest=digest,
                error=error,
            ),
        )
    exported = _fresh(job.node_dir / "export")
    export(job.store, commit, exported)
    try:
        verdict_fields = _evaluate(ctx, exported, job.node_dir / "eval", "smoke", ctx.smoke_gate)
    finally:
        _rm_tree(exported)
    return _write_score(
        job, _record(job, stats=stats, commit=commit, digest=digest, **verdict_fields)
    )


def _evaluate(
    ctx: NodeContext, snapshot_dir: Path, out_dir: Path, phase: str, gate: int
) -> dict[str, Any]:
    """Evaluate a frozen snapshot in `phase`; return fail class, score and measurements."""
    t0 = time.monotonic()
    try:
        raw = evaluate_snapshot(ctx.evaluator, snapshot_dir, phase, out_dir)
    except Exception as exc:  # noqa: BLE001 - one failing evaluation must not abort the cycle
        raw = {
            "status": EvalStatus.EVALUATOR_FAILURE.value,
            "error": f"{type(exc).__name__}: {exc}",
        }
    verdict = classify(raw, gate)
    return {
        "fail_class": verdict.fail_class,
        "evaluated": raw.get("status") == EvalStatus.OK,
        "score": verdict.score,
        "successes": verdict.successes,
        "episodes": verdict.episodes,
        "sizes": verdict.sizes,
        "per_task": per_task_counts(raw),
        "workspace_sha256": raw.get("workspace_sha256"),
        "status": raw.get("status"),
        "error": verdict.error,
        "eval_seconds": round(time.monotonic() - t0, 3),
    }


def promote_node(ctx: NodeContext, node_dir: Path, smoke: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a smoke-passing node's commit at development size; write `development.json`.

    The cycle store sits two levels above the node directory (`<cycle>/nodes/<cell>`).
    """
    store = node_dir.parent.parent / STORE_NAME
    exported = _fresh(node_dir / "export")
    export(store, smoke["commit"], exported)
    try:
        fields = _evaluate(ctx, exported, node_dir / "development", "development", ctx.dev_gate)
    finally:
        _rm_tree(exported)
    record = {
        "cell_id": smoke["cell_id"],
        "commit": smoke["commit"],
        "digest": smoke.get("digest"),
        **fields,
    }
    # Per-task ids come from the evaluator; they are redacted like every other committed string.
    record["per_task"] = {
        redact_node_paths(str(task_id), node_dir): dict(counts)
        for task_id, counts in (record.get("per_task") or {}).items()
    }
    if record.get("error"):
        record["error"] = redact_node_paths(str(record["error"]), node_dir)
    (node_dir / "development.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record
