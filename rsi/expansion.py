"""One Dream-RSI node expansion: fork the parent's snapshot, run the agent, freeze, evaluate."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.agent_backends import REPO_ROOT, AgentSpec
from rsi.api import (
    AGENT_FAILURE,
    AGENT_TIMEOUT,
    IMPLEMENTATION_FAILURE,
    OK,
    RUNTIME_FAILURE,
    gate_fail_class,
)
from rsi.online_evidence import build_discovery_prompt
from rsi.runner import PHASES_ALLOWED
from rsi.sandbox import (
    AgentRunResult,
    SandboxTimeoutError,
    discovery_visibility,
    repo_source_binds,
    run_sandboxed_agent,
    run_sandboxed_evaluator,
    timed_out_result,
)
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

Evaluator = Callable[[Path, str, Path], dict[str, Any]]
EVAL_ERROR_TAIL = 4000
AGENT_STDERR_TAIL = 600
# The Mistral openai-completions path can coerce a structured content block to the JS string
# `[object Object]`; a run whose output is mostly that token did not return usable text.
PLACEHOLDER_TOKEN = "[object Object]"
PLACEHOLDER_MIN_TOKENS = 4
PLACEHOLDER_SHARE_LIMIT = 0.5
# A final answer shorter than this cannot describe a proposal; such a fragment is not usable
# text, so the stdout check still runs (the observed failure left a lone "[" behind).
USABLE_TEXT_MIN_CHARS = 16
# Base paths that must never appear in a committed record, with the marker that replaces them.
HOST_PATH_MARKERS: tuple[tuple[Path, str], ...] = (
    (REPO_ROOT, "<repo>"),
    (Path.home(), "<home>"),
    (Path(tempfile.gettempdir()), "<tmp>"),
)


class AgentOutputInvalid(RuntimeError):
    """Raised when every agent in a batch returned placeholder-only output; stops the cycle."""


def placeholder_output_reason(*texts: str | None) -> str | None:
    """Reason when an agent's text is placeholder-only or mostly `[object Object]`, else None."""
    for text in texts:
        if not text:
            continue
        tokens = text.count(PLACEHOLDER_TOKEN)
        if tokens < PLACEHOLDER_MIN_TOKENS:
            continue
        placeholder_chars = tokens * len(PLACEHOLDER_TOKEN)
        rest = sum(1 for char in text.replace(PLACEHOLDER_TOKEN, "") if not char.isspace())
        share = placeholder_chars / (placeholder_chars + rest)
        if share >= PLACEHOLDER_SHARE_LIMIT:
            return (
                f"agent output is {share:.0%} `{PLACEHOLDER_TOKEN}` placeholders "
                f"({tokens} tokens): the backend returned no usable text"
            )
    return None


def _usable_text(text: str) -> bool:
    """True when an agent's final text carries more than a placeholder fragment."""
    rest = text.replace(PLACEHOLDER_TOKEN, "")
    return sum(1 for char in rest if not char.isspace()) >= USABLE_TEXT_MIN_CHARS


def _agent_output_reason(res: AgentRunResult | None) -> str | None:
    """Placeholder reason for one run: the final text decides, the stream is only a fallback.

    Placeholder-heavy thinking or tool output does not invalidate a usable final answer. The
    stdout tail is consulted only when the final text is missing, a fragment, or placeholders.
    """
    if res is None:
        return None
    reason = placeholder_output_reason(res.text)
    if reason is not None:
        return reason
    if _usable_text(res.text):
        return None
    return placeholder_output_reason(res.stdout)


@dataclass(frozen=True)
class NodeContext:
    """Cycle-wide settings shared by every node expansion in one cycle."""

    cycle_id: str
    agent: AgentSpec
    agent_timeout: float
    eval_timeout: float
    gpu_slots_dir: Path
    gpu_slots: int
    smoke_trials: int
    smoke_gate: int
    dev_gate: int
    evaluator: Evaluator
    study_dir: Path
    agent_cmd: str | list[str] | None = None
    use_bwrap: bool = True


@dataclass(frozen=True)
class NodeJob:
    """Inputs for one expansion: its parent's frozen snapshot and the frozen batch evidence."""

    cell_id: str
    parent_id: str
    store: Path
    parent_commit: str | None
    node_dir: Path
    evidence_nodes: list[dict[str, Any]]
    tree_index: dict[str, Any]
    best_score: float


@dataclass(frozen=True)
class Verdict:
    """Fail class, score and measurements derived from one evaluator result."""

    fail_class: str
    score: float
    successes: int | None
    episodes: int | None
    size_reduction: float | None
    error: str | None
    sizes: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigestRegistry:
    """Digests of every node snapshot claimed in this cycle, shared by concurrent expansions."""

    owners: dict[str, str]
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self, digest: str, cell_id: str) -> str | None:
        """Register `digest` for `cell_id`; return the existing owner instead when there is one."""
        with self._lock:
            owner = self.owners.get(digest)
            if owner is None:
                self.owners[digest] = cell_id
            return owner


def sandboxed_evaluator(
    eval_timeout: float, gpu_slots_dir: Path, gpu_slots: int, study_dir: Path | None = None
) -> Evaluator:
    """Evaluator that runs the fixed `solution` evaluator in the GPU sandbox.

    `study_dir` names the study whose `protocol.json` the evaluator reads.
    """

    def _run(snapshot_dir: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        return run_sandboxed_evaluator(
            snapshot_dir,
            phase,
            out_dir,
            timeout=eval_timeout,
            gpu_slots_dir=gpu_slots_dir,
            gpu_slot_count=gpu_slots,
            study_dir=study_dir,
        )

    return _run


def evaluate_snapshot(
    evaluator: Evaluator, snapshot_dir: Path, phase: str, out_dir: Path
) -> dict[str, Any]:
    """Run the evaluator on one frozen snapshot; only the protocol's phases may be requested."""
    if phase not in PHASES_ALLOWED:
        raise ValueError(f"phase {phase!r} not allowed; expected one of {sorted(PHASES_ALLOWED)}")
    return evaluator(snapshot_dir, phase, out_dir)


def classify(raw: dict[str, Any], gate: int) -> Verdict:
    """Map one evaluator result (solution.py `result.json` fields) to a fail class and score."""
    status = raw.get("status")
    if status == OK:
        try:
            succ = int(raw["successes"])
            episodes = int(raw["episodes"])
            measured = raw["sizes"]
            reduction = float(measured["size_reduction"])
        except (KeyError, TypeError, ValueError):
            return Verdict(
                RUNTIME_FAILURE,
                0.0,
                None,
                None,
                None,
                "ok result lacks successes, episodes or size_reduction",
            )
        sizes = {
            "resident_bytes": measured.get("resident_bytes"),
            "baseline_resident_bytes": measured.get("baseline_resident_bytes"),
            "size_reduction": reduction,
        }
        fail_class = gate_fail_class(succ, gate)
        if fail_class == OK:
            return Verdict(OK, reduction, succ, episodes, reduction, None, sizes)
        return Verdict(
            fail_class, 0.0, succ, episodes, reduction, f"smoke={succ}/{episodes}<{gate}", sizes
        )
    error = str(raw.get("error") or status or "no status")[-EVAL_ERROR_TAIL:]
    if status in ("solution_error", "size_violation"):
        return Verdict(IMPLEMENTATION_FAILURE, 0.0, None, None, None, error)
    return Verdict(RUNTIME_FAILURE, 0.0, None, None, None, error)


def _per_task(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per-task successes and episodes from the evaluator's rollout block, if it reported one."""
    rollout = raw.get("rollout")
    per_task = rollout.get("per_task") if isinstance(rollout, dict) else None
    out: dict[str, dict[str, int]] = {}
    for task_id, counts in (per_task or {}).items():
        if not isinstance(counts, dict):
            continue
        successes = counts.get("successes")
        episodes = counts.get("episodes")
        if isinstance(successes, int) and isinstance(episodes, int):
            out[str(task_id)] = {"successes": successes, "episodes": episodes}
    return out


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


def redact_host_paths(text: str, *bases: tuple[Path, str]) -> str:
    """Replace each absolute base path in `text` with its marker, longest path first."""
    out = text
    for path, marker in sorted(bases, key=lambda item: -len(str(item[0]))):
        if str(path) != "/":
            out = out.replace(str(path), marker)
    return out


def _redact(text: str, node_dir: Path) -> str:
    """Replace absolute node, cycle, study and host paths in a message with markers.

    The node directory is `<study>/cycles/<cycle>/nodes/<cell>`, so the enclosing paths are
    derived from it rather than passed in.
    """
    node = Path(node_dir).resolve()
    bases = [(node, "<node>"), *HOST_PATH_MARKERS]
    cycle = node.parent.parent
    if cycle.parent.name == "cycles":
        bases.extend([(cycle, "<cycle>"), (cycle.parent.parent, "<study>")])
    return redact_host_paths(text, *bases)


def _write_score(job: NodeJob, record: dict[str, Any]) -> dict[str, Any]:
    """Write `score.json` and, when the node failed, `error.txt`; both carry no host paths."""
    if record.get("error"):
        record["error"] = _redact(str(record["error"]), job.node_dir)
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
            job, _record(job, IMPLEMENTATION_FAILURE, _agent_stats(None, False), error=error)
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
                    AGENT_FAILURE,
                    stats,
                    error=_agent_failure_error(res),
                    agent_returncode=res.returncode,
                    agent_backend=res.backend,
                    agent_model=res.model,
                ),
            )
        # A placeholder-only transcript is a backend/model-output fault, not a proposal-quality
        # result: the workspace is not evaluated and the node is not an `implementation_failure`.
        invalid = _agent_output_reason(res)
        if invalid is not None:
            return _write_score(
                job,
                _record(
                    job,
                    AGENT_FAILURE,
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
        fail = AGENT_TIMEOUT if timed_out else IMPLEMENTATION_FAILURE
        return _write_score(job, _record(job, fail, stats, error=str(exc)))
    finally:
        _rm_tree(work)

    owner = registry.claim(digest, job.cell_id)
    if owner is not None and owner != job.cell_id:
        error = f"workspace identical to {owner}"
        return _write_score(
            job,
            _record(job, IMPLEMENTATION_FAILURE, stats, commit=commit, digest=digest, error=error),
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
        raw = {"status": "evaluator_failure", "error": f"{type(exc).__name__}: {exc}"}
    verdict = classify(raw, gate)
    return {
        "fail_class": verdict.fail_class,
        "evaluated": raw.get("status") == OK,
        "score": verdict.score,
        "successes": verdict.successes,
        "episodes": verdict.episodes,
        "sizes": verdict.sizes,
        "per_task": _per_task(raw),
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
        _redact(str(task_id), node_dir): dict(counts)
        for task_id, counts in (record.get("per_task") or {}).items()
    }
    if record.get("error"):
        record["error"] = _redact(str(record["error"]), node_dir)
    (node_dir / "development.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record
