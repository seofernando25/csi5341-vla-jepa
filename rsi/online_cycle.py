"""Dream-RSI online cycle driver: plan grid, expand the tree, promote, write manifest; plus the outer loop."""

from __future__ import annotations

import functools
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from rsi.agent_backends import REPO_ROOT, AgentSpec, load_agent_spec
from rsi.api import GridPlan, GridPlanningContext, PrefixView
from rsi.expansion import (
    HOST_PATH_MARKERS,
    AgentOutputInvalid,
    Evaluator,
    NodeContext,
    promote_node,
    redact_host_paths,
    sandboxed_evaluator,
)
from rsi.online_evidence import agent_totals, recorded_utc
from rsi.online_question import OnlineDiscoveryQuestion
from rsi.policy_dev import INCUMBENT_TEMPLATE, load_policy_class, save_tree_json
from rsi.runner import DREAM_STUDY_DIR, check_phase_lock, load_cycle_config, load_protocol
from rsi.watch_events import redact

DREAM_DIR = DREAM_STUDY_DIR


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    """Trimmed stdout of one `git -C repo` call, or None when git fails."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_revision(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD of the repo at run start, or null with `code_dirty: true` when the tree is dirty."""
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    dirty = status is None or bool(status)
    return {"code_revision": None if dirty else head, "code_dirty": dirty}


@functools.cache
def environment() -> dict[str, Any]:
    """Python, torch and CUDA versions of the running interpreter, read once per process."""
    torch_version, cuda_version = _torch_versions()
    return {"python": platform.python_version(), "torch": torch_version, "cuda": cuda_version}


def _torch_versions() -> tuple[str | None, str | None]:
    """`torch.__version__` and its CUDA build, or nulls when torch cannot be imported."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a missing or broken torch must not stop the loop
        return None, None
    return torch.__version__, torch.version.cuda


def gpu_inventory() -> dict[str, Any]:
    """Product name and count of the host GPUs, or nulls when nvidia-smi cannot be queried."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"name": None, "count": None}
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode or not names:
        return {"name": None, "count": None}
    return {"name": names[0], "count": len(names)}


def _study_repo(study_dir: Path) -> Path:
    """Git toplevel of the study directory, or the project repo when it is not in one."""
    top = _git(study_dir, "rev-parse", "--show-toplevel")
    return Path(top) if top else REPO_ROOT


def provenance(study_dir: Path) -> dict[str, Any]:
    """Frozen run inputs: code revision, protocol hash, checkpoint, GPU, environment, seed, suite."""
    path = Path(study_dir) / "protocol.json"
    proto = load_protocol(study_dir) if path.is_file() else {}
    return {
        **code_revision(_study_repo(study_dir)),
        "protocol_sha256": _sha256(path) if path.is_file() else None,
        "checkpoint": proto.get("checkpoint"),
        "checkpoint_revision": proto.get("checkpoint_revision"),
        "suite": proto.get("suite"),
        "seed": proto.get("seed"),
        "gpu": gpu_inventory(),
        "environment": environment(),
    }


def relative_path(path: Path, study_dir: Path) -> str:
    """Path relative to the study or repo root, so records carry no absolute host path."""
    for base in (Path(study_dir), REPO_ROOT):
        try:
            return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            continue
    return Path(path).name


def redact_error(exc: BaseException, study_dir: Path) -> str:
    """One-line exception text with host paths replaced and credentials masked.

    A driver exception can name any file it touched, so the study and host roots are replaced
    before the text reaches the manifest.
    """
    text = redact_host_paths(
        f"{type(exc).__name__}: {exc}", (Path(study_dir).resolve(), "<study>"), *HOST_PATH_MARKERS
    )
    return redact(text)


def _recent_manifests(study_dir: Path) -> tuple[dict[str, Any], ...]:
    cycles_dir = study_dir / "cycles"
    if not cycles_dir.is_dir():
        return ()
    out: list[dict[str, Any]] = []
    for sub in sorted(cycles_dir.iterdir()):
        mp = sub / "live_cycle_manifest.json"
        if mp.is_file():
            out.append(json.loads(mp.read_text(encoding="utf-8")))
    return tuple(out)


def _recent_beta_sweeps(study_dir: Path) -> tuple[dict[str, Any], ...]:
    policies_dir = study_dir / "policies"
    if not policies_dir.is_dir():
        return ()
    out: list[dict[str, Any]] = []
    for sub in sorted(policies_dir.iterdir()):
        sp = sub / "evidence" / "beta_sweep.json"
        if sp.is_file():
            out.append(json.loads(sp.read_text(encoding="utf-8")))
    return tuple(out)


def _grid_plan(cfg_grid: dict[str, Any], policy: Any, context: GridPlanningContext) -> GridPlan:
    if cfg_grid:
        return GridPlan(
            max_branches=int(cfg_grid["max_branches"]),
            max_attempts_per_branch=int(cfg_grid["max_attempts_per_branch"]),
            max_parallelism=int(cfg_grid["max_parallelism"]),
            rounds_cap=int(cfg_grid["rounds_cap"]),
            rationale=str(cfg_grid.get("rationale", "pre-registered static grid plan")),
        )
    return policy.plan_grid(context)


def _clamp_plan(
    plan: GridPlan, hard_max_branches: int, hard_max_depth: int, max_parallelism: int
) -> GridPlan:
    return GridPlan(
        max_branches=max(1, min(hard_max_branches, int(plan.max_branches))),
        max_attempts_per_branch=max(0, min(hard_max_depth, int(plan.max_attempts_per_branch))),
        max_parallelism=max(1, min(max_parallelism, int(plan.max_parallelism))),
        rounds_cap=max(1, int(plan.rounds_cap)),
        rationale=plan.rationale,
    )


def _promote(
    ctx: NodeContext, cycle_dir: Path, passing: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Development-size evaluation of the top smoke-passing nodes, in rank order."""
    ranked = sorted(passing, key=lambda r: (-float(r["score"]), -int(r["successes"])))
    return [promote_node(ctx, cycle_dir / "nodes" / rec["cell_id"], rec) for rec in ranked[:top_k]]


def run_online_cycle(
    *,
    cycle_id: str,
    study_dir: Path = DREAM_DIR,
    policy_path: Path | None = None,
    iteration: int = 0,
    probe_budget: int = 8,
    max_parallelism: int = 4,
    hard_max_branches: int = 8,
    hard_max_depth: int = 4,
    promote_top_k: int = 1,
    max_hours: float | None = None,
    max_agent_usd: float | None = None,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    agent_timeout: float | None = None,
    eval_timeout: float | None = None,
    use_bwrap: bool = True,
    evaluator: Evaluator | None = None,
    grid_plan: GridPlan | None = None,
    provenance_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one online Dream-RSI cycle (`plan grid -> expand tree -> promote -> manifest`).

    `provenance_block` is the loop's start-of-run provenance. Without it the block is computed
    here, before this cycle writes any study evidence, so `code_revision` is the commit at start.
    """
    study_root = Path(study_dir)
    prov = provenance_block if provenance_block is not None else provenance(study_root)
    cycle_dir = study_root / "cycles" / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cycle_config(study_root)
    check_phase_lock("smoke", study_root)

    pol_path = Path(policy_path) if policy_path is not None else INCUMBENT_TEMPLATE
    policy = load_policy_class(pol_path, cycle_id)({})
    baked_beta = getattr(policy, "beta", None)

    manifests = _recent_manifests(study_root)
    context = GridPlanningContext(
        iteration=iteration,
        max_parallelism=max_parallelism,
        recent_live_manifests=manifests[-3:],
        recent_beta_sweeps=_recent_beta_sweeps(study_root)[-3:],
        default_budget=probe_budget,
    )
    plan = grid_plan or _grid_plan(cfg.grid_plan, policy, context)
    eff_plan = _clamp_plan(plan, hard_max_branches, hard_max_depth, max_parallelism)
    (cycle_dir / "grid_plan.json").write_text(
        json.dumps(asdict(eff_plan), indent=2) + "\n", encoding="utf-8"
    )

    spec = agent or load_agent_spec(study_root)
    discovery_spec = replace(spec, tools=cfg.discovery_tools)
    gpu_dir = cycle_dir / "gpu-slots"
    node_eval_timeout = float(eval_timeout if eval_timeout is not None else cfg.eval_timeout_s)
    ctx = NodeContext(
        cycle_id=cycle_id,
        agent=discovery_spec,
        agent_timeout=float(agent_timeout if agent_timeout is not None else cfg.agent_timeout_s),
        eval_timeout=node_eval_timeout,
        gpu_slots_dir=gpu_dir,
        gpu_slots=cfg.gpu_slots,
        smoke_trials=cfg.smoke_trials,
        smoke_gate=cfg.smoke_gate,
        dev_gate=cfg.dev_gate,
        evaluator=evaluator or sandboxed_evaluator(node_eval_timeout, gpu_dir, cfg.gpu_slots),
        agent_cmd=agent_cmd,
        use_bwrap=use_bwrap,
    )
    question = OnlineDiscoveryQuestion(
        cycle_id=cycle_id, grid_plan=eff_plan, cycle_dir=cycle_dir, ctx=ctx
    )
    solve_error: str | None = None
    stop_reason: str | None = None
    try:
        policy.solve(PrefixView(question), budget=probe_budget)
    except AgentOutputInvalid as exc:
        solve_error = redact_error(exc, study_root)
        stop_reason = "agent_output_invalid"
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest; the cycle still writes its outputs
        solve_error = redact_error(exc, study_root)
    finally:
        tree = question.export_tree()
        save_tree_json(tree, cycle_dir / "tree.json")

    passing = [r for r in question.node_records.values() if r.get("fail_class") == "ok"]
    dev_results = _promote(ctx, cycle_dir, passing, promote_top_k) if promote_top_k > 0 else []

    nodes = question.manifest_nodes()
    prev_best = max((m.get("best_score", 0.0) for m in manifests), default=0.0)
    manifest = {
        "protocol": "dream-rsi-online-cycle",
        "cycle_id": cycle_id,
        "iteration": iteration,
        "run": {
            "probe_budget": probe_budget,
            "max_parallelism": max_parallelism,
            "max_hours": max_hours,
            "max_agent_usd": max_agent_usd,
            "promote_top_k": promote_top_k,
        },
        "provenance": prov,
        "policy_path": relative_path(pol_path, study_root),
        "policy_sha256": _sha256(pol_path),
        "baked_in_beta": baked_beta,
        "grid_plan": asdict(eff_plan),
        "rounds_cap": question.rounds_cap,
        "probes_used": question.budget_spent,
        "decision_rounds": question.decision_rounds,
        "effective_sequential_rounds": question.effective_sequential_rounds,
        "solve_error": solve_error,
        "stop_reason": stop_reason,
        "opened_branches": question.opened_branches(),
        "best_score": round(question.best_so_far, 6),
        "deepest_improvement": question.deepest_improvement(),
        "plateaued": len(manifests) >= 1 and question.best_so_far <= prev_best,
        "world_id": tree.world_id,
        "world_nodes": tree.num_nonroot,
        "gpu_slots": cfg.gpu_slots,
        "agent_timeout_s": ctx.agent_timeout,
        "eval_timeout_s": ctx.eval_timeout,
        "totals": {
            "agent_seconds": round(sum(n.get("agent_seconds") or 0.0 for n in nodes), 3),
            "eval_seconds": round(sum(n.get("eval_seconds") or 0.0 for n in nodes), 3),
            "agent_timed_out": sum(1 for n in nodes if n.get("agent_timed_out")),
        },
        "agent": agent_totals(None if agent_cmd is not None else discovery_spec, nodes),
        "nodes": nodes,
        "development_promotions": dev_results,
        "recorded_utc": recorded_utc(),
    }
    (cycle_dir / "live_cycle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
