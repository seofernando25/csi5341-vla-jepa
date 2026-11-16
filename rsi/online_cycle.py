"""Dream-RSI online cycle driver: plan the grid, expand the tree, promote, write the manifest."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from rsi.agents import AgentSpec, load_agent_spec
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
from rsi.expansion.states import FailClass
from rsi.grid_plan import grid_caps, protocol_plan, request_plan
from rsi.online_evidence import agent_totals, recorded_utc
from rsi.online_question import OnlineDiscoveryQuestion
from rsi.paths import REPO_ROOT
from rsi.policy_dev import INCUMBENT_TEMPLATE, load_policy_class, save_tree_json
from rsi.provenance import provenance, sha256_file
from rsi.redact import redact
from rsi.runner import DREAM_STUDY_DIR, check_phase_lock, load_cycle_config

DREAM_DIR = DREAM_STUDY_DIR


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


def recent_manifests(study_dir: Path) -> tuple[dict[str, Any], ...]:
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
    `grid_plan` is a harness override of the policy's `plan_grid`; the protocol values clamp it.
    """
    study_root = Path(study_dir)
    prov = provenance_block if provenance_block is not None else provenance(study_root)
    cycle_dir = study_root / "cycles" / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cycle_config(study_root)
    check_phase_lock("smoke", study_root)

    pol_path = Path(policy_path) if policy_path is not None else INCUMBENT_TEMPLATE
    policy = load_policy_class(pol_path, cycle_id)(
        {"grid_plan": dict(cfg.grid_plan)} if cfg.grid_plan else {}
    )
    baked_beta = getattr(policy, "beta", None)

    manifests = recent_manifests(study_root)
    caps = grid_caps(
        cfg.grid_plan,
        gpu_slots=cfg.gpu_slots,
        probe_budget=probe_budget,
        max_parallelism=max_parallelism,
        hard_max_branches=hard_max_branches,
        hard_max_depth=hard_max_depth,
    )
    context = GridPlanningContext(
        iteration=iteration,
        max_parallelism=max_parallelism,
        fallback_plan=protocol_plan(cfg.grid_plan, caps),
        recent_live_manifests=manifests[-3:],
        recent_beta_sweeps=_recent_beta_sweeps(study_root)[-3:],
    )
    # The selected policy decides the next grid; the protocol values only clamp it. The record is
    # written before the tree runs, so a resumed cycle reuses the plan it started with.
    plan = request_plan(
        pol_path,
        context,
        caps,
        cycle_dir=cycle_dir,
        use_bwrap=use_bwrap,
        override=grid_plan,
    )
    eff_plan = GridPlan(**plan["applied"])

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
        evaluator=evaluator
        or sandboxed_evaluator(node_eval_timeout, gpu_dir, cfg.gpu_slots, study_dir=study_root),
        study_dir=study_root,
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

    passing = [r for r in question.node_records.values() if r.get("fail_class") == FailClass.OK]
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
        "policy_sha256": sha256_file(pol_path),
        "baked_in_beta": baked_beta,
        "grid_plan": plan,
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
