"""Dream-RSI outer loop: resumable stages (cycle, policy development), stop rules and the study lock."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsi.agents import AgentSpec, load_agent_spec
from rsi.expansion import Evaluator
from rsi.loop.records import (
    agent_cost,
    mark_cycle_done,
    mark_dev_done,
    new_iteration,
    new_state,
    policy_ref,
    recorded_policy_path,
    verified_policy,
    write_summary,
)
from rsi.loop.states import (
    CYCLE_STAGE,
    LOOP,
    POLICY_DEV_STAGE,
    STAGE_FINISHED,
    LoopStatus,
    StageStatus,
    iteration_done,
)
from rsi.loop.store import StudyLock, load_loop_state, save_loop_state, stop_requested, utc_now
from rsi.online_cycle import DREAM_DIR, recent_manifests, run_online_cycle
from rsi.policy_dev import (
    DEFAULT_POLICY_TIMEOUT_S,
    INCUMBENT_TEMPLATE,
    develop_policies_llm,
    load_tree_json,
)
from rsi.provenance import provenance
from rsi.tree import DiscoveryTree


def _spent(state: dict[str, Any]) -> tuple[float, list[str]]:
    """Sum recorded agent cost and name every finished stage whose cost is unknown."""
    total = 0.0
    unknown: list[str] = []
    for it in state["iterations"]:
        for key in ("cycle", "policy_dev"):
            stage = it[key]
            value = stage.get("agent_cost_usd")
            if value is None:
                if stage.get("status") in STAGE_FINISHED:
                    unknown.append(f"{it['cycle_id']}/{key}")
            else:
                total += float(value)
    return total, unknown


def _guard(
    study_root: Path,
    started: float,
    max_hours: float | None,
    max_agent_usd: float | None,
    state: dict[str, Any],
) -> str | None:
    """Stop reason to apply before the next stage, or None to continue."""
    if stop_requested(study_root):
        return "STOP file present"
    if max_hours is not None and time.monotonic() - started >= max_hours * 3600:
        return f"max_hours {max_hours} reached"
    if max_agent_usd is not None:
        spent, unknown = _spent(state)
        if unknown:
            # An unknown cost cannot be checked against the budget, so the guard must stop.
            return (
                f"max_agent_usd {max_agent_usd} cannot be enforced: agent cost unknown for "
                f"{', '.join(unknown)}"
            )
        if spent >= max_agent_usd:
            return f"max_agent_usd {max_agent_usd} reached (spent {spent:.6f})"
    return None


def _load_trees(study_root: Path) -> list[DiscoveryTree]:
    trees = [load_tree_json(p) for p in sorted((study_root / "cycles").glob("iter_*/tree.json"))]
    return [t for t in trees if t.num_nonroot > 0]


def _run_policy_dev(
    study_root: Path, rec: dict[str, Any], dev_fn: Callable[..., dict[str, Any]], kw: dict[str, Any]
) -> None:
    """Finish the policy-development stage for one iteration (skip when no tree has nodes)."""
    cid = rec["cycle_id"]
    selection = study_root / "policies" / cid / "selection.json"
    if selection.is_file():
        res = json.loads(selection.read_text(encoding="utf-8"))
        mark_dev_done(
            rec,
            recorded_policy_path(study_root, res["selected_policy_path"]),
            agent_cost(res),
            StageStatus.DONE,
        )
        return
    if not _load_trees(study_root):
        mark_dev_done(rec, verified_policy(rec["policy_in"]), 0.0, StageStatus.SKIPPED)
        return
    POLICY_DEV_STAGE.advance(rec["policy_dev"], StageStatus.RUNNING)
    rec["policy_dev"].update(started_utc=utc_now())
    res = dev_fn(
        _load_trees(study_root),
        cycle_id=cid,
        incumbent_path=verified_policy(rec["policy_in"]),
        output_dir=study_root / "policies" / cid,
        live_manifests=list(recent_manifests(study_root)),
        record_root=study_root,
        **kw,
    )
    mark_dev_done(
        rec,
        recorded_policy_path(study_root, res["selected_policy_path"]),
        agent_cost(res),
        StageStatus.DONE,
    )


def run_dream_rsi_loop(
    *,
    study_dir: Path = DREAM_DIR,
    cycles: int = 3,
    start_iteration: int = 0,
    probe_budget: int = 8,
    max_parallelism: int = 4,
    promote_top_k: int = 1,
    m_policy_variants: int = 4,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    agent_timeout: float | None = None,
    eval_timeout: float | None = None,
    policy_timeout: float = DEFAULT_POLICY_TIMEOUT_S,
    use_bwrap: bool = True,
    evaluator: Evaluator | None = None,
    max_hours: float | None = None,
    max_agent_usd: float | None = None,
    cycle_fn: Callable[..., dict[str, Any]] | None = None,
    policy_dev_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Iterate to `cycles` finished iterations, resuming from `loop_state.json` under a study lock."""
    study_root = Path(study_dir)
    # Before the lock file, the initial policy copy or any cycle evidence: the tree is still
    # as the operator left it, so `code_revision` is the commit the run started from.
    prov = provenance(study_root)
    spec = agent or load_agent_spec(study_root)
    run = {
        "probe_budget": probe_budget,
        "max_parallelism": max_parallelism,
        "max_hours": max_hours,
        "max_agent_usd": max_agent_usd,
        "promote_top_k": promote_top_k,
    }
    with StudyLock(study_root):
        return _run_locked(
            study_root,
            cycles=cycles,
            start_iteration=start_iteration,
            run=run,
            prov=prov,
            cycle_kw={
                "probe_budget": probe_budget,
                "max_parallelism": max_parallelism,
                "promote_top_k": promote_top_k,
                "max_hours": max_hours,
                "max_agent_usd": max_agent_usd,
                "provenance_block": prov,
                "agent_cmd": agent_cmd,
                "agent": spec,
                "agent_timeout": agent_timeout,
                "eval_timeout": eval_timeout,
                "use_bwrap": use_bwrap,
                "evaluator": evaluator,
            },
            dev_kw={
                "m": m_policy_variants,
                "max_parallelism": max_parallelism,
                "agent_cmd": agent_cmd,
                "agent": spec,
                "timeout": policy_timeout,
                "use_bwrap": use_bwrap,
            },
            max_hours=max_hours,
            max_agent_usd=max_agent_usd,
            cycle_fn=cycle_fn or run_online_cycle,
            dev_fn=policy_dev_fn or develop_policies_llm,
        )


def _run_locked(
    study_root: Path,
    *,
    cycles: int,
    start_iteration: int,
    run: dict[str, Any],
    prov: dict[str, Any],
    cycle_kw: dict[str, Any],
    dev_kw: dict[str, Any],
    max_hours: float | None,
    max_agent_usd: float | None,
    cycle_fn: Callable[..., dict[str, Any]],
    dev_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    init_dir = study_root / "policies" / "v000_initial"
    init_dir.mkdir(parents=True, exist_ok=True)
    if not (init_dir / "policy.py").exists():
        shutil.copy2(INCUMBENT_TEMPLATE, init_dir / "policy.py")
    state = load_loop_state(study_root)
    if state is None:
        state = new_state(cycles, start_iteration, policy_ref(init_dir / "policy.py"), run, prov)
    LOOP.advance(state, LoopStatus.RUNNING)
    state.update(
        stop_reason=None,
        started_utc=utc_now(),
        cycles_target=cycles,
        run=run,
        provenance=prov,
    )
    state.setdefault("first_run", dict(run))
    state.setdefault("first_started_utc", state["started_utc"])
    save_loop_state(study_root, state)
    reason: str | None = None
    try:
        for offset in range(cycles):
            if len(state["iterations"]) <= offset:
                it = start_iteration + offset
                policy_in = state["initial_policy"]
                if offset:
                    prev = state["iterations"][-1]["policy_dev"]
                    policy_in = {"path": prev["selected_policy_path"], "sha256": prev["sha256"]}
                state["iterations"].append(new_iteration(it, policy_in))
                save_loop_state(study_root, state)
            rec = state["iterations"][offset]
            if iteration_done(rec):
                continue
            manifest_path = study_root / "cycles" / rec["cycle_id"] / "live_cycle_manifest.json"
            if rec["cycle"]["status"] != StageStatus.DONE and not manifest_path.is_file():
                reason = _guard(study_root, started, max_hours, max_agent_usd, state)
                if reason:
                    break
                policy = verified_policy(rec["policy_in"])
                CYCLE_STAGE.advance(rec["cycle"], StageStatus.RUNNING)
                rec["cycle"]["started_utc"] = utc_now()
                save_loop_state(study_root, state)
                cycle_fn(
                    cycle_id=rec["cycle_id"],
                    study_dir=study_root,
                    policy_path=policy,
                    iteration=rec["iteration"],
                    **cycle_kw,
                )
            if rec["cycle"]["status"] != StageStatus.DONE:
                cycle_manifest = mark_cycle_done(rec, manifest_path)
                save_loop_state(study_root, state)
                if cycle_manifest.get("stop_reason"):
                    # The cycle stopped itself (e.g. every agent returned placeholder text);
                    # another cycle would repeat the same non-result.
                    reason = f"{rec['cycle_id']}: {cycle_manifest['stop_reason']}"
                    break
            if rec["policy_dev"]["status"] in STAGE_FINISHED:
                continue
            selection = study_root / "policies" / rec["cycle_id"] / "selection.json"
            if not selection.is_file() and _load_trees(study_root):
                reason = _guard(study_root, started, max_hours, max_agent_usd, state)
                if reason:
                    break
            _run_policy_dev(study_root, rec, dev_fn, dev_kw)
            save_loop_state(study_root, state)
    except Exception as exc:
        LOOP.advance(state, LoopStatus.ERROR)
        state["error"] = f"{type(exc).__name__}: {exc}"
        save_loop_state(study_root, state)
        raise
    if reason is not None:
        LOOP.advance(state, LoopStatus.STOPPED)
    else:
        LOOP.advance(state, LoopStatus.DONE)
    state["stop_reason"] = reason
    save_loop_state(study_root, state)
    return write_summary(study_root, state, state["stop_reason"])
