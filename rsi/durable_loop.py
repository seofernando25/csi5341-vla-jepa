"""Dream-RSI outer loop: resumable stages (cycle, policy development), stop rules and the study lock."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsi.agent_backends import AgentSpec, load_agent_spec
from rsi.expansion import Evaluator
from rsi.online_cycle import DREAM_DIR, _recent_manifests, provenance, run_online_cycle
from rsi.policy_dev import (
    INCUMBENT_TEMPLATE,
    develop_policies_llm,
    load_tree_json,
)
from rsi.service import (
    StudyLock,
    load_loop_state,
    save_loop_state,
    stop_requested,
    utc_now,
    write_json_atomic,
)
from rsi.tree import DiscoveryTree


class PolicyMismatchError(RuntimeError):
    """A recorded policy file is missing or no longer matches its recorded sha256."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _verified(ref: dict[str, str]) -> Path:
    path = Path(ref["path"])
    if not path.is_file():
        raise PolicyMismatchError(f"policy file {path} is missing")
    actual = _sha256(path)
    if actual != ref["sha256"]:
        raise PolicyMismatchError(
            f"policy file {path} sha256 {actual} does not match recorded {ref['sha256']}"
        )
    return path


def _new_state(
    cycles: int,
    start_iteration: int,
    initial: dict[str, str],
    run: dict[str, Any],
    prov: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "dream-rsi-loop-state",
        "status": "running",
        "stop_reason": None,
        "cycles_target": cycles,
        "start_iteration": start_iteration,
        "run": run,
        "provenance": prov,
        "initial_policy": initial,
        "first_started_utc": utc_now(),
        "started_utc": utc_now(),
        "iterations": [],
    }


def _new_iteration(iteration: int, policy_in: dict[str, str]) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "cycle_id": f"iter_{iteration:03d}",
        "policy_in": policy_in,
        "cycle": {"status": "pending", "started_utc": None, "finished_utc": None},
        "policy_dev": {
            "status": "pending",
            "started_utc": None,
            "finished_utc": None,
            "selected_policy_path": None,
            "sha256": None,
            "agent_cost_usd": None,
        },
    }


def _spent(state: dict[str, Any]) -> tuple[float, list[str]]:
    """Sum recorded agent cost and name every finished stage whose cost is unknown."""
    total = 0.0
    unknown: list[str] = []
    for it in state["iterations"]:
        for key in ("cycle", "policy_dev"):
            stage = it[key]
            value = stage.get("agent_cost_usd")
            if value is None:
                if stage.get("status") in ("done", "skipped"):
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


def _mark_cycle_done(rec: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agent = manifest.get("agent") or {}
    rec["cycle"].update(
        status="done",
        finished_utc=utc_now(),
        best_score=manifest.get("best_score"),
        agent_cost_usd=agent.get("cost_usd"),
        agent_turns=agent.get("num_turns"),
    )
    return manifest


def _mark_dev_done(rec: dict[str, Any], path: Path, cost: float | None, status: str) -> None:
    ref = _policy_ref(path)
    rec["policy_dev"].update(
        status=status,
        finished_utc=utc_now(),
        selected_policy_path=ref["path"],
        sha256=ref["sha256"],
        agent_cost_usd=cost or 0.0,
    )


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
        _mark_dev_done(rec, Path(res["selected_policy_path"]), _cost(res), "done")
        return
    if not _load_trees(study_root):
        _mark_dev_done(rec, _verified(rec["policy_in"]), 0.0, "skipped")
        return
    rec["policy_dev"].update(status="running", started_utc=utc_now())
    res = dev_fn(
        _load_trees(study_root),
        cycle_id=cid,
        incumbent_path=_verified(rec["policy_in"]),
        output_dir=study_root / "policies" / cid,
        live_manifests=list(_recent_manifests(study_root)),
        **kw,
    )
    _mark_dev_done(rec, Path(res["selected_policy_path"]), _cost(res), "done")


def _cost(payload: dict[str, Any]) -> float | None:
    return (payload.get("agent") or {}).get("cost_usd")


def _summary(study_root: Path, state: dict[str, Any], reason: str | None) -> dict[str, Any]:
    cycles = [
        {
            "cycle_id": it["cycle_id"],
            "best_score": it["cycle"].get("best_score"),
            "policy_dev": it["policy_dev"],
        }
        for it in state["iterations"]
    ]
    summary = {
        "protocol": "dream-rsi-loop",
        "study_dir": str(study_root),
        "status": state["status"],
        "stop_reason": reason,
        "final_active_policy": state["iterations"][-1]["policy_dev"]["selected_policy_path"]
        if state["iterations"]
        else state["initial_policy"]["path"],
        "cycles": cycles,
        "recorded_utc": utc_now(),
    }
    write_json_atomic(study_root / "loop_summary.json", summary)
    return summary


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
    policy_timeout: float = 240.0,
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
        state = _new_state(cycles, start_iteration, _policy_ref(init_dir / "policy.py"), run, prov)
    state.update(
        status="running",
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
                state["iterations"].append(_new_iteration(it, policy_in))
                save_loop_state(study_root, state)
            rec = state["iterations"][offset]
            if rec["cycle"]["status"] == "done" and rec["policy_dev"]["status"] in (
                "done",
                "skipped",
            ):
                continue
            manifest_path = study_root / "cycles" / rec["cycle_id"] / "live_cycle_manifest.json"
            if rec["cycle"]["status"] != "done" and not manifest_path.is_file():
                reason = _guard(study_root, started, max_hours, max_agent_usd, state)
                if reason:
                    break
                policy = _verified(rec["policy_in"])
                rec["cycle"].update(status="running", started_utc=utc_now())
                save_loop_state(study_root, state)
                cycle_fn(
                    cycle_id=rec["cycle_id"],
                    study_dir=study_root,
                    policy_path=policy,
                    iteration=rec["iteration"],
                    **cycle_kw,
                )
            if rec["cycle"]["status"] != "done":
                cycle_manifest = _mark_cycle_done(rec, manifest_path)
                save_loop_state(study_root, state)
                if cycle_manifest.get("stop_reason"):
                    # The cycle stopped itself (e.g. every agent returned placeholder text);
                    # another cycle would repeat the same non-result.
                    reason = f"{rec['cycle_id']}: {cycle_manifest['stop_reason']}"
                    break
            if rec["policy_dev"]["status"] in ("done", "skipped"):
                continue
            selection = study_root / "policies" / rec["cycle_id"] / "selection.json"
            if not selection.is_file() and _load_trees(study_root):
                reason = _guard(study_root, started, max_hours, max_agent_usd, state)
                if reason:
                    break
            _run_policy_dev(study_root, rec, dev_fn, dev_kw)
            save_loop_state(study_root, state)
    except Exception as exc:
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save_loop_state(study_root, state)
        raise
    if reason is not None:
        state.update(status="stopped", stop_reason=reason)
    else:
        state.update(status="done", stop_reason=None)
    save_loop_state(study_root, state)
    return _summary(study_root, state, state["stop_reason"])
