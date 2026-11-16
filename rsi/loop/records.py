"""The records `loop_state.json` holds: policy references, new state and iteration blocks, stage completion and the summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.loop.states import CYCLE_STAGE, POLICY_DEV_STAGE, LoopStatus, StageStatus
from rsi.loop.store import utc_now, write_json_atomic
from rsi.provenance import sha256_file


class PolicyMismatchError(RuntimeError):
    """A recorded policy file is missing or no longer matches its recorded sha256."""


def policy_ref(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def recorded_policy_path(study_root: Path, value: str) -> Path:
    """Resolve a recorded policy path: absolute as written before 2026-11-15, else study-relative."""
    path = Path(value)
    return path if path.is_absolute() else Path(study_root) / path


def verified_policy(ref: dict[str, str]) -> Path:
    path = Path(ref["path"])
    if not path.is_file():
        raise PolicyMismatchError(f"policy file {path} is missing")
    actual = sha256_file(path)
    if actual != ref["sha256"]:
        raise PolicyMismatchError(
            f"policy file {path} sha256 {actual} does not match recorded {ref['sha256']}"
        )
    return path


def new_state(
    cycles: int,
    start_iteration: int,
    initial: dict[str, str],
    run: dict[str, Any],
    prov: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "dream-rsi-loop-state",
        "status": LoopStatus.RUNNING.value,
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


def new_iteration(iteration: int, policy_in: dict[str, str]) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "cycle_id": f"iter_{iteration:03d}",
        "policy_in": policy_in,
        "cycle": {"status": StageStatus.PENDING.value, "started_utc": None, "finished_utc": None},
        "policy_dev": {
            "status": StageStatus.PENDING.value,
            "started_utc": None,
            "finished_utc": None,
            "selected_policy_path": None,
            "sha256": None,
            "agent_cost_usd": None,
        },
    }


def mark_cycle_done(rec: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agent = manifest.get("agent") or {}
    CYCLE_STAGE.advance(rec["cycle"], StageStatus.DONE)
    rec["cycle"].update(
        finished_utc=utc_now(),
        best_score=manifest.get("best_score"),
        agent_cost_usd=agent.get("cost_usd"),
        agent_turns=agent.get("num_turns"),
    )
    return manifest


def mark_dev_done(rec: dict[str, Any], path: Path, cost: float | None, status: StageStatus) -> None:
    """Close the policy-development stage as `done` or `skipped` with the selected policy."""
    ref = policy_ref(path)
    POLICY_DEV_STAGE.advance(rec["policy_dev"], status)
    rec["policy_dev"].update(
        finished_utc=utc_now(),
        selected_policy_path=ref["path"],
        sha256=ref["sha256"],
        agent_cost_usd=cost or 0.0,
    )


def agent_cost(payload: dict[str, Any]) -> float | None:
    return (payload.get("agent") or {}).get("cost_usd")


def write_summary(study_root: Path, state: dict[str, Any], reason: str | None) -> dict[str, Any]:
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
