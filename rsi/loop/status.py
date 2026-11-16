"""Read-only status of a study's loop: launcher liveness, progress, best score, cost and the log tail."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.loop.launcher import LOG_NAME, pid_alive, read_service, unit_state
from rsi.loop.states import LaunchMode, current_stage, iteration_done, launch_mode
from rsi.loop.store import load_loop_state, lock_holder


def _progress(study_dir: Path, state: dict[str, Any] | None) -> dict[str, Any]:
    cycles_dir = Path(study_dir) / "cycles"
    out: dict[str, Any] = {
        "iteration": None,
        "stage": "not started",
        "cycle_id": None,
        "nodes_done": None,
        "nodes_total": None,
    }
    if state is not None:
        pending = [it for it in state.get("iterations", []) if not iteration_done(it)]
        out["iterations_done"] = len(state.get("iterations", [])) - len(pending)
        out["cycles_target"] = state.get("cycles_target")
        if pending:
            cur = pending[0]
            out.update(iteration=cur["iteration"], cycle_id=cur["cycle_id"])
            out["stage"] = current_stage(cur).value
            nodes = cycles_dir / cur["cycle_id"] / "nodes"
            if nodes.is_dir():
                dirs = [p for p in nodes.iterdir() if p.is_dir()]
                out["nodes_total"] = len(dirs)
                out["nodes_done"] = sum((p / "score.json").is_file() for p in dirs)
        else:
            out["stage"] = "done" if state.get("iterations") else "not started"
    return out


def _best_and_cost(study_dir: Path, state: dict[str, Any] | None) -> tuple[float | None, float]:
    scores: list[float] = []
    for path in (Path(study_dir) / "cycles").glob("*/nodes/*/score.json"):
        value = json.loads(path.read_text(encoding="utf-8")).get("score")
        if value is not None:
            scores.append(float(value))
    cost = 0.0
    if state is not None:
        for it in state.get("iterations", []):
            cost += float(it["cycle"].get("agent_cost_usd") or 0.0)
            cost += float(it["policy_dev"].get("agent_cost_usd") or 0.0)
    return (max(scores) if scores else None), cost


def _elapsed_s(state: dict[str, Any] | None, running: bool) -> float | None:
    if state is None or not state.get("first_started_utc"):
        return None
    first = datetime.fromisoformat(state["first_started_utc"])
    end = datetime.now(UTC) if running else datetime.fromisoformat(state["updated_utc"])
    return round((end - first).total_seconds(), 1)


def _log_tail(study_dir: Path, lines: int = 10) -> list[str]:
    path = Path(study_dir) / LOG_NAME
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def status(study_dir: Path) -> dict[str, Any]:
    """Unit or pid liveness, loop progress, best score, cost, elapsed time and the log tail."""
    study = Path(study_dir)
    state = load_loop_state(study)
    holder = lock_holder(study)
    svc = read_service(study)
    mode = launch_mode(svc)
    unit: dict[str, Any] = {"mode": mode.value if mode else None}
    if mode is LaunchMode.SYSTEMD:
        unit.update(unit=svc["unit"], active=unit_state(svc["unit"]))
    elif mode is LaunchMode.DETACHED:
        unit.update(pid=svc["pid"], alive=pid_alive(int(svc["pid"])))
    best, cost = _best_and_cost(study, state)
    return {
        "study_dir": str(study),
        "running": holder is not None,
        "lock_holder_pid": holder,
        "launcher": unit,
        "loop_status": state.get("status") if state else None,
        "stop_reason": state.get("stop_reason") if state else None,
        **_progress(study, state),
        "best_score": best,
        "agent_cost_usd": round(cost, 6),
        "elapsed_s": _elapsed_s(state, holder is not None),
        "log_tail": _log_tail(study),
    }
