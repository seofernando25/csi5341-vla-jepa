"""The watch page's state payload: loop, protocol, GPU and every cycle with its nodes and policy steps."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.loop.states import current_stage, iteration_done, observed_status
from rsi.loop.store import lock_holder
from rsi.watch.files import PART, as_int, as_num, as_str, read_json
from rsi.watch.nodes import cycle_nodes, mark_stopped
from rsi.watch.policy import policy_dev_block, variant_dev_block

_GPU: list[Any] = [0.0, None]


def _branches(value: Any) -> list[int] | None:
    """Branch indices listed in a cycle manifest, or None when the field is absent."""
    if not isinstance(value, list):
        return None
    return [b for b in value if isinstance(b, int) and not isinstance(b, bool)]


def _iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _gpu_block() -> dict[str, Any] | None:
    """nvidia-smi summary, cached for 2 s; None when the query fails or no GPU is present."""
    if time.monotonic() - _GPU[0] < 2.0:
        return _GPU[1]
    value: dict[str, Any] | None = None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        fields = (
            [f.strip() for f in out.stdout.splitlines()[0].split(",")]
            if out.returncode == 0
            else []
        )
        if len(fields) == 4 and all(f.isdigit() for f in fields[1:]):
            value = {
                "name": fields[0],
                "memory_used_mb": int(fields[1]),
                "memory_total_mb": int(fields[2]),
                "utilization_pct": int(fields[3]),
            }
    except (OSError, subprocess.SubprocessError, IndexError):
        value = None
    _GPU[0], _GPU[1] = time.monotonic(), value
    return value


def _cycle(study: Path, cid: str, iteration: int | None, it: dict[str, Any]) -> dict[str, Any]:
    cdir = study / "cycles" / cid
    cyc = it.get("cycle") if isinstance(it.get("cycle"), dict) else {}
    manifest = read_json(study, cdir / "live_cycle_manifest.json")
    done = manifest is not None or cyc.get("status") == "done"
    policy_in = it.get("policy_in") if isinstance(it.get("policy_in"), dict) else {}
    nodes = cycle_nodes(study, cdir)
    return {
        "cycle_id": cid,
        "iteration": iteration,
        "status": "done" if done else "running",
        "decision_rounds": as_int(manifest.get("decision_rounds")) if manifest else None,
        "opened_branches": _branches(manifest.get("opened_branches")) if manifest else None,
        "probes_used": as_int(manifest.get("probes_used")) if manifest else None,
        "probes_used_live": sum(
            1 for n in nodes if n["cell_id"] != "root" and n["score"] is not None
        ),
        "effective_sequential_rounds": (
            as_int(manifest.get("effective_sequential_rounds")) if manifest else None
        ),
        "agent_timeout_s": as_num(manifest.get("agent_timeout_s")) if manifest else None,
        "policy_sha256": as_str(policy_in.get("sha256")),
        "nodes": nodes,
        "policy_dev": policy_dev_block(
            study,
            study / "policies" / cid,
            it.get("policy_dev") if isinstance(it.get("policy_dev"), dict) else {},
        ),
    }


def _elapsed(loop_raw: dict[str, Any] | None, running: bool) -> float | None:
    if loop_raw is None:
        return None
    start = _iso(loop_raw.get("first_started_utc") or loop_raw.get("started_utc"))
    end = datetime.now(UTC) if running else _iso(loop_raw.get("updated_utc"))
    if start is None or end is None:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return round((end - start).total_seconds(), 1) if end >= start else None


def lock_owner(study: Path) -> str | None:
    try:
        return lock_holder(study)
    except OSError:
        return None


def _protocol_block(study: Path) -> dict[str, Any]:
    proto = read_json(study, study / "protocol.json") or {}
    drsi = proto.get("dream_rsi") if isinstance(proto.get("dream_rsi"), dict) else {}
    gates = proto.get("gates") if isinstance(proto.get("gates"), dict) else {}
    phases = proto.get("phases") if isinstance(proto.get("phases"), dict) else {}
    grid = drsi.get("grid_plan") if isinstance(drsi.get("grid_plan"), dict) else {}
    agent = drsi.get("agent") if isinstance(drsi.get("agent"), dict) else {}
    return {
        "smoke_gate": as_int(gates.get("smoke_min_successes")),
        "smoke_trials": as_int((phases.get("smoke") or {}).get("trials")),
        "dev_gate": as_int(gates.get("development_min_successes")),
        "dev_trials": as_int((phases.get("development") or {}).get("trials")),
        "max_parallelism": as_int(grid.get("max_parallelism")),
        "rounds_cap": as_int(grid.get("rounds_cap")),
        "max_branches": as_int(grid.get("max_branches")),
        "max_attempts_per_branch": as_int(grid.get("max_attempts_per_branch")),
        "gpu_slots": as_int(drsi.get("gpu_slots")),
        "agent": {
            "backend": as_str(agent.get("backend")),
            "model": as_str(agent.get("model")),
            "effort": as_str(agent.get("effort")),
        },
    }


def _baseline(study: Path, cycles: list[dict[str, Any]]) -> dict[str, Any]:
    """Baseline score is not recorded (the root's 0.0 is a placeholder); its bytes are on disk."""
    for cycle in cycles:
        for node in cycle["nodes"]:
            if node["cell_id"] == "root" or node["status"] != "done":
                continue
            ndir = study / "cycles" / cycle["cycle_id"] / "nodes" / node["cell_id"]
            score = read_json(study, ndir / "score.json") or {}
            sizes = score.get("sizes") if isinstance(score.get("sizes"), dict) else {}
            value = as_int(sizes.get("baseline_resident_bytes"))
            if value is None:
                result = read_json(study, ndir / "eval" / "result.json") or {}
                esizes = result.get("sizes") if isinstance(result.get("sizes"), dict) else {}
                value = as_int(result.get("baseline_resident_bytes")) or as_int(
                    esizes.get("baseline_resident_bytes")
                )
            if value is not None:
                return {"score": None, "resident_bytes": value}
    return {"score": None, "resident_bytes": None}


def _calibration_state(study: Path, now: datetime) -> dict[str, Any] | None:
    """A finished policy-development root served directly: one cycle, no nodes, no loop state."""
    dev = variant_dev_block(study)
    if dev is None:
        return None
    selection = read_json(study, study / "selection.json") or {}
    agent = selection.get("agent") if isinstance(selection.get("agent"), dict) else {}
    variants = [v for v in selection.get("variants") or [] if isinstance(v, dict)]
    sha = next(
        (
            as_str(v.get("policy_sha256"))
            for v in variants
            if v.get("slot") == selection.get("selected_slot")
        ),
        None,
    )
    cycle = {
        "cycle_id": study.name,
        "iteration": None,
        "status": "done",
        "decision_rounds": None,
        "opened_branches": None,
        "probes_used": None,
        "probes_used_live": 0,
        "effective_sequential_rounds": None,
        "agent_timeout_s": None,
        "policy_sha256": sha,
        "nodes": [],
        "policy_dev": dev,
    }
    loop = {
        "status": "done",
        "stop_reason": None,
        "iteration": None,
        "stage": None,
        "cycles_target": None,
        "elapsed_s": None,
        "agent_cost_usd": round(as_num(agent.get("cost_usd")) or 0.0, 6),
        "best_score": None,
        "running": False,
        "probe_budget": None,
        "baseline": {"score": None, "resident_bytes": None},
    }
    return {
        "study_dir": study.name,
        "now": now.isoformat(),
        "loop": loop,
        "protocol": _protocol_block(study),
        "gpu": _gpu_block(),
        "cycles": [cycle],
    }


def build_state(study_dir: Path) -> dict[str, Any]:
    """Watch state per the watch_api contract; missing or half-written files are skipped."""
    study = Path(study_dir)
    now = datetime.now(UTC)
    raw = read_json(study, study / "loop_state.json")
    if raw is None:
        calibration = _calibration_state(study, now)
        if calibration is not None:
            return calibration
    running = lock_owner(study) is not None
    iterations = [it for it in (raw or {}).get("iterations", []) if isinstance(it, dict)]
    cycles = [
        _cycle(study, str(it.get("cycle_id")), as_int(it.get("iteration")), it)
        for it in iterations
        if it.get("cycle_id") and PART.match(str(it.get("cycle_id")))
    ]
    known = {c["cycle_id"] for c in cycles}
    cdir_root = study / "cycles"
    if cdir_root.is_dir():
        for path in sorted(cdir_root.iterdir()):
            if path.is_dir() and path.name not in known and PART.match(path.name):
                cycles.append(_cycle(study, path.name, None, {}))
    nodes = [n for c in cycles for n in c["nodes"]]
    if not running:
        for node in nodes:
            mark_stopped(node)
    scores = [n["score"] for n in nodes if n["status"] == "done" and n["score"] is not None]
    cost = sum(n["agent_cost_usd"] or 0.0 for n in nodes)
    cost += sum(
        as_num((it.get("policy_dev") or {}).get("agent_cost_usd")) or 0.0 for it in iterations
    )
    current = next((it for it in iterations if not iteration_done(it)), None)
    stage = None
    if current is not None:
        stage = current_stage(current).value
    iteration = None
    if current is not None:
        iteration = as_int(current.get("iteration"))
    elif iterations:
        iteration = as_int(iterations[-1].get("iteration"))
    status = observed_status((raw or {}).get("status"), running).value
    run = (raw or {}).get("run") if isinstance((raw or {}).get("run"), dict) else {}
    first = (raw or {}).get("first_run") if isinstance((raw or {}).get("first_run"), dict) else {}
    probe_budget = as_int(run.get("probe_budget"))
    if probe_budget is None:
        probe_budget = as_int(first.get("probe_budget"))
    loop = {
        "status": status,
        "stop_reason": (raw or {}).get("stop_reason"),
        "iteration": iteration,
        "stage": stage if running else None,
        "cycles_target": as_int((raw or {}).get("cycles_target")),
        "elapsed_s": _elapsed(raw, running),
        "agent_cost_usd": round(cost, 6),
        "best_score": max(scores) if scores else None,
        "running": running,
        "probe_budget": probe_budget,
        "baseline": _baseline(study, cycles),
    }
    return {
        "study_dir": study.name,
        "now": now.isoformat(),
        "loop": loop,
        "protocol": _protocol_block(study),
        "gpu": _gpu_block(),
        "cycles": cycles,
    }
