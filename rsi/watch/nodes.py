"""One cycle's node records for the watch page, derived from the files each expansion leaves behind."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.expansion.states import IN_FLIGHT, NODE, FailClass, NodeStatus, observed_node_status
from rsi.watch.files import PART, as_int, as_num, as_str, read_json, read_text


def _blank(
    cell_id: str, parent: str | None, branch: int | None, attempt: int | None
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "parent_id": parent,
        "branch": branch,
        "attempt": attempt,
        "round": None,
        "status": NodeStatus.PENDING.value,
        "fail_class": None,
        "score": None,
        "size_reduction": None,
        "successes": None,
        "episodes": None,
        "resident_mb": None,
        "baseline_mb": None,
        "per_task": None,
        "peak_cuda_bytes": None,
        "stage_seconds": None,
        "commit": None,
        "agent_seconds": None,
        "eval_seconds": None,
        "agent_cost_usd": None,
        "agent_turns": None,
        "agent_timed_out": False,
        "error": None,
        "headline": None,
        "promoted": None,
        "started_utc": None,
        "elapsed_s": None,
    }


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None


def mark_stopped(node: dict[str, Any]) -> None:
    """A node cannot be in flight once no loop process holds the study lock.

    The filesystem probe cannot tell a live agent from one whose loop died, so `build_state`
    demotes in-flight nodes to `stopped` and drops the live clock. Pending and finished nodes
    keep their status.
    """
    if node["status"] in IN_FLIGHT:
        NODE.advance(node, NodeStatus.STOPPED)
        node["elapsed_s"] = None


def _headline(study: Path, path: Path) -> str | None:
    text = read_text(study, path, cap=20_000) or ""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()[:200] or None
    return None


def _per_task(value: Any) -> dict[str, dict[str, int | None]] | None:
    """Per-task successes from `score.json`, keyed by task id; absent when not recorded."""
    if not isinstance(value, dict) or not value:
        return None
    out = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out[str(key)] = {
                "successes": as_int(item.get("successes")),
                "episodes": as_int(item.get("episodes")),
            }
    return out or None


def _stages(value: Any) -> dict[str, float | None] | None:
    """`eval/result.json` stage timings, keeping only numeric entries."""
    if not isinstance(value, dict) or not value:
        return None
    return {str(key): as_num(item) for key, item in value.items()}


def _eval_detail(study: Path, ndir: Path, rec: dict[str, Any]) -> None:
    """Fill the fields `score.json` omits from `eval/result.json`, when the evaluation wrote one."""
    if all(rec[key] is not None for key in ("peak_cuda_bytes", "stage_seconds", "baseline_mb")):
        return
    result = read_json(study, ndir / "eval" / "result.json")
    if result is None:
        return
    sizes = result.get("sizes") if isinstance(result.get("sizes"), dict) else {}
    if rec["peak_cuda_bytes"] is None:
        rec["peak_cuda_bytes"] = as_int(sizes.get("peak_cuda_bytes"))
    if rec["stage_seconds"] is None:
        rec["stage_seconds"] = _stages(result.get("stage_seconds"))
    if rec["baseline_mb"] is None:
        baseline = as_num(result.get("baseline_resident_bytes"))
        if baseline is None:
            baseline = as_num(sizes.get("baseline_resident_bytes"))
        rec["baseline_mb"] = round(baseline / 1e6, 2) if baseline is not None else None


def _node(
    study: Path,
    ndir: Path,
    cell_id: str,
    parent: str | None,
    branch: int | None,
    attempt: int | None,
) -> dict[str, Any]:
    rec = _blank(cell_id, parent, branch, attempt)
    score = read_json(study, ndir / "score.json")
    rec["status"] = observed_node_status(ndir, scored=score is not None).value
    if score is not None:
        sizes = score.get("sizes") if isinstance(score.get("sizes"), dict) else {}
        resident = as_num(sizes.get("resident_bytes"))
        baseline = as_num(sizes.get("baseline_resident_bytes"))
        rec.update(
            fail_class=as_str(score.get("fail_class")),
            score=as_num(score.get("score")),
            size_reduction=as_num(sizes.get("size_reduction")),
            successes=as_int(score.get("successes")),
            episodes=as_int(score.get("episodes")),
            resident_mb=round(resident / 1e6, 2) if resident is not None else None,
            baseline_mb=round(baseline / 1e6, 2) if baseline is not None else None,
            per_task=_per_task(score.get("per_task")),
            peak_cuda_bytes=as_int(sizes.get("peak_cuda_bytes")),
            commit=as_str(score.get("commit")),
            agent_seconds=as_num(score.get("agent_seconds")),
            eval_seconds=as_num(score.get("eval_seconds")),
            agent_cost_usd=as_num(score.get("agent_cost_usd")),
            agent_turns=as_int(score.get("agent_turns")),
            agent_timed_out=score.get("agent_timed_out") is True,
            error=as_str(score.get("error")),
            round=as_int(score.get("round")),
        )
        _eval_detail(study, ndir, rec)
    if rec["status"] in IN_FLIGHT:
        marker = (
            ndir / "logs" / "stdout.log" if rec["status"] == NodeStatus.AGENT else ndir / "eval"
        )
        started = _mtime(marker)
        if started is not None:
            rec["started_utc"] = started.isoformat()
            rec["elapsed_s"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    if ndir.is_dir():
        rec["headline"] = _headline(study, ndir / "proposal.md")
        dev = read_json(study, ndir / "development.json")
        if dev is not None:
            rec["promoted"] = {
                "successes": as_int(dev.get("successes")) or 0,
                "episodes": as_int(dev.get("episodes")) or 0,
            }
    return rec


def cycle_nodes(study: Path, cdir: Path) -> list[dict[str, Any]]:
    out = [
        {
            **_blank("root", None, None, None),
            "status": NodeStatus.DONE.value,
            "fail_class": FailClass.OK.value,
            "score": 0.0,
        }
    ]
    grid = read_json(study, cdir / "grid_plan.json") or {}
    branches, attempts = (
        as_int(grid.get("max_branches")),
        as_int(grid.get("max_attempts_per_branch")),
    )
    slots: list[tuple[str, str, int, int]] = []
    if branches is not None and attempts is not None:
        for b in range(branches):
            parent = "root"
            for a in range(attempts + 1):
                cell = f"{cdir.name}-b{b}-a{a}"
                slots.append((cell, parent, b, a))
                parent = cell
    ndir_root = cdir / "nodes"
    for cell, parent, b, a in slots:
        out.append(_node(study, ndir_root / cell, cell, parent, b, a))
    known = {cell for cell, *_ in slots}
    if ndir_root.is_dir():
        for path in sorted(ndir_root.iterdir()):
            if path.is_dir() and path.name not in known and PART.match(path.name):
                out.append(_node(study, path, path.name, None, None, None))
    return out
