"""Prompt, node-record and manifest builders for the online Dream-RSI cycle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.agent_backends import AgentSpec, agent_record
from rsi.api import OK, CellMeta, Observation

DISCOVERY_PROMPT = Path(__file__).resolve().parent / "prompts" / "discovery.md"


def build_discovery_prompt(
    *,
    cell_id: str,
    cycle_id: str,
    parent_id: str,
    smoke_trials: int,
    smoke_gate: int,
    best_score: float,
    gpu_slots: int,
    minutes: float,
) -> str:
    """Format the orchestrator's discovery prompt for one node."""
    return DISCOVERY_PROMPT.read_text(encoding="utf-8").format(
        cell_id=cell_id,
        cycle_id=cycle_id,
        parent_id=parent_id,
        smoke_trials=smoke_trials,
        smoke_gate=smoke_gate,
        best_score=f"{best_score:.6f}",
        gpu_slots=gpu_slots,
        minutes=f"{minutes:.0f}",
    )


def revealed_nodes(
    nodes_dir: Path, revealed_order: list[str], slots: dict[str, CellMeta]
) -> list[dict[str, Any]]:
    """Evidence entries (cell, parent, node directory) for every node revealed so far."""
    return [
        {
            "cell_id": cid,
            "parent_id": slots[cid].parent_id or "root",
            "node_dir": str(nodes_dir / cid),
        }
        for cid in revealed_order
    ]


def tree_index(cycle_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact index of the revealed nodes, written as `tree.json` in every worker's evidence."""
    return {
        "cycle_id": cycle_id,
        "nodes": [
            {
                "cell_id": r["cell_id"],
                "parent_id": r["parent_id"],
                "fail_class": r["fail_class"],
                "score": r["score"],
            }
            for r in records
        ],
    }


def one_line_summary(record: dict[str, Any]) -> str:
    """The node's single-line result, shown in the observation summary."""
    sizes = record.get("sizes") or {}
    reduction = sizes.get("size_reduction")
    red = "na" if reduction is None else f"{float(reduction):.4f}"
    succ = record.get("successes")
    eps = record.get("episodes")
    smoke = "na" if succ is None else f"{succ}/{eps}"
    return (
        f"{record['cell_id']}: {record['fail_class']} smoke={smoke} "
        f"size_reduction={red} agent_timed_out={record.get('agent_timed_out', False)}"
    )


def observation(cm: CellMeta, record: dict[str, Any], parent_score: float | None) -> Observation:
    """Prefix-only observation of a finished node; its score is zero unless it passed the gate."""
    score = float(record.get("score") or 0.0)
    fail_class = str(record["fail_class"])
    error = record.get("error")
    return Observation(
        cell_id=cm.cell_id,
        branch=cm.branch,
        attempt=cm.attempt,
        parent_id=cm.parent_id,
        score=score,
        evaluated=bool(record.get("evaluated", False)),
        valid=fail_class == OK,
        fail_class=fail_class,
        error=error,
        delta_vs_baseline=score,
        delta_vs_parent=score - parent_score if parent_score is not None else score,
        n_valid=record.get("successes"),
        n_total=record.get("episodes"),
        mechanism_family="",
        summary=one_line_summary(record),
    )


def agent_totals(spec: AgentSpec | None, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Agent identity plus summed per-node cost and turns, for the cycle manifest."""
    base = agent_record(spec, [])
    costs = [r["agent_cost_usd"] for r in records if r.get("agent_cost_usd") is not None]
    turns = [r["agent_turns"] for r in records if r.get("agent_turns") is not None]
    return {
        **base,
        "calls": len(records),
        "num_turns": sum(turns) if turns else None,
        "cost_usd": round(sum(costs), 6) if costs else None,
        "timed_out": sum(1 for r in records if r.get("agent_timed_out")),
    }


def recorded_utc() -> str:
    """UTC timestamp for manifests."""
    return datetime.now(UTC).isoformat()
