"""Dream-RSI cycle records under `studies/program-v1/cycles/`.

Reads only the files the loop writes (`tree.json`, `live_cycle_manifest.json`,
`nodes/<cell>/score.json`, `proposal.md`) and omits fields that are absent. Returns a status stub
when no cycle exists; never raises on a missing study directory.
"""

from pathlib import Path

from report.plots.data import STUDY, missing, optional_json, rel


def node_record(node_dir: Path) -> dict:
    """One Dream-RSI node: score fields that exist plus proposal metadata."""
    score = optional_json(node_dir / "score.json") or {}
    proposal = node_dir / "proposal.md"
    proposal_bytes = proposal.stat().st_size if proposal.is_file() else None
    record = {
        "cell_id": score.get("cell_id") or node_dir.name,
        "parent_id": score.get("parent_id"),
        "fail_class": score.get("fail_class"),
        "score": score.get("score"),
        "successes": score.get("successes"),
        "episodes": score.get("episodes"),
        "agent_seconds": score.get("agent_seconds"),
        "agent_cost_usd": score.get("agent_cost_usd"),
        "proposal": rel(proposal) if proposal_bytes is not None else None,
        "proposal_bytes": proposal_bytes,
    }
    return {key: value for key, value in record.items() if value is not None}


def dream_rsi_cycles(study: Path | None = None) -> dict:
    """Every clean-room Dream-RSI cycle, newest last."""
    root = Path(study) if study is not None else STUDY
    cycles_dir = root / "cycles"
    if not cycles_dir.is_dir():
        return {**missing(cycles_dir, "Dream-RSI cycles"), "cycles": [], "count": 0}
    cycles = []
    for cycle_dir in sorted(p for p in cycles_dir.iterdir() if p.is_dir()):
        manifest = optional_json(cycle_dir / "live_cycle_manifest.json") or {}
        tree = optional_json(cycle_dir / "tree.json") or {}
        nodes_dir = cycle_dir / "nodes"
        nodes = (
            [node_record(p) for p in sorted(nodes_dir.iterdir()) if p.is_dir()]
            if nodes_dir.is_dir()
            else []
        )
        sources = [
            rel(path)
            for path in (cycle_dir / "tree.json", cycle_dir / "live_cycle_manifest.json")
            if path.is_file()
        ]
        world_nodes = manifest.get("world_nodes")
        if world_nodes is None and isinstance(tree.get("nodes"), list):
            world_nodes = len(tree["nodes"])
        cycle = {
            "cycle_id": manifest.get("cycle_id") or tree.get("cycle_id") or cycle_dir.name,
            "iteration": manifest.get("iteration"),
            "status": "done" if manifest else "incomplete",
            "best_score": manifest.get("best_score"),
            "probes_used": manifest.get("probes_used"),
            "decision_rounds": manifest.get("decision_rounds"),
            "effective_sequential_rounds": manifest.get("effective_sequential_rounds"),
            "world_nodes": world_nodes,
            "policy_path": manifest.get("policy_path"),
            "baked_in_beta": manifest.get("baked_in_beta"),
            "recorded_utc": manifest.get("recorded_utc"),
            "nodes": nodes,
            "sources": sources,
        }
        cycles.append({key: value for key, value in cycle.items() if value is not None})
    return {
        "status": "ok" if cycles else "empty",
        "source": rel(cycles_dir),
        "count": len(cycles),
        "cycles": cycles,
    }
