"""Dream-RSI cycle records under `studies/program-v1/cycles/`.

Reads only the files the loop writes (`tree.json`, `live_cycle_manifest.json`,
`nodes/<cell>/score.json`, `proposal.md`) and omits fields that are absent. When
`exploit/remeasure.json` exists, the audited nodes take their post-fix score and carry their audit
class; exploit nodes are marked invalid and never set a cycle's best. Returns a status stub when no
cycle exists; never raises on a missing study directory.
"""

from pathlib import Path

from report.plots.data import STUDY, audit_record, missing, optional_json, rel


def development_gate(root: Path) -> int:
    """Development successes a node needs before it can be selected, from the study protocol."""
    protocol = optional_json(root / "protocol.json") or {}
    gates = protocol.get("gates") or {}
    return int(gates.get("development_min_successes", 77))


def node_record(node_dir: Path, audit: dict[str, dict] | None = None) -> dict:
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
    development = optional_json(node_dir / "development.json") or {}
    if development:
        record["development_status"] = development.get("status")
        if development.get("successes") is not None:
            record["development_successes"] = int(development["successes"])
            record["development_episodes"] = int(development.get("episodes") or 0)
    entry = (audit or {}).get(record["cell_id"])
    if entry:
        fixed = (entry.get("new") or {}).get("score")
        if fixed is not None and record.get("score") != fixed:
            record["score_old"] = record.get("score")
            record["score"] = fixed
        record["audit_class"] = entry.get("class")
        record["valid"] = entry.get("class") != "exploit"
    return {key: value for key, value in record.items() if value is not None}


def dream_rsi_cycles(study: Path | None = None) -> dict:
    """Every clean-room Dream-RSI cycle, newest last."""
    root = Path(study) if study is not None else STUDY
    cycles_dir = root / "cycles"
    if not cycles_dir.is_dir():
        return {**missing(cycles_dir, "Dream-RSI cycles"), "cycles": [], "count": 0}
    cycles = []
    audit = audit_record(root)
    for cycle_dir in sorted(p for p in cycles_dir.iterdir() if p.is_dir()):
        manifest = optional_json(cycle_dir / "live_cycle_manifest.json") or {}
        tree = optional_json(cycle_dir / "tree.json") or {}
        nodes_dir = cycle_dir / "nodes"
        nodes = (
            [node_record(p, audit) for p in sorted(nodes_dir.iterdir()) if p.is_dir()]
            if nodes_dir.is_dir()
            else []
        )
        valid = [
            node["score"]
            for node in nodes
            if node.get("valid", True) and node.get("score") is not None
        ]
        gate = development_gate(root)
        passing = [
            node["score"]
            for node in nodes
            if node.get("valid", True)
            and node.get("score") is not None
            and (node.get("development_successes") or 0) >= gate
        ]
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
            "best_score": (
                max(passing)
                if passing
                else None
                if any(n.get("development_status") is not None for n in nodes)
                else max(valid)
                if valid
                else manifest.get("best_score")
            ),
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
