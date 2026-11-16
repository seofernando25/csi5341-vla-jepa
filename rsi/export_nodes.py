"""Copy compact per-node latency records out of a (possibly live) program-v4 run.

Reads `<run>/cycles/*/nodes/*/{score.json,proposal.md,eval/result.json}` and writes one
`nodes.json` table with a snapshot time. Read-only on the run; nodes without a `score.json` are
still being evaluated and are left out. Nodes are ordered by the time their evaluation recorded.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _title(proposal: Path) -> str:
    try:
        first = proposal.read_text().splitlines()[0]
    except (OSError, IndexError):
        return ""
    return first.lstrip("# ").removeprefix("Proposal:").strip()


def node_row(node: Path) -> dict | None:
    """Compact record of one scored node, or None while it has no `score.json`."""
    score = _json(node / "score.json")
    if not score:
        return None
    result = _json(node / "eval" / "result.json")
    latency = result.get("latency") or {}
    speedup = (latency.get("candidate") or {}).get("speedup") or {}
    guard = "ok" if score.get("score") else (score.get("error") or latency.get("zero_reason"))
    return {
        "cell_id": node.name,
        "cycle": int(node.parent.parent.name.split("_")[-1]),
        "recorded_utc": (result.get("environment") or {}).get("recorded_at"),
        "score": score.get("score"),
        "speedup": {key: speedup.get(key) for key in ("point", "low", "high")},
        "guard": guard,
        "violations": latency.get("violations") or [],
        "smoke": f"{score.get('successes')}/{score.get('episodes')}",
        "fail_class": score.get("fail_class"),
        "title": _title(node / "proposal.md"),
    }


def export(run: Path) -> dict:
    """The node table of `run` (a study directory) as of now."""
    state = _json(run / "loop_state.json")
    rows = [
        row for node in sorted(run.glob("cycles/*/nodes/*")) if (row := node_row(node)) is not None
    ]
    rows.sort(key=lambda row: (row["recorded_utc"] or "", row["cell_id"]))
    done = [p.parent.name for p in run.glob("cycles/*/live_cycle_manifest.json")]
    return {
        "study": run.name,
        "snapshot_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "status": state.get("status"),
        "cycles_target": state.get("cycles_target"),
        "cycles_complete": len(done),
        "nodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="study directory of the run, e.g. a worktree's")
    parser.add_argument("--out", type=Path, default=Path("studies/program-v4/nodes.json"))
    args = parser.parse_args()
    table = export(args.run)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"{len(table['nodes'])} nodes -> {args.out}")


if __name__ == "__main__":
    main()
