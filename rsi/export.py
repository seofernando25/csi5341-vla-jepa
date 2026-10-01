"""Export compact, attributable results without raw runtime artifacts."""

import json
import os
import tempfile
from pathlib import Path

from rsi.core import atomic


def export_results(runner, output=None):
    init = runner.verify()
    target = Path(output) if output else runner.repo / "studies/dream-rsi"
    target.mkdir(parents=True, exist_ok=True)
    # A destination belongs to exactly one journal, including its initialization identity.
    from rsi.core import digest

    identity = digest(init)
    provenance = target / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text())["study_id"] != identity:
        raise ValueError("export destination belongs to another study")
    atomic(
        provenance,
        {
            "study_id": identity,
            "backbone": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "synthetic": init["synthetic"],
            "git_sha": init["git_sha"],
            "config": init["config"],
            "harness": init["harness"],
            "base": init["base"],
            "discovery_context_updates": runner.events("discovery_context_updated"),
            "baseline": [e["metrics"] for e in runner.events("baseline")],
            "promotion_baseline": [e["metrics"] for e in runner.events("promotion_baseline")],
            "budget_extensions": [e["limits"] for e in runner.events("budget_extended")],
        },
    )
    records = []
    for event in runner.events("outcome"):
        node = event["node"]
        records.append(
            {
                "study_id": identity,
                "attempt_id": event["attempt"],
                "cycle": event["outer"],
                "parent": node["parent"],
                "status": node["status"],
                "score": node["score"] if node["status"] == "ok" else None,
                "proposal": node.get("proposal"),
                "summary": node.get("summary"),
                "workspace_hash": event.get("workspace_hash"),
                "promotion": runner.promotion_for(event["attempt"]),
            }
        )
    destination = target / "attempts.jsonl"
    fd, temporary = tempfile.mkstemp(dir=target, prefix=".attempts-")
    try:
        with os.fdopen(fd, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(target), "attempts": len(records), "synthetic": init["synthetic"]}
