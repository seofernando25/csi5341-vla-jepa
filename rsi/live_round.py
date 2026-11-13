"""Live-round driver: control vs selected on a fresh grid, then promote and export."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.dream import Candidate
from rsi.live import (
    DEV_GATE,
    SMOKE_GATE,
    build_round1_candidates,
    build_round2_candidates,
    gpu_name,
    journals_for,
    promote_to_development,
    run_arm,
    world_from_stage_journal,
)
from rsi.policies import InitialPolicy
from rsi.policies.optimal import OptimalPolicy

ROUND_BUILDERS = {"r1": build_round1_candidates, "r2": build_round2_candidates}


def _code_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_live_round(
    round_id: str,
    selected: Candidate,
    probe_budget_each: int = 6,
    promote_top_k: int = 1,
    output_path: str | Path = "studies/dream-rsi/live_round1.json",
) -> dict[str, Any]:
    """Run control (frozen InitialPolicy) then selected on a fresh grid, same budget."""
    candidates = ROUND_BUILDERS[round_id]()
    recipes_dir = Path("outputs/compression") / f"{round_id}_recipes"
    journals = journals_for(round_id)
    control_policy = InitialPolicy()
    selected_policy = selected.cls(dict(selected.config))

    control = run_arm(
        candidates, control_policy, "control", probe_budget_each, round_id, journals, recipes_dir
    )
    selected_arm = run_arm(
        candidates,
        selected_policy,
        "selected",
        probe_budget_each,
        round_id,
        journals,
        recipes_dir,
    )
    recipes = {c.cell_id: c.recipe for c in candidates}
    control_dev = promote_to_development(
        control, recipes, recipes_dir, journals["stage"], top_k=promote_top_k
    )
    selected_dev = promote_to_development(
        selected_arm, recipes, recipes_dir, journals["stage"], top_k=promote_top_k
    )
    world, world_records = world_from_stage_journal(
        journals["stage"], candidates, f"qwen-live-{round_id}"
    )
    payload = {
        "protocol": "dream-rsi-live-round",
        "round_id": round_id,
        "control_policy": {"name": "InitialPolicy", "config": control_policy.config},
        "selected_policy": {"name": selected.name, "config": selected.config},
        "probe_budget_each": probe_budget_each,
        "smoke_gate": SMOKE_GATE,
        "dev_gate": DEV_GATE,
        "machine": gpu_name(),
        "code_head": _code_head(),
        "recorded_utc": datetime.now(UTC).isoformat(),
        "control": control,
        "selected": selected_arm,
        "development_promotions": {"control": control_dev, "selected": selected_dev},
        "world": {"world_id": world.world_id, "nodes": world.num_nonroot},
        "world_records": world_records,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m rsi.live_round")
    parser.add_argument("--round", default="r1")
    parser.add_argument("--budget-each", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no-gate-retry", action="store_true", default=True)
    parser.add_argument("--name", default="OptimalPolicy[r1-retry-flip]")
    parser.add_argument("--output", default="studies/dream-rsi/live_round1.json")
    args = parser.parse_args(argv)
    selected = Candidate(
        args.name,
        OptimalPolicy,
        {"beta": args.beta, "gate_retry": not args.no_gate_retry},
    )
    payload = run_live_round(args.round, selected, args.budget_each, output_path=args.output)
    print(json.dumps({k: payload[k] for k in ("round_id", "machine", "world")}, indent=2))
    for arm in ("control", "selected"):
        r = payload[arm]
        print(
            f"{arm}: probes={r['probes_used']} rounds={r['decision_rounds']} best={r['best_smoke_reduction']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
