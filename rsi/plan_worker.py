"""Run one policy's `plan_grid` on a recorded context and write the plan (inside the sandbox).

CLI: `python -m rsi.plan_worker POLICY CONTEXT OUT [--config CONFIG]`. The worker receives only
the cross-cycle context, never a question object, and holds no network or GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rsi.api import GridPlan, GridPlanningContext
from rsi.policy_audit import load_policy_class


def load_context(raw: dict[str, Any]) -> GridPlanningContext:
    """Rebuild the planning context from the JSON the harness wrote."""
    return GridPlanningContext(
        iteration=int(raw["iteration"]),
        max_parallelism=int(raw["max_parallelism"]),
        fallback_plan=GridPlan(**raw["fallback_plan"]),
        recent_live_manifests=tuple(raw.get("recent_live_manifests") or ()),
        recent_beta_sweeps=tuple(raw.get("recent_beta_sweeps") or ()),
    )


def main(argv: list[str] | None = None) -> int:
    """Audit, load and plan; any failure becomes one stderr line and a non-zero exit."""
    parser = argparse.ArgumentParser(prog="rsi.plan_worker")
    parser.add_argument("policy")
    parser.add_argument("context")
    parser.add_argument("out")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        context = load_context(json.loads(Path(args.context).read_text(encoding="utf-8")))
        policy = load_policy_class(Path(args.policy), "plan")(config)
        plan = policy.plan_grid(context)
        if not isinstance(plan, GridPlan):
            raise TypeError(f"plan_grid returned {type(plan).__name__}, not GridPlan")
        Path(args.out).write_text(json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - any policy failure becomes one stderr line
        reason = str(exc).strip().splitlines()
        print(
            f"plan failed: {type(exc).__name__}: {reason[0] if reason else ''}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
