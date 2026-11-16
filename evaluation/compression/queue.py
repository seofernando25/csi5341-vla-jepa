"""Run an ordered queue of (recipe, phase) cells through the canonical `vla-compress run` path.

Queue files live in studies/program-v1/ as {"protocol", "purpose", "cells": [{"recipe", "phase"}]}.
A cell is one recipe measured in one phase; the file order is the run order.

Resumable: a cell whose run.json says `completed` is skipped. The queue stops on the first
failure that is not a measured outcome, so a crash cannot silently consume the GPU budget.
The reserved phase (`final`) runs only when the queue is invoked with `--allow-reserved`.

A cell is `pending` (never attempted), `done` (run.json `completed`) or `failed` (attempted,
not completed). Running a pending cell makes it done or failed; a failed cell stays failed
until its directory is removed by hand, which makes it pending again. `next_action` is the
whole decision table of the run loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

from evaluation.common import ROOT, read_json
from evaluation.compression.recipes import validate
from evaluation.fsm import StateMachine

STUDY = ROOT / "studies/program-v1"
QUEUE = STUDY / "queue.json"
RESERVED_PHASE = "final"
PHASES = ("screen", "smoke", "development", "final", "size", "benchmark")
# Planning estimate: 7.5 s per rollout trial plus 2 min per invocation to load the policy.
PHASE_MINUTES = {"screen": 4, "smoke": 4, "development": 15, "final": 65, "benchmark": 5, "size": 2}


class QueueError(RuntimeError):
    """The queue file cannot be run as written."""


class CellState(StrEnum):
    """Where one (recipe, phase) cell stands, read from its run.json."""

    PENDING = "pending"
    FAILED = "failed"
    DONE = "done"


class CellAction(StrEnum):
    """What the run loop does with one cell."""

    SKIP = "skip"
    RUN = "run"
    REJECTED = "rejected"
    STOP = "stop"


CELL = StateMachine(
    "queue cell",
    CellState,
    {
        CellState.PENDING: {CellState.DONE, CellState.FAILED},
        CellState.FAILED: {CellState.PENDING},
    },
    initial={CellState.PENDING},
)
COMPLETED = "completed"  # run.json `status` of a finished measurement


def load(path):
    """Read, validate and de-duplicate a queue file; raise QueueError on any defect."""
    path = Path(path)
    try:
        queue = read_json(path)
    except OSError as exc:
        raise QueueError(f"{path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QueueError(f"{path}: invalid JSON ({exc})") from exc
    entries = queue.get("cells")
    if not isinstance(entries, list) or not entries:
        raise QueueError(f"{path}: needs a non-empty 'cells' list")
    seen, recipes = set(), {}
    for entry in entries:
        recipe, phase = entry.get("recipe"), entry.get("phase")
        if phase not in PHASES:
            raise QueueError(f"{path}: unknown phase {phase!r} in {entry}")
        if (recipe, phase) in seen:
            raise QueueError(f"{path}: duplicate cell {recipe}/{phase}")
        seen.add((recipe, phase))
        if recipe not in recipes:
            recipes[recipe] = recipe_path(recipe)
            validate(read_json(recipes[recipe]), recipes[recipe])
    return {"path": path, "protocol": queue.get("protocol"), "cells": entries}


def recipe_path(recipe):
    """Recipe file for a plan id; a missing file is a queue defect, not a runtime failure."""
    path = STUDY / "recipes" / f"{recipe}.json"
    if not path.exists():
        raise QueueError(f"recipe not authored: {path.relative_to(ROOT)}")
    return path


def result_path(recipe, phase, measurements_root=None):
    root = Path(measurements_root) if measurements_root is not None else STUDY / "measurements"
    return root / recipe / phase / "run.json"


def record(recipe, phase, measurements_root=None):
    """The cell's run.json, or None when the cell has not been attempted."""
    path = result_path(recipe, phase, measurements_root)
    return read_json(path) if path.exists() else None


def state(recipe, phase, measurements_root=None):
    """`done`, `failed` (attempted, not completed) or `pending` (never attempted)."""
    path = result_path(recipe, phase, measurements_root)
    if not path.exists():
        return CellState.FAILED if path.parent.exists() else CellState.PENDING
    return CellState.DONE if read_json(path).get("status") == COMPLETED else CellState.FAILED


def measured_outcome(attempt):
    """A nonfinite-action rejection is a measured outcome; anything else is a runtime failure."""
    return bool(
        attempt
        and attempt.get("error_type") == "ValueError"
        and str(attempt.get("error", "")).startswith("Nonfinite actions")
    )


def next_action(cell_state, attempt=None):
    """The run loop's decision for one cell; `attempt` is its run.json when it failed."""
    cell_state = CELL.parse(cell_state)
    if cell_state is CellState.DONE:
        return CellAction.SKIP
    if cell_state is CellState.PENDING:
        return CellAction.RUN
    return CellAction.REJECTED if measured_outcome(attempt) else CellAction.STOP


def split_reserved(queue, allow_reserved):
    """Cells to run, and reserved cells withheld because no flag named the phase."""
    if allow_reserved:
        return list(queue["cells"]), []
    keep = [cell for cell in queue["cells"] if cell["phase"] != RESERVED_PHASE]
    withheld = [cell for cell in queue["cells"] if cell["phase"] == RESERVED_PHASE]
    return keep, withheld


def command(recipe, phase, measurements_root=None):
    """The canonical measurement command, as `vla-compress run` spells it."""
    argv = [
        sys.executable,
        "-m",
        "evaluation.compression.study",
        "run",
        "--recipe",
        str(recipe_path(recipe)),
        "--phase",
        phase,
    ]
    if measurements_root is not None:
        argv += ["--measurements-root", str(measurements_root)]
    return argv


def plan(queue, allow_reserved=False, measurements_root=None):
    """Ordered cells with their state, withheld reserved cells and the estimated minutes."""
    selected, withheld = split_reserved(queue, allow_reserved)
    rows = [
        {
            "recipe": cell["recipe"],
            "phase": cell["phase"],
            "state": state(cell["recipe"], cell["phase"], measurements_root),
            "minutes": PHASE_MINUTES[cell["phase"]],
        }
        for cell in selected
    ]
    return {
        "queue": str(queue["path"]),
        "cells": rows,
        "withheld_reserved": withheld,
        "total_minutes": sum(row["minutes"] for row in rows),
        "minutes": sum(row["minutes"] for row in rows if row["state"] != CellState.DONE),
    }


def run_queue(
    queue, allow_reserved=False, dry_run=False, measurements_root=None, invoke=None, out=print
):
    """Run pending cells in order; return 0 when every cell is done or a measured rejection."""
    if (
        not dry_run
        and any(c["phase"] == RESERVED_PHASE for c in queue["cells"])
        and not allow_reserved
    ):
        raise QueueError("queue contains reserved cells; pass --allow-reserved to run them")
    steps = plan(queue, allow_reserved, measurements_root)
    out(
        f"{steps['queue']}: {len(steps['cells'])} cells, "
        f"{steps['total_minutes']} min planned, {steps['minutes']} min remaining"
    )
    if steps["withheld_reserved"]:
        out(f"withheld {len(steps['withheld_reserved'])} reserved cells (pass --allow-reserved)")
    if dry_run:
        for index, row in enumerate(steps["cells"], 1):
            out(
                f"{index:3d}  {row['recipe']:38s} {row['phase']:12s} "
                f"{row['minutes']:3d} min  {row['state']}"
            )
        return 0
    invoke = invoke or subprocess.run
    ran = skipped = rejected = 0
    for index, row in enumerate(steps["cells"], 1):
        label = f"{index}/{len(steps['cells'])} {row['recipe']}/{row['phase']}"
        previous = None
        if row["state"] == CellState.FAILED:
            previous = record(row["recipe"], row["phase"], measurements_root)
        action = next_action(row["state"], previous)
        if action is CellAction.SKIP:
            skipped += 1
            out(f"skip {label}: compact result exists")
            continue
        if action is CellAction.REJECTED:
            rejected += 1
            out(f"stop arm {label}: measured rejection ({previous['error']})")
            continue
        if action is CellAction.STOP:
            out(f"stop queue {label}: previous attempt failed, not a measured outcome")
            return 1
        out(f"run  {label}")
        completed = invoke(command(row["recipe"], row["phase"], measurements_root), cwd=ROOT)
        if completed.returncode != 0:
            failure = record(row["recipe"], row["phase"], measurements_root)
            if next_action(CellState.FAILED, failure) is CellAction.REJECTED:
                rejected += 1
                out(f"stop arm {label}: measured rejection ({failure['error']})")
                continue
            out(f"stop queue {label}: exit {completed.returncode}")
            return 1
        ran += 1
    out(f"done: {ran} run, {skipped} skipped, {rejected} measured rejections")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE, help="Ordered queue file.")
    parser.add_argument(
        "--allow-reserved",
        action="store_true",
        help="Run reserved (final) cells. Required for any queue that contains them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    parser.add_argument("--measurements-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        queue = load(args.queue)
        return run_queue(
            queue,
            allow_reserved=args.allow_reserved,
            dry_run=args.dry_run,
            measurements_root=args.measurements_root,
        )
    except QueueError as exc:
        print(f"queue error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
