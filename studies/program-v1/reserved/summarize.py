"""Reserved-500 summary for the frozen selection (studies/program-v1/freeze.json).

Reads the committed episodes.csv of each selected cell and writes summary.json: successes,
Wilson 95 % interval per cell, and the task-stratified paired difference against B16 on the
same task/state/seed keys. Development and reserved counts stay in separate fields.

Example: uv run --no-sync python studies/program-v1/reserved/summarize.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.analysis import paired_difference, wilson

STUDY = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "summary.json"
CELLS = {
    "B16": {"kind": "recipe", "cycle": None},
    "iter_001-b1-a3": {"kind": "agent_solution", "cycle": "iter_001"},
    "iter_000-b0-a3": {"kind": "agent_solution", "cycle": "iter_000"},
    "precision-lang-int8": {"kind": "recipe", "cycle": None},
    "precision-lang-nf4": {"kind": "recipe", "cycle": None},
    "ladder-nf4-graph-1step-packed": {"kind": "recipe", "cycle": None},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episodes_path(cell: str, kind: str) -> Path:
    base = STUDY / ("reserved" if kind == "agent_solution" else "measurements") / cell / "final"
    return base / "episodes.csv"


def episodes(cell: str, kind: str) -> list[dict]:
    with episodes_path(cell, kind).open(newline="") as handle:
        return list(csv.DictReader(handle))


def memory_removed(cell: str, kind: str, baseline_bytes: int) -> dict:
    """Memory figure the cell's own record states, with the rule that produced it."""
    if kind == "agent_solution":
        record = read_json(STUDY / "reserved" / cell / "final" / "result.json")
        sizes = record["sizes"]
        return {
            "rule": "fixed evaluator (walk, live payloads, live CUDA, peak CUDA)",
            "resident_bytes": sizes["resident_bytes"],
            "baseline_bytes": sizes["baseline_resident_bytes"],
            "removed": sizes["size_reduction"],
        }
    summary = read_json(STUDY / "measurements" / cell / "benchmark" / "run.json")["summary"]
    resident = summary["resident_tensor_storage_bytes"]
    return {
        "rule": "resident tensor storage (storage_inventory)",
        "resident_bytes": resident,
        "baseline_bytes": baseline_bytes,
        "removed": max(0.0, 1 - resident / baseline_bytes),
    }


def development(cell: str, kind: str, cycle: str | None) -> dict | None:
    if kind == "agent_solution":
        path = STUDY / "cycles" / cycle / "nodes" / cell / "development.json"
    else:
        path = STUDY / "measurements" / cell / "development" / "run.json"
    if not path.exists():
        return None
    record = read_json(path)
    if kind == "agent_solution":
        successes, total = record["successes"], record["episodes"]
    else:
        successes, total = record["summary"]["successes"], record["summary"]["episodes"]
    return {"successes": successes, "episodes": total}


def main() -> None:
    baseline_bytes = read_json(STUDY / "measurements" / "B16" / "benchmark" / "run.json")[
        "summary"
    ]["resident_tensor_storage_bytes"]
    base = episodes("B16", "recipe")
    rows, missing = [], []
    for cell, spec in CELLS.items():
        if cell != "B16" and not episodes_path(cell, spec["kind"]).exists():
            missing.append(cell)
            continue
        rows_ = base if cell == "B16" else episodes(cell, spec["kind"])
        successes = sum(int(row["success"]) for row in rows_)
        total = len(rows_)
        row = {
            "cell": cell,
            "kind": spec["kind"],
            "development": development(cell, spec["kind"], spec["cycle"]),
            "reserved": {
                "successes": successes,
                "episodes": total,
                "wilson95": list(wilson(successes, total)),
                "per_task": {
                    str(task): sum(int(r["success"]) for r in rows_ if int(r["task_id"]) == task)
                    for task in sorted({int(r["task_id"]) for r in rows_})
                },
            },
            "memory": memory_removed(cell, spec["kind"], baseline_bytes),
            "paired_vs_B16": None,
        }
        if cell != "B16":
            pair = paired_difference(base, rows_)
            merged = {(r["task_id"], r["initial_state_hash"]): int(r["success"]) for r in rows_}
            gained = sum(
                1
                for r in base
                if int(r["success"]) == 0 and merged[(r["task_id"], r["initial_state_hash"])] == 1
            )
            lost = sum(
                1
                for r in base
                if int(r["success"]) == 1 and merged[(r["task_id"], r["initial_state_hash"])] == 0
            )
            row["paired_vs_B16"] = {**pair, "gained": gained, "lost": lost}
        rows.append(row)
    OUT.write_text(
        json.dumps(
            {
                "protocol": "program-v1-cleanroom",
                "phase": "reserved",
                "trials_per_cell": 500,
                "freeze": "studies/program-v1/freeze.json",
                "complete": not missing,
                "missing": missing,
                "statistics": "Wilson 95 %; task-stratified paired percentile bootstrap, "
                "10000 resamples, seed 42, identical task/state/seed keys",
                "cells": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT.relative_to(STUDY.parent.parent)}: {len(rows)} cells, missing {missing}")


if __name__ == "__main__":
    main()
