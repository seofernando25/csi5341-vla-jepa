"""Reserved-500 summary for the program-v3 size winners (studies/program-v3/selection.json).

Reads the committed episodes.csv of each selected candidate and writes summary.json: successes,
Wilson 95 % interval per cell, and the task-stratified paired difference against B16 on the same
task/state/seed keys. Development and reserved counts stay in separate fields; reserved trials
never guide selection.

Example: uv run --no-sync python studies/program-v3/reserved/summarize.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.analysis import paired_difference, wilson

STUDY = Path(__file__).resolve().parents[1]
V1 = STUDY.parent / "program-v1"
OUT = Path(__file__).resolve().parent / "summary.json"
CELLS = ["iter_000-b2-a2", "iter_001-b2-a3", "iter_002-b2-a2"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episodes(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def development(cell: str) -> dict | None:
    """The candidate's development record, whichever cycle its node lives in."""
    for path in sorted((STUDY / "cycles").glob(f"*/nodes/{cell}/development.json")):
        record = read_json(path)
        sizes = record.get("sizes") or {}
        return {
            "successes": record["successes"],
            "episodes": record["episodes"],
            "resident_bytes": sizes.get("resident_bytes"),
            "size_removed": sizes.get("size_reduction"),
        }
    return None


def main() -> None:
    base = episodes(V1 / "measurements" / "B16" / "final" / "episodes.csv")
    rows, missing = [], []
    for cell in CELLS:
        path = STUDY / "reserved" / cell / "final" / "episodes.csv"
        if not path.exists():
            missing.append(cell)
            continue
        rows_ = episodes(path)
        successes, total = sum(int(r["success"]) for r in rows_), len(rows_)
        row = {
            "cell": cell,
            "development": development(cell),
            "reserved": {
                "successes": successes,
                "episodes": total,
                "wilson95": list(wilson(successes, total)),
                "per_task": {
                    str(task): sum(int(r["success"]) for r in rows_ if int(r["task_id"]) == task)
                    for task in sorted({int(r["task_id"]) for r in rows_})
                },
            },
            "paired_vs_B16": None,
        }
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
                "program": "program-v3",
                "protocol": "program-v3-debiased",
                "phase": "reserved",
                "trials_per_cell": 500,
                "selection": "studies/program-v3/selection.json",
                "reference": "studies/program-v1/measurements/B16/final/episodes.csv",
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
