"""The queue runner plans, resumes and stops without touching a GPU."""

import json
from types import SimpleNamespace

import pytest

from evaluation.common import ROOT, write_json
from evaluation.compression import queue as q

QUEUE = ROOT / "studies/program-v1/queue.json"
RESERVED = ROOT / "studies/program-v1/queue-reserved.json"


def completed(recipe, phase, measurements, **extra):
    write_json(
        measurements / recipe / phase / "run.json",
        {"phase": phase, "status": "completed", "summary": {}, **extra},
    )


def test_committed_queue_starts_with_the_baseline_then_the_language_ladder():
    steps = q.plan(q.load(QUEUE))
    assert [(row["recipe"], row["phase"]) for row in steps["cells"][:3]] == [
        ("B16", "screen"),
        ("B16", "smoke"),
        ("B16", "development"),
    ]
    assert steps["cells"][3]["recipe"] == "precision-lang-int8"
    assert steps["withheld_reserved"] == []
    assert steps["minutes"] == sum(q.PHASE_MINUTES[row["phase"]] for row in steps["cells"])


def test_committed_queue_is_ordered_baseline_ladder_joint_formats_benchmarks():
    cells = q.load(QUEUE)["cells"]
    order = [cell["recipe"] for cell in cells]
    assert order[0] == "B16"
    assert order.index("precision-joint-va-hqq2") > order.index("precision-action-hqq2")
    assert order.index("format-lang-hqq4") > order.index("precision-joint-va-hqq2")
    assert cells[-1]["phase"] == "benchmark"
    assert {cell["phase"] for cell in cells} <= set(q.PHASES)


def test_reserved_cells_are_withheld_without_the_flag():
    steps = q.plan(q.load(RESERVED))
    assert steps["cells"] == []
    assert len(steps["withheld_reserved"]) == 4
    assert {cell["phase"] for cell in steps["withheld_reserved"]} == {"final"}


def test_reserved_cells_run_only_with_the_flag():
    queue = q.load(RESERVED)
    with pytest.raises(q.QueueError, match="allow-reserved"):
        q.run_queue(queue, invoke=lambda *args, **kwargs: pytest.fail("must not run"))
    assert q.run_queue(queue, dry_run=True, out=lambda line: None) == 0
    steps = q.plan(queue, allow_reserved=True)
    assert {row["phase"] for row in steps["cells"]} == {"final"}


def test_completed_cells_are_skipped_and_do_not_count_as_remaining(tmp_path):
    measurements = tmp_path / "measurements"
    completed("B16", "screen", measurements)
    steps = q.plan(q.load(QUEUE), measurements_root=measurements)
    assert steps["cells"][0]["state"] == "done"
    assert steps["minutes"] == steps["total_minutes"] - q.PHASE_MINUTES["screen"]


def test_a_runtime_failure_stops_the_queue(tmp_path):
    measurements = tmp_path / "measurements"
    write_json(
        measurements / "B16" / "screen" / "run.json",
        {"status": "failed", "error_type": "RuntimeError", "error": "CUDA out of memory"},
    )
    invoked = []
    code = q.run_queue(
        q.load(QUEUE),
        measurements_root=measurements,
        invoke=lambda *args, **kwargs: invoked.append(args) or SimpleNamespace(returncode=0),
        out=lambda line: None,
    )
    assert code == 1
    assert invoked == []


def test_a_nonfinite_screen_is_a_measured_outcome_and_the_queue_continues(tmp_path):
    measurements = tmp_path / "measurements"
    write_json(
        measurements / "B16" / "screen" / "run.json",
        {
            "status": "failed",
            "error_type": "ValueError",
            "error": "Nonfinite actions at bank_index=7",
        },
    )
    invoked = []

    def invoke(argv, cwd=None):
        invoked.append(argv)
        return SimpleNamespace(returncode=0)

    code = q.run_queue(
        q.load(QUEUE),
        measurements_root=measurements,
        invoke=invoke,
        out=lambda line: None,
    )
    assert code == 0
    assert len(invoked) == len(q.load(QUEUE)["cells"]) - 1


def test_command_is_the_canonical_measurement_path():
    argv = q.command("B16", "smoke")
    assert argv[1:] == [
        "-m",
        "evaluation.compression.study",
        "run",
        "--recipe",
        str(q.recipe_path("B16")),
        "--phase",
        "smoke",
    ]


def test_a_missing_recipe_is_a_queue_defect(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"cells": [{"recipe": "not-authored", "phase": "smoke"}]}))
    with pytest.raises(q.QueueError, match="recipe not authored"):
        q.load(path)


def test_duplicate_cells_and_unknown_phases_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {"cells": [{"recipe": "B16", "phase": "smoke"}, {"recipe": "B16", "phase": "smoke"}]}
        )
    )
    with pytest.raises(q.QueueError, match="duplicate"):
        q.load(duplicate)
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"cells": [{"recipe": "B16", "phase": "reserved"}]}))
    with pytest.raises(q.QueueError, match="unknown phase"):
        q.load(unknown)
