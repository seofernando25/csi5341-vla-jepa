"""The failure analysis replays the screen's frames and reports drift, never task success."""

import json
from pathlib import Path

import pytest
import torch

from evaluation.compression import failure
from report.plots import failure as figure

HEADER = (
    "variant,phase,task_id,task_name,trial,initial_state_hash,seed,success,steps,"
    "wall_seconds,status\n"
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_episodes(path: Path, cell: str, successes: dict[int, int], trials: int = 10) -> None:
    """One development episodes.csv with `successes[task]` successes out of `trials`."""
    lines = [HEADER]
    for task, count in successes.items():
        for trial in range(trials):
            lines.append(
                f"{cell},development,{task},task,{trial},hash{trial},1000,"
                f"{int(trial < count)},100,1.0,completed\n"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def test_screen_frames_are_the_bank_indices_outside_calibration():
    bank = [{"task_id": index // 10} for index in range(100)]
    frames = failure.screen_frames({"bank_indices": [0, 1, 10, 11]}, bank)
    assert frames == [index for index in range(100) if index not in (0, 1, 10, 11)]
    assert len(frames) == 96


def test_sample_rows_is_evenly_spaced_and_survives_few_positions():
    rows = failure.sample_rows(torch.arange(128), rows=8)
    assert len(rows) == 8
    assert (int(rows[0]), int(rows[-1])) == (0, 127)
    assert len(failure.sample_rows(torch.arange(3), rows=8)) == 3
    assert len(failure.sample_rows(torch.tensor([], dtype=torch.long))) == 0


def test_linear_cka_of_identical_features_is_one():
    x = torch.randn(32, 16)
    assert failure.linear_cka(x, x.clone()) == pytest.approx(1.0, abs=1e-6)
    assert failure.linear_cka(x, -x) == pytest.approx(1.0, abs=1e-6)


def test_action_summary_counts_gripper_flips_per_step():
    reference = torch.zeros(2, 7, 7)
    physical = reference.clone()
    physical[1, :, 6] = 1.0  # one frame, seven steps, gripper command differs
    summary = failure.action_summary(physical, reference)
    assert summary["frames"] == 2
    assert summary["frames_with_gripper_flip"] == 1
    assert summary["gripper_steps_disagreeing"] == 7
    assert summary["gripper_steps"] == 14
    assert summary["mean_arm_mse"] == 0.0
    assert summary["per_frame_gripper_disagreement"] == [0.0, 1.0]


def test_task_drops_report_the_worst_tasks_against_the_reference(tmp_path):
    study = tmp_path / "program-v1"
    write_episodes(
        study / "measurements" / "B16" / "development" / "episodes.csv",
        "B16",
        {0: 8, 1: 6, 2: 9},
    )
    write_episodes(
        study / "measurements" / "precision-lang-hqq2" / "development" / "episodes.csv",
        "precision-lang-hqq2",
        {0: 8, 1: 2, 2: 9},
    )
    drops = failure.task_drops(study, cells=("precision-lang-hqq2",))
    cell = drops["development"]["cells"]["precision-lang-hqq2"]
    assert cell["delta_vs_reference"] == {"0": 0, "1": -4, "2": 0}
    assert cell["worst_tasks"][0] == {"task_id": 1, "delta": -4}
    assert (cell["tasks_dropped"], cell["tasks_gained"]) == (1, 0)
    assert drops["development"]["reference"] == {"0": 8, "1": 6, "2": 9}
    assert "reserved" not in drops  # no reserved record: the phase is not invented


def test_missing_failure_analysis_is_not_fatal(tmp_path):
    record = figure.failure_record(tmp_path)
    assert record["status"] == "missing"
    assert figure.fig_failure(record) is None


def test_figure_reads_the_committed_record(tmp_path, monkeypatch):
    study = tmp_path / "program-v1"
    cells = [
        {
            "cell": f"precision-lang-{name}",
            "nominal_bits": bits,
            "cka": {
                "all": [0.99] * 4,
                "image": [0.97] * 4,
                "text": [0.999] * 4,
                "conditioning": 0.98,
            },
            "actions": {
                "frames": 80,
                "mean_arm_mse": error,
                "median_arm_mse": error,
                "max_arm_mse": error,
                "frames_with_gripper_flip": 0,
                "gripper_steps_disagreeing": 0,
                "gripper_steps": 560,
            },
            "attention": {"mean_mass": 0.04, "spearman_median": 0.9},
            "reproduction": {"status": "ok", "max_abs_arm_mse_difference": 0.0},
        }
        for name, bits, error in (
            ("int8", 8, 1.6e-5),
            ("nf4", 4, 2.0e-4),
            ("hqq3", 3, 7.8e-4),
            ("hqq2", 2, 3.6e-2),
        )
    ]
    write(
        study / "failure" / "failure.json",
        {
            "status": "completed",
            "protocol": {"frames": 80},
            "reference": {"cell": "B16"},
            "cells": cells,
            "tasks": {"development": {"reference": {"0": 8}, "cells": {}}},
            "does_not_show": ["LIBERO success"],
        },
    )
    record = figure.failure_record(study)
    assert record["status"] == "ok"
    assert record["frames"] == 80
    assert [cell["nominal_bits"] for cell in record["cells"]] == [8, 4, 3, 2]
    assert record["cells"][0]["cka_decoder"] == 0.99
    assert record["cells"][0]["cka_conditioning"] == 0.98
    assert record["does_not_show"] == ["LIBERO success"]

    out = tmp_path / "figures"
    out.mkdir()
    monkeypatch.setattr(figure, "OUT", out)
    ledger = figure.fig_failure(record)
    assert (out / "failure.pdf").exists()
    assert [cell["nominal_bits"] for cell in ledger["cells"]] == [8, 4, 3, 2]


def test_figure_needs_two_cells(tmp_path, monkeypatch):
    record = {"status": "ok", "cells": [{"cell": "a", "nominal_bits": 8}]}
    monkeypatch.setattr(figure, "OUT", tmp_path)
    assert figure.fig_failure(record) is None
