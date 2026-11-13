"""CPU-only tests for the parallel rollout harness (no GPU, env, or model)."""

import subprocess

import pytest
import torch

from evaluation.parallel_rollout import (
    check_gpu_guard,
    compare_trials,
    gpu_used_bytes,
    stack_batches,
)


def _batch(value, task="lift"):
    return {
        "observation.image": torch.full((1, 3, 4, 4), value),
        "observation.state": torch.full((1, 8), value),
        "task": [task],
    }


def test_stack_batches_concatenates_rows():
    stacked = stack_batches([_batch(1.0, "a"), _batch(2.0, "b"), _batch(3.0, "c")])
    assert stacked["observation.image"].shape == (3, 3, 4, 4)
    assert stacked["observation.state"].shape == (3, 8)
    assert stacked["task"] == ["a", "b", "c"]
    assert stacked["observation.image"][1, 0, 0, 0].item() == 2.0


def test_stack_batches_skips_all_none_keys():
    a = _batch(1.0)
    b = _batch(2.0)
    a["action"] = None
    b["action"] = None
    stacked = stack_batches([a, b])
    assert "action" not in stacked
    assert stacked["observation.state"].shape == (2, 8)


def test_stack_batches_nests_dicts_and_collects_unknowns():
    a = {"m": {"s": "x"}, "r": 0.0, "task": ["a"]}
    b = {"m": {"s": "y"}, "r": 1.0, "task": ["b"]}
    stacked = stack_batches([a, b])
    assert stacked["m"] == {"s": ["x", "y"]}
    assert stacked["r"].tolist() == [0.0, 1.0]
    assert stacked["task"] == ["a", "b"]


def test_stack_batches_rejects_mixed_none():
    with pytest.raises(TypeError):
        stack_batches([{"n": 1}, {"n": None}])


def test_batched_queues_match_serial_select_action():
    """Per-env queues from one [N, C, D] chunk equal N serial [1, C, D] forwards."""
    torch.manual_seed(0)
    chunk = torch.randn(3, 16, 7)
    n_action_steps = 5
    queued = chunk.transpose(0, 1)[:n_action_steps]
    parallel_queues = [list(queued[:, i, :].unbind(0)) for i in range(3)]
    for i in range(3):
        serial = list(chunk[i : i + 1].transpose(0, 1)[:n_action_steps])  # 5 x [1, D]
        assert len(parallel_queues[i]) == n_action_steps == len(serial)
        for got, want in zip(parallel_queues[i], serial):
            assert torch.equal(got, want[0])


def test_gpu_used_bytes_parses_smi(monkeypatch):
    class Proc:
        stdout = "7835\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc())
    assert gpu_used_bytes() == 7835 * 1024 * 1024


def test_gpu_used_bytes_none_on_failure(monkeypatch):
    def fail(*a, **k):
        raise OSError("no smi")

    monkeypatch.setattr(subprocess, "run", fail)
    assert gpu_used_bytes() is None
    assert check_gpu_guard() == 0  # unknown GPU state must not block


def test_check_gpu_guard_refuses_full_card(monkeypatch):
    import evaluation.parallel_rollout as pr

    monkeypatch.setattr(pr, "gpu_used_bytes", lambda: 21 * 10**9)
    with pytest.raises(RuntimeError, match="GPU guard"):
        check_gpu_guard()


def test_compare_trials_flags_success_diffs():
    rows = [
        {"task_id": 0, "trial": 0, "success": 1, "steps": 40, "seed": 1000},
        {"task_id": 1, "trial": 0, "success": 0, "steps": 60, "seed": 2000},
    ]
    same = [dict(r) for r in rows]
    assert compare_trials(rows, same)["success_diffs"] == []
    flipped = [dict(r) for r in rows]
    flipped[1]["success"] = 1
    diff = compare_trials(rows, flipped)
    assert diff["success_diffs"] == [{"task_id": 1, "trial": 0, "first": 0, "second": 1}]
    dropped = rows[:1]
    diff = compare_trials(rows, dropped)
    assert diff["missing_in_second"] == [(1, 0)]
