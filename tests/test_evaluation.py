"""Statistics, checkpoint selection and the frozen evaluation splits."""

from collections import Counter

import pytest

from evaluation.adaptation import choose
from evaluation.analysis import (
    core_summary,
    paired_difference,
    success_summary,
    timing_summary,
    wilson,
)
from evaluation.common import ROOT, read_json


def episode(task=0, trial=0, success=1, seed=10):
    return {
        "task_id": str(task),
        "trial": str(trial),
        "initial_state_hash": f"state-{task}-{trial}",
        "seed": str(seed),
        "success": str(success),
        "status": "completed",
    }


def test_wilson_retains_uncertainty_at_boundaries():
    low, high = wilson(0, 50)
    assert low == 0 and 0.06 < high < 0.08
    low, high = wilson(50, 50)
    assert 0.92 < low < 0.94 and high == 1


def test_pairing_rejects_changed_seed_and_duplicate_trial():
    with pytest.raises(ValueError, match="identical"):
        paired_difference([episode()], [episode(seed=11)])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_difference([episode(), episode()], [episode(), episode()])


def test_identical_and_uniformly_worse_paired_outcomes():
    rows = [episode(t, i) for t in range(2) for i in range(5)]
    same = paired_difference(rows, rows)
    assert same["difference_pp"] == 0 and same["ci95_pp"] == [0, 0]
    worse = paired_difference(rows, [dict(r, success="0") for r in rows])
    assert worse["difference_pp"] == -100


def test_unbalanced_and_runtime_failure_outcomes_are_not_success_evidence():
    record = {"protocol": {"task_ids": [0, 1]}}
    with pytest.raises(ValueError, match="balanced"):
        success_summary(record, [episode()])
    with pytest.raises(ValueError, match="Invalid"):
        success_summary(record, [dict(episode(), status="failed"), episode(task=1)])


def test_missing_timing_records_are_rejected():
    record = {"summary": {"predictions_per_repetition": 2, "repetitions": 1}}
    rows = [{"mode": "policy", "repetition": "0", "prediction": "0"}]
    with pytest.raises(ValueError, match="missing"):
        timing_summary(record, rows)


def test_neural_timings_require_full_native_equivalence_check():
    record = {
        "predictions_per_repetition": 1,
        "repetitions": 1,
        "equivalence": {"observations": 1, "identical_flow_noise_seeds": True},
    }
    rows = [{"mode": "neural", "repetition": "0", "prediction": "0", "latency_ms": "1"}]
    with pytest.raises(ValueError, match="full observation bank"):
        core_summary(record, rows)
    record["equivalence"]["observations"] = 100
    assert core_summary(record, rows)["median_ms"] == 1


def test_checkpoint_selection_uses_loss_and_breaks_ties_by_budget():
    rows = [
        {"step": step, "loss": loss, "samples": 200}
        for step, loss in [(1500, 0.5), (5000, 0.4), (10000, 0.4)]
    ]
    assert choose(rows, [1500, 5000, 10000])["step"] == 5000
    with pytest.raises(ValueError, match="incomplete"):
        choose(rows[:2], [1500, 5000, 10000])
    rows[-1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        choose(rows, [1500, 5000, 10000])


def test_committed_initial_states_reserve_distinct_final_trials():
    manifest = read_json(ROOT / "studies/evaluation/initial_states.json")
    assert len(manifest["tasks"]) == 10
    for task in manifest["tasks"]:
        development, final = task["development_hashes"], task["final_hashes"]
        assert len(set(development)) == 10
        assert len(set(final)) == 50
        assert not set(development) & set(final)


def test_training_split_excludes_whole_heldout_trajectories():
    split = read_json(ROOT / "studies/evaluation/training_split.json")
    train, validation = set(split["train_episodes"]), set(split["validation_episodes"])
    assert not train & validation
    assert train | validation == set(range(432))
    assert split["train_frames"] + split["validation_frames"] == 52970


def test_validation_frames_are_task_balanced_and_held_out():
    split = read_json(ROOT / "studies/evaluation/training_split.json")
    samples = read_json(ROOT / "studies/evaluation/validation_samples.json")
    assert len(set(samples["heldout_row_indices"])) == 200
    assert Counter(samples["task_indices"]) == {task: 20 for task in range(10)}
    assert set(samples["episode_indices"]) <= set(split["validation_episodes"])
    assert not set(samples["episode_indices"]) & set(split["train_episodes"])
