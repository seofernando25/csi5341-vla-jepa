"""Validate complete cohorts and frozen confirmation choices on CPU fixtures."""

import csv

import pytest

from evaluation.common import file_hash, read_json, write_json
from evaluation.compression import protocol


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    states = tmp_path / "states.json"
    tasks = [
        {
            "task_id": task,
            "name": f"task-{task}",
            "development_hashes": [f"hash-{task}-{trial}" for trial in range(10)],
            "final_hashes": [f"final-{task}-{trial}" for trial in range(50)],
        }
        for task in range(10)
    ]
    write_json(states, {"tasks": tasks})
    monkeypatch.setattr(protocol, "STATE_PATH", states)
    folder = tmp_path / "candidate" / "development"
    folder.mkdir(parents=True)
    rows = [
        {
            "variant": "candidate",
            "phase": "development",
            "task_id": task,
            "task_name": f"task-{task}",
            "trial": trial,
            "initial_state_hash": f"hash-{task}-{trial}",
            "seed": 1000 + task * 1000 + trial,
            "success": trial % 2,
            "steps": 10,
            "wall_seconds": 1.0,
            "status": "completed",
        }
        for task in range(10)
        for trial in range(10)
    ]
    record = {
        "candidate": {"id": "candidate"},
        "phase": "development",
        "status": "completed",
        "states_sha256": file_hash(states),
        "original_protocol": {"seed": 1000, "final_seed_offset": 10000},
        "summary": {
            "episodes": 100,
            "successes": 50,
            "task_ids": list(range(10)),
            "phase": "development",
            "task_macro_success": 0.5,
        },
    }

    def save():
        write_json(folder / "run.json", record)
        with (folder / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save()
    return folder, rows, record, save


def test_complete_development_cohort_is_validated(cohort):
    folder, rows, record, _ = cohort
    checked, episodes = protocol.validated_episodes(folder)
    assert checked == record
    assert len(episodes) == len(rows) == 100


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 999),
        ("phase", "final"),
        ("initial_state_hash", "wrong"),
        ("success", 2),
        ("variant", "other"),
        ("task_name", "wrong"),
        ("wall_seconds", "nan"),
    ],
)
def test_episode_identity_and_outcomes_cannot_be_mislabeled(cohort, field, value):
    folder, rows, _, save = cohort
    rows[0][field] = value
    save()
    with pytest.raises(ValueError):
        protocol.validated_episodes(folder)


def test_missing_duplicate_and_summary_mismatch_are_rejected(cohort):
    folder, rows, record, save = cohort
    rows[-1] = rows[0].copy()
    save()
    with pytest.raises(ValueError, match="duplicate"):
        protocol.validated_episodes(folder)
    rows.pop()
    save()
    with pytest.raises(ValueError):
        protocol.validated_episodes(folder)
    record["status"] = "failed"
    save()
    with pytest.raises(ValueError, match="completed"):
        protocol.validated_episodes(folder)


def test_recorded_summary_must_recount_completed_episodes(cohort):
    folder, _, record, save = cohort
    record["summary"]["successes"] = 51
    save()
    with pytest.raises(ValueError, match="summary"):
        protocol.validated_episodes(folder)


def test_reserved_trials_require_immutable_recipe_source_and_input_registration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol, "STUDY", tmp_path)
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    monkeypatch.setattr(protocol, "scientific_sources", lambda: {"source.py": "original"})
    recipe = tmp_path / "recipe.json"
    write_json(recipe, {"id": "candidate", "changes": []})
    with pytest.raises(ValueError, match="Freeze"):
        protocol.verify_final_registration(recipe)
    (tmp_path / "calibration.bin").write_bytes(b"fixed")
    frozen = {
        "recipe_sha256": {"candidate": file_hash(recipe)},
        "scientific_sources": {"source.py": "original"},
        "checkpoint_sha256": "checkpoint",
        "deployed_tensor_sha256": {"candidate": "compressed-state"},
        "input_sha256": {"calibration.bin": file_hash(tmp_path / "calibration.bin")},
    }
    write_json(tmp_path / "final_selection.json", frozen)
    assert protocol.verify_final_registration(recipe)["candidate_frozen"] == "candidate"
    assert (
        protocol.verify_final_registration(recipe)["deployed_tensor_sha256"]
        == "compressed-state"
    )
    (tmp_path / "calibration.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input"):
        protocol.verify_final_registration(recipe)
    (tmp_path / "calibration.bin").write_bytes(b"fixed")
    monkeypatch.setattr(protocol, "scientific_sources", lambda: {"source.py": "changed"})
    with pytest.raises(ValueError, match="source"):
        protocol.verify_final_registration(recipe)
    monkeypatch.setattr(protocol, "scientific_sources", lambda: {"source.py": "original"})
    write_json(recipe, {"id": "candidate", "changes": [{"method": "new"}]})
    with pytest.raises(ValueError, match="recipe"):
        protocol.verify_final_registration(recipe)


def test_quantized_storage_is_not_counted_as_logical_parameter_pruning():
    import torch
    from torch import nn

    from evaluation.compression.transforms import (
        Int8Embedding,
        logical_parameter_count,
        storage_inventory,
    )

    embedding = nn.Embedding(100, 64)
    quantized = Int8Embedding(embedding)
    assert logical_parameter_count(embedding) == logical_parameter_count(quantized) == 6400
    assert (
        storage_inventory(quantized)["resident_tensor_storage_bytes"]
        < storage_inventory(embedding)["resident_tensor_storage_bytes"]
    )
    packed = nn.Module()
    packed.in_features, packed.out_features = 128, 64
    packed.register_buffer("packed_weight", torch.zeros(64, 64, dtype=torch.uint8))
    assert logical_parameter_count(packed) == 128 * 64


def test_native_reset_guard_observes_states_and_restores_factory(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from evaluation import run as native

    states = np.arange(12, dtype=np.float64).reshape(3, 4)
    env = SimpleNamespace(
        init_states=True, init_state_id=0, _reset_stride=1, _init_states=states
    )

    def reset(seed=None):
        env.init_state_id += 1
        return seed

    env.reset = reset
    task = {"development_hashes": [protocol.state_hash(row) for row in states]}
    def factory(task_id, phase):
        return env, task
    monkeypatch.setattr(native, "make_environment", factory)
    with protocol.guarded_environments() as membership:
        checked, _ = native.make_environment(2, "development")
        for trial in range(3):
            seed = native.seed_for(2, trial, "development")
            assert checked.reset(seed=seed) == seed
        assert membership == [
            {
                "task_id": 2,
                "trial": trial,
                "seed": native.seed_for(2, trial, "development"),
                "actual_state_sha256": task["development_hashes"][trial],
            }
            for trial in range(3)
        ]
        with pytest.raises(ValueError, match="counter or actual reset seed"):
            checked.reset(seed=0)
    assert native.make_environment is factory


@pytest.mark.parametrize("bits,axis", [(2, 0), (2, 1), (3, 0), (3, 1), (4, 1)])
def test_hqq_packed_cpu_storage_and_linear_output(bits, axis):
    import torch
    from torch import nn

    from evaluation.common import ROOT
    from evaluation.compression.lowbit import HQQPackedLinear, package
    from evaluation.compression.transforms import logical_parameter_count, storage_inventory

    if not (ROOT / "outputs/compression/vendor/hqq/hqq").exists():
        pytest.skip("Pinned optional HQQ source cache is absent")
    package("hqq")
    from hqq.core.quantize import Quantizer

    torch.manual_seed(19)
    source = nn.Linear(128, 64, dtype=torch.bfloat16)
    values, meta = Quantizer.quantize(
        source.weight.detach(),
        nbits=bits,
        group_size=64,
        axis=axis,
        optimize=True,
        round_zero=False,
        bitpack=True,
        compute_dtype=torch.bfloat16,
        device="cpu",
    )
    packed = HQQPackedLinear(source, Quantizer, values, meta)
    x = torch.randn(7, 128, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, packed.dequantize(), source.bias)
    assert torch.isfinite(packed(x)).all()
    assert torch.equal(packed(x), expected)
    assert logical_parameter_count(packed) == logical_parameter_count(source)
    assert (
        storage_inventory(packed)["resident_tensor_storage_bytes"]
        < storage_inventory(source)["resident_tensor_storage_bytes"]
    )


def test_export_distinguishes_original_checkpoint_from_compressed_parameters(
    tmp_path, monkeypatch
):
    from evaluation.compression import analysis

    monkeypatch.setattr(analysis, "STUDY", tmp_path)
    write_json(tmp_path / "coverage.json", {"domains": {}})
    frozen = tmp_path / "final_selection.json"
    write_json(frozen, {"deployed_tensor_sha256": {"B16": "baseline", "compressed": "packed"}})
    ledger = []
    for identity, core, memory in [("B16", 1000, 2000), ("compressed", 750, 800)]:
        summary = {
            "predictions_per_repetition": 1,
            "repetitions": 1,
            "memory_peaks": [{"allocated_bytes": memory, "reserved_bytes": memory}],
            "inference_core_logical_parameter_elements": core,
        }
        folder = tmp_path / "measurements" / identity / "benchmark"
        folder.mkdir(parents=True)
        write_json(
            folder / "run.json",
            {
                "summary": summary,
                "model": {"logical_parameters": 1400},
                "final_registration": {"selection_sha256": file_hash(frozen)},
                "deployed_tensor_fingerprint": {
                    "sha256": "baseline" if identity == "B16" else "packed"
                },
            },
        )
        with (folder / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["mode", "repetition", "prediction", "latency_ms"]
            )
            writer.writeheader()
            for mode in ["policy", "pipeline"]:
                writer.writerow(
                    {"mode": mode, "repetition": 0, "prediction": 0, "latency_ms": 10}
                )
        ledger.append(
            {
                "candidate": identity,
                "family": "fixture",
                "domains": [],
                "changes": [],
                "phases": {"benchmark": {"status": "completed", "error": None}},
            }
        )
    monkeypatch.setattr(analysis, "ledger", lambda: ledger)
    exported = analysis.export()
    measured = exported["candidates"]["compressed"]["phases"]["benchmark"]["summary"]
    assert "logical_parameters" not in measured
    assert measured["original_checkpoint_parameter_elements"] == 1400
    assert measured["inference_core_logical_parameter_elements"] == 750
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[1]["measured_inference_core_removed_fraction"]) == 0.25
    assert float(rows[1]["benchmark_peak_memory_reduction_fraction"]) == 0.6
    record_path = tmp_path / "measurements/compressed/benchmark/run.json"
    altered = read_json(record_path)
    altered["deployed_tensor_fingerprint"]["sha256"] = "other-model"
    write_json(record_path, altered)
    with pytest.raises(ValueError, match="model payload"):
        analysis.export()


def test_shape_priming_preserves_timing_counts_and_repetition_noise(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from evaluation.compression import artifacts, study

    predictions = []

    def predict(batch):
        value = torch.rand(1, 7, 7)
        predictions.append(value.clone())
        return value

    policy = SimpleNamespace(
        reset=lambda: None,
        predict_action_chunk=predict,
        model=SimpleNamespace(action_model=SimpleNamespace(action_horizon=7)),
        config=SimpleNamespace(n_action_steps=7),
    )
    bank = [{"raw": torch.zeros(1), "instruction": str(index)} for index in range(3)]
    monkeypatch.setattr(study, "observation_bank", lambda: bank)
    processor = SimpleNamespace(reset=lambda: None)
    post = lambda value: value
    post.reset = lambda: None
    monkeypatch.setattr(artifacts, "processors", lambda policy: (processor, processor, post))
    monkeypatch.setattr(artifacts, "prepare", lambda *args: {})
    monkeypatch.setattr(artifacts, "inference_autocast", lambda policy: nullcontext())
    monkeypatch.setattr(artifacts, "storage_inventory", lambda policy: {})
    monkeypatch.setattr(artifacts, "serialize", lambda *args: {"bytes": 1})
    for name in ["synchronize", "reset_peak_memory_stats", "empty_cache"]:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ["max_memory_allocated", "max_memory_reserved"]:
        monkeypatch.setattr(torch.cuda, name, lambda: 1)
    summary = artifacts.benchmark(policy, "fixture", tmp_path, predictions=2, repetitions=2)
    with (tmp_path / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 2 * 2
    assert summary["backend_shape_priming"]["observations"] == 3
    assert not summary["backend_shape_priming"]["included_in_timing_csv"]
    warmup = artifacts.PROTOCOL["timing"]["warmup"]
    for mode in range(2):
        for repetition in range(2):
            torch.manual_seed(artifacts.PROTOCOL["seed"] + repetition)
            expected = [torch.rand(1, 7, 7) for _ in range(warmup + 2)]
            offset = 3 + (mode * 2 + repetition) * (warmup + 2)
            assert torch.equal(predictions[offset + warmup], expected[warmup])


def test_tensor_fingerprint_detects_packed_values_and_alias_changes():
    import copy

    import torch
    from torch import nn

    from evaluation.compression.artifacts import tensor_fingerprint

    source = nn.Module()
    weight = nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    source.register_parameter("left", weight)
    source.register_parameter("right", weight)
    source.register_buffer("scalar", torch.tensor(1.0))
    source.register_buffer("packed", torch.tensor([3, 17], dtype=torch.uint8))
    source._ohb_beta_scalar = 1.125
    reproduced = copy.deepcopy(source)
    original = tensor_fingerprint(source)
    assert original == tensor_fingerprint(reproduced)
    assert original["tensor_count"] == 3
    reproduced.packed[0] = 4
    assert original["sha256"] != tensor_fingerprint(reproduced)["sha256"]
    reproduced.packed[0] = 3
    reproduced._ohb_beta_scalar = 1.0
    assert original["sha256"] != tensor_fingerprint(reproduced)["sha256"]
    reproduced._ohb_beta_scalar = 1.125
    reproduced.right = nn.Parameter(reproduced.right.detach().clone())
    assert original["sha256"] != tensor_fingerprint(reproduced)["sha256"]


def test_joint_refinement_runs_before_selection_with_older_parent_plan(tmp_path, monkeypatch):
    from evaluation.compression import boundaries, mission

    parent_execution = {
        "development_smoke_tolerance_episodes": 1,
        "development_maximum_candidates": 20,
    }
    write_json(
        tmp_path / "execution_plan.json",
        {
            **parent_execution,
            "structural_boundary_rounds": 0,
            "post_boundary_joint_candidates": ["refined"],
            "supplemental_deployment_endpoints": ["compiled"],
        },
    )
    baseline = tmp_path / "measurements/B16/smoke/run.json"
    write_json(baseline, {"status": "completed", "summary": {"successes": 8}})
    monkeypatch.setattr(boundaries, "ROOT", tmp_path)
    monkeypatch.setattr(boundaries, "STUDY", tmp_path)
    monkeypatch.setattr(mission, "STUDY", tmp_path)
    calls = []

    def queue(phase, candidates):
        calls.append((phase, candidates))
        summary = (
            {"successes": 7}
            if phase == "smoke"
            else {"peak_allocated_bytes": 900, "pipeline_median_ms": 120}
        )
        for candidate in candidates:
            write_json(
                tmp_path / "measurements" / candidate / phase / "run.json",
                {"status": "completed", "summary": summary},
            )

    monkeypatch.setattr(mission, "queue", queue)
    monkeypatch.setattr(
        mission,
        "ledger",
        lambda: [
            {
                "candidate": candidate,
                "family": "joint compression" if candidate == "refined" else "compiled TorchAO",
                "changes": [],
                "phases": {phase: mission.result(candidate, phase) for phase in ["screen", "smoke"]},
            }
            for candidate in ["refined", "compiled"]
        ],
    )
    boundaries.main()
    assert calls == [
        ("screen", ["refined"]), ("smoke", ["refined"]),
        ("screen", ["compiled"]), ("smoke", ["compiled"]),
    ]
    assert mission.development_selection(parent_execution) == ["B16", "refined", "compiled"]


def test_joint_boundary_choices_use_component_smokes_and_preserve_ranking_criteria():
    from evaluation.compression.refinements import choose

    def row(identity, change, successes, memory=100, status="completed"):
        return {
            "candidate": identity,
            "changes": [change],
            "phases": {
                "screen": {"status": status, "summary": {"peak_allocated_bytes": memory}},
                "smoke": {"status": "completed", "summary": {"successes": successes}},
                "final": {"status": "completed", "summary": {"successes": 500}},
            },
        }

    rows = [
        row("mild", {"method": "channels", "scope": "text_mlp", "fraction": 0.125}, 8),
        row("passing", {"method": "channels", "scope": "text_mlp", "fraction": 0.5}, 7),
        row("failing", {"method": "channels", "scope": "text_mlp", "fraction": 0.75}, 6, 50),
        row("runtime-failure", {"method": "channels", "scope": "text_mlp", "fraction": 0.9}, 8, status="failed"),
        row("energy", {"method": "layers", "scope": "vision", "count": 8}, 6),
        row("cka", {"method": "layers", "scope": "vision", "count": 4, "criterion": "cka"}, 8),
    ]
    selected = choose(rows, 7)
    assert selected["channels", "text_mlp"]["candidate"] == "passing"
    assert selected["layers", "vision"]["changes"][0]["criterion"] == "cka"


def test_joint_boundary_registration_preserves_frozen_recipes(tmp_path, monkeypatch):
    from evaluation.compression import refinements

    monkeypatch.setattr(refinements, "STUDY", tmp_path)
    recipe = tmp_path / "recipes/frozen.json"
    write_json(recipe, {"id": "frozen", "changes": []})
    write_json(tmp_path / "plan.json", {"candidates": ["frozen"]})
    write_json(
        tmp_path / "joint_boundary_selection.json",
        {"candidates": ["frozen"], "recipe_sha256": {"frozen": file_hash(recipe)}},
    )
    monkeypatch.setattr(refinements, "ledger", lambda: pytest.fail("Must not reselect"))
    assert refinements.register({"joint_pruning_boundary_refinements": True}) == ["frozen"]
    write_json(recipe, {"id": "frozen", "changes": [{"method": "changed"}]})
    with pytest.raises(ValueError, match="Preserve"):
        refinements.register({"joint_pruning_boundary_refinements": True})
