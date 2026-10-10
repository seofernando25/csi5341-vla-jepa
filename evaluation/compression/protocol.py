"""State, episode and reserved-confirmation integrity for compression results."""

import csv
import importlib.util
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from evaluation.common import ROOT, file_hash, read_json
from evaluation.libero_setup import state_hash

STUDY = ROOT / "studies/compression"
STATE_PATH = ROOT / "studies/evaluation/initial_states.json"
PHASE_COUNTS = {"smoke": 1, "development": 10, "final": 50}


def verify_state_assets():
    root = (
        Path(next(iter(importlib.util.find_spec("libero").submodule_search_locations))) / "libero"
    )
    manifest = read_json(STATE_PATH)
    checks = []
    for task in manifest["tasks"]:
        groups = {}
        for phase in ["development", "final"]:
            path = root / task[f"{phase}_file"]
            digest = file_hash(path)
            if digest != task[f"{phase}_file_sha256"]:
                raise ValueError(f"LIBERO state file changed: task {task['task_id']}/{phase}")
            states = np.asarray(torch.load(path, weights_only=False))
            hashes = [state_hash(states[index]) for index in task[f"{phase}_indices"]]
            if hashes != task[f"{phase}_hashes"]:
                raise ValueError("Selected state vectors disagree with the frozen manifest")
            groups[phase] = set(hashes)
            checks.append(
                {
                    "task_id": task["task_id"],
                    "phase": phase,
                    "sha256": digest,
                    "selected_vectors_verified": len(hashes),
                }
            )
        if groups["development"] & groups["final"]:
            raise ValueError("Development and final state vectors overlap")
    return {
        "manifest_sha256": file_hash(STATE_PATH),
        "files": checks,
        "disjoint_selected_vectors_verified": True,
    }


@contextmanager
def guarded_environments():
    """Verify the exact state vector/counter used by each native reset."""
    from evaluation import run as native

    original = native.make_environment
    membership = []

    def checked(task_id, phase):
        env, task = original(task_id, phase)
        if not env.init_states or env.init_state_id != 0 or env._reset_stride != 1:
            env.close()
            raise ValueError("Matched sequential rollout requires state index zero and stride one")
        hashes = [state_hash(row) for row in env._init_states]
        if hashes != task[f"{phase}_hashes"]:
            env.close()
            raise ValueError("Environment contains different initial-state vectors")
        reset = env.reset
        calls = 0

        def checked_reset(seed=None, **kwargs):
            nonlocal calls
            if env.init_state_id != calls or seed != native.seed_for(task_id, calls, phase):
                raise ValueError(
                    "Environment state counter or actual reset seed differs from protocol"
                )
            actual_hash = state_hash(env._init_states[calls])
            output = reset(seed=seed, **kwargs)
            if env.init_state_id != calls + 1:
                raise ValueError("Native reset did not advance exactly one initial state")
            membership.append(
                {
                    "task_id": task_id,
                    "trial": calls,
                    "seed": seed,
                    "actual_state_sha256": actual_hash,
                }
            )
            calls += 1
            return output

        env.reset = checked_reset
        return env, task

    native.make_environment = checked
    try:
        yield membership
    finally:
        native.make_environment = original


def validated_episodes(folder):
    folder = Path(folder)
    record = read_json(folder / "run.json")
    phase = record["phase"]
    if record["status"] != "completed" or phase not in PHASE_COUNTS:
        raise ValueError("Only completed, explicitly sized rollout phases enter comparisons")
    if record["states_sha256"] != file_hash(STATE_PATH):
        raise ValueError("Frozen initial-state manifest differs from the recorded run")
    tasks = read_json(STATE_PATH)["tasks"]
    protocol = record["original_protocol"]
    expected_phase = "final" if phase == "final" else "development"
    count = PHASE_COUNTS[phase]
    with (folder / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        task, trial = int(row["task_id"]), int(row["trial"])
        key = task, trial
        if key in seen or task not in range(10) or trial not in range(count):
            raise ValueError("Missing, duplicate or unexpected task/trial membership")
        seen.add(key)
        seed = (
            protocol["seed"]
            + task * 1000
            + trial
            + (protocol["final_seed_offset"] if phase == "final" else 0)
        )
        if (
            row["variant"] != record["candidate"]["id"]
            or row["phase"] != expected_phase
            or row["status"] != "completed"
            or int(row["seed"]) != seed
            or row["initial_state_hash"] != tasks[task][f"{expected_phase}_hashes"][trial]
            or row["task_name"] != tasks[task]["name"]
            or int(row["success"]) not in (0, 1)
            or int(row["steps"]) < 1
            or not math.isfinite(float(row["wall_seconds"]))
            or float(row["wall_seconds"]) <= 0
        ):
            raise ValueError("Episode outcome, state, task, phase or seed is invalid")
    if seen != {(task, trial) for task in range(10) for trial in range(count)}:
        raise ValueError("Rollout cohort is incomplete")
    successes = sum(int(row["success"]) for row in rows)
    if record.get("state_membership") is not None:
        actual = {
            (entry["task_id"], entry["trial"], entry["seed"], entry["actual_state_sha256"])
            for entry in record["state_membership"]
        }
        declared = {
            (int(row["task_id"]), int(row["trial"]), int(row["seed"]), row["initial_state_hash"])
            for row in rows
        }
        if len(record["state_membership"]) != len(rows) or actual != declared:
            raise ValueError("Observed reset membership differs from recorded episodes")
    elif phase == "final":
        raise ValueError("Final episodes require observed reset-state membership")
    summary = record["summary"]
    if (
        summary["episodes"] != len(rows)
        or summary["successes"] != successes
        or summary["phase"] != expected_phase
        or sorted(summary["task_ids"]) != list(range(10))
        or not math.isclose(summary["task_macro_success"], successes / len(rows), abs_tol=1e-12)
    ):
        raise ValueError("Recorded summary differs from episode measurements")
    return record, rows


def matched_controls(left, right):
    for name in ["states_sha256", "checkpoint_sha256", "calibration_sha256", "original_protocol"]:
        if not left.get(name) or left[name] != right.get(name):
            raise ValueError(f"Matched control differs: {name}")
    for name in ["gpu", "driver", "packages", "torch_cuda"]:
        if left["environment"].get(name) != right["environment"].get(name):
            raise ValueError(f"Matched environment differs: {name}")
    for name in ["evaluation/models.py", "evaluation/run.py", "evaluation/common.py"]:
        if (
            left["environment"]["source_manifest"][name]
            != right["environment"]["source_manifest"][name]
        ):
            raise ValueError(f"Baseline evaluator changed: {name}")


def scientific_sources():
    files = [
        ROOT / path
        for path in [
            "evaluation/models.py",
            "evaluation/run.py",
            "evaluation/common.py",
            "evaluation/libero_setup.py",
            "evaluation/protocol.json",
            "evaluation/sources.json",
            "pyproject.toml",
            "uv.lock",
            "studies/compression/methods.json",
        ]
    ]
    files += [
        ROOT / "evaluation/compression" / f"{name}.py"
        for name in [
            "study",
            "transforms",
            "advanced",
            "compiled",
            "attention_matching",
            "tokens",
            "artifacts",
            "protocol",
            "lowbit",
        ]
    ]
    result = {str(path.relative_to(ROOT)): file_hash(path) for path in files}
    for package, relative in {
        "lerobot": [
            "policies/vla_jepa/modeling_vla_jepa.py",
            "policies/vla_jepa/action_head.py",
            "policies/vla_jepa/configuration_vla_jepa.py",
            "policies/vla_jepa/processor_vla_jepa.py",
            "envs/libero.py",
            "utils/random_utils.py",
        ],
        "transformers": [
            "models/qwen3_vl/modeling_qwen3_vl.py",
            "models/qwen3_vl/configuration_qwen3_vl.py",
        ],
        "diffusers": ["models/attention_processor.py", "models/attention.py"],
    }.items():
        root = Path(importlib.util.find_spec(package).origin).parent
        result.update({f"installed/{package}/{name}": file_hash(root / name) for name in relative})
    methods = read_json(STUDY / "methods.json")
    for method, manifest in methods["upstream"].items():
        folder = ROOT / "outputs/compression/vendor" / method
        for name, digest in manifest["files"].items():
            if file_hash(folder / name) != digest:
                raise ValueError(f"Pinned upstream algorithm changed: {method}/{name}")
            result[str((folder / name).relative_to(ROOT))] = digest
    torchao = ROOT / "outputs/compression/vendor_packages/torchao"
    for path in sorted(torchao.rglob("*")):
        if path.suffix in {".py", ".so"}:
            result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def verify_final_registration(recipe_path):
    path = STUDY / "final_selection.json"
    if not path.exists():
        raise ValueError("Freeze the final selection before reserved confirmation trials")
    frozen = read_json(path)
    recipe = read_json(recipe_path)
    expected = frozen["recipe_sha256"].get(recipe["id"])
    if not expected or file_hash(recipe_path) != expected:
        raise ValueError("Candidate or recipe differs from the frozen final selection")
    if scientific_sources() != frozen["scientific_sources"]:
        raise ValueError("Scientific source changed after final selection was frozen")
    for relative, digest in frozen["input_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"Frozen scientific input changed: {relative}")
    return {
        "selection_sha256": file_hash(path),
        "candidate_frozen": recipe["id"],
        "scientific_source_verified": True,
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "deployed_tensor_sha256": frozen["deployed_tensor_sha256"][recipe["id"]],
    }
