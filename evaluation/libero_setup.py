"""Create local simulator configuration and freeze disjoint initial-state manifests."""

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.common import ROOT, file_hash, write_json


def configure():
    root = (
        Path(next(iter(importlib.util.find_spec("libero").submodule_search_locations))) / "libero"
    )
    location = ROOT / "outputs/evaluation/libero_config"
    location.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(ROOT / "data"),
        "assets": str(root / "assets"),
    }
    (location / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(location)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return root


def state_hash(row):
    return hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()


def manifest():
    root = configure()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        final_path = root / "init_files" / task.problem_folder / task.init_states_file
        original_path = final_path.with_suffix(".init")
        final = np.asarray(torch.load(final_path, weights_only=False))
        original = np.asarray(torch.load(original_path, weights_only=False))
        final_hashes = [state_hash(row) for row in final]
        development = [i for i, row in enumerate(original) if state_hash(row) not in final_hashes][
            :10
        ]
        if len(development) != 10 or len(final) != 50:
            raise ValueError(
                "Unexpected LIBERO state counts; review protocol instead of silently changing it"
            )
        tasks.append(
            {
                "task_id": task_id,
                "name": task.name,
                "instruction": task.language,
                "final_file": str(final_path.relative_to(root)),
                "final_file_sha256": file_hash(final_path),
                "development_file": str(original_path.relative_to(root)),
                "development_file_sha256": file_hash(original_path),
                "development_indices": development,
                "development_hashes": [state_hash(original[i]) for i in development],
                "final_indices": list(range(50)),
                "final_hashes": final_hashes,
            }
        )
    return {
        "suite": "libero_spatial",
        "task_order_index": 0,
        "development": "10 original states/task, exact state vectors excluded from official pruned set",
        "final": "all 50 official pruned states/task",
        "tasks": tasks,
    }


if __name__ == "__main__":
    output = ROOT / "studies/evaluation/initial_states.json"
    write_json(output, manifest())
    print(output.relative_to(ROOT))
