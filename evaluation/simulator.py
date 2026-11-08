"""LIBERO-Spatial environments with the frozen initial-state manifest.

Example: python -m evaluation.simulator   # rewrites studies/evaluation/initial_states.json
"""

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.common import PROTOCOL, ROOT, read_json, write_json
from evaluation.sources import artifact

MANIFEST = ROOT / "studies/evaluation/initial_states.json"


def configure():
    """Point LIBERO at its installed files and select headless rendering."""
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


def make_environment(task_id, phase):
    """Environment whose resets walk the manifest's states for `phase`, in order."""
    root = configure()
    import libero.libero
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    libero.libero._assets_path_cache = artifact("simulator_assets")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(
        suite,
        task_id,
        "libero_spatial",
        obs_type="pixels_agent_pos",
        observation_height=PROTOCOL["observation_height"],
        observation_width=PROTOCOL["observation_width"],
    )
    task = read_json(MANIFEST)["tasks"][task_id]
    states = np.asarray(torch.load(root / task[f"{phase}_file"], weights_only=False))
    env._init_states = states[task[f"{phase}_indices"]]
    return env, task


def batch_raw(raw):
    if isinstance(raw, dict):
        return {k: batch_raw(v) for k, v in raw.items()}
    return np.expand_dims(raw, 0)


def tensor_raw(raw):
    if isinstance(raw, dict):
        return {k: tensor_raw(v) for k, v in raw.items()}
    return torch.from_numpy(np.asarray(raw).copy())


def numpy_raw(raw):
    if isinstance(raw, dict):
        return {k: numpy_raw(v) for k, v in raw.items()}
    return raw.numpy()


def manifest():
    """Development states: the first 10 original states absent from the official 50 per task."""
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
        development = [i for i, row in enumerate(original) if state_hash(row) not in final_hashes]
        development = development[:10]
        if len(development) != 10 or len(final) != 50:
            raise ValueError("Unexpected LIBERO state counts")
        tasks.append(
            {
                "task_id": task_id,
                "name": task.name,
                "instruction": task.language,
                "final_file": str(final_path.relative_to(root)),
                "development_file": str(original_path.relative_to(root)),
                "development_indices": development,
                "development_hashes": [state_hash(original[i]) for i in development],
                "final_indices": list(range(50)),
                "final_hashes": final_hashes,
            }
        )
    return {
        "suite": "libero_spatial",
        "task_order_index": 0,
        "development": "10 original states/task, exact state vectors excluded from official set",
        "final": "all 50 official pruned states/task",
        "tasks": tasks,
    }


if __name__ == "__main__":
    write_json(MANIFEST, manifest())
    print(MANIFEST.relative_to(ROOT))
