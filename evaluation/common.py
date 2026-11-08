"""JSON I/O, run records and environment metadata shared by evaluation commands."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


PROTOCOL = read_json(ROOT / "evaluation/protocol.json")


def run_id(label):
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + label


@contextmanager
def recorded_run(path, record):
    """Keep `record` in `path` as running, then completed, or failed with the error."""
    record["status"] = "running"
    write_json(path, record)
    started = time.perf_counter()
    try:
        yield record
        if record["status"] == "running":
            record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        write_json(path, record)


def episode_seed(task_id, trial, phase, protocol=None):
    """Simulator seed of one episode; final trials are offset from development trials."""
    protocol = protocol or PROTOCOL
    offset = protocol["final_seed_offset"] if phase == "final" else 0
    return protocol["seed"] + task_id * 1000 + trial + offset


def environment():
    import torch

    packages = {}
    for name in [
        "torch",
        "transformers",
        "lerobot",
        "hf-libero",
        "bitsandbytes",
        "mujoco",
        "robosuite",
        "numpy",
        "matplotlib",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        revision = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "git_revision": revision,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        x = torch.ones(16, device="cuda")
        result.update(
            gpu=gpu.name,
            gpu_total_bytes=gpu.total_memory,
            cuda_compute_verified=x.sum().item() == 16,
        )
        result["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    return result
