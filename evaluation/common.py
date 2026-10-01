"""Provenance helpers shared by the evaluation and report commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
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


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest():
    files = [ROOT / name for name in ["pyproject.toml", "uv.lock"]]
    for folder in ["evaluation", "src"]:
        files += [p for p in (ROOT / folder).rglob("*") if p.suffix in {".py", ".json"}]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


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
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "git_revision": revision,
        "source_manifest": source_manifest(),
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
