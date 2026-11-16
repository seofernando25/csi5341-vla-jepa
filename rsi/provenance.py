"""Frozen run inputs recorded with every loop and cycle: code revision, protocol hash, GPU and environment."""

from __future__ import annotations

import functools
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from rsi.paths import REPO_ROOT
from rsi.runner import load_protocol


def sha256_file(path: Path) -> str:
    """Hex sha256 of one file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    """Trimmed stdout of one `git -C repo` call, or None when git fails."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_revision(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD of the repo at run start, or null with `code_dirty: true` when the tree is dirty."""
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    dirty = status is None or bool(status)
    return {"code_revision": None if dirty else head, "code_dirty": dirty}


@functools.cache
def environment() -> dict[str, Any]:
    """Python, torch and CUDA versions of the running interpreter, read once per process."""
    torch_version, cuda_version = _torch_versions()
    return {"python": platform.python_version(), "torch": torch_version, "cuda": cuda_version}


def _torch_versions() -> tuple[str | None, str | None]:
    """`torch.__version__` and its CUDA build, or nulls when torch cannot be imported."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a missing or broken torch must not stop the loop
        return None, None
    return torch.__version__, torch.version.cuda


def gpu_inventory() -> dict[str, Any]:
    """Product name and count of the host GPUs, or nulls when nvidia-smi cannot be queried."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"name": None, "count": None}
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode or not names:
        return {"name": None, "count": None}
    return {"name": names[0], "count": len(names)}


def _study_repo(study_dir: Path) -> Path:
    """Git toplevel of the study directory, or the project repo when it is not in one."""
    top = _git(study_dir, "rev-parse", "--show-toplevel")
    return Path(top) if top else REPO_ROOT


def provenance(study_dir: Path) -> dict[str, Any]:
    """Frozen run inputs: code revision, protocol hash, checkpoint, GPU, environment, seed, suite."""
    path = Path(study_dir) / "protocol.json"
    proto = load_protocol(study_dir) if path.is_file() else {}
    return {
        **code_revision(_study_repo(study_dir)),
        "protocol_sha256": sha256_file(path) if path.is_file() else None,
        "checkpoint": proto.get("checkpoint"),
        "checkpoint_revision": proto.get("checkpoint_revision"),
        "suite": proto.get("suite"),
        "seed": proto.get("seed"),
        "gpu": gpu_inventory(),
        "environment": environment(),
    }
