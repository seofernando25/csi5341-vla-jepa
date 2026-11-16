"""GPU resources for sandboxed code: device nodes, cached model assets and a flock slot semaphore."""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rsi.paths import REPO_ROOT, SANDBOX_HF_HUB, SANDBOX_REPO

GPU_DEVICE_GLOBS = (
    "/dev/nvidia[0-9]*",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps",
    "/dev/dri",
)
SANDBOX_HF_HOME = "/tmp/.cache/huggingface"
SANDBOX_SLOTS = "/tmp/gpu-slots"
SANDBOX_LIBERO_CONFIG = f"{SANDBOX_REPO}/outputs/evaluation/libero_config"
SANDBOX_LIBERO_DATA = f"{SANDBOX_REPO}/data"
GPU_SLOTS_DIR = REPO_ROOT / "outputs" / "gpu-slots"
GPU_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "HF_HOME": SANDBOX_HF_HOME,
    "HF_HUB_CACHE": SANDBOX_HF_HUB,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ASSET_MODEL_KEYS = ("baseline", "qwen", "world_model")
SLOT_POLL_SECONDS = 0.2


def gpu_device_paths() -> list[str]:
    """Return the existing NVIDIA and DRI device nodes that CUDA and EGL need."""
    found: list[str] = []
    for pattern in GPU_DEVICE_GLOBS:
        found.extend(sorted(glob.glob(pattern)))
    return [path for path in found if os.path.exists(path)]


def hf_hub_cache() -> Path:
    """Host Hugging Face hub cache directory, honouring HF_HUB_CACHE and HF_HOME."""
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def asset_cache_dirs(repo_root: Path) -> dict[Path, str]:
    """Map pinned model and simulator-asset cache folders to their read-only sandbox paths."""
    sources_file = Path(repo_root) / "evaluation" / "sources.json"
    if not sources_file.is_file():
        return {}
    sources = json.loads(sources_file.read_text(encoding="utf-8"))
    models = sources["models"]
    entries = [(models[key]["repo"], "model") for key in ASSET_MODEL_KEYS]
    entries.append((sources["simulator_assets"]["repo"], "dataset"))
    cache = hf_hub_cache()
    binds: dict[Path, str] = {}
    for repo, kind in entries:
        folder = cache / f"{kind}s--{repo.replace('/', '--')}"
        if folder.is_dir():
            binds[folder] = f"{SANDBOX_HF_HUB}/{folder.name}"
    return binds


def gpu_bind_args(repo_root: Path, slots_dir: Path | None) -> list[str]:
    """Return bwrap arguments for GPU devices, /sys/module (CUDA needs it), assets and the slots."""
    args: list[str] = ["--dir", "/sys", "--ro-bind", "/sys/module", "/sys/module"]
    for path in gpu_device_paths():
        args.extend(["--dev-bind", path, path])
    for host, dest in asset_cache_dirs(repo_root).items():
        args.extend(["--ro-bind", str(host), dest])
    args.extend(["--dir", SANDBOX_LIBERO_CONFIG, "--dir", SANDBOX_LIBERO_DATA])
    if slots_dir is not None:
        args.extend(["--bind", str(Path(slots_dir).resolve()), SANDBOX_SLOTS])
    return args


def _try_lock(path: Path) -> int | None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def acquire_slot(slots_dir: Path, n_slots: int, *, timeout: float | None = None) -> int:
    """Hold one of `n_slots` flock files; return its file descriptor (lock lives with it)."""
    if n_slots < 1:
        raise ValueError("n_slots must be at least 1")
    slots = Path(slots_dir)
    slots.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        for index in range(n_slots):
            fd = _try_lock(slots / f"slot_{index}")
            if fd is not None:
                return fd
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"no GPU slot free in {slots} after {timeout:.1f}s")
        time.sleep(SLOT_POLL_SECONDS)


@contextmanager
def gpu_slot(slots_dir: Path, n_slots: int, *, timeout: float | None = None) -> Iterator[None]:
    """Hold one GPU slot for the block; the kernel releases it when the process dies."""
    fd = acquire_slot(slots_dir, n_slots, timeout=timeout)
    try:
        yield
    finally:
        os.close(fd)


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a command while holding a GPU slot.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="hold a slot while the command runs")
    run.add_argument("--slots-dir", type=Path, default=Path(SANDBOX_SLOTS))
    run.add_argument("--n", type=int, required=True)
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")
    fd = acquire_slot(args.slots_dir, args.n, timeout=args.timeout)
    # exec keeps the inherited lock fd, so the slot is held exactly as long as the command runs.
    os.set_inheritable(fd, True)
    os.execvp(command[0], command)
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
