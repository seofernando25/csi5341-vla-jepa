"""Tests for GPU slots and the sandboxed GPU path (CPU-only unless RSI_GPU_LIVE=1)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from rsi.gpu import acquire_slot, gpu_slot
from rsi.paths import REPO_ROOT
from rsi.sandbox import build_bwrap_argv, run_sandboxed_python

requires_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
live_gpu = pytest.mark.skipif(os.environ.get("RSI_GPU_LIVE") != "1", reason="set RSI_GPU_LIVE=1")


def test_semaphore_never_exceeds_slots(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0
    done: list[int] = []

    def holder(index: int) -> None:
        nonlocal active, peak
        with gpu_slot(tmp_path, 2, timeout=30):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.2)
            with lock:
                active -= 1
        done.append(index)

    threads = [threading.Thread(target=holder, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert peak == 2
    assert sorted(done) == [0, 1, 2]


def test_full_semaphore_times_out(tmp_path: Path) -> None:
    fds = [acquire_slot(tmp_path, 2), acquire_slot(tmp_path, 2)]
    try:
        with pytest.raises(TimeoutError):
            acquire_slot(tmp_path, 2, timeout=0.3)
    finally:
        for fd in fds:
            os.close(fd)


def test_invalid_slot_count_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        acquire_slot(tmp_path, 0)


def test_slot_released_when_holder_is_sigkilled(tmp_path: Path) -> None:
    code = (
        "import sys, time\n"
        "from rsi.gpu import acquire_slot\n"
        f"acquire_slot(__import__('pathlib').Path({str(tmp_path)!r}), 1)\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        start_new_session=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError):
            acquire_slot(tmp_path, 1, timeout=0.2)
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=10)
        fd = acquire_slot(tmp_path, 1, timeout=10)
        os.close(fd)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
        if holder.stdout is not None:
            holder.stdout.close()


def test_cli_holds_slot_while_command_runs(tmp_path: Path) -> None:
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from rsi.gpu import acquire_slot\n"
        "try:\n"
        f"    acquire_slot(Path({str(tmp_path)!r}), 1, timeout=0)\n"
        "except TimeoutError:\n"
        "    sys.exit(0)\n"
        "sys.exit(7)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    cli = [
        sys.executable,
        "-m",
        "rsi.gpu",
        "run",
        "--slots-dir",
        str(tmp_path),
        "--n",
        "1",
        "--",
        sys.executable,
        "-c",
        probe,
    ]
    result = subprocess.run(
        cli, cwd=REPO_ROOT, env=env, check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    after = subprocess.run(
        [
            sys.executable,
            "-m",
            "rsi.gpu",
            "run",
            "--slots-dir",
            str(tmp_path),
            "--n",
            "1",
            "--",
            "true",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert after.returncode == 0, after.stderr


@requires_bwrap
def test_flock_is_shared_across_bwrap_boundary(tmp_path: Path) -> None:
    slots = tmp_path / "slots"
    probe = (
        "import fcntl, os\n"
        "fd = os.open('/tmp/gpu-slots/slot_0', os.O_RDWR)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    print('free')\n"
        "except BlockingIOError:\n"
        "    print('busy')\n"
    )
    argv = build_bwrap_argv(
        ["/usr/bin/python3", "-c", probe],
        workspace_dir=tmp_path / "ws",
        repo_root=tmp_path,
        gpu=True,
        gpu_slots_dir=slots,
    )
    fd = acquire_slot(slots, 1)
    try:
        busy = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)
        assert busy.stdout.strip() == "busy", busy.stderr
    finally:
        os.close(fd)
    free = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)
    assert free.stdout.strip() == "free", free.stderr


@live_gpu
@pytest.mark.slow
def test_live_cuda_matmul_in_gpu_sandbox(tmp_path: Path) -> None:
    code = "import torch; x = torch.ones(1024, 1024, device='cuda'); print(float((x @ x).sum()))"
    result = run_sandboxed_python(
        ["-c", code], workspace_dir=tmp_path / "ws", gpu=True, timeout=150, cpu_seconds=None
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip().splitlines()[-1]) == 1024.0**3
