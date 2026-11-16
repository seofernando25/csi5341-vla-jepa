"""Red-team regressions: sandbox limits, FIFO-safe output reads, live duplicate retry, K07/K34."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from rsi.policy_dev.replay import worker_failure
from rsi.sandbox import SandboxOutputError, read_agent_output
from rsi.sandbox.process import run_process as _run_process


def _run_python(code: str, *, cpu_seconds: int | None, memory_bytes: int | None) -> int:
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        return _run_process(
            [sys.executable, "-c", code],
            cwd=Path(tempfile.gettempdir()),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            stdout=out,
            stderr=err,
            stdin_payload=None,
            timeout=30.0,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
        )


def test_address_space_limit_stops_a_ballooning_child() -> None:
    rc = _run_python(
        "x = bytearray(800 * 1024 * 1024)", cpu_seconds=None, memory_bytes=256 * 1024 * 1024
    )
    assert rc != 0


def test_cpu_limit_stops_an_endless_loop() -> None:
    started = time.monotonic()
    rc = _run_python("while True:\n    pass", cpu_seconds=1, memory_bytes=None)
    assert rc != 0
    assert time.monotonic() - started < 20


def test_worker_limit_failures_name_the_signal() -> None:
    proc = subprocess.CompletedProcess(args=[], returncode=-24, stdout="", stderr="")
    assert worker_failure(proc) == "replay worker killed by SIGXCPU"
    failed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="x\nreplay failed: MemoryError\n"
    )
    assert worker_failure(failed) == "replay failed: MemoryError"


def test_fifo_in_workspace_does_not_hang_the_host(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "proposal.md")
    result: dict[str, BaseException | str] = {}

    def read() -> None:
        try:
            result["value"] = read_agent_output(tmp_path, "proposal.md")
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
            result["error"] = exc

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(5.0)
    assert not thread.is_alive(), "read_agent_output blocked on a FIFO"
    assert isinstance(result.get("error"), SandboxOutputError)
    assert "not a regular file" in str(result["error"])


def test_regular_output_is_still_read(tmp_path: Path) -> None:
    (tmp_path / "method.py").write_text("x = 1\n", encoding="utf-8")
    assert read_agent_output(tmp_path, "method.py") == "x = 1\n"
