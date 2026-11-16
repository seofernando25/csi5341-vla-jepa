"""Process runner for sandboxed children: rlimits, wall-clock timeout, process-group reaping and bounded output reads."""

from __future__ import annotations

import functools
import os
import resource
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import IO

from rsi.paths import REPO_ROOT, SANDBOX_VENV
from rsi.sandbox.bwrap import build_bwrap_argv

MAX_LOG_CHARS = 200_000
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandboxed process exceeds its wall-clock timeout."""


class SandboxOutputError(ValueError):
    """Raised when an agent output file is not a small, plain, UTF-8 regular file."""


def _limit_child(cpu_seconds: int | None, memory_bytes: int | None) -> None:
    """Cap file size, core dumps, CPU time and address space in the child."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _reap(proc: subprocess.Popen[str]) -> None:
    """Stop a still-running process group, then kill any leftover members."""
    if proc.poll() is None:
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
    _signal_group(proc.pid, signal.SIGKILL)


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
    stdin_payload: str | None,
    timeout: float,
    cpu_seconds: int | None,
    memory_bytes: int | None,
) -> int:
    """Run `argv` in its own session with limits; always reap its process group."""
    started = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=env,
        start_new_session=True,
        # The child only calls setrlimit before exec, which is safe with threads.
        preexec_fn=functools.partial(_limit_child, cpu_seconds, memory_bytes),  # noqa: PLW1509
    )
    try:
        if stdin_payload is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_payload)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        while proc.poll() is None:
            if time.monotonic() - started > timeout:
                raise SandboxTimeoutError(f"Sandboxed process timed out after {timeout:.1f}s")
            time.sleep(0.1)
    finally:
        _reap(proc)
    return int(proc.returncode or 0)


def read_tail(path: Path) -> str:
    """Return the last MAX_LOG_CHARS characters of a log file without reading it whole."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > MAX_LOG_CHARS * 4:
            fh.seek(size - MAX_LOG_CHARS * 4)
        data = fh.read()
    return data.decode("utf-8", errors="replace")[-MAX_LOG_CHARS:]


def read_all(path: Path) -> str:
    """Return a whole log as text; the child's RLIMIT_FSIZE bounds how large it can grow."""
    return path.read_bytes().decode("utf-8", errors="replace")


def host_path_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def read_agent_output(workspace_dir: Path, name: str, max_bytes: int = 65536) -> str:
    """Read a small plain UTF-8 file from the workspace without following symlinks."""
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", "\x00")):
        raise SandboxOutputError(f"invalid output name: {name!r}")
    if ".." in name:
        raise SandboxOutputError(f"invalid output name: {name!r}")
    path = Path(workspace_dir) / name
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise SandboxOutputError(f"cannot open {name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SandboxOutputError(f"{name} is not a regular file")
        if st.st_size > max_bytes:
            raise SandboxOutputError(f"{name} exceeds {max_bytes} bytes")
        data = os.read(fd, max_bytes + 1)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise SandboxOutputError(f"{name} exceeds {max_bytes} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SandboxOutputError(f"{name} is not valid UTF-8") from exc


def run_captured(
    argv: list[str],
    *,
    workspace: Path,
    timeout: float,
    cpu_seconds: int | None,
    memory_bytes: int | None,
) -> tuple[int, str, str]:
    """Run `argv` with a temporary log pair and return (returncode, stdout tail, stderr tail)."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "stdout.txt"
        err_path = Path(tmp) / "stderr.txt"
        with (
            out_path.open("w", encoding="utf-8") as fout,
            err_path.open("w", encoding="utf-8") as ferr,
        ):
            returncode = run_process(
                argv,
                cwd=workspace,
                env=host_path_env(),
                stdout=fout,
                stderr=ferr,
                stdin_payload=None,
                timeout=timeout,
                cpu_seconds=cpu_seconds,
                memory_bytes=memory_bytes,
            )
        return returncode, read_tail(out_path), read_tail(err_path)


def run_sandboxed_python(
    args: list[str],
    *,
    workspace_dir: Path,
    evidence_dir: Path | None = None,
    extra_ro_binds: dict[Path, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 60.0,
    cpu_seconds: int | None = 30,
    memory_bytes: int | None = None,
    gpu: bool = False,
    network: bool = False,
    source_ro: dict[Path, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the repo venv python on untrusted code inside the sandbox.

    `memory_bytes=None` means the CPU default cap, or no address-space cap when `gpu` is set,
    because CUDA reserves large virtual ranges.
    """
    if not (REPO_ROOT / ".venv" / "bin" / "python").exists():
        raise RuntimeError("repo .venv is required to run sandboxed python")
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    argv = build_bwrap_argv(
        [f"{SANDBOX_VENV}/bin/python", *args],
        workspace_dir=workspace,
        evidence_dir=evidence_dir,
        extra_ro_binds=extra_ro_binds,
        network=network,
        env_overrides=env_overrides,
        gpu=gpu,
        source_ro=source_ro,
    )
    if memory_bytes is None and not gpu:
        memory_bytes = MAX_MEMORY_BYTES
    returncode, stdout, stderr = run_captured(
        argv,
        workspace=workspace,
        timeout=timeout,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
    )
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
