"""Durable loop state on disk: atomic JSON writes, `loop_state.json`, the study lock and the STOP file."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

LOCK_NAME = "loop.lock"
STATE_NAME = "loop_state.json"
STOP_NAME = "STOP"


class LoopLockedError(RuntimeError):
    """Another loop holds the study's lock."""


def utc_now() -> str:
    """UTC timestamp as ISO text."""
    return datetime.now(UTC).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON to a temp file in the same directory, then `os.replace` it over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_holder(fd: int) -> str:
    raw = os.pread(fd, 32, 0).decode("utf-8", "replace").strip()
    return raw or "unknown"


class StudyLock:
    """Exclusive flock on `study_dir/loop.lock`; the kernel releases it when the holder dies."""

    def __init__(self, study_dir: Path) -> None:
        self.path = Path(study_dir) / LOCK_NAME
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the lock or raise `LoopLockedError` naming the holder's pid."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder(fd)
            os.close(fd)
            raise LoopLockedError(f"{self.path} is held by pid {holder}") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        """Drop the lock if held."""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def lock_holder(study_dir: Path) -> str | None:
    """Pid text of the live lock holder, or None when the study is not locked."""
    path = Path(study_dir) / LOCK_NAME
    if not path.is_file():
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_holder(fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def load_loop_state(study_dir: Path) -> dict[str, Any] | None:
    """Read `loop_state.json`, or None when the study has no state yet."""
    path = Path(study_dir) / STATE_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_loop_state(study_dir: Path, state: dict[str, Any]) -> None:
    """Persist loop state atomically, stamping `updated_utc`."""
    state["updated_utc"] = utc_now()
    write_json_atomic(Path(study_dir) / STATE_NAME, state)


def stop_requested(study_dir: Path) -> bool:
    """True when the `STOP` file exists in the study directory."""
    return (Path(study_dir) / STOP_NAME).exists()
