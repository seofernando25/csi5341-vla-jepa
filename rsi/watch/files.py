"""Safe reads of study files for the watch server: paths, caps and symlink checks.

Agent-authored trees can contain symlinks; every read goes through `inside` so the watch server
cannot be turned into an arbitrary-file-read primitive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

PART = re.compile(r"^[A-Za-z0-9._-]+$")
HEX = re.compile(r"^[0-9a-f]{7,40}$")
STEP_RE = re.compile(r"^step_(\d+)$")
ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
ARTIFACT_CAP = 200_000
CONTEXT_CAP = 40_000


class SourceLike(Protocol):
    """The three fields a source id carries; `watch.Source` satisfies it."""

    kind: str
    cycle_id: str
    name: str


def inside(study: Path, path: Path) -> Path | None:
    """Resolve `path` and return it only when it stays inside the resolved study root."""
    try:
        real = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return real if real.is_relative_to(study) else None


def read_json(study: Path, path: Path) -> dict[str, Any] | None:
    """Parse one JSON object under `study`; a symlink that escapes the study is not read."""
    safe = inside(study, path)
    if safe is None:
        return None
    try:
        obj = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def read_text(study: Path, path: Path, cap: int = ARTIFACT_CAP) -> str | None:
    """UTF-8 text of one file under `study`, capped; None when missing or outside the study."""
    safe = inside(study, path)
    if safe is None or not safe.is_file():
        return None
    try:
        with safe.open("rb") as fh:
            data = fh.read(cap * 4 + 1)
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")[:cap]


def rel(study: Path, path: Path) -> str:
    """Study-relative POSIX path so the page never sees a machine-specific absolute path."""
    try:
        return path.relative_to(study).as_posix()
    except ValueError:
        return path.as_posix()


def step_dir(study: Path, source: SourceLike) -> Path:
    """The `policies/<cycle>/step_NN` directory of a policy source."""
    return study / "policies" / source.cycle_id / f"step_{int(source.name):02d}"


def log_path(study: Path, source: SourceLike) -> Path:
    """The agent log of one source: the node log, or the newest policy attempt log."""
    if source.kind == "node":
        return study / "cycles" / source.cycle_id / "nodes" / source.name / "logs" / "stdout.log"
    attempts = study / "policies" / source.cycle_id / f"step_{int(source.name):02d}" / "logs"
    best: tuple[int, Path] | None = None
    if attempts.is_dir():
        for path in attempts.iterdir():
            match = ATTEMPT_RE.match(path.name)
            has_log = match is not None and (path / "stdout.log").is_file()
            if has_log and (best is None or int(match.group(1)) > best[0]):
                best = (int(match.group(1)), path / "stdout.log")
    return best[1] if best else attempts / "attempt_0" / "stdout.log"


def as_num(value: Any) -> float | None:
    """A JSON number as float; booleans and anything else are None."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def as_int(value: Any) -> int | None:
    """A JSON integer; booleans and anything else are None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def as_str(value: Any) -> str | None:
    """Any recorded value as text; None stays None."""
    return None if value is None else str(value)
