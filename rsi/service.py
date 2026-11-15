"""Durable Dream-RSI runs: atomic state files, the study lock, stop files, and detached launch."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

LOCK_NAME = "loop.lock"
STATE_NAME = "loop_state.json"
STOP_NAME = "STOP"
LOG_NAME = "loop.log"
SERVICE_NAME = "service.json"
PIDFILE_NAME = "loop.pid"
PASS_ENV = ("PATH", "HOME")
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def unit_name(study_dir: Path) -> str:
    """Systemd unit name derived from the resolved study directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(Path(study_dir).resolve()).lower()).strip("-")
    return f"rsi-{slug}"


def loop_argv(study_dir: Path, loop_args: list[str], python: str) -> list[str]:
    """Command that runs one `loop` invocation for the study."""
    return [python, "-m", "rsi", "loop", "--study-dir", str(study_dir), *loop_args]


def systemd_argv(
    study_dir: Path,
    loop_args: list[str],
    *,
    repo: Path,
    python: str,
    env: dict[str, str],
) -> list[str]:
    """`systemd-run --user` argv for a restarting loop unit with appended logs and minimal env."""
    log = Path(study_dir) / LOG_NAME
    setenv = [f"--setenv={key}={env[key]}" for key in PASS_ENV if key in env]
    return [
        "systemd-run",
        "--user",
        "--unit",
        unit_name(study_dir),
        "--collect",
        "--property=Restart=on-failure",
        "--property=RestartSec=30",
        "--property=StartLimitBurst=5",
        f"--working-directory={repo}",
        f"--property=StandardOutput=append:{log}",
        f"--property=StandardError=append:{log}",
        *setenv,
        "--",
        *loop_argv(study_dir, loop_args, python),
    ]


def systemd_available() -> bool:
    """True when `systemd-run` exists and the user manager answers."""
    if shutil.which("systemd-run") is None:
        return False
    probe = subprocess.run(
        ["systemctl", "--user", "show-environment"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def _unit_state(unit: str) -> str:
    probe = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ActiveState", "--value", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() or "unknown"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_service(study_dir: Path) -> dict[str, Any]:
    path = Path(study_dir) / SERVICE_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def start(study_dir: Path, loop_args: list[str]) -> dict[str, Any]:
    """Launch the loop detached (systemd unit, or setsid fallback) and record how it was started."""
    study = Path(study_dir).resolve()
    if (holder := lock_holder(study)) is not None:
        raise LoopLockedError(f"{study} is already running (lock held by pid {holder})")
    (study / STOP_NAME).unlink(missing_ok=True)
    study.mkdir(parents=True, exist_ok=True)
    env = {key: os.environ[key] for key in PASS_ENV if key in os.environ}
    python = sys.executable
    if systemd_available():
        argv = systemd_argv(study, loop_args, repo=REPO_ROOT, python=python, env=env)
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"systemd-run failed: {done.stderr.strip()}")
        record = {"mode": "systemd", "unit": unit_name(study)}
    else:
        argv = ["setsid", "nohup", *loop_argv(study, loop_args, python)]
        with (study / LOG_NAME).open("ab") as log:
            proc = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        (study / PIDFILE_NAME).write_text(f"{proc.pid}\n", encoding="utf-8")
        record = {"mode": "detached", "pid": proc.pid}
    write_json_atomic(study / SERVICE_NAME, {**record, "started_utc": utc_now()})
    return record


def _progress(study_dir: Path, state: dict[str, Any] | None) -> dict[str, Any]:
    cycles_dir = Path(study_dir) / "cycles"
    out: dict[str, Any] = {
        "iteration": None,
        "stage": "not started",
        "cycle_id": None,
        "nodes_done": None,
        "nodes_total": None,
    }
    if state is not None:
        pending = [it for it in state.get("iterations", []) if not _iteration_done(it)]
        out["iterations_done"] = len(state.get("iterations", [])) - len(pending)
        out["cycles_target"] = state.get("cycles_target")
        if pending:
            cur = pending[0]
            out.update(iteration=cur["iteration"], cycle_id=cur["cycle_id"])
            out["stage"] = "cycle" if cur["cycle"]["status"] != "done" else "policy_dev"
            nodes = cycles_dir / cur["cycle_id"] / "nodes"
            if nodes.is_dir():
                dirs = [p for p in nodes.iterdir() if p.is_dir()]
                out["nodes_total"] = len(dirs)
                out["nodes_done"] = sum((p / "score.json").is_file() for p in dirs)
        else:
            out["stage"] = "done" if state.get("iterations") else "not started"
    return out


def _iteration_done(it: dict[str, Any]) -> bool:
    return it["cycle"]["status"] == "done" and it["policy_dev"]["status"] in ("done", "skipped")


def _best_and_cost(study_dir: Path, state: dict[str, Any] | None) -> tuple[float | None, float]:
    scores: list[float] = []
    for path in (Path(study_dir) / "cycles").glob("*/nodes/*/score.json"):
        value = json.loads(path.read_text(encoding="utf-8")).get("score")
        if value is not None:
            scores.append(float(value))
    cost = 0.0
    if state is not None:
        for it in state.get("iterations", []):
            cost += float(it["cycle"].get("agent_cost_usd") or 0.0)
            cost += float(it["policy_dev"].get("agent_cost_usd") or 0.0)
    return (max(scores) if scores else None), cost


def _elapsed_s(state: dict[str, Any] | None, running: bool) -> float | None:
    if state is None or not state.get("first_started_utc"):
        return None
    first = datetime.fromisoformat(state["first_started_utc"])
    end = datetime.now(UTC) if running else datetime.fromisoformat(state["updated_utc"])
    return round((end - first).total_seconds(), 1)


def _log_tail(study_dir: Path, lines: int = 10) -> list[str]:
    path = Path(study_dir) / LOG_NAME
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def status(study_dir: Path) -> dict[str, Any]:
    """Unit or pid liveness, loop progress, best score, cost, elapsed time and the log tail."""
    study = Path(study_dir)
    state = load_loop_state(study)
    holder = lock_holder(study)
    svc = _read_service(study)
    if svc.get("mode") == "systemd":
        unit: dict[str, Any] = {
            "mode": "systemd",
            "unit": svc["unit"],
            "active": _unit_state(svc["unit"]),
        }
    elif svc.get("mode") == "detached":
        unit = {"mode": "detached", "pid": svc["pid"], "alive": _pid_alive(int(svc["pid"]))}
    else:
        unit = {"mode": None}
    best, cost = _best_and_cost(study, state)
    return {
        "study_dir": str(study),
        "running": holder is not None,
        "lock_holder_pid": holder,
        "launcher": unit,
        "loop_status": state.get("status") if state else None,
        "stop_reason": state.get("stop_reason") if state else None,
        **_progress(study, state),
        "best_score": best,
        "agent_cost_usd": round(cost, 6),
        "elapsed_s": _elapsed_s(state, holder is not None),
        "log_tail": _log_tail(study),
    }


def stop(study_dir: Path, grace: float = 120.0) -> dict[str, Any]:
    """Touch `STOP`, wait up to `grace` seconds for a clean exit, then stop the unit or group."""
    study = Path(study_dir)
    study.mkdir(parents=True, exist_ok=True)
    (study / STOP_NAME).touch()
    if lock_holder(study) is None:
        return {"stopped": True, "method": "not-running"}
    deadline = time.monotonic() + grace
    while lock_holder(study) is not None and time.monotonic() < deadline:
        time.sleep(1.0)
    if lock_holder(study) is None:
        return {"stopped": True, "method": "clean"}
    svc = _read_service(study)
    if svc.get("mode") == "systemd":
        subprocess.run(["systemctl", "--user", "stop", svc["unit"]], check=False)
        method = "unit-stop"
    elif svc.get("mode") == "detached":
        os.killpg(int(svc["pid"]), signal.SIGTERM)
        method = "signal"
    else:
        return {"stopped": False, "method": "unknown-holder", "lock_holder_pid": lock_holder(study)}
    time.sleep(2.0)
    return {"stopped": lock_holder(study) is None, "method": method}
