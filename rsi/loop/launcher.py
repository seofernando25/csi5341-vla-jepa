"""Detached launch of the loop: a restarting `systemd --user` unit, or a `setsid` process group, and its stop."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rsi.loop.states import ESCALATION, STOP, LaunchMode, StopStep, launch_mode
from rsi.loop.store import STOP_NAME, LoopLockedError, lock_holder, utc_now, write_json_atomic
from rsi.paths import REPO_ROOT

LOG_NAME = "loop.log"
SERVICE_NAME = "service.json"
PIDFILE_NAME = "loop.pid"
PASS_ENV = ("PATH", "HOME")


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


def unit_state(unit: str) -> str:
    probe = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ActiveState", "--value", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() or "unknown"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_service(study_dir: Path) -> dict[str, Any]:
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
        record = {"mode": LaunchMode.SYSTEMD.value, "unit": unit_name(study)}
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
        record = {"mode": LaunchMode.DETACHED.value, "pid": proc.pid}
    write_json_atomic(study / SERVICE_NAME, {**record, "started_utc": utc_now()})
    return record


def stop(study_dir: Path, grace: float = 120.0) -> dict[str, Any]:
    """Touch `STOP`, wait up to `grace` seconds for a clean exit, then stop the unit or group."""
    study = Path(study_dir)
    study.mkdir(parents=True, exist_ok=True)
    (study / STOP_NAME).touch()
    step = STOP.check(None, StopStep.REQUESTED)
    if lock_holder(study) is None:
        return {"stopped": True, "method": STOP.check(step, StopStep.NOT_RUNNING).value}
    step = STOP.check(step, StopStep.WAITING)
    deadline = time.monotonic() + grace
    while lock_holder(study) is not None and time.monotonic() < deadline:
        time.sleep(1.0)
    if lock_holder(study) is None:
        return {"stopped": True, "method": STOP.check(step, StopStep.CLEAN).value}
    svc = read_service(study)
    mode = launch_mode(svc)
    if mode is None:
        method = STOP.check(step, StopStep.UNKNOWN_HOLDER)
        return {"stopped": False, "method": method.value, "lock_holder_pid": lock_holder(study)}
    method = STOP.check(step, ESCALATION[mode])
    if mode is LaunchMode.SYSTEMD:
        subprocess.run(["systemctl", "--user", "stop", svc["unit"]], check=False)
    else:
        os.killpg(int(svc["pid"]), signal.SIGTERM)
    time.sleep(2.0)
    return {"stopped": lock_holder(study) is None, "method": method.value}
