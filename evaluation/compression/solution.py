"""Fixed evaluator for an agent-written solution.py that defines compress(policy, calibration).

run_solution loads the B16 policy and frozen calibration, hashes the workspace, imports
solution.py as a fresh module and calls compress under a wall-clock limit. The returned
object is measured: resident bytes are every tensor reachable from it, from the solution
module and from modules imported out of the workspace, each storage counted once, and the
larger of that and the live CUDA storage in the process. Size is compared with the same walk
on the untouched policy. Violations: off-device or meta tensors, memory maps, and files the
solution opens or creates outside the workspace. Those are detected by diffing open fds,
/proc/self/maps and directory listings of the workspace, out_dir and the temp dir around
compress, so a file opened and closed without touching a watched tree is not seen.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import random
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from evaluation.common import environment, read_json, recorded_run
from evaluation.compression import calibration as calib
from evaluation.compression import study
from evaluation.compression.inventory import ResidentWalk, live_cuda_bytes
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline
from evaluation.rollout import rollout

PHASES = ("check", "smoke", "development")
DEFAULT_TIME_LIMIT_S = 900.0
HASH_FILE_LIMIT = 50 << 20
REQUIRED_DEVICE = "cuda"
PROTOCOL_PATH = calib.STUDY / "protocol.json"
ALWAYS_ALLOWED = ("/dev", "/proc", "/sys")
POLICY_API = ("select_action", "predict_action_chunk")
CHECK_KEYS = (
    "diagnostic_observations",
    "maximum_normalized_action_difference",
    "mean_physical_arm_change_mse",
    "maximum_physical_arm_change_mse",
    "mean_gripper_disagreement",
    "pipeline_median_ms",
    "pipeline_p95_ms",
)


class _Verdict(Exception):
    """A solution-side failure with its result status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def workspace_hash(workspace: Path) -> tuple[str, list[str]]:
    """sha256 over sorted relative paths and contents; files over 50 MiB by size and mtime."""
    digest, large = hashlib.sha256(), []
    for path in _workspace_files(workspace):
        name = path.relative_to(workspace).as_posix()
        stat = path.stat()
        digest.update(name.encode() + b"\0")
        if stat.st_size > HASH_FILE_LIMIT:
            large.append(name)
            digest.update(f"big:{stat.st_size}:{stat.st_mtime_ns}".encode() + b"\0")
        else:
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest(), large


def _workspace_files(workspace: Path) -> list[Path]:
    files = [p for p in workspace.rglob("*") if p.is_file()]
    kept = [p for p in files if "__pycache__" not in p.relative_to(workspace).parts]
    return sorted(kept, key=lambda p: p.relative_to(workspace).as_posix())


def measure(policy: object, roots: tuple[object, ...]) -> dict:
    """Resident bytes of policy and extra roots; violations found by the walk."""
    walk = ResidentWalk(REQUIRED_DEVICE)
    walk.add(policy, *roots)
    resident = walk.nbytes
    if REQUIRED_DEVICE == "cuda":
        resident = max(resident, live_cuda_bytes())
    return {"resident_bytes": resident, "violations": walk.violations}


def _in_workspace(module: ModuleType, workspace: Path) -> bool:
    file = getattr(module, "__file__", None)
    return bool(file) and os.path.abspath(file).startswith(str(workspace) + os.sep)


def _workspace_modules(workspace: Path) -> list[ModuleType]:
    return [m for m in list(sys.modules.values()) if _in_workspace(m, workspace)]


def _forget_workspace_modules(workspace: Path) -> None:
    for name, module in list(sys.modules.items()):
        if _in_workspace(module, workspace):
            del sys.modules[name]


def _open_targets() -> set[tuple[str, str]]:
    result = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/"):
            result.add((fd, target))
    return result


def _mapped_files() -> set[str]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            paths.add(parts[5])
    return paths


def _tree_state(roots: list[Path]) -> dict[str, tuple[int, int]]:
    state = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    stat = os.stat(path, follow_symlinks=False)
                except OSError:
                    continue
                state[path] = (stat.st_size, stat.st_mtime_ns)
    return state


class _Probe:
    """Files opened outside the allowed roots, or created and changed in watched trees, during a call."""

    def __init__(self, workspace: Path, out_dir: Path) -> None:
        self.allowed = (str(workspace), sys.prefix, sys.base_prefix, *ALWAYS_ALLOWED)
        self.watched = [workspace, out_dir, Path(tempfile.gettempdir())]

    def start(self) -> None:
        self.fds, self.maps, self.trees = (
            _open_targets(),
            _mapped_files(),
            _tree_state(self.watched),
        )

    def finish(self) -> list[str]:
        found = [
            f"opened outside workspace: {t}"
            for _, t in _open_targets() - self.fds
            if self._foreign(t)
        ]
        found += [
            f"opened outside workspace (mapped): {p}"
            for p in _mapped_files() - self.maps
            if self._foreign(p)
        ]
        now = _tree_state(self.watched)
        changed = sorted(p for p in set(now) | set(self.trees) if now.get(p) != self.trees.get(p))
        found += [f"created or changed: {p}" for p in changed]
        return found

    def _foreign(self, path: str) -> bool:
        return not any(
            path == root or path.startswith(root.rstrip("/") + "/") for root in self.allowed
        )


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    def expire(signum: int, frame: object) -> None:
        raise TimeoutError(f"compress exceeded {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _import_solution(workspace: Path, digest: str) -> ModuleType:
    path = workspace / "solution.py"
    if not path.is_file():
        raise _Verdict("solution_error", "workspace has no solution.py")
    spec = importlib.util.spec_from_file_location(f"vla_solution_{digest[:16]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "compress", None)):
        raise _Verdict("solution_error", "solution.py defines no compress(policy, calibration)")
    return module


def _per_task(episodes_csv: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with episodes_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            entry = result.setdefault(row["task_id"], {"successes": 0, "episodes": 0})
            entry["successes"] += int(row["success"])
            entry["episodes"] += 1
    return result


class _Run:
    """One evaluation of a workspace; fills the result record stage by stage."""

    def __init__(
        self, workspace: Path, phase: str, out_dir: Path, record: dict, limit: float
    ) -> None:
        self.workspace, self.phase, self.out_dir = workspace, phase, out_dir
        self.record, self.limit = record, limit
        self.protocol = read_json(PROTOCOL_PATH)
        self.log: list[str] = []
        self.baseline_bytes = 0
        self.baseline_violations: Counter[str] = Counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            self.record["stage_seconds"][name] = seconds
            self.log.append(f"{name}: {seconds:.3f} s")

    def evaluate(self) -> None:
        record = self.record
        record["environment"] = environment()
        sys.path.insert(0, str(self.workspace))
        try:
            with self.stage("hash"):
                digest, large = workspace_hash(self.workspace)
            record.update(workspace_sha256=digest, workspace_large_files=large)
            calib.setup()
            with self.stage("load"):
                calibration = calib.load()
                policy, record["model"] = load_policy("B16")
            with self.stage("baseline"):
                baseline = measure(policy, ())
                self.baseline_bytes = baseline["resident_bytes"]
                self.baseline_violations = Counter(baseline["violations"])
            record["baseline_resident_bytes"] = self.baseline_bytes
            self.candidate(policy, calibration)
        except _Verdict as verdict:
            record.update(status=verdict.status, error=str(verdict))
        except Exception:  # noqa: BLE001 - solution code may raise anything; recorded, not propagated
            record.update(status="solution_error", error=traceback.format_exc()[-2000:])
        finally:
            _forget_workspace_modules(self.workspace)
            sys.path.remove(str(self.workspace))

    def candidate(self, policy: object, calibration: dict) -> None:
        record = self.record
        with self.stage("import"):
            module = _import_solution(self.workspace, record["workspace_sha256"])
        seed = self.protocol["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        record["seed"] = seed
        probe = _Probe(self.workspace, self.out_dir)
        if REQUIRED_DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with self.stage("compress"):
            probe.start()
            with _time_limit(self.limit):
                returned = module.compress(policy, calibration)
            opened = probe.finish()
        if returned is None or not all(hasattr(returned, name) for name in POLICY_API):
            raise _Verdict("solution_error", f"compress must return an object with {POLICY_API}")
        with self.stage("measure"):
            roots = (vars(module), *(vars(m) for m in _workspace_modules(self.workspace)))
            sized = measure(returned, roots)
        peak = torch.cuda.max_memory_allocated() if REQUIRED_DEVICE == "cuda" else 0
        resident = sized["resident_bytes"]
        reduction = max(0.0, 1 - resident / self.baseline_bytes) if self.baseline_bytes else 0.0
        record["sizes"] = {
            "resident_bytes": resident,
            "baseline_resident_bytes": self.baseline_bytes,
            "size_reduction": reduction,
            "peak_cuda_bytes": peak,
        }
        new = Counter(sized["violations"]) - self.baseline_violations
        violations = opened + sorted(new.elements())
        if violations:
            raise _Verdict("size_violation", "; ".join(violations[:20]))
        if self.phase == "check":
            self.check(returned, calibration)
        else:
            self.rollouts(returned)
        record["status"] = "ok"

    def check(self, policy: object, calibration: dict) -> None:
        with self.stage("check"):
            summary, _ = study.screen(policy, calibration, compiled=False)
        self.record["check"] = {key: summary[key] for key in CHECK_KEYS}
        self.record["action_error"] = summary["maximum_normalized_action_difference"]

    def rollouts(self, policy: object) -> None:
        trials = self.protocol["phases"][self.phase]["trials"]
        tasks = self.protocol["task_ids"]
        if trials % len(tasks):
            raise ValueError("protocol trials must divide evenly across tasks")
        name = f"solution-{self.record['workspace_sha256'][:16]}"
        with self.stage("rollout"):
            # Smoke and development share the development seeds and states, as study.run does.
            summary = rollout(
                name,
                tasks,
                "development",
                trials // len(tasks),
                policy,
                Pipeline(policy),
                self.out_dir,
            )
        self.record.update(successes=summary["successes"], episodes=summary["episodes"])
        self.record["rollout"] = {**summary, "per_task": _per_task(self.out_dir / "episodes.csv")}


def run_solution(
    workspace: Path | str,
    phase: str,
    out_dir: Path | str,
    *,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> dict:
    """Evaluate workspace/solution.py in one phase; writes out_dir/result.json and log.txt.

    Solution failures are recorded in the result, never raised. Phase final is not accepted.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    workspace, out_dir = Path(workspace).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"phase": phase, "time_limit_s": time_limit_s, "stage_seconds": {}}
    run = _Run(workspace, phase, out_dir, record, time_limit_s)
    with recorded_run(out_dir / "result.json", record):
        try:
            run.evaluate()
        finally:
            (out_dir / "log.txt").write_text("".join(f"{line}\n" for line in run.log))
    return record
