"""Timing instruments for the latency score (program-v4).

The clock, the CUDA synchronizer and the peak-memory readers are captured when this module is
imported, before solution code runs; replacing any of them fails the node. Each side is timed in
`blocks` blocks, after warm-up, on frames that were never timed before; `latency_plan` lays the
frames out and turns the blocks into a speedup. The co-tenant watch samples the other processes'
GPU use between blocks: another process above a set share of the device fails the node, because
the measurement is not trustworthy then.

Every count is per task and every task in the suite gets the same count, so no task can carry
the median: timed calls, equivalence, arm and perturbation frames are drawn equally from each.
"""

from __future__ import annotations

import os
import resource
import statistics
import subprocess
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.pipeline import inference_autocast

PMON = ("nvidia-smi", "pmon", "-c", "1")
APPS = ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")


def _noop() -> None:
    return None


def _zero() -> int:
    return 0


@dataclass(frozen=True)
class Clock:
    """Time, CPU and CUDA readers captured before any solution code is imported."""

    perf: Callable[[], float]
    perf_ns: Callable[[], int]
    monotonic: Callable[[], float]
    getrusage: Callable[..., Any]
    sync: Callable[[], None]
    event: Callable[..., Any] | None
    reset_peak: Callable[[], None]
    max_allocated: Callable[[], int]
    empty_cache: Callable[[], None]
    cuda: bool


def _capture(cuda: bool | None = None) -> Clock:
    cuda = torch.cuda.is_available() if cuda is None else cuda
    return Clock(
        perf=time.perf_counter,
        perf_ns=time.perf_counter_ns,
        monotonic=time.monotonic,
        getrusage=resource.getrusage,
        sync=torch.cuda.synchronize if cuda else _noop,
        event=torch.cuda.Event if cuda else None,
        reset_peak=torch.cuda.reset_peak_memory_stats if cuda else _noop,
        max_allocated=torch.cuda.max_memory_allocated if cuda else _zero,
        empty_cache=torch.cuda.empty_cache if cuda else _noop,
        cuda=cuda,
    )


CLOCK = _capture()
CPU_CLOCK = _capture(cuda=False)


def rebinds(clock: Clock = CLOCK) -> list[str]:
    """Names of time or CUDA readers replaced since `clock` was captured."""
    pairs = [
        ("time.perf_counter", time.perf_counter, clock.perf),
        ("time.perf_counter_ns", time.perf_counter_ns, clock.perf_ns),
        ("time.monotonic", time.monotonic, clock.monotonic),
        ("resource.getrusage", resource.getrusage, clock.getrusage),
    ]
    if clock.cuda:
        pairs += [
            ("torch.cuda.synchronize", torch.cuda.synchronize, clock.sync),
            ("torch.cuda.Event", torch.cuda.Event, clock.event),
            (
                "torch.cuda.reset_peak_memory_stats",
                torch.cuda.reset_peak_memory_stats,
                clock.reset_peak,
            ),
            (
                "torch.cuda.max_memory_allocated",
                torch.cuda.max_memory_allocated,
                clock.max_allocated,
            ),
            ("torch.cuda.empty_cache", torch.cuda.empty_cache, clock.empty_cache),
        ]
    return [
        f"{name} was replaced during compress"
        for name, current, captured in pairs
        if current is not captured
    ]


@dataclass(frozen=True)
class Limits:
    """Pre-registered timing counts and guard thresholds from the study protocol."""

    warmup_per_task: int
    timed_per_task: int
    blocks: int
    bootstrap_resamples: int
    bootstrap_seed: int
    max_frames_per_task: int
    reference_trials: int
    reference_tasks: tuple[int, ...]
    frame_stride: int
    reset_budget_ms: float
    reset_budget_fraction: float
    overhead_factor: float
    overhead_ms: float
    cpu_factor: float
    cpu_ms: float
    child_cpu_seconds: float
    novel_frame_factor: float
    novel_frame_ms: float
    gpu_over_wall_factor: float
    gpu_over_wall_ms: float
    action_atol: float
    min_action_spread: float
    equivalence_per_task: int
    max_new_threads: int
    peak_tolerance: float
    arm_action_mse_max: float
    arm_per_task: int
    perturbation_per_task: int
    perturbation_levels: int
    perturbation_min_correlation: float
    other_gpu_utilisation_max_pct: float
    sandwich_drift_max: float

    @classmethod
    def from_protocol(cls, protocol: dict, phase: str = "smoke") -> Limits:
        timing = dict(protocol["score"]["timing"])
        check = timing.pop("check", {})
        if phase == "check":
            timing.update(check)
        guards = protocol["score"]["guards"]
        return cls(
            warmup_per_task=int(timing["warmup_calls_per_task"]),
            timed_per_task=int(timing["timed_calls_per_task"]),
            blocks=int(timing["blocks"]),
            bootstrap_resamples=int(timing["bootstrap_resamples"]),
            bootstrap_seed=int(timing["bootstrap_seed"]),
            max_frames_per_task=int(timing["max_frames_per_task"]),
            reference_trials=int(timing["reference_trials"]),
            reference_tasks=tuple(int(task) for task in timing["reference_tasks"]),
            frame_stride=int(timing["frame_stride"]),
            reset_budget_ms=float(guards["reset_budget_ms"]),
            reset_budget_fraction=float(guards["reset_budget_fraction"]),
            overhead_factor=float(guards["overhead_factor"]),
            overhead_ms=float(guards["overhead_ms"]),
            cpu_factor=float(guards["cpu_factor"]),
            cpu_ms=float(guards["cpu_ms"]),
            child_cpu_seconds=float(guards["child_cpu_seconds"]),
            novel_frame_factor=float(guards["novel_frame_factor"]),
            novel_frame_ms=float(guards["novel_frame_ms"]),
            gpu_over_wall_factor=float(guards["gpu_over_wall_factor"]),
            gpu_over_wall_ms=float(guards["gpu_over_wall_ms"]),
            action_atol=float(guards["action_atol"]),
            min_action_spread=float(guards["min_action_spread"]),
            equivalence_per_task=int(guards["equivalence_frames_per_task"]),
            max_new_threads=int(guards["max_new_threads"]),
            peak_tolerance=float(guards["peak_tolerance"]),
            arm_action_mse_max=float(guards["arm_action_mse_max"]),
            arm_per_task=int(guards["arm_check_frames_per_task"]),
            perturbation_per_task=int(guards["perturbation_frames_per_task"]),
            perturbation_levels=int(guards["perturbation_levels"]),
            perturbation_min_correlation=float(guards["perturbation_min_correlation"]),
            other_gpu_utilisation_max_pct=float(guards["other_gpu_utilisation_max_pct"]),
            sandwich_drift_max=float(guards["sandwich_drift_max"]),
        )

    @property
    def tasks(self) -> int:
        return len(self.reference_tasks)


class FrameCollector:
    """Rollout observer: keeps sampled batches and the policy time of every step."""

    def __init__(self, limit: int, stride: int) -> None:
        self.limit, self.stride = int(limit), max(1, int(stride))
        self.frames: list[Any] = []
        self.total_select_seconds = 0.0
        self.seen = 0

    def __call__(self, batch: Any, seconds: float) -> None:
        self.seen += 1
        self.total_select_seconds += seconds
        if self.seen % self.stride or len(self.frames) >= self.limit:
            return
        self.frames.append(batch)


def autocast(clock: Clock):
    """Inference autocast on a CUDA clock; a no-op otherwise."""
    return inference_autocast() if clock.cuda else nullcontext()


def _cpu(clock: Clock) -> tuple[float, float]:
    """Process and reaped-children CPU seconds (user + system)."""
    own = clock.getrusage(resource.RUSAGE_SELF)
    children = clock.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime, children.ru_utime + children.ru_stime


def child_pids() -> set[int]:
    """Live processes whose parent is this process."""
    found: set[int] = set()
    parent = str(os.getpid())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) > 1 and fields[1] == parent:
            found.add(int(entry.name))
    return found


def summarize(values: list[float]) -> dict[str, float]:
    """Median, p95, mean, min and max of one pool of call times."""
    ordered = sorted(values)
    return {
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "mean_ms": float(statistics.fmean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def warm_up(policy: object, frames: list[Any], indices: list[int], seed: int, clock: Clock) -> None:
    """Untimed calls that prime the policy on frames disjoint from the timed blocks."""
    clock.sync()
    for index in indices:
        policy.reset()
        torch.manual_seed(seed + index)
        with torch.inference_mode(), autocast(clock):
            policy.select_action(frames[index])
    clock.sync()


def time_block(
    policy: object, frames: list[Any], indices: list[int], seed: int, clock: Clock = CLOCK
) -> dict:
    """Time one block of full select_action calls on distinct frames, with a device sync."""
    wall: list[float] = []
    gpu: list[float] = []
    cpu_before = _cpu(clock)
    for index in indices:
        start_event = clock.event(enable_timing=True) if clock.event else None
        end_event = clock.event(enable_timing=True) if clock.event else None
        policy.reset()
        torch.manual_seed(seed + index)
        clock.sync()
        if start_event is not None:
            start_event.record()
        started = clock.perf_ns()
        with torch.inference_mode(), autocast(clock):
            action = policy.select_action(frames[index])
        # Reading the value forces the computation, so a lazy answer is timed too.
        float(action.float().sum().item())
        clock.sync()
        wall.append((clock.perf_ns() - started) / 1e6)
        if end_event is not None and start_event is not None:
            end_event.record()
            clock.sync()
            gpu.append(float(start_event.elapsed_time(end_event)))
    cpu_after = _cpu(clock)
    return {
        "wall": wall,
        "gpu": gpu,
        "cpu_seconds": max(0.0, cpu_after[0] - cpu_before[0]),
        "child_seconds": max(0.0, cpu_after[1] - cpu_before[1]),
    }


def timing_summary(results: list[dict], warmup: int) -> dict:
    """Pooled figures over timed blocks, plus the CPU and GPU totals the guards read."""
    wall = [value for result in results for value in result["wall"]]
    gpu = [value for result in results for value in result["gpu"]]
    timing = summarize(wall)
    timing.update(
        calls=len(wall),
        warmup=warmup,
        blocks=len(results),
        block_medians_ms=[float(np.median(result["wall"])) for result in results],
        cpu_ms_per_call=1000.0 * sum(result["cpu_seconds"] for result in results) / len(wall),
        child_cpu_seconds=sum(result["child_seconds"] for result in results),
        wall_total_ms=float(sum(wall)),
        gpu_total_ms=float(sum(gpu)) if gpu else None,
    )
    if gpu:
        gpu_summary = summarize(gpu)
        timing["gpu_median_ms"] = gpu_summary["median_ms"]
        timing["gpu_p95_ms"] = gpu_summary["p95_ms"]
        timing["gpu_mean_ms"] = gpu_summary["mean_ms"]
    return timing


def _pmon_other(text: str, ours: set[int]) -> float:
    """Sum of the SM utilisation percentages of compute processes that are not ours."""
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if pid in ours:
            continue
        try:
            total += float(fields[3])
            found = True
        except ValueError:
            continue
    return total if found else 0.0


def _apps_other(text: str, ours: set[int]) -> float:
    """MiB held by compute processes that are not ours."""
    total = 0.0
    for line in text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid, mib = int(fields[0]), float(fields[1])
        except ValueError:
            continue
        if pid not in ours:
            total += mib
    return total


class GpuWatch:
    """Other processes' GPU use, sampled between timing blocks with nvidia-smi.

    `pmon` reports per-process SM utilisation, so the evaluator's own kernels are excluded by
    pid. When it is unavailable the device-wide figure is recorded instead and the guard is not
    applied, because the evaluator's own work is part of that figure.
    """

    def __init__(self, level_pct: float) -> None:
        self.level = float(level_pct)
        self.ours = {os.getpid()} | child_pids()
        self.samples: list[dict] = []
        self.error: str | None = None

    def sample(self) -> None:
        entry: dict = {"other_utilisation_pct": None, "other_memory_mib": None}
        try:
            completed = subprocess.run(
                PMON, capture_output=True, text=True, timeout=20, check=False
            )
            entry["other_utilisation_pct"] = _pmon_other(completed.stdout, self.ours)
        except (OSError, subprocess.SubprocessError) as error:
            self.error = f"pmon: {error}"
        try:
            completed = subprocess.run(
                APPS, capture_output=True, text=True, timeout=20, check=False
            )
            entry["other_memory_mib"] = _apps_other(completed.stdout, self.ours)
        except (OSError, subprocess.SubprocessError) as error:
            self.error = f"{self.error}; apps: {error}" if self.error else f"apps: {error}"
        self.samples.append(entry)

    def summary(self) -> dict:
        utilisation = [
            sample["other_utilisation_pct"]
            for sample in self.samples
            if sample["other_utilisation_pct"] is not None
        ]
        memory = [
            sample["other_memory_mib"]
            for sample in self.samples
            if sample["other_memory_mib"] is not None
        ]
        return {
            "level_pct": self.level,
            "samples": len(self.samples),
            "other_utilisation_max_pct": max(utilisation) if utilisation else None,
            "other_utilisation_mean_pct": float(np.mean(utilisation)) if utilisation else None,
            "other_memory_max_mib": max(memory) if memory else None,
            "per_sample": self.samples,
            "error": self.error,
        }

    def violation(self) -> str | None:
        peak = self.summary()["other_utilisation_max_pct"]
        if peak is not None and peak > self.level:
            return (
                f"another process used {peak:.0f} % of the GPU during timing, "
                f"above {self.level:.0f} %"
            )
        return None
