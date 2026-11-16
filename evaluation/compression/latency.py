"""Latency score and anti-gaming guards for the fixed evaluator (program-v4).

The score is the reference median observation-to-action latency divided by the candidate's,
both measured in this process on frames the evaluator collected from episodes it ran. The
reference is the untouched B16 policy on the stock path; the candidate is whatever `compress`
returned. The score is zero unless the smoke gate passes and the candidate's peak CUDA bytes do
not exceed the reference's.

Timing is block-based. Each side is timed in `blocks` blocks of `timed_calls / blocks` calls,
after warm-up, on frames that were never timed before. When a second copy of the reference fits
beside the candidate, the blocks alternate reference, candidate, reference, candidate (ABAB), so
a co-tenant burst lands in both sides. When it does not fit, the reference is timed in blocks
before `compress` and again after the candidate episodes and the two pools are combined
(sandwich). The speedup is reported with a 95 % cluster-bootstrap interval over blocks and the
score is its lower bound, so timing noise cannot create a win.

Each guard is a failing condition, not a warning:

- the clock, the CUDA synchronizer and the peak-memory readers are captured when this module is
  imported, before solution code runs; replacing any of them fails the node;
- warm-up and timed calls use disjoint frames, so every timed call sees an observation that was
  never timed before and a lookup table keyed on the observation gains nothing;
- `select_action` must agree with `predict_action_chunk` on the same frame and seed, must be
  reproducible, must not answer the previous frame, and must change with the observation;
- a novel frame (a blend of two collected frames) must not cost more than a set factor above the
  median, which catches a cache with a slow fallback;
- the candidate's physical arm actions on fresh frames must stay within a set distance of the
  untouched policy's, and its arm changes under a small pixel perturbation must correlate with the
  untouched policy's, which catches a lookup table that returns a stored answer;
- `reset` must stay cheap, so work cannot move out of the timed call into it;
- the episode batch's wall time per step, minus the policy's own time in the episodes, must not
  exceed the reference's by more than a set factor, so background or asynchronous work is
  charged to the candidate;
- CPU seconds per call and child-process CPU seconds are capped against the reference's, so work
  cannot move to the CPU or another process, and new threads during timing fail;
- another process using more than a set share of the GPU during the timing blocks fails the node,
  because the measurement is not trustworthy;
- peak CUDA bytes must not exceed the reference's.
"""

from __future__ import annotations

import os
import resource
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.pipeline import Pipeline, inference_autocast, observation_bank
from evaluation.rollout import rollout

SCORE_KIND = "latency_speedup"
MIN_TIMED_CALLS = 8
RESET_PROBES = 8
NOVEL_PAIRS = 4
PMON = ("nvidia-smi", "pmon", "-c", "1")
APPS = ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
INTERVAL_PERCENTILES = (2.5, 97.5)


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

    warmup_calls: int
    timed_calls: int
    blocks: int
    bootstrap_resamples: int
    bootstrap_seed: int
    max_frames: int
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
    equivalence_frames: int
    max_new_threads: int
    peak_tolerance: float
    arm_action_mse_max: float
    arm_check_frames: int
    perturbation_frames: int
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
            warmup_calls=int(timing["warmup_calls"]),
            timed_calls=int(timing["timed_calls"]),
            blocks=int(timing["blocks"]),
            bootstrap_resamples=int(timing["bootstrap_resamples"]),
            bootstrap_seed=int(timing["bootstrap_seed"]),
            max_frames=int(timing["max_frames"]),
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
            equivalence_frames=int(guards["equivalence_frames"]),
            max_new_threads=int(guards["max_new_threads"]),
            peak_tolerance=float(guards["peak_tolerance"]),
            arm_action_mse_max=float(guards["arm_action_mse_max"]),
            arm_check_frames=int(guards["arm_check_frames"]),
            perturbation_frames=int(guards["perturbation_frames"]),
            perturbation_levels=int(guards["perturbation_levels"]),
            perturbation_min_correlation=float(guards["perturbation_min_correlation"]),
            other_gpu_utilisation_max_pct=float(guards["other_gpu_utilisation_max_pct"]),
            sandwich_drift_max=float(guards["sandwich_drift_max"]),
        )


class FrameCollector:
    """Rollout observer: keeps sampled batches and the policy time of every step."""

    def __init__(self, limit: int, stride: int) -> None:
        self.limit, self.stride = int(limit), max(1, int(stride))
        self.frames: list[Any] = []
        self.select_seconds: list[float] = []
        self.total_select_seconds = 0.0
        self.seen = 0

    def __call__(self, batch: Any, seconds: float) -> None:
        self.seen += 1
        self.total_select_seconds += seconds
        if self.seen % self.stride or len(self.frames) >= self.limit:
            return
        self.frames.append(batch)
        self.select_seconds.append(seconds)


def _autocast(clock: Clock):
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


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "mean_ms": float(statistics.fmean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def overhead_violation(reference_ms: float, candidate_ms: float, limits: Limits) -> str | None:
    """Episode wall time per step above the policy's own time, against the reference's."""
    limit = max(reference_ms * limits.overhead_factor, reference_ms + limits.overhead_ms)
    if candidate_ms <= limit:
        return None
    return (
        f"episode wall time is {candidate_ms:.2f} ms/step above the policy's own time, "
        f"above {limit:.2f} ms/step"
    )


def cpu_violation(reference_ms: float, candidate_ms: float, limits: Limits) -> str | None:
    """CPU seconds per call against the reference's, so work cannot move to the CPU."""
    limit = max(reference_ms * limits.cpu_factor, reference_ms + limits.cpu_ms)
    if candidate_ms <= limit:
        return None
    return f"CPU time {candidate_ms:.2f} ms/call is above {limit:.2f} ms/call"


def child_cpu_violation(seconds: float, limits: Limits) -> str | None:
    """CPU seconds of reaped child processes during the timed calls."""
    if seconds <= limits.child_cpu_seconds:
        return None
    return f"child processes used {seconds:.2f} s of CPU during timing"


def peak_violation(reference_bytes: int, candidate_bytes: int, limits: Limits) -> str | None:
    """Peak CUDA bytes against the reference's; the score is zero above it."""
    limit = reference_bytes * (1.0 + limits.peak_tolerance)
    if candidate_bytes <= limit:
        return None
    return f"peak CUDA {candidate_bytes} B is above the reference {reference_bytes} B"


def speedup(
    interval: dict,
    *,
    gate: int,
    successes: int,
    episodes: int,
    reference_peak: int,
    candidate_peak: int,
    limits: Limits,
) -> tuple[float, str | None]:
    """The score (the lower bound of the speedup interval) and the reason it is zero, if it is."""
    if gate and successes < gate:
        return 0.0, f"smoke gate: {successes}/{episodes} < {gate}"
    above = peak_violation(reference_peak, candidate_peak, limits)
    if above is not None:
        return 0.0, above
    return float(interval["low"]), None


def calls_per_block(limits: Limits) -> int:
    """Timed calls in one block; at least one."""
    return max(1, limits.timed_calls // max(1, limits.blocks))


def frame_plan(count: int, limits: Limits, first: int = 0) -> tuple[list[int], list[list[int]]]:
    """Warm-up indices and the timed blocks, disjoint and drawn from `frames[first:]`.

    Warm-up indices are spread over the whole pool so every task's shapes are seen; the timed
    indices are the rest, split by stride so every block covers the same spread of frames.
    """
    pool = list(range(first, count))
    needed = limits.blocks * calls_per_block(limits)
    if len(pool) < needed + 1:
        raise RuntimeError(f"need {needed + 1} frames after index {first}, collected {count}")
    warmup = min(limits.warmup_calls, len(pool) - needed)
    step = max(1, len(pool) // max(1, warmup))
    warm = pool[::step][:warmup]
    warm_set = set(warm)
    timed = [index for index in pool if index not in warm_set][:needed]
    blocks = [timed[offset :: limits.blocks] for offset in range(limits.blocks)]
    return warm, blocks


def warm_up(policy: object, frames: list[Any], indices: list[int], seed: int, clock: Clock) -> None:
    """Untimed calls that prime the policy on frames disjoint from the timed blocks."""
    clock.sync()
    for index in indices:
        policy.reset()
        torch.manual_seed(seed + index)
        with torch.inference_mode(), _autocast(clock):
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
        with torch.inference_mode(), _autocast(clock):
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


def _timing_summary(results: list[dict], warmup: int) -> dict:
    """Pooled figures over timed blocks, plus the CPU and GPU totals the guards read."""
    wall = [value for result in results for value in result["wall"]]
    gpu = [value for result in results for value in result["gpu"]]
    timing = _summary(wall)
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
        gpu_summary = _summary(gpu)
        timing["gpu_median_ms"] = gpu_summary["median_ms"]
        timing["gpu_p95_ms"] = gpu_summary["p95_ms"]
        timing["gpu_mean_ms"] = gpu_summary["mean_ms"]
    return timing


def time_calls(
    policy: object,
    frames: list[Any],
    limits: Limits,
    seed: int,
    clock: Clock = CLOCK,
    first: int = 0,
) -> dict:
    """Time every block of the plan on one policy and pool the figures."""
    warm, blocks = frame_plan(len(frames), limits, first)
    warm_up(policy, frames, warm, seed, clock)
    results = [time_block(policy, frames, block, seed, clock) for block in blocks]
    return _timing_summary(results, len(warm))


def bootstrap_speedup(
    reference_blocks: list[list[float]],
    candidate_blocks: list[list[float]],
    limits: Limits,
    *,
    paired: bool = True,
) -> dict:
    """Point estimate and 95 % cluster-bootstrap interval of the speedup over blocks.

    `paired` resamples one block index and takes both sides of that block, which is the
    interleaved design. Without it the two sides are resampled independently, which is the
    sandwich design where the blocks are not adjacent in time.
    """
    reference = [np.asarray(block, dtype=float) for block in reference_blocks]
    candidate = [np.asarray(block, dtype=float) for block in candidate_blocks]
    point = float(np.median(np.concatenate(reference)) / np.median(np.concatenate(candidate)))
    rng = np.random.default_rng(limits.bootstrap_seed)
    draws = np.empty(limits.bootstrap_resamples, dtype=float)
    for draw in range(limits.bootstrap_resamples):
        if paired:
            pick = rng.integers(0, len(reference), len(reference))
            left = np.concatenate([reference[index] for index in pick])
            right = np.concatenate([candidate[index] for index in pick])
        else:
            left = np.concatenate(
                [reference[index] for index in rng.integers(0, len(reference), len(reference))]
            )
            right = np.concatenate(
                [candidate[index] for index in rng.integers(0, len(candidate), len(candidate))]
            )
        draws[draw] = float(np.median(left) / np.median(right))
    low, high = np.percentile(draws, INTERVAL_PERCENTILES)
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "blocks": len(reference),
        "candidate_blocks": len(candidate),
        "resamples": limits.bootstrap_resamples,
        "paired": paired,
        "method": "95 % percentile cluster bootstrap over timing blocks",
    }


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


def _call(policy: object, batch: Any, seed: int, clock: Clock) -> torch.Tensor:
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode(), _autocast(clock):
        action = policy.select_action(batch)
    return action.float().detach().cpu().flatten()


def _chunk(policy: object, batch: Any, seed: int, clock: Clock) -> torch.Tensor:
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode(), _autocast(clock):
        chunk = policy.predict_action_chunk(batch)
    return chunk.float().detach().cpu().flatten()


def _close(left: torch.Tensor, right: torch.Tensor, atol: float) -> bool:
    return left.shape == right.shape and bool(torch.allclose(left, right, atol=atol, rtol=0.0))


def equivalence(policy: object, frames: list[Any], limits: Limits, seed: int, clock: Clock = CLOCK):
    """Reproducibility, order, path and input-sensitivity checks on held-out frames."""
    count = min(limits.equivalence_frames, len(frames))
    if count < 2:
        return {
            "violations": ["fewer than two frames for the equivalence check"],
            "frames": count,
            "reproducible": False,
            "order_invariant": False,
            "path_equivalent": False,
            "action_spread": 0.0,
            "seed_sensitivity": 0.0,
        }
    atol = limits.action_atol
    first = _call(policy, frames[0], seed, clock)
    _call(policy, frames[1], seed, clock)
    after_other = _call(policy, frames[0], seed, clock)
    repeat = _call(policy, frames[0], seed, clock)
    chunk = _chunk(policy, frames[0], seed, clock)
    actions = [_call(policy, frames[index], seed, clock) for index in range(count)]
    spread = 0.0
    for left in range(count):
        for right in range(left + 1, count):
            spread = max(spread, float((actions[left] - actions[right]).abs().max()))
    other_seed = _call(policy, frames[0], seed + 1, clock)
    violations = []
    if not _close(repeat, first, atol):
        violations.append("the same frame and seed gave two different actions")
    if not _close(after_other, first, atol):
        violations.append("a call on another frame changed the answer for the first frame")
    if not _close(chunk[: first.numel()], first, atol):
        violations.append("select_action disagrees with predict_action_chunk on the same frame")
    if spread < limits.min_action_spread:
        violations.append("actions do not depend on the observation")
    return {
        "violations": violations,
        "frames": count,
        "reproducible": _close(repeat, first, atol),
        "order_invariant": _close(after_other, first, atol),
        "path_equivalent": _close(chunk[: first.numel()], first, atol),
        "action_spread": spread,
        "seed_sensitivity": float((other_seed - first).abs().max()),
    }


def reset_cost(policy: object, frames: list[Any], clock: Clock = CLOCK) -> float:
    """Median milliseconds of `reset` over a few probes; work moved there is charged."""
    times = []
    for index in range(min(RESET_PROBES, len(frames))):
        clock.sync()
        started = clock.perf_ns()
        policy.reset()
        clock.sync()
        times.append((clock.perf_ns() - started) / 1e6)
    return float(statistics.median(times))


def _blend(left: Any, right: Any, factor: float = 0.5) -> Any:
    """Element-wise blend of two batches; strings and integers keep the left value."""
    if isinstance(left, dict):
        return {
            key: _blend(left[key], right[key], factor) if key in right else left[key]
            for key in left
        }
    if isinstance(left, list):
        return [_blend(a, b, factor) for a, b in zip(left, right)]
    if isinstance(left, tuple):
        return tuple(_blend(a, b, factor) for a, b in zip(left, right))
    if (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.shape == right.shape
        and left.is_floating_point()
    ):
        return left * (1.0 - factor) + right * factor
    return left


def novel_frame_cost(
    policy: object,
    frames: list[Any],
    median_ms: float,
    limits: Limits,
    seed: int,
    clock: Clock = CLOCK,
) -> dict:
    """Time observations the candidate cannot have seen; a cache with a slow fallback shows here."""
    pairs = min(NOVEL_PAIRS, len(frames) // 2)
    times = []
    for index in range(pairs):
        batch = _blend(frames[2 * index], frames[2 * index + 1])
        policy.reset()
        torch.manual_seed(seed + 1000 + index)
        clock.sync()
        started = clock.perf_ns()
        with torch.inference_mode(), _autocast(clock):
            action = policy.select_action(batch)
        float(action.float().sum().item())
        clock.sync()
        times.append((clock.perf_ns() - started) / 1e6)
    measured = float(statistics.median(times)) if times else 0.0
    limit = max(median_ms * limits.novel_frame_factor, median_ms + limits.novel_frame_ms)
    return {
        "frames": pairs,
        "median_ms": measured,
        "limit_ms": limit,
        "violation": measured > limit,
    }


def arm_chunk(policy: object, batch: Any, seed: int, clock: Clock = CLOCK) -> torch.Tensor:
    """One predicted action chunk, on the CPU, under the given seed."""
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode(), _autocast(clock):
        chunk = policy.predict_action_chunk(batch)
    return chunk.float().detach().cpu()


def physical_arm(chunk: torch.Tensor, pipeline: Any) -> torch.Tensor:
    """Post-processed arm dimensions (the first six) of a chunk, on the CPU."""
    physical = pipeline.post(chunk.float().clone()).float()
    return physical[..., :6].detach().cpu()


def arm_actions(
    policy: object,
    frames: list[Any],
    count: int,
    seed: int,
    pipeline: Any,
    clock: Clock = CLOCK,
    offset: int = 0,
) -> list[torch.Tensor]:
    """Physical arm actions of the policy on `count` frames starting at `offset`."""
    return [
        physical_arm(
            arm_chunk(policy, frames[offset + index], seed + offset + index, clock), pipeline
        )
        for index in range(count)
    ]


def arm_error(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    """Mean over frames of the squared physical arm-action difference against the reference."""
    if not reference or len(reference) != len(candidate):
        return float("inf")
    return float(
        np.mean(
            [
                float((left - right).square().mean())
                for left, right in zip(reference, candidate, strict=True)
            ]
        )
    )


def perturb_raw(raw: Any, levels: int, generator: torch.Generator) -> Any:
    """Integer pixel noise in [-levels, levels] on the uint8 image tensors of a raw observation."""
    if isinstance(raw, dict):
        return {key: perturb_raw(value, levels, generator) for key, value in raw.items()}
    if torch.is_tensor(raw) and raw.dtype == torch.uint8:
        noise = torch.randint(
            -levels, levels + 1, raw.shape, generator=generator, dtype=torch.int16
        )
        return (raw.to(torch.int16) + noise).clamp_(0, 255).to(torch.uint8)
    return raw


def perturbation_pairs(
    bank: list[Any], indices: list[int], levels: int, seed: int
) -> list[tuple[Any, Any]]:
    """Original and pixel-perturbed visible bank samples, with the same noise for every policy."""
    generator = torch.Generator().manual_seed(seed)
    return [
        (bank[index], {**bank[index], "raw": perturb_raw(bank[index]["raw"], levels, generator)})
        for index in indices
    ]


def perturbation_deltas(
    policy: object,
    originals: list[Any],
    perturbed: list[Any],
    seed: int,
    pipeline: Any,
    clock: Clock = CLOCK,
) -> list[torch.Tensor]:
    """Physical arm change under the perturbation, per frame, with the flow noise held fixed."""
    deltas = []
    for index, (original, changed) in enumerate(zip(originals, perturbed, strict=True)):
        before = physical_arm(arm_chunk(policy, original, seed + 900 + index, clock), pipeline)
        after = physical_arm(arm_chunk(policy, changed, seed + 900 + index, clock), pipeline)
        deltas.append((after - before).flatten())
    return deltas


def perturbation_correlation(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    """Pearson correlation of the arm changes under the same perturbation.

    A lookup table returns the stored action for both frames, so its changes are zero and the
    correlation is zero.
    """
    if not reference or len(reference) != len(candidate):
        return 0.0
    left = torch.cat(reference).double()
    right = torch.cat(candidate).double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    return float((left * right).sum() / denominator)


def _bank_indices(bank: list[Any], count: int) -> list[int]:
    """Visible bank frames spread over the bank, so every task is represented."""
    total = len(bank)
    if total == 0 or count <= 0:
        return []
    step = max(1, total // count)
    return list(range(0, total, step))[:count]


@dataclass
class Reference:
    """Reference pass: frames, timing, action baselines and the figures compared against."""

    limits: Limits
    frames: list[Any]
    timing: dict
    episode_wall_seconds: float
    overhead_per_step_ms: float
    peak_cuda_bytes: int
    steps: int
    successes: int
    episodes: int
    warm: list[int] = field(default_factory=list)
    block_frames: list[list[int]] = field(default_factory=list)
    blocks: list[list[float]] = field(default_factory=list)
    arm: list[torch.Tensor] = field(default_factory=list)
    probe: dict = field(default_factory=dict)
    pipeline: Any = None
    watch: GpuWatch | None = None

    def as_record(self) -> dict:
        return {
            "model": "B16",
            "path": "stock",
            "frames": len(self.frames),
            "steps": self.steps,
            "episodes": self.episodes,
            "successes": self.successes,
            "timing": self.timing,
            "episode_wall_seconds": self.episode_wall_seconds,
            "overhead_per_step_ms": self.overhead_per_step_ms,
            "peak_cuda_bytes": self.peak_cuda_bytes,
            "arm_frames": len(self.arm),
            "probe_frames": len(self.probe.get("originals", [])),
        }


def run_reference(policy: object, protocol: dict, phase: str, out_dir: Path, clock: Clock = CLOCK):
    """Run reference episodes, collect fresh frames and time the untouched policy in blocks."""
    limits = Limits.from_protocol(protocol, phase)
    collector = FrameCollector(limits.max_frames, limits.frame_stride)
    folder = Path(out_dir) / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(policy)
    watch = GpuWatch(limits.other_gpu_utilisation_max_pct)
    clock.empty_cache()
    clock.reset_peak()
    started = clock.perf_ns()
    summary = rollout(
        "latency-reference",
        list(limits.reference_tasks),
        "development",
        limits.reference_trials,
        policy,
        pipeline,
        folder,
        trace=collector,
    )
    wall = (clock.perf_ns() - started) / 1e9
    warm, block_frames = frame_plan(len(collector.frames), limits, limits.equivalence_frames)
    warm_up(policy, collector.frames, warm, protocol["seed"], clock)
    results = []
    for block in block_frames:
        watch.sample()
        results.append(time_block(policy, collector.frames, block, protocol["seed"], clock))
    timing = _timing_summary(results, len(warm))
    count = max(0, min(limits.arm_check_frames, len(collector.frames) - limits.equivalence_frames))
    arm = arm_actions(
        policy,
        collector.frames,
        count,
        protocol["seed"],
        pipeline,
        clock,
        offset=limits.equivalence_frames,
    )
    bank = observation_bank()
    indices = _bank_indices(bank, limits.perturbation_frames)
    pairs = perturbation_pairs(bank, indices, limits.perturbation_levels, protocol["seed"] + 7000)
    probe = {
        "originals": [pipeline.sample(original) for original, _ in pairs],
        "perturbed": [pipeline.sample(perturbed) for _, perturbed in pairs],
    }
    probe["reference_deltas"] = perturbation_deltas(
        policy, probe["originals"], probe["perturbed"], protocol["seed"], pipeline, clock
    )
    peak = clock.max_allocated()
    steps = max(1, collector.seen)
    return Reference(
        limits=limits,
        frames=collector.frames,
        timing=timing,
        episode_wall_seconds=wall,
        overhead_per_step_ms=1000.0 * max(0.0, wall - collector.total_select_seconds) / steps,
        peak_cuda_bytes=peak,
        steps=collector.seen,
        successes=summary["successes"],
        episodes=summary["episodes"],
        warm=warm,
        block_frames=block_frames,
        blocks=[result["wall"] for result in results],
        arm=arm,
        probe=probe,
        pipeline=pipeline,
        watch=watch,
    )


def _load_second(loader: Callable[[], object] | None, peak_bytes: int, clock: Clock):
    """A second untouched policy, or None when it does not fit or cannot be loaded."""
    if loader is None:
        return None
    if clock.cuda:
        free, _ = torch.cuda.mem_get_info()
        if free < peak_bytes * 1.02:
            return None
    try:
        return loader()
    except (RuntimeError, MemoryError):
        clock.empty_cache()
        return None


def _interleaved(
    reference_policy: object,
    candidate: object,
    frames: list[Any],
    blocks: list[list[int]],
    seed: int,
    clock: Clock,
    watch: GpuWatch,
) -> tuple[list[list[float]], list[list[float]], list[dict]]:
    """Time reference and candidate blocks in alternation, so a burst lands in both sides.

    The order inside a pair alternates, so a clock ramp at the start of a block does not
    systematically favour the side that always runs second.
    """
    reference_blocks: list[list[float]] = []
    candidate_blocks: list[list[float]] = []
    results: list[dict] = []
    for index, block in enumerate(blocks):
        watch.sample()
        if index % 2:
            candidate_result = time_block(candidate, frames, block, seed, clock)
            reference_result = time_block(reference_policy, frames, block, seed, clock)
        else:
            reference_result = time_block(reference_policy, frames, block, seed, clock)
            candidate_result = time_block(candidate, frames, block, seed, clock)
        reference_blocks.append(reference_result["wall"])
        candidate_blocks.append(candidate_result["wall"])
        results.append(candidate_result)
    return reference_blocks, candidate_blocks, results


def run_candidate(
    policy: object,
    protocol: dict,
    reference: Reference,
    out_dir: Path,
    *,
    phase: str,
    tasks: list[int],
    trials: int,
    gate: int,
    clock: Clock = CLOCK,
    reference_loader: Callable[[], object] | None = None,
) -> dict:
    """Guards, block timing and episodes for one candidate; returns the latency record."""
    limits = reference.limits
    seed = protocol["seed"]
    violations = list(rebinds(clock))
    checks = equivalence(policy, reference.frames, limits, seed, clock)
    violations += checks["violations"]
    reset_ms = reset_cost(policy, reference.frames, clock)
    reset_limit = max(
        limits.reset_budget_ms, limits.reset_budget_fraction * reference.timing["median_ms"]
    )
    if reset_ms > reset_limit:
        violations.append(f"reset costs {reset_ms:.2f} ms, above {reset_limit:.2f} ms")
    arm = arm_actions(
        policy,
        reference.frames,
        len(reference.arm),
        seed,
        reference.pipeline,
        clock,
        offset=limits.equivalence_frames,
    )
    error = arm_error(reference.arm, arm)
    if error > limits.arm_action_mse_max:
        violations.append(f"arm action error {error:.3g} is above {limits.arm_action_mse_max:g}")
    deltas = perturbation_deltas(
        policy,
        reference.probe["originals"],
        reference.probe["perturbed"],
        seed,
        reference.pipeline,
        clock,
    )
    correlation = perturbation_correlation(reference.probe["reference_deltas"], deltas)
    if correlation < limits.perturbation_min_correlation:
        violations.append(
            f"perturbation response correlation {correlation:.3f} is below "
            f"{limits.perturbation_min_correlation:g}"
        )
    watch = reference.watch or GpuWatch(limits.other_gpu_utilisation_max_pct)
    clock.empty_cache()
    second = _load_second(reference_loader, reference.peak_cuda_bytes, clock)
    threads_before = threading.active_count()
    children_before = child_pids()
    warm_up(policy, reference.frames, reference.warm, seed, clock)
    if second is not None:
        warm_up(second, reference.frames, reference.warm, seed, clock)
        reference_blocks, candidate_blocks, results = _interleaved(
            second, policy, reference.frames, reference.block_frames, seed, clock, watch
        )
        mode = "interleaved"
    else:
        results = []
        for block in reference.block_frames:
            watch.sample()
            results.append(time_block(policy, reference.frames, block, seed, clock))
        reference_blocks = reference.blocks
        candidate_blocks = [result["wall"] for result in results]
        mode = "sandwich"
    # The second reference must be gone before the episode peak is measured, or the candidate
    # would be charged for it.
    second = None
    clock.empty_cache()
    timing = _timing_summary(results, len(reference.warm))
    threads_after = threading.active_count()
    children_after = child_pids()
    cpu_message = cpu_violation(
        reference.timing["cpu_ms_per_call"], timing["cpu_ms_per_call"], limits
    )
    if cpu_message is not None:
        violations.append(cpu_message)
    child_seconds = sum(result["child_seconds"] for result in results)
    child_message = child_cpu_violation(child_seconds, limits)
    if child_message is not None:
        violations.append(child_message)
    new_threads = max(0, threads_after - threads_before)
    if new_threads > limits.max_new_threads:
        violations.append(f"{new_threads} new threads were started during timing")
    new_pids = children_after - children_before
    if new_pids:
        violations.append(f"child processes started during timing: {sorted(new_pids)}")
    if timing.get("gpu_total_ms") is not None:
        gpu_limit = (
            timing["wall_total_ms"] * limits.gpu_over_wall_factor
            + limits.gpu_over_wall_ms * timing["calls"]
        )
        if timing["gpu_total_ms"] > gpu_limit:
            violations.append(
                f"GPU time {timing['gpu_total_ms']:.2f} ms is above {gpu_limit:.2f} ms "
                "for the timed calls"
            )
    novel = novel_frame_cost(policy, reference.frames, timing["median_ms"], limits, seed, clock)
    if novel["violation"]:
        violations.append(
            f"a novel frame costs {novel['median_ms']:.2f} ms, above {novel['limit_ms']:.2f} ms"
        )
    collector = FrameCollector(0, 1)
    pipeline = Pipeline(policy)
    clock.empty_cache()
    clock.reset_peak()
    started = clock.perf_ns()
    summary = rollout(
        "latency-candidate",
        tasks,
        phase,
        trials,
        policy,
        pipeline,
        Path(out_dir),
        trace=collector,
    )
    wall = (clock.perf_ns() - started) / 1e9
    peak = clock.max_allocated()
    steps = max(1, collector.seen)
    overhead = 1000.0 * max(0.0, wall - collector.total_select_seconds) / steps
    overhead_message = overhead_violation(reference.overhead_per_step_ms, overhead, limits)
    if overhead_message is not None:
        violations.append(overhead_message)
    watch_message = watch.violation()
    if watch_message is not None:
        violations.append(watch_message)
    interval = bootstrap_speedup(
        reference_blocks, candidate_blocks, limits, paired=mode == "interleaved"
    )
    timing["speedup"] = interval
    timing["mode"] = mode
    timing["sandwich_pending"] = mode == "sandwich"
    timing["reference_blocks_after"] = 0
    timing["reference_block_medians_ms"] = [float(np.median(block)) for block in reference_blocks]
    timing["candidate_blocks"] = candidate_blocks
    timing["gpu_watch"] = watch.summary()
    score, reason = speedup(
        interval,
        gate=gate,
        successes=summary["successes"],
        episodes=summary["episodes"],
        reference_peak=reference.peak_cuda_bytes,
        candidate_peak=peak,
        limits=limits,
    )
    overhead_limit = max(
        reference.overhead_per_step_ms * limits.overhead_factor,
        reference.overhead_per_step_ms + limits.overhead_ms,
    )
    return {
        "score": score,
        "score_kind": SCORE_KIND,
        "zero_reason": reason,
        "violations": violations,
        "gate": gate,
        "reference": reference.as_record(),
        "candidate": {
            **timing,
            "peak_cuda_bytes": peak,
            "episode_wall_seconds": wall,
            "episode_steps": collector.seen,
            "overhead_per_step_ms": overhead,
            "overhead_limit_ms": overhead_limit,
            "reset_ms": reset_ms,
            "reset_limit_ms": reset_limit,
            "novel_frame": novel,
            "child_cpu_seconds": child_seconds,
            "new_threads": new_threads,
            "new_child_pids": sorted(new_pids),
            "arm_action_error": error,
            "arm_action_error_limit": limits.arm_action_mse_max,
            "perturbation_correlation": correlation,
            "perturbation_correlation_min": limits.perturbation_min_correlation,
        },
        "equivalence": checks,
        "successes": summary["successes"],
        "episodes": summary["episodes"],
        "rollout": summary,
    }


def finish_sandwich(
    report: dict,
    reference: Reference,
    protocol: dict,
    loader: Callable[[], object] | None,
    clock: Clock = CLOCK,
) -> dict:
    """Time the reference again after the candidate episodes, when the two do not fit together.

    The caller frees the candidate first. The after blocks join the before blocks in one pooled
    reference median; the interval resamples the two sides independently because they are not
    adjacent in time.
    """
    timing = report.get("candidate") or {}
    if timing.get("mode") != "sandwich" or not timing.get("sandwich_pending"):
        return report
    timing["sandwich_pending"] = False
    second = _load_second(loader, reference.peak_cuda_bytes, clock)
    if second is None:
        timing["reference_blocks_after"] = 0
        report["violations"].append(
            "the reference could not be re-timed after the candidate: "
            "the sandwich measurement is incomplete"
        )
        report["score"] = 0.0
        return report
    warm_up(second, reference.frames, reference.warm, protocol["seed"], clock)
    watch = reference.watch
    after = []
    try:
        for block in reference.block_frames:
            if watch is not None:
                watch.sample()
            after.append(
                time_block(second, reference.frames, block, protocol["seed"], clock)["wall"]
            )
    except (RuntimeError, MemoryError):
        timing["reference_blocks_after"] = 0
        report["violations"].append(
            "the reference could not be re-timed after the candidate: "
            "the sandwich measurement is incomplete"
        )
        report["score"] = 0.0
        return report
    timing["reference_blocks_after"] = len(after)
    before_median = float(np.median([value for block in reference.blocks for value in block]))
    after_median = float(np.median([value for block in after for value in block]))
    drift = abs(after_median / before_median - 1.0)
    timing["sandwich_drift"] = drift
    if drift > reference.limits.sandwich_drift_max:
        report["violations"].append(
            f"the reference moved {100.0 * drift:.0f} % between the before and after blocks, "
            f"above {100.0 * reference.limits.sandwich_drift_max:.0f} %"
        )
    interval = bootstrap_speedup(
        reference.blocks + after, timing["candidate_blocks"], reference.limits, paired=False
    )
    timing["speedup"] = interval
    pooled = _summary([value for block in reference.blocks + after for value in block])
    reference.timing.update(pooled)
    timing["reference_median_ms"] = pooled["median_ms"]
    if watch is not None:
        timing["gpu_watch"] = watch.summary()
        message = watch.violation()
        if message is not None and message not in report["violations"]:
            report["violations"].append(message)
    report["score"], report["zero_reason"] = speedup(
        interval,
        gate=int(report.get("gate", 0)),
        successes=int(report.get("successes", 0)),
        episodes=int(report.get("episodes", 0)),
        reference_peak=reference.peak_cuda_bytes,
        candidate_peak=int(report["candidate"]["peak_cuda_bytes"]),
        limits=reference.limits,
    )
    if report["violations"]:
        report["score"] = 0.0
    return report
