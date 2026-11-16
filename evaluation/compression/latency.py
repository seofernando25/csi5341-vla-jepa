"""Latency score and anti-gaming guards for the fixed evaluator (program-v4).

The score is the reference median observation-to-action latency divided by the candidate's,
both measured in this process on frames the evaluator collected from episodes it ran. The
reference is the untouched B16 policy on the stock path; the candidate is whatever `compress`
returned. The score is zero unless the smoke gate passes and the candidate's peak CUDA bytes
do not exceed the reference's.

Each guard is a failing condition, not a warning:

- the clock, the CUDA synchronizer and the peak-memory readers are captured when this module is
  imported, before solution code runs; replacing any of them fails the node;
- warm-up and timed calls use disjoint frames, so every timed call sees an observation that was
  never timed before and a lookup table keyed on the observation gains nothing;
- `select_action` must agree with `predict_action_chunk` on the same frame and seed, must be
  reproducible, must not answer the previous frame, and must change with the observation;
- a novel frame (a blend of two collected frames) must not cost more than a set factor above the
  median, which catches a cache with a slow fallback;
- `reset` must stay cheap, so work cannot move out of the timed call into it;
- the episode batch's wall time per step, minus the policy's own time in the episodes, must not
  exceed the reference's by more than a set factor, so background or asynchronous work is
  charged to the candidate;
- CPU seconds per call and child-process CPU seconds are capped against the reference's, so work
  cannot move to the CPU or another process, and new threads during timing fail;
- peak CUDA bytes must not exceed the reference's.
"""

from __future__ import annotations

import os
import resource
import statistics
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.pipeline import Pipeline, inference_autocast
from evaluation.rollout import rollout

SCORE_KIND = "latency_speedup"
MIN_TIMED_CALLS = 8
RESET_PROBES = 8
NOVEL_PAIRS = 4


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
    reference_ms: float,
    candidate_ms: float,
    *,
    gate: int,
    successes: int,
    episodes: int,
    reference_peak: int,
    candidate_peak: int,
    limits: Limits,
) -> tuple[float, str | None]:
    """The score and the reason it is zero, if it is."""
    if gate and successes < gate:
        return 0.0, f"smoke gate: {successes}/{episodes} < {gate}"
    above = peak_violation(reference_peak, candidate_peak, limits)
    if above is not None:
        return 0.0, above
    return reference_ms / candidate_ms, None


def time_calls(
    policy: object,
    frames: list[Any],
    limits: Limits,
    seed: int,
    clock: Clock = CLOCK,
    first: int = 0,
) -> dict:
    """Time full select_action calls on distinct frames, after warm-up, with a device sync.

    `first` leaves the equivalence and novel-frame probes out of the timed pool, so no timed
    frame was ever passed to the policy before. Warm-up indices are spread over the whole pool
    (so every task's shapes are seen) and the timed indices are the rest.
    """
    pool = list(range(first, len(frames)))
    warmup = min(limits.warmup_calls, max(0, len(pool) - MIN_TIMED_CALLS))
    step = max(1, len(pool) // max(1, warmup))
    warm = pool[::step][:warmup]
    warm_set = set(warm)
    timed_indices = [index for index in pool if index not in warm_set][: limits.timed_calls]
    if len(timed_indices) < MIN_TIMED_CALLS:
        raise RuntimeError(f"need {MIN_TIMED_CALLS} distinct frames, collected {len(frames)}")
    clock.sync()
    for index in warm:
        policy.reset()
        torch.manual_seed(seed + index)
        with torch.inference_mode(), _autocast(clock):
            policy.select_action(frames[index])
    clock.sync()
    wall: list[float] = []
    gpu: list[float] = []
    cpu_before = _cpu(clock)
    for index in timed_indices:
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
    timing = _summary(wall)
    timing.update(
        calls=len(timed_indices),
        warmup=warmup,
        cpu_ms_per_call=1000.0 * max(0.0, cpu_after[0] - cpu_before[0]) / len(timed_indices),
        child_cpu_seconds=max(0.0, cpu_after[1] - cpu_before[1]),
        wall_total_ms=float(sum(wall)),
        gpu_total_ms=float(sum(gpu)) if gpu else None,
    )
    if gpu:
        gpu_summary = _summary(gpu)
        timing["gpu_median_ms"] = gpu_summary["median_ms"]
        timing["gpu_p95_ms"] = gpu_summary["p95_ms"]
        timing["gpu_mean_ms"] = gpu_summary["mean_ms"]
    return timing


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


@dataclass
class Reference:
    """Reference pass: frames, timing and the figures the candidate is compared against."""

    limits: Limits
    frames: list[Any]
    timing: dict
    episode_wall_seconds: float
    overhead_per_step_ms: float
    peak_cuda_bytes: int
    steps: int
    successes: int
    episodes: int

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
        }


def run_reference(policy: object, protocol: dict, phase: str, out_dir: Path, clock: Clock = CLOCK):
    """Run reference episodes, collect fresh frames and time the untouched policy on them."""
    limits = Limits.from_protocol(protocol, phase)
    collector = FrameCollector(limits.max_frames, limits.frame_stride)
    folder = Path(out_dir) / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(policy)
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
    timing = time_calls(
        policy, collector.frames, limits, protocol["seed"], clock, first=limits.equivalence_frames
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
    )


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
) -> dict:
    """Guards, timing and episodes for one candidate; returns the latency record."""
    limits = reference.limits
    violations = list(rebinds(clock))
    checks = equivalence(policy, reference.frames, limits, protocol["seed"], clock)
    violations += checks["violations"]
    reset_ms = reset_cost(policy, reference.frames, clock)
    reset_limit = max(
        limits.reset_budget_ms, limits.reset_budget_fraction * reference.timing["median_ms"]
    )
    if reset_ms > reset_limit:
        violations.append(f"reset costs {reset_ms:.2f} ms, above {reset_limit:.2f} ms")
    threads_before = threading.active_count()
    children_before = child_pids()
    cpu_before = _cpu(clock)
    clock.empty_cache()
    clock.reset_peak()
    timing = time_calls(
        policy, reference.frames, limits, protocol["seed"], clock, first=limits.equivalence_frames
    )
    cpu_after = _cpu(clock)
    threads_after = threading.active_count()
    children_after = child_pids()
    cpu_message = cpu_violation(
        reference.timing["cpu_ms_per_call"], timing["cpu_ms_per_call"], limits
    )
    if cpu_message is not None:
        violations.append(cpu_message)
    child_seconds = max(0.0, cpu_after[1] - cpu_before[1])
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
    novel = novel_frame_cost(
        policy, reference.frames, timing["median_ms"], limits, protocol["seed"], clock
    )
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
    score, reason = speedup(
        reference.timing["median_ms"],
        timing["median_ms"],
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
        },
        "equivalence": checks,
        "successes": summary["successes"],
        "episodes": summary["episodes"],
        "rollout": summary,
    }
