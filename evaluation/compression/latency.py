"""Latency score for the fixed evaluator (program-v4).

The score is the reference median observation-to-action latency divided by the candidate's, both
measured in this process on frames the evaluator collected from episodes it ran. The reference is
the untouched B16 policy on the stock path; the candidate is whatever `compress` returned. The
score is zero unless the smoke gate passes and the candidate's peak CUDA bytes do not exceed the
reference's.

Timing is block-based. Each side is timed in `blocks` blocks of `timed_calls / blocks` calls,
after warm-up, on frames that were never timed before. When a second copy of the reference fits
beside the candidate, the blocks alternate reference, candidate, reference, candidate (ABAB), so
a co-tenant burst lands in both sides. When it does not fit, the reference is timed in blocks
before `compress` and again after the candidate episodes and the two pools are combined
(sandwich). The speedup is reported with a 95 % cluster-bootstrap interval over blocks and the
score is its lower bound, so timing noise cannot create a win.

The instruments are in `latency_timing.py` (captured clock, frame plan, blocks, bootstrap, the
co-tenant watch) and the failing conditions in `latency_guards.py` (equivalence, novel-frame
cost, arm-action error, perturbation response, reset cost, episode overhead, CPU and child CPU,
peak CUDA). This module runs the reference pass, the candidate pass and the sandwich, and turns
them into the score.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.compression.latency_guards import (
    arm_actions,
    arm_error,
    bank_indices,
    child_cpu_violation,
    cpu_violation,
    equivalence,
    novel_frame_cost,
    overhead_violation,
    peak_violation,
    perturbation_correlation,
    perturbation_deltas,
    perturbation_pairs,
    reset_cost,
)
from evaluation.compression.latency_timing import (
    CLOCK,
    Clock,
    FrameCollector,
    GpuWatch,
    Limits,
    bootstrap_speedup,
    child_pids,
    frame_plan,
    rebinds,
    summarize,
    time_block,
    timing_summary,
    warm_up,
)
from evaluation.pipeline import Pipeline, observation_bank
from evaluation.rollout import rollout

SCORE_KIND = "latency_speedup"


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


@dataclass(frozen=True)
class Episodes:
    """One rollout batch: its outcome, its wall time and the policy's own time in it."""

    frames: list[Any]
    summary: dict
    wall_seconds: float
    steps: int
    overhead_per_step_ms: float


def _episodes(
    policy: object,
    pipeline: object,
    name: str,
    tasks: list[int],
    phase: str,
    trials: int,
    out_dir: Path,
    clock: Clock,
    limit: int = 0,
    stride: int = 1,
) -> Episodes:
    """Run the episodes and measure what the guards read; `limit` keeps sampled frames."""
    collector = FrameCollector(limit, stride)
    clock.empty_cache()
    clock.reset_peak()
    started = clock.perf_ns()
    summary = rollout(
        name, list(tasks), phase, trials, policy, pipeline, Path(out_dir), trace=collector
    )
    wall = (clock.perf_ns() - started) / 1e9
    steps = max(1, collector.seen)
    return Episodes(
        frames=collector.frames,
        summary=summary,
        wall_seconds=wall,
        steps=collector.seen,
        overhead_per_step_ms=1000.0 * max(0.0, wall - collector.total_select_seconds) / steps,
    )


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
    folder = Path(out_dir) / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(policy)
    watch = GpuWatch(limits.other_gpu_utilisation_max_pct)
    episodes = _episodes(
        policy,
        pipeline,
        "latency-reference",
        list(limits.reference_tasks),
        "development",
        limits.reference_trials,
        folder,
        clock,
        limit=limits.max_frames,
        stride=limits.frame_stride,
    )
    warm, block_frames = frame_plan(len(episodes.frames), limits, limits.equivalence_frames)
    warm_up(policy, episodes.frames, warm, protocol["seed"], clock)
    results = []
    for block in block_frames:
        watch.sample()
        results.append(time_block(policy, episodes.frames, block, protocol["seed"], clock))
    timing = timing_summary(results, len(warm))
    count = max(0, min(limits.arm_check_frames, len(episodes.frames) - limits.equivalence_frames))
    arm = arm_actions(
        policy,
        episodes.frames,
        count,
        protocol["seed"],
        pipeline,
        clock,
        offset=limits.equivalence_frames,
    )
    bank = observation_bank()
    indices = bank_indices(bank, limits.perturbation_frames)
    pairs = perturbation_pairs(bank, indices, limits.perturbation_levels, protocol["seed"] + 7000)
    probe = {
        "originals": [pipeline.sample(original) for original, _ in pairs],
        "perturbed": [pipeline.sample(perturbed) for _, perturbed in pairs],
    }
    probe["reference_deltas"] = perturbation_deltas(
        policy, probe["originals"], probe["perturbed"], protocol["seed"], pipeline, clock
    )
    return Reference(
        limits=limits,
        frames=episodes.frames,
        timing=timing,
        episode_wall_seconds=episodes.wall_seconds,
        overhead_per_step_ms=episodes.overhead_per_step_ms,
        peak_cuda_bytes=clock.max_allocated(),
        steps=episodes.steps,
        successes=episodes.summary["successes"],
        episodes=episodes.summary["episodes"],
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


def _behaviour(policy: object, reference: Reference, seed: int, clock: Clock) -> dict:
    """Guards on what the candidate answers, before it is timed."""
    limits = reference.limits
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
    return {
        "violations": violations,
        "equivalence": checks,
        "reset_ms": reset_ms,
        "reset_limit_ms": reset_limit,
        "arm_action_error": error,
        "perturbation_correlation": correlation,
    }


def _timed(
    policy: object,
    reference: Reference,
    seed: int,
    clock: Clock,
    reference_loader: Callable[[], object] | None,
) -> dict:
    """Block timing of the candidate against the reference, and the guards that read it."""
    limits = reference.limits
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
    timing = timing_summary(results, len(reference.warm))
    threads_after = threading.active_count()
    children_after = child_pids()
    violations = []
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
    return {
        "timing": timing,
        "reference_blocks": reference_blocks,
        "candidate_blocks": candidate_blocks,
        "mode": mode,
        "watch": watch,
        "novel": novel,
        "child_cpu_seconds": child_seconds,
        "new_threads": new_threads,
        "new_child_pids": sorted(new_pids),
        "violations": violations,
    }


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
    behaviour = _behaviour(policy, reference, seed, clock)
    timed = _timed(policy, reference, seed, clock, reference_loader)
    violations = behaviour["violations"] + timed["violations"]
    timing = timed["timing"]
    watch = timed["watch"]
    episodes = _episodes(
        policy,
        Pipeline(policy),
        "latency-candidate",
        tasks,
        phase,
        trials,
        Path(out_dir),
        clock,
    )
    overhead_message = overhead_violation(
        reference.overhead_per_step_ms, episodes.overhead_per_step_ms, limits
    )
    if overhead_message is not None:
        violations.append(overhead_message)
    watch_message = watch.violation()
    if watch_message is not None:
        violations.append(watch_message)
    interval = bootstrap_speedup(
        timed["reference_blocks"],
        timed["candidate_blocks"],
        limits,
        paired=timed["mode"] == "interleaved",
    )
    timing["speedup"] = interval
    timing["mode"] = timed["mode"]
    timing["sandwich_pending"] = timed["mode"] == "sandwich"
    timing["reference_blocks_after"] = 0
    timing["reference_block_medians_ms"] = [
        float(np.median(block)) for block in timed["reference_blocks"]
    ]
    timing["candidate_blocks"] = timed["candidate_blocks"]
    timing["gpu_watch"] = watch.summary()
    peak = clock.max_allocated()
    score, reason = speedup(
        interval,
        gate=gate,
        successes=episodes.summary["successes"],
        episodes=episodes.summary["episodes"],
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
            "episode_wall_seconds": episodes.wall_seconds,
            "episode_steps": episodes.steps,
            "overhead_per_step_ms": episodes.overhead_per_step_ms,
            "overhead_limit_ms": overhead_limit,
            "reset_ms": behaviour["reset_ms"],
            "reset_limit_ms": behaviour["reset_limit_ms"],
            "novel_frame": timed["novel"],
            "child_cpu_seconds": timed["child_cpu_seconds"],
            "new_threads": timed["new_threads"],
            "new_child_pids": timed["new_child_pids"],
            "arm_action_error": behaviour["arm_action_error"],
            "arm_action_error_limit": limits.arm_action_mse_max,
            "perturbation_correlation": behaviour["perturbation_correlation"],
            "perturbation_correlation_min": limits.perturbation_min_correlation,
        },
        "equivalence": behaviour["equivalence"],
        "successes": episodes.summary["successes"],
        "episodes": episodes.summary["episodes"],
        "rollout": episodes.summary,
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
    pooled = summarize([value for block in reference.blocks + after for value in block])
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
