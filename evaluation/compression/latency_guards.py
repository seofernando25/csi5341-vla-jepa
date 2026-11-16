"""Failing conditions on a latency candidate (program-v4).

Each is a failing condition, not a warning:

- `select_action` must agree with `predict_action_chunk` on the same frame and seed, must be
  reproducible, must not answer the previous frame, and must change with the observation;
- a novel frame (a blend of two collected frames) must not cost more than a set factor above the
  median, which catches a cache with a slow fallback;
- the candidate's physical arm actions on fresh frames must stay within a set distance of the
  untouched policy's, and its arm changes under a small pixel perturbation must correlate with the
  untouched policy's, which catches a lookup table that returns a stored answer. Equivalence, arm
  error and correlation are judged on each task, and the worst task decides, so a candidate that
  degrades one task cannot hide it in a pooled figure;
- `reset` must stay cheap, so work cannot move out of the timed call into it;
- the episode batch's wall time per step, minus the policy's own time in the episodes, must not
  exceed the reference's by more than a set factor, so background or asynchronous work is charged
  to the candidate;
- CPU seconds per call and child-process CPU seconds are capped against the reference's, so work
  cannot move to the CPU or another process;
- peak CUDA bytes must not exceed the reference's.

The first group is about what the candidate answers, the second about where its work happens.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import torch

from evaluation.compression.latency_timing import CLOCK, Clock, Limits, autocast

RESET_PROBES = 8
NOVEL_PAIRS = 4


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


def _seeded(policy: object, batch: Any, seed: int, clock: Clock, call: Any) -> torch.Tensor:
    """One policy call on the CPU, through `call`, under `seed`."""
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode(), autocast(clock):
        value = call(batch)
    return value.float().detach().cpu()


def _action(policy: object, batch: Any, seed: int, clock: Clock) -> torch.Tensor:
    """One `select_action` on the CPU, flattened."""
    return _seeded(policy, batch, seed, clock, policy.select_action).flatten()


def _chunk(policy: object, batch: Any, seed: int, clock: Clock) -> torch.Tensor:
    """One `predict_action_chunk` on the CPU."""
    return _seeded(policy, batch, seed, clock, policy.predict_action_chunk)


def _close(left: torch.Tensor, right: torch.Tensor, atol: float) -> bool:
    return left.shape == right.shape and bool(torch.allclose(left, right, atol=atol, rtol=0.0))


def equivalence(policy: object, frames: list[Any], limits: Limits, seed: int, clock: Clock = CLOCK):
    """Reproducibility, order, path and input-sensitivity checks on held-out frames."""
    count = min(limits.equivalence_per_task, len(frames))
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
    first = _action(policy, frames[0], seed, clock)
    _action(policy, frames[1], seed, clock)
    after_other = _action(policy, frames[0], seed, clock)
    repeat = _action(policy, frames[0], seed, clock)
    chunk = _chunk(policy, frames[0], seed, clock).flatten()
    actions = [_action(policy, frames[index], seed, clock) for index in range(count)]
    spread = 0.0
    for left in range(count):
        for right in range(left + 1, count):
            spread = max(spread, float((actions[left] - actions[right]).abs().max()))
    other_seed = _action(policy, frames[0], seed + 1, clock)
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
        with torch.inference_mode(), autocast(clock):
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


def physical_arm(chunk: torch.Tensor, pipeline: Any) -> torch.Tensor:
    """Post-processed arm dimensions (the first six) of a chunk, on the CPU."""
    physical = pipeline.post(chunk.float().clone()).float()
    return physical[..., :6].detach().cpu()


def arm_actions(
    policy: object,
    frames: list[Any],
    indices: list[int],
    seed: int,
    pipeline: Any,
    clock: Clock = CLOCK,
) -> list[torch.Tensor]:
    """Physical arm actions of the policy on the frames at `indices`, one seed per frame."""
    return [
        physical_arm(_chunk(policy, frames[index], seed + index, clock), pipeline)
        for index in indices
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


def arm_error_by_task(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], labels: list[int]
) -> dict[int, float]:
    """`arm_error` of each task's frames; the limit applies to the worst task, not the pool."""
    return {
        task: arm_error(
            [value for value, label in zip(reference, labels) if label == task],
            [value for value, label in zip(candidate, labels) if label == task],
        )
        for task in sorted(set(labels))
    }


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
        before = physical_arm(_chunk(policy, original, seed + 900 + index, clock), pipeline)
        after = physical_arm(_chunk(policy, changed, seed + 900 + index, clock), pipeline)
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


def perturbation_by_task(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], labels: list[int]
) -> dict[int, float]:
    """`perturbation_correlation` of each task's frames; the limit applies to the worst task."""
    return {
        task: perturbation_correlation(
            [value for value, label in zip(reference, labels) if label == task],
            [value for value, label in zip(candidate, labels) if label == task],
        )
        for task in sorted(set(labels))
    }


def bank_indices(bank: list[Any], count: int, tasks: tuple[int, ...]) -> list[int]:
    """`count` visible bank frames of each task, spread over that task's frames."""
    chosen: list[int] = []
    for task in tasks:
        own = [index for index, sample in enumerate(bank) if int(sample["task_id"]) == task]
        if own and count > 0:
            step = max(1, len(own) // count)
            chosen += own[::step][:count]
    return chosen
