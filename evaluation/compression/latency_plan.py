"""Frame layout and speedup statistics of the latency score (program-v4).

Every task of the suite gets the same number of frames in every role: equivalence, arm check,
warm-up and timed. A timed block holds the same number of calls from each task, in alternation, and
the same frames time the reference and the candidate. The speedup is computed per task, as the
median reference time over the median candidate time on that task's calls, and aggregated as the
geometric mean, so a gain on one task moves the score by its share and no task dominates. The
interval is a percentile cluster bootstrap that resamples whole blocks; every block holds every
task, so each resample keeps the stratification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.compression.latency_timing import Limits

INTERVAL_PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Plan:
    """Which collected frames play which role; `labels[i]` is the task of frame `i`."""

    labels: list[int]
    equivalence: dict[int, list[int]]
    arm: dict[int, list[int]]
    warm: list[int]
    blocks: list[list[int]]

    @property
    def arm_indices(self) -> list[int]:
        return [index for indices in self.arm.values() for index in indices]

    @property
    def block_labels(self) -> list[list[int]]:
        return [[self.labels[index] for index in block] for block in self.blocks]

    def alternating(self, groups: dict[int, list[int]]) -> list[int]:
        """Frames of every task in turn: task 0, task 1, ..., task 0, ..."""
        return [index for turn in zip(*groups.values()) for index in turn]


def frame_plan(labels: list[int], limits: Limits) -> Plan:
    """Split every task's frames into equivalence, arm, warm-up and timed roles, all disjoint.

    Within a task the frames keep their episode order: equivalence first, then arm, then the
    pool. Warm-up frames are spread over the pool so the shapes are seen; the timed frames are the
    rest, split by stride so every block covers the same spread of the episode.
    """
    if limits.timed_per_task % limits.blocks:
        raise ValueError("timed calls per task must divide into the blocks")
    equivalence: dict[int, list[int]] = {}
    arm: dict[int, list[int]] = {}
    timed: dict[int, list[int]] = {}
    warm: list[int] = []
    first, second = limits.equivalence_per_task, limits.arm_per_task
    needed = first + second + limits.warmup_per_task + limits.timed_per_task
    for task in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == task]
        if len(indices) < needed:
            raise RuntimeError(f"task {task}: need {needed} frames, collected {len(indices)}")
        pool = indices[first + second :]
        step = max(1, len(pool) // max(1, limits.warmup_per_task))
        chosen = pool[::step][: limits.warmup_per_task]
        equivalence[task] = indices[:first]
        arm[task] = indices[first : first + second]
        warm += chosen
        timed[task] = [index for index in pool if index not in set(chosen)][: limits.timed_per_task]
    blocks = []
    for offset in range(limits.blocks):
        slices = {task: indices[offset :: limits.blocks] for task, indices in timed.items()}
        blocks.append([index for turn in zip(*slices.values()) for index in turn])
    return Plan(list(labels), equivalence, arm, warm, blocks)


def task_speedups(
    reference: np.ndarray,
    reference_labels: np.ndarray,
    candidate: np.ndarray,
    candidate_labels: np.ndarray,
    tasks: list[int],
) -> list[float]:
    """Median reference time over median candidate time, per task."""
    return [
        float(
            np.median(reference[reference_labels == task])
            / np.median(candidate[candidate_labels == task])
        )
        for task in tasks
    ]


def aggregate(ratios: list[float]) -> float:
    """Geometric mean: a factor on one task moves the score by its share only."""
    return float(np.exp(np.mean(np.log(ratios))))


def _pool(blocks: list[np.ndarray], tags: list[np.ndarray], pick: np.ndarray):
    return (
        np.concatenate([blocks[index] for index in pick]),
        np.concatenate([tags[index] for index in pick]),
    )


def bootstrap_speedup(
    reference_blocks: list[list[float]],
    candidate_blocks: list[list[float]],
    limits: Limits,
    *,
    paired: bool = True,
    labels: list[list[int]] | None = None,
    reference_labels: list[list[int]] | None = None,
) -> dict:
    """Aggregate speedup and its 95 % cluster-bootstrap interval over blocks.

    `labels` gives the task of every call of the candidate blocks, `reference_labels` the same for
    the reference blocks (default: the same); without labels every call is one task. `paired`
    resamples one block index and takes both sides of that block, which is the interleaved
    design. Without it the two sides are resampled independently, the sandwich design.
    """
    reference = [np.asarray(block, dtype=float) for block in reference_blocks]
    candidate = [np.asarray(block, dtype=float) for block in candidate_blocks]
    if labels is None:
        labels = [[0] * len(block) for block in candidate]
    if reference_labels is None:
        reference_labels = labels
    reference_tags = [np.asarray(block, dtype=int) for block in reference_labels]
    candidate_tags = [np.asarray(block, dtype=int) for block in labels]
    tasks = sorted({int(task) for block in candidate_tags for task in block})
    whole = np.arange(len(reference))
    point_ratios = task_speedups(
        *_pool(reference, reference_tags, whole),
        *_pool(candidate, candidate_tags, np.arange(len(candidate))),
        tasks,
    )
    rng = np.random.default_rng(limits.bootstrap_seed)
    draws = np.empty(limits.bootstrap_resamples, dtype=float)
    for draw in range(limits.bootstrap_resamples):
        pick = rng.integers(0, len(reference), len(reference))
        other = pick if paired else rng.integers(0, len(candidate), len(candidate))
        draws[draw] = aggregate(
            task_speedups(
                *_pool(reference, reference_tags, pick),
                *_pool(candidate, candidate_tags, other),
                tasks,
            )
        )
    low, high = np.percentile(draws, INTERVAL_PERCENTILES)
    return {
        "point": aggregate(point_ratios),
        "low": float(low),
        "high": float(high),
        "per_task": {str(task): ratio for task, ratio in zip(tasks, point_ratios)},
        "worst_task": min(point_ratios),
        "blocks": len(reference),
        "candidate_blocks": len(candidate),
        "resamples": limits.bootstrap_resamples,
        "paired": paired,
        "method": (
            "geometric mean over tasks of the per-task median speedup; "
            "95 % percentile cluster bootstrap over timing blocks"
        ),
    }
