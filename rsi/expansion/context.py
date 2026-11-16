"""Inputs of one node expansion: cycle settings, the node job, the digest registry and the evaluator."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.agents import AgentSpec
from rsi.runner import PHASES_ALLOWED
from rsi.sandbox import run_sandboxed_evaluator

Evaluator = Callable[[Path, str, Path], dict[str, Any]]


@dataclass(frozen=True)
class NodeContext:
    """Cycle-wide settings shared by every node expansion in one cycle."""

    cycle_id: str
    agent: AgentSpec
    agent_timeout: float
    eval_timeout: float
    gpu_slots_dir: Path
    gpu_slots: int
    smoke_trials: int
    smoke_gate: int
    dev_gate: int
    evaluator: Evaluator
    study_dir: Path
    agent_cmd: str | list[str] | None = None
    use_bwrap: bool = True


@dataclass(frozen=True)
class NodeJob:
    """Inputs for one expansion: its parent's frozen snapshot and the frozen batch evidence."""

    cell_id: str
    parent_id: str
    store: Path
    parent_commit: str | None
    node_dir: Path
    evidence_nodes: list[dict[str, Any]]
    tree_index: dict[str, Any]
    best_score: float


@dataclass
class DigestRegistry:
    """Digests of every node snapshot claimed in this cycle, shared by concurrent expansions."""

    owners: dict[str, str]
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self, digest: str, cell_id: str) -> str | None:
        """Register `digest` for `cell_id`; return the existing owner instead when there is one."""
        with self._lock:
            owner = self.owners.get(digest)
            if owner is None:
                self.owners[digest] = cell_id
            return owner


def sandboxed_evaluator(
    eval_timeout: float, gpu_slots_dir: Path, gpu_slots: int, study_dir: Path | None = None
) -> Evaluator:
    """Evaluator that runs the fixed `solution` evaluator in the GPU sandbox.

    `study_dir` names the study whose `protocol.json` the evaluator reads.
    """

    def _run(snapshot_dir: Path, phase: str, out_dir: Path) -> dict[str, Any]:
        return run_sandboxed_evaluator(
            snapshot_dir,
            phase,
            out_dir,
            timeout=eval_timeout,
            gpu_slots_dir=gpu_slots_dir,
            gpu_slot_count=gpu_slots,
            study_dir=study_dir,
        )

    return _run


def evaluate_snapshot(
    evaluator: Evaluator, snapshot_dir: Path, phase: str, out_dir: Path
) -> dict[str, Any]:
    """Run the evaluator on one frozen snapshot; only the protocol's phases may be requested."""
    if phase not in PHASES_ALLOWED:
        raise ValueError(f"phase {phase!r} not allowed; expected one of {sorted(PHASES_ALLOWED)}")
    return evaluator(snapshot_dir, phase, out_dir)
