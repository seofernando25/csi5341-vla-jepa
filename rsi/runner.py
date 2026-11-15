"""Dream-RSI protocol loader: the cycle constants read from one study's `protocol.json`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DREAM_STUDY_DIR = Path("studies/program-v1")
PHASES_ALLOWED = frozenset({"smoke", "development"})


@dataclass(frozen=True)
class CycleConfig:
    """Cycle constants read from one study's `protocol.json`."""

    grid_plan: dict[str, Any]
    smoke_gate: int
    dev_gate: int
    smoke_trials: int
    discovery_tools: str
    policy_dev_tools: str
    agent_timeout_s: float
    eval_timeout_s: float
    gpu_slots: int


def load_protocol(study_dir: Path) -> dict[str, Any]:
    """Read `protocol.json` from a study directory."""
    return json.loads((Path(study_dir) / "protocol.json").read_text(encoding="utf-8"))


def load_cycle_config(study_dir: Path) -> CycleConfig:
    """Read every Dream-RSI cycle constant from the study's `protocol.json`."""
    proto = load_protocol(study_dir)
    drsi = proto["dream_rsi"]
    return CycleConfig(
        grid_plan=dict(drsi.get("grid_plan", {})),
        smoke_gate=int(proto["gates"]["smoke_min_successes"]),
        dev_gate=int(proto["gates"]["development_min_successes"]),
        smoke_trials=int(proto["phases"]["smoke"]["trials"]),
        discovery_tools=str(drsi["discovery_tools"]),
        policy_dev_tools=str(drsi["policy_dev_tools"]),
        agent_timeout_s=float(drsi["agent_timeout_s"]),
        eval_timeout_s=float(drsi["eval_timeout_s"]),
        gpu_slots=int(drsi["gpu_slots"]),
    )


def check_phase_lock(phase: str, study_dir: Path) -> None:
    """Refuse phases outside the cycle's set and any study whose reserved lock is absent."""
    if phase not in PHASES_ALLOWED:
        raise ValueError(f"phase {phase!r} not allowed; expected one of {sorted(PHASES_ALLOWED)}")
    if load_protocol(study_dir).get("reserved_lock") is not True:
        raise ValueError(f"{study_dir / 'protocol.json'} lacks reserved_lock: true")
