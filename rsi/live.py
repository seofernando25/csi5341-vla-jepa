"""Live Dream-RSI rounds: frozen InitialPolicy control vs selected OptimalPolicy variant.

Each round probes a fresh Qwen compression grid cell set with the same probe budget for
both arms, journals every run (phase, n, bytes, machine), promotes smoke passes to
development under a pre-stated gate, and exports the round as a replay world that joins
history. GPU use is guarded: memory is checked before each probe and the runner backs
off when the shared card is tight. Nothing is ever killed.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.api import gate_fail_class
from rsi.qwen_search import (
    B16_RESIDENT_BYTES,
    LiveQwenQuestion,
    Stage2Candidate,
    _t2_base_changes,
    run_compression_subprocess,
)
from rsi.tree import DiscoveryTree, TreeNode

SMOKE_GATE = 8
DEV_GATE = 77
GPU_BACKOFF_USED_FRACTION = 0.85


def gpu_name() -> str:
    """GPU model string; never touches other processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        return out[0].strip() if out else "unknown"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def gpu_tight() -> bool:
    """True when the first GPU reports more than the backoff fraction of memory used."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        used, total = (float(x) for x in out[0].split(","))
        return used / total > GPU_BACKOFF_USED_FRACTION
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return False


def journals_for(round_id: str) -> dict[str, Path]:
    base = Path("outputs/compression") / round_id
    return {
        "control": base / "control_journal.jsonl",
        "selected": base / "selected_journal.jsonl",
        "stage": base / "stage_journal.jsonl",
    }


def _channels(family: str, specs: list[tuple[str, float]]) -> list[dict[str, Any]]:
    changes = [{"method": "channels", "scope": scope, "fraction": frac} for scope, frac in specs]
    return [*changes, *_t2_base_changes()]


def build_round1_candidates() -> list[Stage2Candidate]:
    """Fresh 3-branch x 3-attempt grid around the Stage 2 winner (T2 + 10% text + 25% vision).

    All cell IDs are new (`r1-*`); no committed measurements exist for them.
    """
    specs: list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]] = [
        (
            "r1_joint",
            [
                ("r1-t12-v30", _channels("r1_joint", [("text_mlp", 0.12), ("vision_mlp", 0.30)])),
                ("r1-t15-v35", _channels("r1_joint", [("text_mlp", 0.15), ("vision_mlp", 0.35)])),
                ("r1-t20-v45", _channels("r1_joint", [("text_mlp", 0.20), ("vision_mlp", 0.45)])),
            ],
        ),
        (
            "r1_vision",
            [
                ("r1-v35", _channels("r1_vision", [("vision_mlp", 0.35)])),
                ("r1-v45", _channels("r1_vision", [("vision_mlp", 0.45)])),
                ("r1-v60", _channels("r1_vision", [("vision_mlp", 0.60)])),
            ],
        ),
        (
            "r1_group32",
            [
                (
                    "r1-t10-v25-g32",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.10},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.25},
                        {"method": "world_remove"},
                        {"method": "compact_embedding"},
                        {"method": "bnb", "scope": "text", "bits": 4},
                        {
                            "method": "hqq",
                            "scope": "vision_blocks",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {
                            "method": "hqq",
                            "scope": "action_mlp",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {"method": "bnb", "scope": "action_attention", "bits": 4},
                        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
                        {"method": "bnb", "scope": "merger", "bits": 4},
                    ],
                ),
                (
                    "r1-t15-v35-g32",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.15},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.35},
                        {"method": "world_remove"},
                        {"method": "compact_embedding"},
                        {"method": "bnb", "scope": "text", "bits": 4},
                        {
                            "method": "hqq",
                            "scope": "vision_blocks",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {
                            "method": "hqq",
                            "scope": "action_mlp",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {"method": "bnb", "scope": "action_attention", "bits": 4},
                        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
                        {"method": "bnb", "scope": "merger", "bits": 4},
                    ],
                ),
                (
                    "r1-t20-v45-g32",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.20},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.45},
                        {"method": "world_remove"},
                        {"method": "compact_embedding"},
                        {"method": "bnb", "scope": "text", "bits": 4},
                        {
                            "method": "hqq",
                            "scope": "vision_blocks",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {
                            "method": "hqq",
                            "scope": "action_mlp",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 32,
                            "backend": "pytorch",
                        },
                        {"method": "bnb", "scope": "action_attention", "bits": 4},
                        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
                        {"method": "bnb", "scope": "merger", "bits": 4},
                    ],
                ),
            ],
        ),
    ]
    candidates: list[Stage2Candidate] = []
    seq = 0
    for b_idx, (fam, chain) in enumerate(specs):
        parent_id = "root"
        for a_idx, (cid, changes) in enumerate(chain):
            candidates.append(
                Stage2Candidate(
                    cell_id=cid,
                    branch=b_idx,
                    attempt=a_idx,
                    parent_id=parent_id,
                    seq=seq,
                    family=fam,
                    recipe={"id": cid, "family": fam, "changes": changes},
                )
            )
            parent_id = cid
            seq += 1
    return candidates


def _g32_changes(text_frac: float, vision_frac: float) -> list[dict[str, Any]]:
    return [
        {"method": "channels", "scope": "text_mlp", "fraction": text_frac},
        {"method": "channels", "scope": "vision_mlp", "fraction": vision_frac},
        {"method": "world_remove"},
        {"method": "compact_embedding"},
        {"method": "bnb", "scope": "text", "bits": 4},
        {
            "method": "hqq",
            "scope": "vision_blocks",
            "bits": 2,
            "axis": 0,
            "group_size": 32,
            "backend": "pytorch",
        },
        {
            "method": "hqq",
            "scope": "action_mlp",
            "bits": 2,
            "axis": 0,
            "group_size": 32,
            "backend": "pytorch",
        },
        {"method": "bnb", "scope": "action_attention", "bits": 4},
        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
        {"method": "bnb", "scope": "merger", "bits": 4},
    ]


def _chain(
    specs: list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]],
    start_seq: int = 0,
) -> list[Stage2Candidate]:
    candidates: list[Stage2Candidate] = []
    seq = start_seq
    for b_idx, (fam, chain) in enumerate(specs):
        parent_id = "root"
        for a_idx, (cid, changes) in enumerate(chain):
            candidates.append(
                Stage2Candidate(
                    cell_id=cid,
                    branch=b_idx,
                    attempt=a_idx,
                    parent_id=parent_id,
                    seq=seq,
                    family=fam,
                    recipe={"id": cid, "family": fam, "changes": changes},
                )
            )
            parent_id = cid
            seq += 1
    return candidates


def build_round2_candidates() -> list[Stage2Candidate]:
    """Fresh 2-branch x 3-attempt refinements around the Round 1 winner (r1-t20-v45-g32).

    Round 1 selected arm: 3/5 smoke passes on the g32 branch, best `r1-t20-v45-g32`
    (`8/10`, `853.1 MB`, dev `83/100`). Round 2 tightens pruning on that base.
    All cell IDs are new (`r2-*`); no committed measurements exist for them.
    """
    return _chain(
        [
            (
                "r2_tight",
                [
                    ("r2-t25-v50-g32", _g32_changes(0.25, 0.50)),
                    ("r2-t30-v55-g32", _g32_changes(0.30, 0.55)),
                    ("r2-t30-v60-g32", _g32_changes(0.30, 0.60)),
                ],
            ),
            (
                "r2_mix",
                [
                    ("r2-t20-v50-g32", _g32_changes(0.20, 0.50)),
                    ("r2-t25-v45-g32", _g32_changes(0.25, 0.45)),
                    ("r2-t25-v55-g32", _g32_changes(0.25, 0.55)),
                ],
            ),
        ]
    )


def make_evaluator(
    policy_tag: str,
    round_id: str,
    stage_journal: Path,
    recipes_dir: Path,
) -> Callable[[Stage2Candidate], dict[str, Any]]:
    """Wrap the compression subprocess runner with GPU backoff and stage journaling."""

    def evaluate(candidate: Stage2Candidate) -> dict[str, Any]:
        for _ in range(3):
            if not gpu_tight():
                break
            time.sleep(120)
        res = run_compression_subprocess(candidate.recipe, "smoke", recipes_dir)
        res["policy"] = policy_tag
        res["round_id"] = round_id
        res["machine"] = gpu_name()
        res["recorded_utc"] = datetime.now(UTC).isoformat()
        stage_journal.parent.mkdir(parents=True, exist_ok=True)
        with stage_journal.open("a") as f:
            f.write(json.dumps(res) + "\n")
        return res

    return evaluate


def run_arm(
    candidates: list[Stage2Candidate],
    policy: Any,
    policy_tag: str,
    probe_budget: int,
    round_id: str,
    journals: dict[str, Path],
    recipes_dir: Path,
) -> dict[str, Any]:
    """Run one arm (control or selected) on the round grid with a fixed probe budget."""
    question = LiveQwenQuestion(
        candidates=candidates,
        evaluator=make_evaluator(policy_tag, round_id, journals["stage"], recipes_dir),
        journal_path=journals[policy_tag],
        max_parallelism=4,
        smoke_gate=SMOKE_GATE,
    )
    sim = policy.solve(question, budget=probe_budget)
    return {
        "policy": policy_tag,
        "probes_used": sim.total_probes,
        "decision_rounds": sim.decision_rounds,
        "best_smoke_reduction": sim.best_score,
        "results": list(question._cache.values()),
    }


def promote_to_development(
    arm_result: dict[str, Any],
    recipes: dict[str, dict[str, Any]],
    recipes_dir: Path,
    stage_journal: Path,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    """Promote smoke passes (>=SMOKE_GATE/10) to development (100 trials)."""
    passing = [
        r
        for r in arm_result["results"]
        if r.get("status") == "completed" and r.get("successes", 0) >= SMOKE_GATE
    ]
    passing.sort(key=lambda r: (r["resident_bytes"], -r["successes"]))
    dev_results: list[dict[str, Any]] = []
    for rec in passing[:top_k]:
        res = run_compression_subprocess(recipes[rec["candidate"]], "development", recipes_dir)
        res["policy"] = arm_result["policy"]
        res["machine"] = gpu_name()
        res["recorded_utc"] = datetime.now(UTC).isoformat()
        with stage_journal.open("a") as f:
            f.write(json.dumps(res) + "\n")
        dev_results.append(res)
    return dev_results


def world_from_stage_journal(
    stage_journal: Path,
    candidates: list[Stage2Candidate],
    world_id: str,
    smoke_gate: int = SMOKE_GATE,
) -> tuple[DiscoveryTree, list[dict[str, Any]]]:
    """Build a replay world from one live round's smoke records (first record per cell).

    Replicate smoke runs of one cell are not pooled: each run is its own 10-trial smoke
    measurement and the world keeps the first.
    """
    seen: dict[str, dict[str, Any]] = {}
    for line in stage_journal.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("phase") != "smoke" or rec.get("status") != "completed":
            continue
        seen.setdefault(rec["candidate"], rec)
    by_id = {c.cell_id: c for c in candidates}
    nodes: list[TreeNode] = []
    records: list[dict[str, Any]] = []
    for cid, rec in seen.items():
        cand = by_id[cid]
        succ = int(rec["successes"])
        episodes = int(rec["episodes"])
        fail_class = gate_fail_class(succ, smoke_gate)
        passed = fail_class == "ok"
        reduction = max(0.0, 1.0 - float(rec["resident_bytes"]) / B16_RESIDENT_BYTES)
        nodes.append(
            TreeNode(
                cell_id=cid,
                branch=cand.branch,
                attempt=cand.attempt,
                parent_id=cand.parent_id,
                seq=len(nodes),
                score=reduction if passed else 0.0,
                evaluated=True,
                valid=passed,
                fail_class=fail_class,
                error=None if passed else f"smoke={succ}<{smoke_gate}",
                mechanism_family=cand.family,
                summary=f"{cid}: smoke={succ}/{episodes}, red={reduction:.3f}",
                cycle=0,
                successes=succ,
                episodes=episodes,
            )
        )
        records.append(
            {
                "candidate": cid,
                "parent": cand.parent_id,
                "policy": rec.get("policy"),
                "smoke_successes": succ,
                "smoke_episodes": episodes,
                "fail_class": fail_class,
                "resident_bytes": int(rec["resident_bytes"]),
                "machine": rec.get("machine"),
                "recorded_utc": rec.get("recorded_utc"),
            }
        )
    return DiscoveryTree(world_id=world_id, baseline_score=0.0, nodes=nodes), records
