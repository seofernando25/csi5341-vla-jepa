"""Stage 2 boundary grid and its committed-measurement replay world.

The grid spans the restricted-vocabulary bases T1
(`deploy-nf4-vision-text-channels-restricted`) and T2
(`deploy-hqq2-vision-action-restricted`) of the Study B compression track. Cell IDs
keep their recorded `c1`/`c2` spelling because the committed measurements under
`studies/program-v1/measurements/s2-*` use them. `stage2_smoke_world` reads only
committed 10-trial smoke `run.json` files, so the replay benchmark stays CPU-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.api import gate_fail_class
from rsi.runner import B16_RESIDENT_BYTES
from rsi.tree import DiscoveryTree, TreeNode

MEASUREMENTS_DIR = Path("studies/program-v1/measurements")
T2_BASE_ID = "deploy-hqq2-vision-action-restricted"


@dataclass(frozen=True)
class Stage2Candidate:
    """One candidate compression recipe in the Stage 2 search grid."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str
    seq: int
    family: str
    recipe: dict[str, Any]


def _t2_base_changes() -> list[dict[str, Any]]:
    """Restricted-vocabulary HQQ2 base (T2): drops the world model and compacts the embedding."""
    return [
        {"method": "world_remove"},
        {"method": "compact_embedding"},
        {"method": "bnb", "scope": "text", "bits": 4},
        {
            "method": "hqq",
            "scope": "vision_blocks",
            "bits": 2,
            "axis": 0,
            "group_size": 64,
            "backend": "pytorch",
        },
        {
            "method": "hqq",
            "scope": "action_mlp",
            "bits": 2,
            "axis": 0,
            "group_size": 64,
            "backend": "pytorch",
        },
        {"method": "bnb", "scope": "action_attention", "bits": 4},
        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
        {"method": "bnb", "scope": "merger", "bits": 4},
    ]


def _t1_base_changes() -> list[dict[str, Any]]:
    """Restricted-vocabulary NF4 channel-pruned base (T1)."""
    return [
        {"method": "world_remove"},
        {"method": "compact_embedding"},
        {"method": "channels", "scope": "text_mlp", "fraction": 0.25},
        {"method": "channels", "scope": "vision_mlp", "fraction": 0.50},
        {"method": "bnb", "scope": "text", "bits": 4},
        {"method": "bnb", "scope": "vision_blocks", "bits": 4},
        {"method": "bnb", "scope": "merger", "bits": 4},
        {"method": "bnb", "scope": "action", "bits": 4},
    ]


def _ch(scope: str, fraction: float) -> dict[str, Any]:
    return {"method": "channels", "scope": scope, "fraction": fraction}


def _t2(*extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(e) for e in extra] + _t2_base_changes()


def _t2_layers(count: int) -> list[dict[str, Any]]:
    return _t2({"method": "layers", "scope": "text", "count": count, "criterion": "cka"})


def build_stage2_candidates() -> list[Stage2Candidate]:
    """Build the 5-branch x 3-attempt boundary grid between T1 (838 MB) and T2 (977 MB)."""
    specs: list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]] = [
        (
            "c2_text_mlp",
            [
                ("s2-c2-tmlp10", _t2(_ch("text_mlp", 0.10))),
                ("s2-c2-tmlp20", _t2(_ch("text_mlp", 0.20))),
                ("s2-c2-tmlp30", _t2(_ch("text_mlp", 0.30))),
            ],
        ),
        (
            "c2_vision_mlp",
            [
                ("s2-c2-vmlp25", _t2(_ch("vision_mlp", 0.25))),
                ("s2-c2-vmlp50", _t2(_ch("vision_mlp", 0.50))),
                ("s2-c2-vmlp65", _t2(_ch("vision_mlp", 0.65))),
            ],
        ),
        (
            "c2_joint_channels",
            [
                ("s2-c2-t10-v25", _t2(_ch("text_mlp", 0.10), _ch("vision_mlp", 0.25))),
                ("s2-c2-t15-v35", _t2(_ch("text_mlp", 0.15), _ch("vision_mlp", 0.35))),
                ("s2-c2-t20-v50", _t2(_ch("text_mlp", 0.20), _ch("vision_mlp", 0.50))),
            ],
        ),
        (
            "c2_text_layers_cka",
            [
                ("s2-c2-tlay2", _t2_layers(2)),
                ("s2-c2-tlay4", _t2_layers(4)),
                ("s2-c2-tlay6", _t2_layers(6)),
            ],
        ),
        (
            "c1_hqq2_hybrid",
            [
                (
                    "s2-c1-hqq2-act",
                    [
                        *_t1_base_changes()[:-1],
                        {
                            "method": "hqq",
                            "scope": "action_mlp",
                            "bits": 2,
                            "axis": 0,
                            "group_size": 64,
                            "backend": "pytorch",
                        },
                        {"method": "bnb", "scope": "action_attention", "bits": 4},
                        {"method": "bnb", "scope": "action_interfaces", "bits": 4},
                    ],
                ),
                (
                    "s2-c1-hqq2-vis-act",
                    _t2(_ch("text_mlp", 0.25), _ch("vision_mlp", 0.50)),
                ),
                (
                    "s2-c1-hqq2-tlay2",
                    _t2(
                        {"method": "layers", "scope": "text", "count": 2, "criterion": "cka"},
                        _ch("text_mlp", 0.25),
                        _ch("vision_mlp", 0.50),
                    ),
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


def _text_mlp_removed(run: dict[str, Any]) -> int | None:
    """Removed text-MLP channels per layer, or None if the run prunes anything else."""
    channels = [t for t in run.get("transformations", []) if t.get("method") == "channels"]
    if not channels or any(not t["module"].startswith("text.") for t in channels):
        return None
    removed = {t["original_width"] - t["kept_width"] for t in channels}
    return removed.pop() if len(removed) == 1 else None


def stage2_smoke_world(
    measurements_dir: str | Path = MEASUREMENTS_DIR,
    smoke_gate: int = 8,
    world_id: str = "qwen-stage2-smoke",
) -> tuple[DiscoveryTree, list[dict[str, Any]]]:
    """Build a replay world from the committed Stage 2 smoke measurements (10 trials each).

    Nodes are the grid candidates that have a smoke `run.json`; edges are the grid's
    parent->child chains. Score is gate-qualified size reduction versus B16. A node
    without a size measurement gets bytes extrapolated linearly in removed
    text-MLP channels from its measured parent and the T2 base; its record says `derived`.
    Returns the tree and one provenance record per node.
    """
    root = Path(measurements_dir)

    def load(cid: str, phase: str) -> dict[str, Any] | None:
        path = root / cid / phase / "run.json"
        return json.loads(path.read_text()) if path.exists() else None

    def size_bytes(cid: str) -> float | None:
        run = load(cid, "size")
        return None if run is None else float(run["summary"]["resident_tensor_storage_bytes"])

    nodes: list[TreeNode] = []
    records: list[dict[str, Any]] = []
    measured: dict[str, dict[str, Any]] = {}
    for cand in build_stage2_candidates():
        smoke = load(cand.cell_id, "smoke")
        if smoke is None or (cand.parent_id != "root" and cand.parent_id not in measured):
            continue
        measured[cand.cell_id] = smoke
        resident = size_bytes(cand.cell_id)
        source = "measured"
        if resident is None:
            base = size_bytes(T2_BASE_ID)
            parent = size_bytes(cand.parent_id)
            removed = _text_mlp_removed(smoke)
            parent_removed = _text_mlp_removed(measured.get(cand.parent_id, {}))
            if None in (base, parent, removed, parent_removed):
                raise ValueError(f"{cand.cell_id}: no size measurement and no linear derivation")
            resident = base - (base - parent) * removed / parent_removed
            source = "derived"
        succ = int(smoke["summary"]["successes"])
        episodes = int(smoke["summary"]["episodes"])
        fail_class = gate_fail_class(succ, smoke_gate)
        passed = fail_class == "ok"
        reduction = max(0.0, 1.0 - resident / B16_RESIDENT_BYTES)
        nodes.append(
            TreeNode(
                cell_id=cand.cell_id,
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
                summary=f"{cand.cell_id}: smoke={succ}/{episodes}, red={reduction:.3f}",
                cycle=0,
                successes=succ,
                episodes=episodes,
            )
        )
        records.append(
            {
                "candidate": cand.cell_id,
                "parent": cand.parent_id,
                "smoke_successes": succ,
                "smoke_episodes": episodes,
                "fail_class": fail_class,
                "resident_bytes": round(resident),
                "resident_bytes_source": source,
                "recorded_at": smoke["environment"]["recorded_at"],
            }
        )
    return DiscoveryTree(world_id=world_id, baseline_score=0.0, nodes=nodes), records
