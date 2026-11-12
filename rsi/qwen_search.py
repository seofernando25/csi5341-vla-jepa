"""Stage 2: Live Dream-RSI Qwen quantization/pruning boundary search (between C1 and C2)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.api import CellMeta, Observation
from rsi.policies import OptimalPolicy

B16_RESIDENT_BYTES = 5532276494.0


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


def _c2_base_changes() -> list[dict[str, Any]]:
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


def _c1_base_changes() -> list[dict[str, Any]]:
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


def build_stage2_candidates() -> list[Stage2Candidate]:
    """Build 5-branch x 3-attempt boundary grid between C1 (838 MB) and C2 (978 MB)."""
    specs: list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]] = [
        (
            "c2_text_mlp",
            [
                (
                    "s2-c2-tmlp10",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.10},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-tmlp20",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.20},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-tmlp30",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.30},
                        *_c2_base_changes(),
                    ],
                ),
            ],
        ),
        (
            "c2_vision_mlp",
            [
                (
                    "s2-c2-vmlp25",
                    [
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.25},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-vmlp50",
                    [
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.50},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-vmlp65",
                    [
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.65},
                        *_c2_base_changes(),
                    ],
                ),
            ],
        ),
        (
            "c2_joint_channels",
            [
                (
                    "s2-c2-t10-v25",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.10},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.25},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-t15-v35",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.15},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.35},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-t20-v50",
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.20},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.50},
                        *_c2_base_changes(),
                    ],
                ),
            ],
        ),
        (
            "c2_text_layers_cka",
            [
                (
                    "s2-c2-tlay2",
                    [
                        {"method": "layers", "scope": "text", "count": 2, "criterion": "cka"},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-tlay4",
                    [
                        {"method": "layers", "scope": "text", "count": 4, "criterion": "cka"},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c2-tlay6",
                    [
                        {"method": "layers", "scope": "text", "count": 6, "criterion": "cka"},
                        *_c2_base_changes(),
                    ],
                ),
            ],
        ),
        (
            "c1_hqq2_hybrid",
            [
                (
                    "s2-c1-hqq2-act",
                    [
                        *_c1_base_changes()[:-1],
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
                    [
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.25},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.50},
                        *_c2_base_changes(),
                    ],
                ),
                (
                    "s2-c1-hqq2-tlay2",
                    [
                        {"method": "layers", "scope": "text", "count": 2, "criterion": "cka"},
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.25},
                        {"method": "channels", "scope": "vision_mlp", "fraction": 0.50},
                        *_c2_base_changes(),
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


def run_compression_subprocess(
    recipe: dict[str, Any],
    phase: str,
    recipes_dir: Path,
) -> dict[str, Any]:
    """Run one compression candidate in an isolated subprocess to prevent VRAM leaks."""
    recipes_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipes_dir / f"{recipe['id']}.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")

    run_json = Path("studies/compression/measurements") / recipe["id"] / phase / "run.json"
    wall = 0.0
    if not run_json.exists():
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.compression.study",
                "run",
                "--recipe",
                str(recipe_path),
                "--phase",
                phase,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0 or not run_json.exists():
            return {
                "candidate": recipe["id"],
                "phase": phase,
                "status": "runtime_failure",
                "error": (proc.stderr or proc.stdout)[-400:],
                "wall_seconds": round(wall, 2),
            }

    data = json.loads(run_json.read_text())
    summary = data.get("summary") or {}
    wall = wall or float(data.get("wall_seconds", 0.0))

    # Measure resident tensor storage bytes via --phase size if not yet measured
    resident_bytes = summary.get("resident_tensor_storage_bytes")
    if resident_bytes is None and phase in {"smoke", "development"}:
        size_json = Path("studies/compression/measurements") / recipe["id"] / "size" / "run.json"
        if not size_json.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.compression.study",
                    "run",
                    "--recipe",
                    str(recipe_path),
                    "--phase",
                    "size",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if size_json.exists():
            size_data = json.loads(size_json.read_text())
            resident_bytes = (size_data.get("summary") or {}).get("resident_tensor_storage_bytes")

    return {
        "candidate": recipe["id"],
        "phase": phase,
        "status": data.get("status", "ok"),
        "successes": int(summary.get("successes", 0)),
        "episodes": int(summary.get("episodes", 0)),
        "resident_bytes": float(resident_bytes or B16_RESIDENT_BYTES),
        "wall_seconds": round(wall, 2),
    }


class LiveQwenQuestion:
    """Dream-RSI question interface backed by live GPU compression rollouts."""

    def __init__(
        self,
        candidates: list[Stage2Candidate],
        evaluator: Callable[[Stage2Candidate], dict[str, Any]],
        journal_path: Path,
        max_parallelism: int = 4,
        smoke_gate: int = 8,
    ) -> None:
        self.candidates = {c.cell_id: c for c in candidates}
        self.roots = [c.cell_id for c in candidates if c.parent_id == "root"]
        self.children: dict[str, list[str]] = {}
        for c in candidates:
            if c.parent_id != "root":
                self.children.setdefault(c.parent_id, []).append(c.cell_id)
        self.evaluator = evaluator
        self.journal_path = journal_path
        self.baseline_score = 0.0
        self.max_parallelism = max_parallelism
        self.smoke_gate = smoke_gate

        self._cache: dict[str, dict[str, Any]] = {}
        if journal_path.exists():
            for line in journal_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["candidate"]] = rec

        self._revealed: dict[str, Observation] = {}
        self._revealed_order: list[str] = []
        self.best_so_far = 0.0
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def reset(self) -> None:
        self._revealed.clear()
        self._revealed_order.clear()
        self.best_so_far = 0.0
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def observed(self) -> dict[str, Observation]:
        return dict(self._revealed)

    def legal_roots(self) -> list[str]:
        return [cid for cid in self.roots if cid not in self._revealed]

    def opened_branches(self) -> list[int]:
        seen, seen_set = [], set()
        for cid in self._revealed_order:
            b = self._revealed[cid].branch
            if b not in seen_set:
                seen_set.add(b)
                seen.append(b)
        return seen

    def legal_actions(self) -> list[str]:
        actions = self.legal_roots()
        for cid in self._revealed_order:
            for child_id in self.children.get(cid, ()):
                if child_id not in self._revealed:
                    actions.append(child_id)
        return actions

    def meta(self, cell_id: str) -> CellMeta:
        c = self.candidates[cell_id]
        return CellMeta(
            cell_id=c.cell_id,
            branch=c.branch,
            attempt=c.attempt,
            parent_id=c.parent_id,
            seq=c.seq,
        )

    def probe_batch(
        self,
        cells: list[str],
        on_reveal: Callable[[Observation], Any] | None = None,
    ) -> list[Observation]:
        if not cells:
            return []
        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(len(cells) / self.max_parallelism)
        out: list[Observation] = []

        for cid in cells:
            c = self.candidates[cid]
            if cid in self._cache:
                res = self._cache[cid]
            else:
                res = self.evaluator(c)
                self._cache[cid] = res
                self.journal_path.parent.mkdir(parents=True, exist_ok=True)
                with self.journal_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")

            if res.get("status") != "completed":
                score = 0.0
                fail_class = "runtime_failure"
                err = str(res.get("error", "runtime failure"))
                passed = False
            else:
                succ = int(res["successes"])
                passed = succ >= self.smoke_gate
                red = max(0.0, 1.0 - float(res["resident_bytes"]) / B16_RESIDENT_BYTES)
                score = red if passed else 0.0
                fail_class = "ok" if passed else "gate_failed"
                err = None if passed else f"smoke={succ}<{self.smoke_gate}"

            parent_obs = self._revealed.get(c.parent_id)
            delta_parent = (
                score - parent_obs.score
                if parent_obs is not None and parent_obs.score is not None
                else score
            )
            obs = Observation(
                cell_id=cid,
                branch=c.branch,
                attempt=c.attempt,
                parent_id=c.parent_id,
                score=score,
                evaluated=True,
                valid=passed,
                fail_class=fail_class,
                error=err,
                delta_vs_baseline=score,
                delta_vs_parent=delta_parent,
                n_valid=int(res.get("successes", 0)),
                n_total=int(res.get("episodes", 10)),
                mechanism_family=c.family,
                summary=f"{cid}: status={res.get('status')} score={score:.4f}",
            )
            self._revealed[cid] = obs
            self._revealed_order.append(cid)
            self.budget_spent += 1
            if passed and score > self.best_so_far:
                self.best_so_far = score
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        return out


def run_stage2_live_search(
    smoke_budget: int = 10,
    promote_top_k: int = 1,
    beta: float = 0.6,
    output_path: Path = Path("studies/dream-rsi/qwen_stage2.json"),
) -> dict[str, Any]:
    """Run live Stage 2 Dream-RSI Qwen boundary search and promote top candidate to dev."""
    candidates = build_stage2_candidates()
    recipes_dir = Path("outputs/compression/stage2_recipes")
    journal_path = Path("outputs/compression/stage2_journal.jsonl")

    question = LiveQwenQuestion(
        candidates=candidates,
        evaluator=lambda c: run_compression_subprocess(c.recipe, "smoke", recipes_dir),
        journal_path=journal_path,
        max_parallelism=4,
        smoke_gate=8,
    )
    policy = OptimalPolicy({"beta": beta})
    sim = policy.solve(question, budget=smoke_budget)

    smoke_records = list(question._cache.values())
    passing_smoke = [
        r for r in smoke_records if r.get("status") == "completed" and r.get("successes", 0) >= 8
    ]
    passing_smoke.sort(key=lambda r: (r["resident_bytes"], -r["successes"]))

    dev_results: list[dict[str, Any]] = []
    cand_map = {c.cell_id: c for c in candidates}
    for rec in passing_smoke[:promote_top_k]:
        cid = rec["candidate"]
        dev_res = run_compression_subprocess(cand_map[cid].recipe, "development", recipes_dir)
        dev_results.append(dev_res)

    payload = {
        "protocol": "dream-rsi-qwen-stage2",
        "beta": beta,
        "smoke_budget": smoke_budget,
        "probes_used": sim.total_probes,
        "decision_rounds": sim.decision_rounds,
        "best_smoke_reduction": sim.best_score,
        "smoke_results": smoke_records,
        "development_promotions": dev_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
