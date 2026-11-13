"""Discovery tree representation and deterministic replay simulator (Dream-RSI Section 2)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.api import CellMeta, Observation, gate_fail_class


@dataclass(frozen=True)
class TreeNode:
    """Internal full-tree record (hidden from policy until revealed)."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str | None
    seq: int
    score: float | None
    evaluated: bool
    valid: bool
    fail_class: str
    error: str | None
    mechanism_family: str
    summary: str
    cycle: int
    tags: tuple[str, ...] = ()
    successes: int | None = None
    episodes: int | None = None


class DiscoveryTree:
    """Recorded discovery tree T_i distilled into a branch x attempt replay grid."""

    def __init__(
        self,
        world_id: str,
        baseline_score: float,
        nodes: list[TreeNode],
    ) -> None:
        self.world_id = world_id
        self.baseline_score = float(baseline_score)
        self.nodes: dict[str, TreeNode] = {n.cell_id: n for n in nodes}
        self.order: list[str] = [n.cell_id for n in nodes]
        self.children: dict[str, list[str]] = {}
        self.roots: list[str] = []
        for n in nodes:
            if n.parent_id is None or n.parent_id == "root":
                self.roots.append(n.cell_id)
            else:
                self.children.setdefault(n.parent_id, []).append(n.cell_id)

    @property
    def num_nonroot(self) -> int:
        return len(self.nodes)

    @property
    def best_score(self) -> float:
        scores = [n.score for n in self.nodes.values() if n.score is not None]
        return max(scores) if scores else self.baseline_score

    @property
    def max_gain(self) -> float:
        """S_i: best improvement over baseline_score (or raw score if baseline == 0)."""
        gains = [
            max(0.0, n.score - self.baseline_score)
            for n in self.nodes.values()
            if n.score is not None and n.fail_class == "ok"
        ]
        return max(gains) if gains else 0.0

    def to_observation(self, node: TreeNode) -> Observation:
        parent = self.nodes.get(node.parent_id) if node.parent_id not in (None, "root") else None
        delta_base = (node.score - self.baseline_score) if node.score is not None else None
        if node.score is not None and parent is not None and parent.score is not None:
            delta_parent = node.score - parent.score
        elif node.score is not None and node.parent_id in (None, "root"):
            delta_parent = node.score - self.baseline_score
        else:
            delta_parent = None
        return Observation(
            cell_id=node.cell_id,
            branch=node.branch,
            attempt=node.attempt,
            parent_id=node.parent_id,
            score=node.score,
            evaluated=node.evaluated,
            valid=node.valid,
            fail_class=node.fail_class,
            error=node.error,
            delta_vs_baseline=delta_base,
            delta_vs_parent=delta_parent,
            n_valid=node.successes if node.successes is not None else int(node.valid),
            n_total=node.episodes if node.episodes is not None else int(node.evaluated),
            mechanism_family=node.mechanism_family,
            summary=node.summary,
            cycle=node.cycle,
        )

    def to_meta(self, node: TreeNode) -> CellMeta:
        return CellMeta(
            cell_id=node.cell_id,
            branch=node.branch,
            attempt=node.attempt,
            parent_id=node.parent_id,
            seq=node.seq,
            tags=node.tags,
        )


class ReplayQuestion:
    """Deterministic prefix-only replay environment over a DiscoveryTree."""

    def __init__(self, tree: DiscoveryTree, max_parallelism: int = 4) -> None:
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be >= 1")
        self.tree = tree
        self.baseline_score = tree.baseline_score
        self.max_parallelism = max_parallelism
        self._revealed: dict[str, Observation] = {}
        self._revealed_order: list[str] = []
        self.best_so_far: float = tree.baseline_score
        self.budget_spent: int = 0
        self.decision_rounds: int = 0
        self.effective_sequential_rounds: int = 0

    def reset(self) -> None:
        self._revealed.clear()
        self._revealed_order.clear()
        self.best_so_far = self.tree.baseline_score
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def observed(self) -> dict[str, Observation]:
        return dict(self._revealed)

    def legal_roots(self) -> list[str]:
        return [cid for cid in self.tree.roots if cid not in self._revealed]

    def opened_branches(self) -> list[int]:
        seen: list[int] = []
        seen_set: set[int] = set()
        for cid in self._revealed_order:
            b = self._revealed[cid].branch
            if b not in seen_set:
                seen_set.add(b)
                seen.append(b)
        return seen

    def legal_actions(self) -> list[str]:
        actions = self.legal_roots()
        for cid in self._revealed_order:
            for child_id in self.tree.children.get(cid, ()):
                if child_id not in self._revealed:
                    actions.append(child_id)
        return actions

    def meta(self, cell_id: str) -> CellMeta:
        if cell_id not in self.tree.nodes:
            raise KeyError(f"unknown cell_id: {cell_id!r}")
        return self.tree.to_meta(self.tree.nodes[cell_id])

    def probe_batch(
        self,
        cells: list[str],
        on_reveal: Callable[[Observation], Any] | None = None,
    ) -> list[Observation]:
        if not cells:
            return []
        if len(cells) > self.max_parallelism:
            raise ValueError(
                f"batch size {len(cells)} exceeds max_parallelism={self.max_parallelism}"
            )
        if len(set(cells)) != len(cells):
            raise ValueError("batch contains duplicate cell ids")
        legal = set(self.legal_actions())
        for cid in cells:
            if cid not in legal:
                raise ValueError(f"illegal cell in batch: {cid!r}")

        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(len(cells) / self.max_parallelism)

        revealed_now: list[Observation] = []
        for cid in cells:
            node = self.tree.nodes[cid]
            obs = self.tree.to_observation(node)
            self._revealed[cid] = obs
            self._revealed_order.append(cid)
            self.budget_spent += 1
            if obs.score is not None and obs.fail_class == "ok" and obs.score > self.best_so_far:
                self.best_so_far = obs.score
            revealed_now.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        return revealed_now


def _normalize_fail_class(record: dict[str, Any]) -> tuple[str, str | None, bool, bool]:
    status = record.get("status", "unknown")
    if status == "ok":
        return "ok", None, True, True
    if status == "implementation_failure":
        return (
            "implementation_failure",
            record.get("summary") or "implementation failure",
            True,
            False,
        )
    if status == "runtime_failure":
        return "runtime_failure", record.get("summary") or "runtime failure", True, False
    if status == "interrupted":
        return "interrupted", record.get("summary") or "interrupted", False, False
    return str(status), record.get("summary"), False, False


def from_attempts_jsonl(
    path: str | Path,
    baseline_score: float = -0.4074,
    world_id: str = "libero-smolvlm-full",
) -> DiscoveryTree:
    """Load a DiscoveryTree from a Dream-RSI attempts.jsonl search journal."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return _records_to_tree(records, baseline_score=baseline_score, world_id=world_id)


def partition_attempts_by_root_cycle(
    path: str | Path,
    baseline_score: float = -0.4074,
) -> list[DiscoveryTree]:
    """Partition attempts.jsonl into independent replay worlds grouped by root creation cycle."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {r["attempt_id"]: r for r in records}

    def root_cycle(r: dict[str, Any]) -> int:
        cur = r
        while cur["parent"] not in (None, "root") and cur["parent"] in by_id:
            cur = by_id[cur["parent"]]
        return int(cur.get("cycle", 0))

    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        buckets.setdefault(root_cycle(r), []).append(r)

    return [
        _records_to_tree(
            buckets[cyc],
            baseline_score=baseline_score,
            world_id=f"libero-smolvlm-cycle{cyc}",
        )
        for cyc in sorted(buckets)
    ]


def split_cycle_branches(
    trees: list[DiscoveryTree], min_nodes: int = 2
) -> tuple[list[DiscoveryTree], list[DiscoveryTree]]:
    """Split cycle trees into per-root-branch worlds (disjoint node sets).

    Provenance: each root of a SmolVLM cycle tree starts an independent search line,
    so a root's descendant subtree replays standalone. Branches below `min_nodes`
    carry no refine-vs-abandon decision and pool per cycle into a fragments world.
    Returns (branch_worlds, fragment_worlds).
    """
    branches: list[DiscoveryTree] = []
    fragments: list[DiscoveryTree] = []
    for tree in trees:
        small: list[TreeNode] = []
        for root_id in tree.roots:
            sub = [cid for cid in tree.order if cid in _subtree_ids(tree, root_id)]
            nodes = [tree.nodes[cid] for cid in sub]
            if len(nodes) >= min_nodes:
                branches.append(
                    DiscoveryTree(f"{tree.world_id}-branch{tree.nodes[root_id].branch}", 0.0, nodes)
                )
            else:
                small.extend(nodes)
        if small:
            fragments.append(DiscoveryTree(f"{tree.world_id}-fragments", 0.0, small))
    for t in branches + fragments:
        t.baseline_score = trees[0].baseline_score if trees else 0.0
    return branches, fragments


def _subtree_ids(tree: DiscoveryTree, root_id: str) -> set[str]:
    out = {root_id}
    stack = [root_id]
    while stack:
        for child in tree.children.get(stack.pop(), ()):
            out.add(child)
            stack.append(child)
    return out


def from_live_records(
    records: list[dict[str, Any]],
    world_id: str,
    provenance: str,
    baseline_score: float = 0.0,
    smoke_gate: int = 8,
) -> DiscoveryTree:
    """Build a replay world from a live round's compact `world_records`.

    Provenance (recorded in `provenance`, e.g. round id + journal path): branch groups
    cells by root ancestor, attempt counts depth below that root; score is
    gate-qualified size reduction versus B16. Only 10-trial smoke records enter;
    development promotions never do.
    """
    from rsi.qwen_search import B16_RESIDENT_BYTES

    parent = {r["candidate"]: r.get("parent") or "root" for r in records}

    def root_of(cid: str) -> str:
        cur = cid
        while parent.get(cur) not in (None, "root"):
            cur = parent[cur]
        return cur

    def depth(cid: str) -> int:
        d, cur = 0, cid
        while parent.get(cur) not in (None, "root"):
            cur = parent[cur]
            d += 1
        return d

    branch_of: dict[str, int] = {}
    for cid in parent:
        r = root_of(cid)
        if r not in branch_of:
            branch_of[r] = len(branch_of)
    nodes: list[TreeNode] = []
    for seq, r in enumerate(records):
        cid = r["candidate"]
        succ = int(r["smoke_successes"])
        fail_class = gate_fail_class(succ, smoke_gate)
        passed = fail_class == "ok"
        reduction = max(0.0, 1.0 - float(r["resident_bytes"]) / B16_RESIDENT_BYTES)
        nodes.append(
            TreeNode(
                cell_id=cid,
                branch=branch_of[root_of(cid)],
                attempt=depth(cid),
                parent_id=parent[cid],
                seq=seq,
                score=reduction if passed else 0.0,
                evaluated=True,
                valid=passed,
                fail_class=fail_class,
                error=None if passed else f"smoke={succ}<{smoke_gate}",
                mechanism_family=provenance,
                summary=f"{cid}: smoke={succ}/{r['smoke_episodes']}, red={reduction:.3f}",
                cycle=0,
                successes=succ,
                episodes=int(r["smoke_episodes"]),
            )
        )
    return DiscoveryTree(world_id=world_id, baseline_score=baseline_score, nodes=nodes)


def _records_to_tree(
    records: list[dict[str, Any]],
    baseline_score: float,
    world_id: str,
) -> DiscoveryTree:
    branch_map: dict[str, int] = {}
    attempt_map: dict[str, int] = {}
    next_branch = 0
    nodes: list[TreeNode] = []

    for seq, rec in enumerate(records):
        cid = rec["attempt_id"]
        parent = rec.get("parent")
        if parent in (None, "root") or parent not in branch_map:
            branch = next_branch
            next_branch += 1
            attempt = 0
            norm_parent = "root"
        else:
            branch = branch_map[parent]
            attempt = attempt_map[parent] + 1
            norm_parent = parent
        branch_map[cid] = branch
        attempt_map[cid] = attempt

        fail_class, error, evaluated, valid = _normalize_fail_class(rec)
        proposal = rec.get("proposal") or {}
        fam = str(proposal.get("mechanism_family") or "")
        tags: list[str] = [f"cycle:{rec.get('cycle', 0)}"]
        if proposal.get("relation_to_parent"):
            tags.append(str(proposal["relation_to_parent"]))
        nodes.append(
            TreeNode(
                cell_id=cid,
                branch=branch,
                attempt=attempt,
                parent_id=norm_parent,
                seq=seq,
                score=float(rec["score"]) if rec.get("score") is not None else None,
                evaluated=evaluated,
                valid=valid,
                fail_class=fail_class,
                error=error,
                mechanism_family=fam,
                summary=str(rec.get("summary") or ""),
                cycle=int(rec.get("cycle", 0)),
                tags=tuple(tags),
            )
        )
    return DiscoveryTree(world_id=world_id, baseline_score=baseline_score, nodes=nodes)


def make_synthetic_tree(
    world_id: str,
    branches: list[list[tuple[float | None, str]]],
    baseline_score: float = 0.0,
) -> DiscoveryTree:
    """Create a synthetic DiscoveryTree from [(score, fail_class), ...] branch chains."""
    nodes: list[TreeNode] = []
    seq = 0
    for b_idx, chain in enumerate(branches):
        parent_id = "root"
        for a_idx, (score, fail_class) in enumerate(chain):
            cid = f"b{b_idx}_a{a_idx}"
            ok = fail_class == "ok"
            nodes.append(
                TreeNode(
                    cell_id=cid,
                    branch=b_idx,
                    attempt=a_idx,
                    parent_id=parent_id,
                    seq=seq,
                    score=score,
                    evaluated=fail_class != "interrupted",
                    valid=ok and score is not None,
                    fail_class=fail_class,
                    error=None if ok else fail_class,
                    mechanism_family=f"family_{b_idx}",
                    summary=f"synthetic {cid} ({fail_class})",
                    cycle=0,
                )
            )
            parent_id = cid
            seq += 1
    return DiscoveryTree(world_id=world_id, baseline_score=baseline_score, nodes=nodes)


def from_compression_summary(
    path: str | Path = "studies/compression/summary.csv",
    smoke_gate: int = 8,
    dev_gate: int = 77,
) -> tuple[DiscoveryTree, DiscoveryTree]:
    """Distill Study B's Qwen compression summary.csv into smoke and dev DiscoveryTrees.

    Each family forms a branch ordered from mildest (largest bytes) to most aggressive
    (smallest bytes). Score is gate-qualified size reduction: 1 - bytes / B16_bytes.
    These chains are synthesized from a grid study, not from a recorded search order. A smoke
    count one below its gate is `gate_near_miss`; the development world has no near-miss class.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    b16 = next(r for r in rows if r["candidate"] == "B16")
    b16_bytes = float(b16["resident_tensor_storage_bytes"])

    def build_tree(world_id: str, col: str, gate: int, near_miss: int) -> DiscoveryTree:
        filtered = [
            r
            for r in rows
            if r["candidate"] != "B16" and r[col] and r["resident_tensor_storage_bytes"]
        ]
        by_family: dict[str, list[dict[str, str]]] = {}
        for r in filtered:
            by_family.setdefault(r["family"], []).append(r)

        nodes: list[TreeNode] = []
        seq = 0
        for b_idx, fam in enumerate(sorted(by_family)):
            chain = sorted(
                by_family[fam],
                key=lambda r: (-float(r["resident_tensor_storage_bytes"]), r["candidate"]),
            )
            parent_id = "root"
            for a_idx, r in enumerate(chain):
                cid = r["candidate"]
                succ = int(r[col])
                passed = succ >= gate
                reduction = max(0.0, 1.0 - float(r["resident_tensor_storage_bytes"]) / b16_bytes)
                score = reduction if passed else 0.0
                fail_class = gate_fail_class(succ, gate, near_miss)
                nodes.append(
                    TreeNode(
                        cell_id=cid,
                        branch=b_idx,
                        attempt=a_idx,
                        parent_id=parent_id,
                        seq=seq,
                        score=score,
                        evaluated=True,
                        valid=passed,
                        fail_class=fail_class,
                        error=None if passed else f"{col}={succ}<{gate}",
                        mechanism_family=fam,
                        summary=f"{cid}: {col}={succ}, red={reduction:.3f}",
                        cycle=0,
                        successes=succ,
                        episodes=int(r[col.replace("successes", "episodes")]),
                    )
                )
                parent_id = cid
                seq += 1
        return DiscoveryTree(world_id=world_id, baseline_score=0.0, nodes=nodes)

    return (
        build_tree("qwen-compression-smoke", "smoke_successes", smoke_gate, near_miss=1),
        build_tree("qwen-compression-dev", "development_successes", dev_gate, near_miss=0),
    )


def compression_smoke_by_family(
    path: str | Path = "studies/compression/summary.csv",
    smoke_gate: int = 8,
    min_nodes: int = 4,
) -> list[DiscoveryTree]:
    """Split the Qwen compression smoke tree into per-family worlds (disjoint nodes).

    Provenance: each family chain in `from_compression_summary` is synthesized from
    the Study B grid (mildest to most aggressive), not from a recorded search order,
    so a family replays standalone. Families below `min_nodes` carry no
    refine-vs-abandon decision and are excluded (kept in the pooled smoke world).
    Single-branch worlds only test stopping patience, stated as a caveat.
    """
    full = from_compression_summary(path, smoke_gate=smoke_gate)[0]
    by_family: dict[str, list[TreeNode]] = {}
    for cid in full.order:
        node = full.nodes[cid]
        by_family.setdefault(node.mechanism_family, []).append(node)
    worlds = []
    for fam in sorted(by_family):
        nodes = by_family[fam]
        if len(nodes) < min_nodes:
            continue
        worlds.append(DiscoveryTree(f"qwen-smoke-{_slug(fam)}", 0.0, nodes))
    return worlds


def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
