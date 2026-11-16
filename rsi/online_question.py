"""Dream-RSI discovery question: a branch x attempt slot skeleton expanded online, one batch per probe."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rsi.api import (
    IMPLEMENTATION_FAILURE,
    CellMeta,
    GridPlan,
    Observation,
)
from rsi.expansion import (
    AgentOutputInvalid,
    DigestRegistry,
    NodeContext,
    NodeJob,
    expand_node,
)
from rsi.online_evidence import observation, revealed_nodes, tree_index
from rsi.runner import load_protocol
from rsi.sandbox import discovery_visibility
from rsi.tree import DiscoveryTree, TreeNode
from rsi.workspace import code_digest, commit_root, init_store


class OnlineDiscoveryQuestion:
    """Question whose `probe_batch` expands the chosen slots concurrently in their own workspaces."""

    def __init__(
        self, *, cycle_id: str, grid_plan: GridPlan, cycle_dir: Path, ctx: NodeContext
    ) -> None:
        self.cycle_id = cycle_id
        self.grid_plan = grid_plan
        self.cycle_dir = Path(cycle_dir)
        self.nodes_dir = self.cycle_dir / "nodes"
        self.ctx = ctx
        self.rounds_cap = grid_plan.rounds_cap
        self.max_parallelism = grid_plan.max_parallelism
        # A latency study scores speedup, so its identity baseline is 1.0, not 0.0.
        self.baseline_score = float(
            (load_protocol(ctx.study_dir).get("score") or {}).get("baseline", 0.0)
        )

        self._slots: dict[str, CellMeta] = {}
        self.roots: list[str] = []
        self.children: dict[str, list[str]] = {}
        seq = 0
        for b in range(grid_plan.max_branches):
            parent = "root"
            for a in range(grid_plan.max_attempts_per_branch + 1):
                cid = f"{cycle_id}-b{b}-a{a}"
                self._slots[cid] = CellMeta(
                    cell_id=cid, branch=b, attempt=a, parent_id=parent, seq=seq
                )
                if parent == "root":
                    self.roots.append(cid)
                else:
                    self.children.setdefault(parent, []).append(cid)
                parent = cid
                seq += 1

        self.store = init_store(self.cycle_dir)
        check_tool = discovery_visibility(self.ctx.study_dir).workspace_check
        self.root_commit = commit_root(self.store, check_tool)
        self.registry = DigestRegistry({code_digest(self.store, self.root_commit): "root"})
        self.node_records: dict[str, dict[str, Any]] = {}
        self.agent_records: list[dict[str, Any]] = []

        self._revealed: dict[str, Observation] = {}
        self._revealed_order: list[str] = []
        self.best_so_far = self.baseline_score
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def reset(self) -> None:
        self._revealed.clear()
        self._revealed_order.clear()
        self.best_so_far = self.baseline_score
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def observed(self) -> dict[str, Observation]:
        return dict(self._revealed)

    def _round_cap_reached(self) -> bool:
        return self.decision_rounds >= self.rounds_cap

    def legal_roots(self) -> list[str]:
        if self._round_cap_reached():
            return []
        return [cid for cid in self.roots if cid not in self._revealed]

    def opened_branches(self) -> list[int]:
        seen: list[int] = []
        for cid in self._revealed_order:
            b = self._revealed[cid].branch
            if b not in seen:
                seen.append(b)
        return seen

    def legal_actions(self) -> list[str]:
        if self._round_cap_reached():
            return []
        actions = self.legal_roots()
        for cid in self._revealed_order:
            for child_id in self.children.get(cid, ()):
                if child_id not in self._revealed:
                    actions.append(child_id)
        return actions

    def meta(self, cell_id: str) -> CellMeta:
        if cell_id not in self._slots:
            raise KeyError(f"unknown cell_id: {cell_id!r}")
        return self._slots[cell_id]

    def _parent_commit(self, parent_id: str) -> str | None:
        if parent_id == "root":
            return self.root_commit
        record = self.node_records.get(parent_id)
        return record.get("commit") if record else None

    def _tree_index(self) -> dict[str, Any]:
        """Revealed-node index for tree.json, with each node's commit id."""
        index = tree_index(self.cycle_id, [self.node_records[c] for c in self._revealed_order])
        for entry, cid in zip(index["nodes"], self._revealed_order, strict=True):
            entry["commit"] = self.node_records[cid].get("commit")
        return index

    def _expand(self, job: NodeJob) -> dict[str, Any]:
        """Run one expansion; an unexpected exception becomes that node's runtime failure."""
        try:
            return expand_node(self.ctx, job, self.registry)
        except Exception as exc:  # noqa: BLE001 - one broken node must not abort the cycle
            job.node_dir.mkdir(parents=True, exist_ok=True)
            return {
                "cell_id": job.cell_id,
                "parent_id": job.parent_id,
                "fail_class": IMPLEMENTATION_FAILURE,
                "evaluated": False,
                "score": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "agent_timed_out": False,
                "agent_cost_usd": None,
                "agent_turns": None,
            }

    def _check_batch(self, cells: list[str]) -> None:
        if not cells:
            raise ValueError("empty batch")
        if self.decision_rounds >= self.rounds_cap:
            raise RuntimeError(f"round cap reached: rounds_cap={self.rounds_cap}")
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

    def probe_batch(self, cells: list[str], on_reveal: Any = None) -> list[Observation]:
        """Expand the batch concurrently against one frozen prefix, then reveal in batch order."""
        if not cells:
            return []
        self._check_batch(cells)
        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(len(cells) / self.max_parallelism)

        evidence_nodes = revealed_nodes(self.nodes_dir, self._revealed_order, self._slots)
        index = self._tree_index()
        jobs = [
            NodeJob(
                cell_id=cid,
                parent_id=self._slots[cid].parent_id or "root",
                store=self.store,
                parent_commit=self._parent_commit(self._slots[cid].parent_id or "root"),
                node_dir=self.nodes_dir / cid,
                evidence_nodes=evidence_nodes,
                tree_index=index,
                best_score=self.best_so_far,
            )
            for cid in cells
        ]
        with ThreadPoolExecutor(max_workers=self.max_parallelism) as pool:
            records = list(pool.map(self._expand, jobs))

        out: list[Observation] = []
        for cid, record in zip(cells, records, strict=True):
            cm = self._slots[cid]
            parent_obs = self._revealed.get(cm.parent_id) if cm.parent_id != "root" else None
            parent_score = parent_obs.score if parent_obs is not None else None
            obs = observation(cm, record, parent_score)
            self.node_records[cid] = record
            self.agent_records.append(record)
            self._revealed[cid] = obs
            self._revealed_order.append(cid)
            self.budget_spent += 1
            if obs.valid and obs.score is not None and obs.score > self.best_so_far:
                self.best_so_far = obs.score
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        if records and all(record.get("agent_output_invalid") for record in records):
            # Every worker in this batch returned placeholder text: the provider is corrupting
            # content, so another batch would burn budget for the same non-result.
            raise AgentOutputInvalid(
                f"every agent in batch {', '.join(cells)} returned placeholder-only output"
            )
        return out

    def export_tree(self) -> DiscoveryTree:
        """Freeze the revealed nodes of this cycle into a `DiscoveryTree` for policy development."""
        nodes: list[TreeNode] = []
        for idx, cid in enumerate(self._revealed_order):
            obs = self._revealed[cid]
            nodes.append(
                TreeNode(
                    cell_id=cid,
                    branch=obs.branch,
                    attempt=obs.attempt,
                    parent_id=obs.parent_id,
                    seq=idx,
                    score=obs.score,
                    evaluated=obs.evaluated,
                    valid=obs.valid,
                    fail_class=obs.fail_class,
                    error=obs.error,
                    mechanism_family=obs.mechanism_family,
                    summary=obs.summary,
                    cycle=0,
                    successes=obs.n_valid,
                    episodes=obs.n_total,
                )
            )
        return DiscoveryTree(
            world_id=f"dream-{self.cycle_id}",
            baseline_score=self.baseline_score,
            nodes=nodes,
        )

    def deepest_improvement(self) -> int:
        """Largest attempt index among valid nodes with a positive score."""
        return max(
            (o.attempt for o in self._revealed.values() if o.valid and o.score and o.score > 0),
            default=0,
        )

    def manifest_nodes(self) -> list[dict[str, Any]]:
        """Per-node records in reveal order, as written to the cycle manifest."""
        return [self.node_records[c] for c in self._revealed_order]
