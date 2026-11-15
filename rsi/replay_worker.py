"""Replay one policy file over recorded worlds and write a JSON result (runs inside the sandbox)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rsi.api import LLMDesignedMethod, Observation
from rsi.objective import (
    DEFAULT_BETA_GRID,
    DEFAULT_LAMBDA_PARALLEL,
    BetaPoint,
    WorldReplayScore,
    replay_beta_point,
    score_from_points,
)
from rsi.policy_audit import audit_no_world_literals, audit_policy_source, load_policy_class
from rsi.tree import DiscoveryTree, ReplayQuestion, TreeNode


def save_tree_json(tree: DiscoveryTree, path: Path) -> Path:
    """Persist a DiscoveryTree to JSON so replay worlds and the sandboxed worker share one schema."""
    payload = {
        "world_id": tree.world_id,
        "baseline_score": tree.baseline_score,
        "nodes": [
            {**asdict(tree.nodes[cid]), "tags": list(tree.nodes[cid].tags)} for cid in tree.order
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_tree_json(path: Path) -> DiscoveryTree:
    """Load a DiscoveryTree saved by `save_tree_json`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [TreeNode(**{**n, "tags": tuple(n.get("tags", ()))}) for n in raw["nodes"]]
    return DiscoveryTree(
        world_id=raw["world_id"],
        baseline_score=float(raw["baseline_score"]),
        nodes=nodes,
    )


class _EpisodeTrace:
    """Harness-side batch log for one (world, beta) episode; the policy never sees it."""

    def __init__(self, world_id: str, beta: float) -> None:
        self._world_id = world_id
        self._beta = beta
        self.events: list[dict[str, Any]] = []

    def __call__(self, question: ReplayQuestion, cells: list[str], out: list[Observation]) -> None:
        self.events.append(
            {
                "world_id": self._world_id,
                "beta": self._beta,
                "round": question.decision_rounds,
                "effective_sequential_rounds": question.effective_sequential_rounds,
                "prefix_size": question.budget_spent - len(cells),
                "batch": list(cells),
                "batch_size": len(cells),
                "max_parallelism": question.max_parallelism,
                "outcomes": [
                    {
                        "cell_id": o.cell_id,
                        "branch": o.branch,
                        "attempt": o.attempt,
                        "parent_id": o.parent_id,
                        "fail_class": o.fail_class,
                        "score": round(o.score, 4) if o.score is not None else None,
                        "delta_vs_parent": (
                            round(o.delta_vs_parent, 4) if o.delta_vs_parent is not None else None
                        ),
                    }
                    for o in out
                ],
            }
        )


def _fresh_loader(policy_path: Path) -> Callable[[], type[LLMDesignedMethod]]:
    """Return a loader that re-audits and re-executes the policy module on every call."""
    tags = itertools.count()
    return lambda: load_policy_class(policy_path, unique_tag=f"replay{next(tags)}")


def _run_episode(
    load: Callable[[], type[LLMDesignedMethod]],
    tree: DiscoveryTree,
    beta: float,
    max_parallelism: int,
) -> tuple[BetaPoint, dict[str, Any]]:
    """One (world, beta) replay with a fresh module, so no state carries between episodes."""
    trace = _EpisodeTrace(tree.world_id, float(beta))
    point = replay_beta_point(load(), tree, beta, max_parallelism=max_parallelism, recorder=trace)
    episode = {
        "world_id": tree.world_id,
        "beta": float(beta),
        "total_probes": point.probes,
        "decision_rounds": point.decision_rounds,
        "effective_sequential_rounds": point.effective_sequential_rounds,
        "best_score": point.best_score,
        "max_gain": tree.max_gain,
        "steps": trace.events,
    }
    return point, episode


def _world_entry(score: WorldReplayScore) -> dict[str, Any]:
    return {
        "v": score.reward,
        "hypervolume": score.normalized_hypervolume,
        "penalty": score.parallel_penalty,
        "degenerate": score.degenerate,
        "points": [
            {
                "beta": p.beta,
                "probes": p.probes,
                "attainment": p.attainment,
                "rounds": p.effective_sequential_rounds,
            }
            for p in score.beta_points
        ],
    }


def replay_once(
    load: Callable[[], type[LLMDesignedMethod]],
    trees: list[DiscoveryTree],
    *,
    max_parallelism: int,
    lambda_parallel: float,
) -> tuple[dict[str, Any], list[str]]:
    """Score one policy on every tree; returns the worker result and trace lines."""
    worlds: dict[str, Any] = {}
    lines: list[str] = []
    for tree in trees:
        points: list[BetaPoint] = []
        for beta in DEFAULT_BETA_GRID:
            point, episode = _run_episode(load, tree, beta, max_parallelism)
            points.append(point)
            lines.append(json.dumps(episode))
        worlds[tree.world_id] = _world_entry(score_from_points(tree, points, lambda_parallel))
    rewards = [w["v"] for w in worlds.values()]
    result = {
        "max_parallelism": max_parallelism,
        "lambda_parallel": lambda_parallel,
        "worlds": worlds,
        "mean_v": sum(rewards) / float(len(rewards)),
    }
    return result, lines


def run_worker(
    policy_path: Path,
    world_paths: list[Path],
    *,
    max_parallelism: int,
    lambda_parallel: float,
) -> tuple[dict[str, Any], list[str]]:
    """Audit, load and replay a policy twice; raise if the two replays differ."""
    source = Path(policy_path).read_text(encoding="utf-8")
    audit_policy_source(source)
    trees = [load_tree_json(p) for p in world_paths]
    audit_no_world_literals(source, trees)
    load = _fresh_loader(Path(policy_path))
    first = replay_once(
        load, trees, max_parallelism=max_parallelism, lambda_parallel=lambda_parallel
    )
    second = replay_once(
        load, trees, max_parallelism=max_parallelism, lambda_parallel=lambda_parallel
    )
    if first != second:
        raise RuntimeError("non-deterministic replay: two runs disagree")
    result = dict(first[0])
    result["policy_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return result, first[1]


def main(argv: list[str] | None = None) -> int:
    """CLI: `replay_worker POLICY OUT WORLD... [--max-parallelism W] [--lambda L] [--traces P]`."""
    parser = argparse.ArgumentParser(prog="rsi.replay_worker")
    parser.add_argument("policy")
    parser.add_argument("out")
    parser.add_argument("worlds", nargs="+")
    parser.add_argument("--max-parallelism", type=int, default=4)
    parser.add_argument("--lambda", dest="lambda_parallel", type=float, default=None)
    parser.add_argument("--traces", default=None)
    args = parser.parse_args(argv)
    lam = DEFAULT_LAMBDA_PARALLEL if args.lambda_parallel is None else args.lambda_parallel
    try:
        result, traces = run_worker(
            Path(args.policy),
            [Path(w) for w in args.worlds],
            max_parallelism=args.max_parallelism,
            lambda_parallel=lam,
        )
    except Exception as exc:  # noqa: BLE001 - any policy failure becomes one stderr line
        reason = str(exc).strip().splitlines()
        print(
            f"replay failed: {type(exc).__name__}: {reason[0] if reason else ''}", file=sys.stderr
        )
        return 1
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.traces:
        Path(args.traces).write_text("\n".join(traces) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
