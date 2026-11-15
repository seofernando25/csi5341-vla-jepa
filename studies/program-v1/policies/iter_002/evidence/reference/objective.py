"""Dream-RSI Equation (1) replay objective and beta-sweep evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from rsi.api import LLMDesignedMethod, Observation, PrefixView
from rsi.tree import DiscoveryTree, ReplayQuestion

DEFAULT_BETA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_LAMBDA_PARALLEL: float = 0.05


@dataclass(frozen=True)
class BetaPoint:
    """Single replay outcome for one (world, beta) pair."""

    beta: float
    probes: int
    decision_rounds: int
    effective_sequential_rounds: int
    best_score: float
    attainment: float


@dataclass(frozen=True)
class WorldReplayScore:
    """Eq. (1) replay score V_i^m for one policy on one DiscoveryTree."""

    world_id: str
    total_nodes: int
    max_gain: float
    normalized_hypervolume: float
    parallel_penalty: float
    lambda_parallel: float
    degenerate: bool
    reward: float
    pareto_frontier: tuple[tuple[int, float], ...]
    beta_points: tuple[BetaPoint, ...]


def pareto_frontier(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return distinct nondominated (n, a) points ordered by increasing n."""
    unique = sorted(set(points), key=lambda p: (p[0], -p[1]))
    nondominated: list[tuple[int, float]] = []
    for n, a in unique:
        dominated = any((n2 <= n and a2 >= a) and (n2 < n or a2 > a) for n2, a2 in unique)
        if not dominated:
            nondominated.append((n, a))
    nondominated.sort(key=lambda p: p[0])
    return nondominated


def compute_eq1_score(
    total_nodes: int,
    points: list[BetaPoint],
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
) -> tuple[float, float, float, list[tuple[int, float]]]:
    """Compute (reward, normalized_hypervolume, parallel_penalty, frontier) per Eq. (1)."""
    if total_nodes <= 0:
        raise ValueError("total_nodes B_i must be > 0")
    if not points:
        return 0.0, 0.0, 0.0, []

    frontier = pareto_frontier([(p.probes, p.attainment) for p in points])
    hv = 0.0
    prev_a = 0.0
    for n_bar, a_bar in frontier:
        if a_bar > prev_a:
            hv += (a_bar - prev_a) * max(0, total_nodes - n_bar)
            prev_a = a_bar
    hv /= float(total_nodes)

    penalty_terms = [
        p.effective_sequential_rounds / float(p.probes) for p in points if p.probes > 0
    ]
    penalty = sum(penalty_terms) / float(len(points))
    reward = hv - lambda_parallel * penalty
    return reward, hv, penalty, frontier


EpisodeRecorder = Callable[[ReplayQuestion, list[str], list[Observation]], None]


def replay_beta_point(
    policy_cls: type[LLMDesignedMethod],
    tree: DiscoveryTree,
    beta: float,
    *,
    max_parallelism: int = 4,
    extra_config: dict[str, Any] | None = None,
    recorder: EpisodeRecorder | None = None,
) -> BetaPoint:
    """Run one policy instance on one world at one beta; accounting comes from the real question."""
    cfg = dict(extra_config or {})
    cfg["beta"] = float(beta)
    question = ReplayQuestion(tree, max_parallelism=max_parallelism)
    view_recorder = (
        None if recorder is None else (lambda cells, outs: recorder(question, cells, outs))
    )
    policy_cls(cfg).solve(PrefixView(question, view_recorder), budget=None)

    best = question.best_so_far
    max_gain = tree.max_gain
    if question.budget_spent > 0 and max_gain > 0.0:
        gain = max(0.0, best - tree.baseline_score)
        attainment = min(1.0, max(0.0, gain / max_gain))
    else:
        attainment = 0.0
    return BetaPoint(
        beta=float(beta),
        probes=question.budget_spent,
        decision_rounds=question.decision_rounds,
        effective_sequential_rounds=question.effective_sequential_rounds,
        best_score=best,
        attainment=attainment,
    )


def score_from_points(
    tree: DiscoveryTree, points: list[BetaPoint], lambda_parallel: float
) -> WorldReplayScore:
    """Eq. (1) score of one world from its per-beta replay points."""
    max_gain = tree.max_gain
    reward, hv, penalty, frontier = compute_eq1_score(
        total_nodes=tree.num_nonroot,
        points=points,
        lambda_parallel=lambda_parallel,
    )
    return WorldReplayScore(
        world_id=tree.world_id,
        total_nodes=tree.num_nonroot,
        max_gain=max_gain,
        normalized_hypervolume=hv,
        parallel_penalty=penalty,
        lambda_parallel=lambda_parallel,
        degenerate=max_gain <= 0.0,
        reward=reward,
        pareto_frontier=tuple(frontier),
        beta_points=tuple(points),
    )


def evaluate_policy_on_tree(
    policy_cls: type[LLMDesignedMethod],
    tree: DiscoveryTree,
    beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    max_parallelism: int = 4,
    extra_config: dict[str, Any] | None = None,
) -> WorldReplayScore:
    """Evaluate a policy class across beta_grid on one DiscoveryTree."""
    points = [
        replay_beta_point(
            policy_cls,
            tree,
            beta,
            max_parallelism=max_parallelism,
            extra_config=extra_config,
        )
        for beta in beta_grid
    ]
    return score_from_points(tree, points, lambda_parallel)


def evaluate_policy_over_history(
    policy_cls: type[LLMDesignedMethod],
    trees: list[DiscoveryTree],
    beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    max_parallelism: int = 4,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute average replay score V^m = (1/t) sum_i V_i^m over history H_t.

    Degenerate worlds (no ok node beats the baseline) carry no attainment signal and are
    excluded from the mean unless every world is degenerate.
    """
    if not trees:
        raise ValueError("trees must be non-empty")
    all_worlds = [
        evaluate_policy_on_tree(
            policy_cls=policy_cls,
            tree=t,
            beta_grid=beta_grid,
            lambda_parallel=lambda_parallel,
            max_parallelism=max_parallelism,
            extra_config=extra_config,
        )
        for t in trees
    ]
    per_world = [w for w in all_worlds if not w.degenerate] or all_worlds
    n = float(len(per_world))
    mean_reward = sum(w.reward for w in per_world) / n
    mean_hv = sum(w.normalized_hypervolume for w in per_world) / n
    mean_penalty = sum(w.parallel_penalty for w in per_world) / n

    per_beta: list[dict[str, Any]] = []
    for idx, beta in enumerate(beta_grid):
        pts = [w.beta_points[idx] for w in per_world]
        per_beta.append(
            {
                "beta": beta,
                "mean_probes": sum(p.probes for p in pts) / n,
                "mean_rounds": sum(p.effective_sequential_rounds for p in pts) / n,
                "mean_attainment": sum(p.attainment for p in pts) / n,
            }
        )

    return {
        "policy": getattr(policy_cls, "NAME", policy_cls.__name__),
        "num_worlds": len(per_world),
        "num_degenerate_worlds": len(all_worlds) - sum(1 for w in all_worlds if not w.degenerate),
        "max_parallelism": max_parallelism,
        "lambda_parallel": lambda_parallel,
        "pareto": {
            "reward": mean_reward,
            "auc": mean_hv,
            "parallel_penalty": mean_penalty,
        },
        "per_beta": per_beta,
        "worlds": [asdict(w) for w in all_worlds],
    }
