"""Dream-RSI offline dreaming selection: replay terms, policy selection and stability checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rsi.api import LLMDesignedMethod
from rsi.objective import DEFAULT_BETA_GRID, evaluate_policy_over_history
from rsi.policies import (
    EpsilonGreedyPolicy,
    GreedySerialPolicy,
    IncumbentWithExceptions,
    InitialPolicy,
    OptimalPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    ShallowFirstPolicy,
    TwoStageGatePolicy,
)
from rsi.tree import DiscoveryTree

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)
PARALLELISM_GRID: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class Candidate:
    """One policy version: a policy class plus fixed configuration (beta is swept, not fixed)."""

    name: str
    cls: type[LLMDesignedMethod]
    config: dict[str, Any] = field(default_factory=dict)


INCUMBENT = Candidate("InitialPolicy", InitialPolicy)
# Base pool. Per-round M=4 revisions from the policy-development agent (rsi/policy_dev.py)
# join the pool at selection time; the names below are the hand-written starting versions.
CANDIDATE_POOL: tuple[Candidate, ...] = (
    Candidate("OptimalPolicy", OptimalPolicy),
    Candidate("GreedySerialPolicy", GreedySerialPolicy),
    Candidate("RoundRobinPolicy", RoundRobinPolicy),
    Candidate("EpsilonGreedyPolicy[eps=0.2]", EpsilonGreedyPolicy, {"epsilon": 0.2}),
    Candidate("ShallowFirstPolicy", ShallowFirstPolicy),
    Candidate("TwoStageGatePolicy", TwoStageGatePolicy),
    Candidate("IncumbentWithExceptions", IncumbentWithExceptions),
    Candidate("RandomPolicy[seed=0]", RandomPolicy, {"seed": 0}),
    Candidate("RandomPolicy[seed=1]", RandomPolicy, {"seed": 1}),
    Candidate("RandomPolicy[seed=2]", RandomPolicy, {"seed": 2}),
)

STABILITY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def pool_for_seed(seed: int) -> list[Candidate]:
    """Per-seed pool: deterministic candidates plus one seeded draw of each stochastic policy."""
    pool = [c for c in (INCUMBENT, *CANDIDATE_POOL) if "RandomPolicy" not in c.name]
    pool = [c for c in pool if "EpsilonGreedy" not in c.name]
    pool.append(Candidate(f"RandomPolicy[seed={seed}]", RandomPolicy, {"seed": seed}))
    pool.append(
        Candidate(
            f"EpsilonGreedyPolicy[eps=0.2,seed={seed}]",
            EpsilonGreedyPolicy,
            {"epsilon": 0.2, "seed": seed},
        )
    )
    return pool


# terms[policy][world_id] = (normalized hypervolume, parallelism penalty); V = hv - lambda * penalty.
Terms = dict[str, dict[str, tuple[float, float]]]


def replay_terms(
    pool: list[Candidate],
    worlds: list[DiscoveryTree],
    max_parallelism: int,
    beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
) -> Terms:
    """Replay every candidate on every world once; both Eq. (1) terms are lambda-free."""
    ids = [t.world_id for t in worlds]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate world ids in replay: {dupes}")
    terms: Terms = {}
    for cand in pool:
        ev = evaluate_policy_over_history(
            policy_cls=cand.cls,
            trees=worlds,
            beta_grid=beta_grid,
            lambda_parallel=0.0,
            max_parallelism=max_parallelism,
            extra_config=cand.config,
        )
        terms[cand.name] = {
            w["world_id"]: (w["normalized_hypervolume"], w["parallel_penalty"])
            for w in ev["worlds"]
        }
    return terms


def mean_terms(terms: Terms, policy: str, world_ids: list[str]) -> tuple[float, float]:
    """Mean (hypervolume, penalty) of one policy over the given worlds."""
    n = float(len(world_ids))
    return (
        sum(terms[policy][w][0] for w in world_ids) / n,
        sum(terms[policy][w][1] for w in world_ids) / n,
    )


def replay_score(terms: Terms, policy: str, world_ids: list[str], lam: float) -> float:
    hv, penalty = mean_terms(terms, policy, world_ids)
    return hv - lam * penalty


def select_from_terms(terms: Terms, world_ids: list[str], lam: float) -> str:
    """argmax_m V^m over the given worlds; the incumbent is first and wins ties."""
    names = list(terms)
    return max(
        names, key=lambda name: (replay_score(terms, name, world_ids, lam), -names.index(name))
    )


def leave_one_world_out(terms: Terms, world_ids: list[str], lam: float) -> dict[str, Any]:
    """Select on all worlds but one and score the selection on the held-out world."""
    if len(world_ids) < 2:
        raise ValueError("leave-one-world-out needs at least two worlds")
    incumbent = next(iter(terms))
    folds: list[dict[str, Any]] = []
    for held in world_ids:
        selected = select_from_terms(terms, [w for w in world_ids if w != held], lam)
        per_policy = {name: replay_score(terms, name, [held], lam) for name in terms}
        folds.append(
            {
                "held_out_world": held,
                "selected_policy": selected,
                "selected_reward": per_policy[selected],
                "incumbent_reward": per_policy[incumbent],
                "best_policy_on_held_out": max(per_policy, key=per_policy.get),
                "best_reward_on_held_out": max(per_policy.values()),
                "held_out_reward_by_policy": per_policy,
            }
        )
    n = float(len(folds))
    return {
        "folds": folds,
        "mean_selected_reward": sum(f["selected_reward"] for f in folds) / n,
        "mean_incumbent_reward": sum(f["incumbent_reward"] for f in folds) / n,
        "mean_held_out_reward_by_policy": {
            name: sum(f["held_out_reward_by_policy"][name] for f in folds) / n for name in terms
        },
        "folds_selected_beats_incumbent": sum(
            f["selected_reward"] > f["incumbent_reward"] for f in folds
        ),
        "folds_selected_below_incumbent": sum(
            f["selected_reward"] < f["incumbent_reward"] for f in folds
        ),
    }


def sensitivity_grid(
    terms_by_w: dict[int, Terms],
    world_ids: list[str],
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
) -> list[dict[str, Any]]:
    """In-sample and leave-one-world-out selection for every (lambda, W) cell."""
    cells: list[dict[str, Any]] = []
    for w, terms in terms_by_w.items():
        for lam in lambda_grid:
            selected = select_from_terms(terms, world_ids, lam)
            hv, penalty = mean_terms(terms, selected, world_ids)
            cell: dict[str, Any] = {
                "lambda": lam,
                "W": w,
                "selected_policy": selected,
                "selected_hypervolume": hv,
                "selected_parallel_penalty": penalty,
                "selected_reward": hv - lam * penalty,
                "incumbent_reward": replay_score(terms, next(iter(terms)), world_ids, lam),
            }
            if len(world_ids) >= 2:
                lowo = leave_one_world_out(terms, world_ids, lam)
                cell["held_out_selected_policies"] = [f["selected_policy"] for f in lowo["folds"]]
                cell["held_out_mean_selected_reward"] = lowo["mean_selected_reward"]
                cell["held_out_mean_incumbent_reward"] = lowo["mean_incumbent_reward"]
            cells.append(cell)
    return cells


def selection_stability(
    terms_by_seed_w: dict[int, dict[int, Terms]],
    world_ids: list[str],
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
) -> dict[str, Any]:
    """Winner per seed for every (lambda, W) cell; unanimity is stability evidence.

    `terms_by_seed_w[seed][W]` holds replay terms over the same world universe with
    the seed's stochastic draws. Returns per-cell winners plus suite summaries:
    unanimous-cell fraction and the distinct-winner set.
    """
    seeds = sorted(terms_by_seed_w)
    cells: list[dict[str, Any]] = []
    for w in sorted(terms_by_seed_w[seeds[0]]):
        for lam in lambda_grid:
            winners = {
                str(s): select_from_terms(terms_by_seed_w[s][w], world_ids, lam) for s in seeds
            }
            counts = Counter(winners.values())
            incumbent = next(iter(terms_by_seed_w[seeds[0]][w]))
            modal = min(counts, key=lambda name: (-counts[name], name != incumbent, name))
            cells.append(
                {
                    "lambda": lam,
                    "W": w,
                    "winners_by_seed": winners,
                    "modal_winner": modal,
                    "unanimous": len(set(winners.values())) == 1,
                    "distinct_winners": sorted(set(winners.values())),
                }
            )
    n = float(len(cells))
    return {
        "cells": cells,
        "unanimous_fraction": sum(c["unanimous"] for c in cells) / n,
        "distinct_winners": sorted({w for c in cells for w in c["distinct_winners"]}),
    }


def recommend_next_beta(
    current_beta: float,
    recent_manifests: list[dict[str, Any]],
    recent_sweeps: list[dict[str, Any]],
) -> float:
    """Cross-cycle default-beta rule from Dream-RSI Appendix D.2 Listing 2."""
    if len(recent_manifests) < 2 or not recent_sweeps:
        return 0.6
    prev_best = recent_manifests[-2].get("best_score")
    curr_best = recent_manifests[-1].get("best_score")
    if prev_best is None or curr_best is None:
        return 0.6
    if curr_best > prev_best:
        return round(max(0.0, min(1.0, current_beta)), 2)
    # Plateaued: check if higher beta in sweep improved attainment
    last_sweep = recent_sweeps[-1].get("per_beta", [])
    higher = [p for p in last_sweep if p["beta"] > current_beta]
    curr_pts = [p for p in last_sweep if abs(p["beta"] - current_beta) < 1e-6]
    if curr_pts and higher:
        curr_att = curr_pts[0]["mean_attainment"]
        if any(p["mean_attainment"] > curr_att + 0.05 for p in higher):
            return round(min(1.0, current_beta + 0.15), 2)
        if current_beta >= 0.7:
            return round(max(0.0, current_beta - 0.15), 2)
    return round(max(0.0, min(1.0, current_beta)), 2)
