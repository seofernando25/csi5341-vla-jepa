"""Dream-RSI offline dreaming selection engine and benchmark runner."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.api import GridPlanningContext, LLMDesignedMethod
from rsi.objective import DEFAULT_BETA_GRID, DEFAULT_LAMBDA_PARALLEL, evaluate_policy_over_history
from rsi.policies import (
    GreedySerialPolicy,
    InitialPolicy,
    OptimalPolicy,
    RandomPolicy,
    RoundRobinPolicy,
)
from rsi.qwen_search import MEASUREMENTS_DIR, stage2_smoke_world
from rsi.tree import (
    DiscoveryTree,
    from_attempts_jsonl,
    from_compression_summary,
    make_synthetic_tree,
    partition_attempts_by_root_cycle,
)

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)
PARALLELISM_GRID: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class Candidate:
    """One policy version: a policy class plus fixed configuration (beta is swept, not fixed)."""

    name: str
    cls: type[LLMDesignedMethod]
    config: dict[str, Any] = field(default_factory=dict)


INCUMBENT = Candidate("InitialPolicy", InitialPolicy)
# Hand-written stand-ins for the paper's M policy versions; no policy-development agent wrote them.
CANDIDATE_POOL: tuple[Candidate, ...] = (
    Candidate("OptimalPolicy", OptimalPolicy),
    Candidate("OptimalPolicy[no-gate-retry]", OptimalPolicy, {"gate_retry": False}),
    Candidate("GreedySerialPolicy", GreedySerialPolicy),
    Candidate("RoundRobinPolicy", RoundRobinPolicy),
    Candidate("RandomPolicy[seed=0]", RandomPolicy, {"seed": 0}),
    Candidate("RandomPolicy[seed=1]", RandomPolicy, {"seed": 1}),
    Candidate("RandomPolicy[seed=2]", RandomPolicy, {"seed": 2}),
)

# terms[policy][world_id] = (normalized hypervolume, parallelism penalty); V = hv - lambda * penalty.
Terms = dict[str, dict[str, tuple[float, float]]]


def _as_candidate(policy: Candidate | type[LLMDesignedMethod]) -> Candidate:
    if isinstance(policy, Candidate):
        return policy
    return Candidate(getattr(policy, "NAME", policy.__name__), policy)


def replay_terms(
    pool: list[Candidate],
    worlds: list[DiscoveryTree],
    max_parallelism: int,
    beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
) -> Terms:
    """Replay every candidate on every world once; both Eq. (1) terms are lambda-free."""
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


class DreamEngine:
    """Offline dreaming policy evaluation and selection (paper Section 3, Eq. 1).

    Covers "Offline evaluation" and the argmax of "Policy improvement and selection". The
    policy-development agent that writes new versions is not implemented; callers pass candidates.
    """

    def __init__(
        self,
        beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
        lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
        max_parallelism: int = 4,
    ) -> None:
        self.beta_grid = beta_grid
        self.lambda_parallel = lambda_parallel
        self.max_parallelism = max_parallelism

    def _pool(
        self,
        incumbent: Candidate | type[LLMDesignedMethod],
        candidates: list[Candidate] | list[type[LLMDesignedMethod]],
    ) -> list[Candidate]:
        pool = [_as_candidate(incumbent)]
        for cand in map(_as_candidate, candidates):
            if cand.name not in {p.name for p in pool}:
                pool.append(cand)
        return pool

    def select_policy(
        self,
        incumbent: Candidate | type[LLMDesignedMethod],
        candidates: list[Candidate] | list[type[LLMDesignedMethod]],
        history: list[DiscoveryTree],
    ) -> dict[str, Any]:
        """Evaluate incumbent + candidates on H_t and return argmax V^m (in-sample)."""
        pool = self._pool(incumbent, candidates)
        evaluations = []
        for cand in pool:
            ev = evaluate_policy_over_history(
                policy_cls=cand.cls,
                trees=history,
                beta_grid=self.beta_grid,
                lambda_parallel=self.lambda_parallel,
                max_parallelism=self.max_parallelism,
                extra_config=cand.config,
            )
            ev["policy"] = cand.name
            evaluations.append(ev)
        best_idx = max(
            range(len(evaluations)), key=lambda i: (evaluations[i]["pareto"]["reward"], -i)
        )
        return {
            "selected_policy": evaluations[best_idx]["policy"],
            "selected_class": pool[best_idx].cls,
            "selected_candidate": pool[best_idx],
            "selected_reward": evaluations[best_idx]["pareto"]["reward"],
            "incumbent_reward": evaluations[0]["pareto"]["reward"],
            "evaluations": evaluations,
        }

    def leave_one_world_out(
        self,
        incumbent: Candidate | type[LLMDesignedMethod],
        candidates: list[Candidate] | list[type[LLMDesignedMethod]],
        history: list[DiscoveryTree],
    ) -> dict[str, Any]:
        """Select on t-1 worlds and report V on the held-out world, for each world."""
        terms = replay_terms(
            self._pool(incumbent, candidates), history, self.max_parallelism, self.beta_grid
        )
        return leave_one_world_out(terms, [t.world_id for t in history], self.lambda_parallel)


def build_synthetic_suite() -> list[DiscoveryTree]:
    """Four controlled discovery worlds testing width, depth, and failure recovery."""
    return [
        make_synthetic_tree(
            "synth-recovery-depth",
            branches=[
                [(0.12, "ok"), (0.20, "implementation_failure"), (0.45, "ok"), (0.52, "ok")],
                [(0.18, "ok"), (0.14, "ok"), (0.10, "ok")],
                [(0.05, "runtime_failure"), (0.04, "runtime_failure"), (0.02, "interrupted")],
                [(0.22, "ok"), (0.25, "ok"), (0.23, "ok")],
            ],
            baseline_score=0.0,
        ),
        make_synthetic_tree(
            "synth-wide-needle",
            branches=[
                [(0.05, "ok"), (0.04, "ok")],
                [(0.08, "ok"), (0.07, "ok")],
                [(0.10, "ok"), (0.09, "ok")],
                [(0.35, "ok"), (0.60, "ok"), (0.75, "ok")],
                [(0.12, "ok"), (0.11, "ok")],
                [(0.02, "interrupted")],
            ],
            baseline_score=0.0,
        ),
        make_synthetic_tree(
            "synth-late-bloomer",
            branches=[
                [(0.30, "ok"), (0.31, "ok"), (0.29, "ok")],
                [(0.15, "ok"), (0.28, "ok"), (0.42, "ok"), (0.68, "ok")],
                [(0.20, "ok"), (0.19, "implementation_failure"), (0.18, "ok")],
                [(0.05, "ok"), (0.04, "ok")],
            ],
            baseline_score=0.0,
        ),
        make_synthetic_tree(
            "synth-multi-failure",
            branches=[
                [(0.10, "implementation_failure"), (0.38, "ok"), (0.50, "ok")],
                [(0.25, "ok"), (0.24, "ok"), (0.20, "ok")],
                [(0.18, "runtime_failure"), (0.19, "runtime_failure"), (0.15, "ok")],
                [(0.32, "ok"), (0.36, "ok")],
            ],
            baseline_score=0.0,
        ),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision() -> dict[str, Any]:
    """Git HEAD and whether `rsi/` differs from it; None values outside a Git checkout."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "rsi"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "rsi_modified": None}
    return {"git_head": head, "rsi_modified": bool(dirty)}


def _in_sample(terms: Terms, world_ids: list[str], lam: float) -> dict[str, Any]:
    incumbent = next(iter(terms))
    policies = {}
    for name in terms:
        hv, penalty = mean_terms(terms, name, world_ids)
        policies[name] = {
            "reward": hv - lam * penalty,
            "hypervolume": hv,
            "parallel_penalty": penalty,
        }
    selected = select_from_terms(terms, world_ids, lam)
    return {
        "selected_policy": selected,
        "delta_reward": policies[selected]["reward"] - policies[incumbent]["reward"],
        "ranking": sorted(policies, key=lambda n: -policies[n]["reward"]),
        "policies": policies,
    }


def run_benchmark(
    attempts_path: str | Path = "studies/dream-rsi/attempts.jsonl",
    compression_summary_path: str | Path = "studies/compression/summary.csv",
    output_path: str | Path | None = "studies/dream-rsi/dream_benchmark.json",
    parallelism_levels: tuple[int, ...] = PARALLELISM_GRID,
    measurements_dir: str | Path = MEASUREMENTS_DIR,
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
) -> dict[str, Any]:
    """Replay the candidate pool on recorded and synthetic worlds (CPU only).

    Writes in-sample scores, leave-one-world-out scores and a lambda x W sensitivity grid.
    Smoke and development worlds stay separate suites; no V averages two phases.
    """
    t0 = time.perf_counter()
    full_tree = from_attempts_jsonl(attempts_path)
    cycle_trees = partition_attempts_by_root_cycle(attempts_path)
    synth_trees = build_synthetic_suite()
    sources = {str(attempts_path): _sha256(Path(attempts_path))}

    suites: dict[str, list[DiscoveryTree]] = {
        "smolvlm_full": [full_tree],
        "smolvlm_cycles": cycle_trees,
        "synthetic_suite": synth_trees,
    }
    real_smoke_derived = list(cycle_trees)
    comp_path = Path(compression_summary_path)
    if comp_path.exists():
        qwen_smoke, qwen_dev = from_compression_summary(comp_path)
        suites["qwen_compression_smoke"] = [qwen_smoke]
        suites["qwen_compression_development"] = [qwen_dev]
        real_smoke_derived.append(qwen_smoke)
        # qwen_dev stays out of held-out means: one world cannot leave-one-out,
        # and no V averages smoke with development trials.
        sources[str(compression_summary_path)] = _sha256(comp_path)
    stage2_records: list[dict[str, Any]] = []
    if (Path(measurements_dir) / "s2-c2-tmlp10" / "smoke" / "run.json").exists():
        stage2_tree, stage2_records = stage2_smoke_world(measurements_dir)
        suites["qwen_stage2_smoke"] = [stage2_tree]
        real_smoke_derived.append(stage2_tree)
        for rec in stage2_records:
            run = Path(measurements_dir) / rec["candidate"] / "smoke" / "run.json"
            sources[str(run)] = _sha256(run)
    held_out_suites = {"real_smoke_derived": real_smoke_derived, "synthetic": synth_trees}

    pool = [INCUMBENT, *CANDIDATE_POOL]
    worlds = {t.world_id: t for trees in suites.values() for t in trees}
    terms_by_w = {w: replay_terms(pool, list(worlds.values()), w) for w in parallelism_levels}
    lam = DEFAULT_LAMBDA_PARALLEL

    in_sample = {
        name: {
            f"W={w}": _in_sample(terms_by_w[w], [t.world_id for t in trees], lam)
            for w in parallelism_levels
        }
        for name, trees in suites.items()
    }
    held_out = {
        name: {
            f"W={w}": leave_one_world_out(terms_by_w[w], [t.world_id for t in trees], lam)
            for w in parallelism_levels
        }
        for name, trees in held_out_suites.items()
    }
    sensitivity = {
        name: sensitivity_grid(terms_by_w, [t.world_id for t in trees], lambda_grid)
        for name, trees in held_out_suites.items()
    }

    # Verify GridPlan & cross-cycle beta rule
    sweep = evaluate_policy_over_history(OptimalPolicy, cycle_trees, max_parallelism=4)
    sweep.pop("worlds")
    grid_plan = OptimalPolicy({"beta": 0.6}).plan_grid(
        GridPlanningContext(
            iteration=3,
            max_parallelism=4,
            recent_live_manifests=[
                {"iteration": 1, "best_score": -0.3307},
                {"iteration": 2, "best_score": -0.3190},
            ],
            recent_beta_sweeps=[sweep],
        )
    )
    next_beta = recommend_next_beta(
        0.6, [{"best_score": -0.3307}, {"best_score": -0.3190}], [sweep]
    )

    payload = {
        "protocol": "dream-rsi-arxiv-2609.14858v2",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "host": {"python": platform.python_version(), "machine": platform.machine()},
        "code_revision": _code_revision(),
        "sources_sha256": sources,
        "attempts_source": str(attempts_path),
        "attempts_count": full_tree.num_nonroot,
        "beta_grid": list(DEFAULT_BETA_GRID),
        "lambda_parallel": lam,
        "lambda_grid": list(lambda_grid),
        "parallelism_levels": list(parallelism_levels),
        "incumbent": INCUMBENT.name,
        "candidate_pool": [c.name for c in CANDIDATE_POOL],
        "worlds": {
            name: [
                {"world_id": t.world_id, "nodes": t.num_nonroot, "max_gain": t.max_gain}
                for t in trees
            ]
            for name, trees in suites.items()
        },
        "notes": {
            "score": "Eq. (1) replay score V; not LIBERO success.",
            "in_sample": "Selection and reported V use the same worlds.",
            "held_out": (
                "Selection uses the other worlds of the suite; V is from the held-out world. "
                "Worlds are not independent: the three SmolVLM cycles split one search run, the "
                "qwen-compression chains are synthesized from a grid study, and the synthetic "
                "worlds are hand-made. Development worlds are excluded."
            ),
            "stage2_world": (
                "Committed 10-trial smoke measurements with the grid's parent->child edges; "
                "'derived' bytes are a linear extrapolation, not a measurement."
            ),
        },
        "in_sample": in_sample,
        "held_out": held_out,
        "sensitivity": sensitivity,
        "replay_terms": {
            f"W={w}": {
                name: {
                    world: {"hypervolume": hv, "parallel_penalty": penalty}
                    for world, (hv, penalty) in per_world.items()
                }
                for name, per_world in terms.items()
            }
            for w, terms in terms_by_w.items()
        },
        "stage2_smoke_world": stage2_records,
        "grid_plan_example": {
            "max_branches": grid_plan.max_branches,
            "max_attempts_per_branch": grid_plan.max_attempts_per_branch,
            "max_parallelism": grid_plan.max_parallelism,
            "rounds_cap": grid_plan.rounds_cap,
            "rationale": grid_plan.rationale,
            "recommended_next_beta": next_beta,
        },
    }
    payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return payload
