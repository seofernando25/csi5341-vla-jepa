"""Dream-RSI offline dreaming selection engine and benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.api import GridPlanningContext, LLMDesignedMethod
from rsi.objective import DEFAULT_BETA_GRID, DEFAULT_LAMBDA_PARALLEL, evaluate_policy_over_history
from rsi.policies.initial import InitialPolicy
from rsi.policies.optimal import OptimalPolicy
from rsi.tree import (
    DiscoveryTree,
    from_attempts_jsonl,
    from_compression_summary,
    make_synthetic_tree,
    partition_attempts_by_root_cycle,
)


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
    """Offline dreaming policy evaluation and monotonic selection (Algorithm 1)."""

    def __init__(
        self,
        beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
        lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
        max_parallelism: int = 4,
    ) -> None:
        self.beta_grid = beta_grid
        self.lambda_parallel = lambda_parallel
        self.max_parallelism = max_parallelism

    def select_policy(
        self,
        incumbent: type[LLMDesignedMethod],
        candidates: list[type[LLMDesignedMethod]],
        history: list[DiscoveryTree],
    ) -> dict[str, Any]:
        """Evaluate incumbent + candidates on H_t and return argmax V^m."""
        pool = [incumbent, *[c for c in candidates if c is not incumbent]]
        evaluations = [
            evaluate_policy_over_history(
                policy_cls=cls,
                trees=history,
                beta_grid=self.beta_grid,
                lambda_parallel=self.lambda_parallel,
                max_parallelism=self.max_parallelism,
            )
            for cls in pool
        ]
        best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i]["pareto"]["reward"])
        return {
            "selected_policy": evaluations[best_idx]["policy"],
            "selected_class": pool[best_idx],
            "selected_reward": evaluations[best_idx]["pareto"]["reward"],
            "incumbent_reward": evaluations[0]["pareto"]["reward"],
            "evaluations": evaluations,
        }


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


def run_benchmark(
    attempts_path: str | Path = "studies/dream-rsi/attempts.jsonl",
    compression_summary_path: str | Path = "studies/compression/summary.csv",
    output_path: str | Path | None = "studies/dream-rsi/dream_benchmark.json",
    parallelism_levels: tuple[int, ...] = (1, 2, 4),
) -> dict[str, Any]:
    """Run Dream-RSI integration benchmark on historical and synthetic worlds."""
    t0 = time.perf_counter()
    full_tree = from_attempts_jsonl(attempts_path)
    cycle_trees = partition_attempts_by_root_cycle(attempts_path)
    synth_trees = build_synthetic_suite()

    suites: dict[str, list[DiscoveryTree]] = {
        "smolvlm_full": [full_tree],
        "smolvlm_cycles": cycle_trees,
        "synthetic_suite": synth_trees,
    }
    comp_path = Path(compression_summary_path)
    if comp_path.exists():
        qwen_smoke, qwen_dev = from_compression_summary(comp_path)
        suites["qwen_compression"] = [qwen_smoke, qwen_dev]

    results: dict[str, Any] = {}
    for suite_name, trees in suites.items():
        suite_res: dict[str, Any] = {}
        for w in parallelism_levels:
            engine = DreamEngine(max_parallelism=w)
            sel = engine.select_policy(InitialPolicy, [OptimalPolicy], trees)
            init_eval, opt_eval = sel["evaluations"][0], sel["evaluations"][1]
            suite_res[f"W={w}"] = {
                "selected_policy": sel["selected_policy"],
                "initial": init_eval,
                "optimal": opt_eval,
                "delta_reward": (opt_eval["pareto"]["reward"] - init_eval["pareto"]["reward"]),
                "delta_auc": opt_eval["pareto"]["auc"] - init_eval["pareto"]["auc"],
                "delta_penalty": (
                    opt_eval["pareto"]["parallel_penalty"] - init_eval["pareto"]["parallel_penalty"]
                ),
            }
        results[suite_name] = suite_res

    # Verify GridPlan & cross-cycle beta rule
    opt_policy = OptimalPolicy({"beta": 0.6})
    grid_plan = opt_policy.plan_grid(
        GridPlanningContext(
            iteration=3,
            max_parallelism=4,
            recent_live_manifests=[
                {"iteration": 1, "best_score": -0.3307},
                {"iteration": 2, "best_score": -0.3190},
            ],
            recent_beta_sweeps=[results["smolvlm_cycles"]["W=4"]["optimal"]],
        )
    )
    next_beta = recommend_next_beta(
        0.6,
        [{"best_score": -0.3307}, {"best_score": -0.3190}],
        [results["smolvlm_cycles"]["W=4"]["optimal"]],
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload = {
        "protocol": "dream-rsi-arxiv-2609.14858v2",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "host": {
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "attempts_source": str(attempts_path),
        "attempts_count": full_tree.num_nonroot,
        "beta_grid": list(DEFAULT_BETA_GRID),
        "lambda_parallel": DEFAULT_LAMBDA_PARALLEL,
        "elapsed_ms": round(elapsed_ms, 2),
        "grid_plan_example": {
            "max_branches": grid_plan.max_branches,
            "max_attempts_per_branch": grid_plan.max_attempts_per_branch,
            "max_parallelism": grid_plan.max_parallelism,
            "rounds_cap": grid_plan.rounds_cap,
            "rationale": grid_plan.rationale,
            "recommended_next_beta": next_beta,
        },
        "suites": results,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return payload
