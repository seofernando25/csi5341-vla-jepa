"""Eq. (1) replay objective: hand-computed worlds, Pareto frontier, parallel penalty."""

from __future__ import annotations

import pytest

from rsi.api import LLMDesignedMethod, SimResult, finalize_result
from rsi.objective import (
    BetaPoint,
    compute_eq1_score,
    evaluate_policy_on_tree,
    evaluate_policy_over_history,
    pareto_frontier,
)
from rsi.tree import make_synthetic_tree

TOL = 1e-4


class ScriptedPolicy(LLMDesignedMethod):
    """Probes a fixed list of batches, chosen by the beta in its config."""

    def solve(self, question, budget=None) -> SimResult:
        for batch in self.config["plans"][self.config["beta"]]:
            question.probe_batch(batch)
        return finalize_result(question, SimResult())


def _point(beta: float, probes: int, rounds: int, attainment: float) -> BetaPoint:
    return BetaPoint(
        beta=beta,
        probes=probes,
        decision_rounds=rounds,
        effective_sequential_rounds=rounds,
        best_score=0.0,
        attainment=attainment,
    )


def test_t1_chain_matches_hand_derivation() -> None:
    # Chain of 3 nodes (B = 3), scores 0.2, 0.5, 0.4, S = 0.5, W = 1.
    # beta=0 probes one node (a = 0.4); other betas probe all three (a = 1.0).
    # HV = (0.4 * (3 - 1)) / 3 = 0.26667, penalty = 1, V = 0.26667 - 0.05.
    tree = make_synthetic_tree("T1", [[(0.2, "ok"), (0.5, "ok"), (0.4, "ok")]])
    plans = {0.0: [["b0_a0"]]}
    plans.update({b: [["b0_a0"], ["b0_a1"], ["b0_a2"]] for b in (0.25, 0.5, 0.75, 1.0)})
    score = evaluate_policy_on_tree(
        ScriptedPolicy, tree, max_parallelism=1, extra_config={"plans": plans}
    )
    assert not score.degenerate
    assert score.reward == pytest.approx(0.21667, abs=TOL)


def test_t2_degenerate_world_is_penalty_only() -> None:
    # Two roots scoring 0.0 and -0.1: S = 0, so no node beats the baseline.
    # Every beta probes both (penalty 1), attainment is 0, V = -0.05.
    tree = make_synthetic_tree("T2", [[(0.0, "ok")], [(-0.1, "ok")]])
    plans = {b: [["b0_a0"], ["b1_a0"]] for b in (0.0, 0.25, 0.5, 0.75, 1.0)}
    extra = {"plans": plans}
    score = evaluate_policy_on_tree(ScriptedPolicy, tree, max_parallelism=1, extra_config=extra)
    assert score.degenerate
    assert score.normalized_hypervolume == pytest.approx(0.0)
    assert score.reward == pytest.approx(-0.05, abs=TOL)

    history = evaluate_policy_over_history(
        ScriptedPolicy, [tree], max_parallelism=1, extra_config=extra
    )
    assert history["pareto"]["reward"] == pytest.approx(-0.05, abs=TOL)


def test_t3_ties_with_width_two() -> None:
    # Three roots scoring 0.1, 0.3, 0.3 (B = 3, S = 0.3, W = 2).
    # beta < 0.5 probes {r1}: a = 1/3, penalty 1. beta >= 0.5 probes {r2, r3}: a = 1, penalty 1/2.
    # Frontier (1, 1/3), (2, 1): HV = (2/3 + 2/3) / 3 = 4/9; mean penalty = 0.7;
    # V = 4/9 - 0.05 * 0.7 = 0.40944. The brief's 0.54833 corresponds to B = 4, which is
    # inconsistent with three non-root nodes.
    tree = make_synthetic_tree("T3", [[(0.1, "ok")], [(0.3, "ok")], [(0.3, "ok")]])
    plans = {0.0: [["b0_a0"]], 0.25: [["b0_a0"]]}
    plans.update({b: [["b1_a0", "b2_a0"]] for b in (0.5, 0.75, 1.0)})
    score = evaluate_policy_on_tree(
        ScriptedPolicy, tree, max_parallelism=2, extra_config={"plans": plans}
    )
    assert score.parallel_penalty == pytest.approx(0.7)
    assert score.normalized_hypervolume == pytest.approx(4.0 / 9.0)
    assert score.reward == pytest.approx(0.40944, abs=TOL)


def test_pareto_frontier_drops_dominated_points() -> None:
    points = [(1, 0.2), (2, 0.5), (3, 0.4), (3, 0.6)]
    assert pareto_frontier(points) == [(1, 0.2), (2, 0.5), (3, 0.6)]


def test_pareto_frontier_collapses_duplicate_n() -> None:
    assert pareto_frontier([(2, 0.5), (2, 0.5), (2, 0.8), (1, 0.1)]) == [(1, 0.1), (2, 0.8)]


def test_penalty_uses_ceil_of_batch_over_width() -> None:
    # One batch of 3 probes at width 2 costs ceil(3 / 2) = 2 sequential rounds.
    points = [_point(0.0, probes=3, rounds=2, attainment=0.5)]
    _, _, penalty, _ = compute_eq1_score(total_nodes=5, points=points, lambda_parallel=0.05)
    assert penalty == pytest.approx(2.0 / 3.0)


def test_degenerate_world_excluded_from_history_mean() -> None:
    good = make_synthetic_tree("good", [[(0.5, "ok")], [(0.5, "ok")]])
    flat = make_synthetic_tree("flat", [[(0.0, "ok")], [(-0.1, "ok")]])
    extra = {"plans": {b: [["b0_a0"], ["b1_a0"]] for b in (0.0, 0.25, 0.5, 0.75, 1.0)}}
    history = evaluate_policy_over_history(
        ScriptedPolicy, [good, flat], max_parallelism=1, extra_config=extra
    )
    alone = evaluate_policy_over_history(
        ScriptedPolicy, [good], max_parallelism=1, extra_config=extra
    )
    assert history["num_worlds"] == 1
    assert history["worlds"][1]["degenerate"] is True
    assert history["pareto"]["reward"] == pytest.approx(alone["pareto"]["reward"])
