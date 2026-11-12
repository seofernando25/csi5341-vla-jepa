"""Tests for Dream-RSI discovery trees, replay simulator, Eq. 1 objective, and policies."""

from __future__ import annotations

import pytest

from rsi.api import GridPlan, GridPlanningContext
from rsi.dream import DreamEngine, build_synthetic_suite, recommend_next_beta, run_benchmark
from rsi.objective import BetaPoint, compute_eq1_score, evaluate_policy_on_tree, pareto_frontier
from rsi.policies import InitialPolicy, OptimalPolicy
from rsi.qwen_search import B16_RESIDENT_BYTES, LiveQwenQuestion, build_stage2_candidates
from rsi.tree import (
    ReplayQuestion,
    from_attempts_jsonl,
    from_compression_summary,
    make_synthetic_tree,
    partition_attempts_by_root_cycle,
)


def test_load_attempts_jsonl_and_partition() -> None:
    full = from_attempts_jsonl("studies/dream-rsi/attempts.jsonl")
    assert full.num_nonroot == 27
    assert full.baseline_score == pytest.approx(-0.4074)
    assert full.max_gain > 0.0

    cycles = partition_attempts_by_root_cycle("studies/dream-rsi/attempts.jsonl")
    assert len(cycles) == 3
    assert sum(t.num_nonroot for t in cycles) == 27


def test_replay_question_prefix_and_legality() -> None:
    tree = make_synthetic_tree(
        "prefix-test",
        branches=[
            [(0.1, "ok"), (0.3, "ok")],
            [(0.2, "implementation_failure"), (0.5, "ok")],
        ],
        baseline_score=0.0,
    )
    q = ReplayQuestion(tree, max_parallelism=2)
    assert q.observed() == {}
    assert q.legal_roots() == ["b0_a0", "b1_a0"]
    assert q.legal_actions() == ["b0_a0", "b1_a0"]

    # Cannot probe unrevealed child before parent
    with pytest.raises(ValueError, match="illegal cell"):
        q.probe_batch(["b0_a1"])

    # Cannot exceed max_parallelism
    with pytest.raises(ValueError, match="exceeds max_parallelism"):
        q.probe_batch(["b0_a0", "b1_a0", "b0_a1"])

    # Cannot duplicate
    with pytest.raises(ValueError, match="duplicate"):
        q.probe_batch(["b0_a0", "b0_a0"])

    obs = q.probe_batch(["b0_a0", "b1_a0"])
    assert len(obs) == 2
    assert q.decision_rounds == 1
    assert q.effective_sequential_rounds == 1
    assert q.budget_spent == 2
    assert q.legal_roots() == []
    assert set(q.legal_actions()) == {"b0_a1", "b1_a1"}


def test_eq1_pareto_and_objective_exact() -> None:
    pts = [(2, 0.5), (4, 0.5), (3, 0.4), (5, 1.0)]
    frontier = pareto_frontier(pts)
    assert frontier == [(2, 0.5), (5, 1.0)]

    beta_pts = [
        BetaPoint(
            beta=0.0,
            probes=2,
            decision_rounds=1,
            effective_sequential_rounds=1,
            best_score=0.5,
            attainment=0.5,
        ),
        BetaPoint(
            beta=0.5,
            probes=5,
            decision_rounds=2,
            effective_sequential_rounds=2,
            best_score=1.0,
            attainment=1.0,
        ),
        BetaPoint(
            beta=1.0,
            probes=0,
            decision_rounds=0,
            effective_sequential_rounds=0,
            best_score=0.0,
            attainment=0.0,
        ),
    ]
    # B_i = 10 -> HV = ((0.5 - 0)*(10 - 2) + (1.0 - 0.5)*(10 - 5)) / 10 = (4.0 + 2.5)/10 = 0.65
    # penalty = (1/2 + 2/5 + 0) / 3 = 0.9 / 3 = 0.30
    reward, hv, pen, out_frontier = compute_eq1_score(10, beta_pts, lambda_parallel=0.1)
    assert out_frontier == [(0, 0.0), (2, 0.5), (5, 1.0)]
    assert hv == pytest.approx(0.65)
    assert pen == pytest.approx(0.30)
    assert reward == pytest.approx(0.65 - 0.1 * 0.30)


def test_optimal_policy_schedule_and_grid_plan() -> None:
    pol = OptimalPolicy({"beta": 0.6})
    s_low = pol._schedule(0.0)
    s_high = pol._schedule(1.0)
    assert s_high["target_roots"] > s_low["target_roots"]
    assert s_high["patience_rounds"] > s_low["patience_rounds"]

    plan = pol.plan_grid(
        GridPlanningContext(
            iteration=2,
            max_parallelism=4,
            recent_live_manifests=[{"plateaued": True, "deepest_improvement": 3}],
        )
    )
    assert isinstance(plan, GridPlan)
    assert plan.max_branches == 8
    assert plan.max_attempts_per_branch == 5
    assert plan.max_parallelism == 4

    assert recommend_next_beta(0.6, [{"best_score": 0.1}, {"best_score": 0.2}], [{}]) == 0.6


def test_optimal_beats_or_matches_initial_on_replay() -> None:
    synth = build_synthetic_suite()
    engine = DreamEngine(max_parallelism=4)
    sel = engine.select_policy(InitialPolicy, [OptimalPolicy], synth)
    assert sel["selected_reward"] >= sel["incumbent_reward"]
    assert sel["selected_policy"] == "OptimalPolicy"

    # Verify non-degenerate beta sweep on a multi-branch tree
    score = evaluate_policy_on_tree(OptimalPolicy, synth[0], max_parallelism=4)
    probes_across_beta = {p.beta: p.probes for p in score.beta_points}
    assert max(probes_across_beta.values()) >= min(probes_across_beta.values())


def test_load_compression_summary_and_replay() -> None:
    smoke_tree, dev_tree = from_compression_summary("studies/compression/summary.csv")
    assert smoke_tree.num_nonroot == 162
    assert dev_tree.num_nonroot == 23
    assert smoke_tree.max_gain == pytest.approx(0.823, abs=0.01)
    assert dev_tree.max_gain == pytest.approx(0.849, abs=0.01)

    engine = DreamEngine(max_parallelism=4)
    sel = engine.select_policy(InitialPolicy, [OptimalPolicy], [smoke_tree, dev_tree])
    assert sel["selected_reward"] >= sel["incumbent_reward"]


def test_stage2_live_qwen_question_mock(tmp_path) -> None:
    cands = build_stage2_candidates()
    assert len(cands) == 15

    def fake_eval(c) -> dict:
        return {
            "candidate": c.cell_id,
            "phase": "smoke",
            "status": "completed",
            "successes": 8 if c.attempt == 0 else 6,
            "episodes": 10,
            "resident_bytes": B16_RESIDENT_BYTES * (0.16 - 0.02 * c.attempt),
        }

    q = LiveQwenQuestion(
        candidates=cands,
        evaluator=fake_eval,
        journal_path=tmp_path / "journal.jsonl",
        max_parallelism=4,
    )
    res = OptimalPolicy({"beta": 0.6}).solve(q, budget=6)
    assert res.total_probes == 6
    assert res.best_score == pytest.approx(0.84)


def test_run_benchmark_smoke(tmp_path) -> None:
    out_file = tmp_path / "bench.json"
    payload = run_benchmark(output_path=out_file, parallelism_levels=(1, 4))
    assert out_file.exists()
    assert payload["attempts_count"] == 27
    assert "smolvlm_full" in payload["suites"]
    assert "smolvlm_cycles" in payload["suites"]
    assert "synthetic_suite" in payload["suites"]
    assert "qwen_compression" in payload["suites"]
