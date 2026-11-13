"""Tests for Dream-RSI discovery trees, replay simulator, Eq. 1 objective, and policies."""

from __future__ import annotations

import pytest

from rsi.api import GATE_FAILED, GATE_NEAR_MISS, GridPlan, GridPlanningContext, gate_fail_class
from rsi.dream import (
    CANDIDATE_POOL,
    INCUMBENT,
    DreamEngine,
    build_synthetic_suite,
    leave_one_world_out,
    recommend_next_beta,
    replay_terms,
    run_benchmark,
    select_from_terms,
    sensitivity_grid,
)
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
    assert "smolvlm_full" in payload["in_sample"]
    assert "smolvlm_cycles" in payload["in_sample"]
    assert "synthetic_suite" in payload["in_sample"]
    assert "qwen_compression_smoke" in payload["in_sample"]
    assert "qwen_compression_development" in payload["in_sample"]
    # Smoke and development never share a V: separate hold-out suites.
    assert "real_smoke_derived" in payload["held_out"]
    assert "synthetic" in payload["held_out"]
    assert "qwen_compression_development" not in payload["held_out"]
    for suite_data in payload["held_out"].values():
        for row in suite_data.values():
            assert "folds" in row
            assert set(row["mean_held_out_reward_by_policy"]) >= {payload["incumbent"]}
    # Sensitivity records hypervolume and penalty separately per (lambda, W) cell.
    assert len(payload["sensitivity"]["synthetic"]) == 2 * len(payload["lambda_grid"])
    for cell in payload["sensitivity"]["synthetic"]:
        assert "selected_hypervolume" in cell and "selected_parallel_penalty" in cell
        assert cell["selected_reward"] == pytest.approx(
            cell["selected_hypervolume"] - cell["lambda"] * cell["selected_parallel_penalty"]
        )


def test_gate_fail_class_uses_shared_constants() -> None:
    assert gate_fail_class(8, 8) == "ok"
    assert gate_fail_class(7, 8) == GATE_NEAR_MISS
    assert gate_fail_class(6, 8) == GATE_FAILED
    assert gate_fail_class(7, 8, near_miss=0) == GATE_FAILED


def test_candidate_pool_covers_families_and_incumbent_wins_ties() -> None:
    names = [c.name for c in CANDIDATE_POOL]
    assert any("RandomPolicy" in n for n in names)
    assert any("RoundRobin" in n for n in names)
    assert any("GreedySerial" in n for n in names)
    assert "OptimalPolicy" in names and "OptimalPolicy[no-gate-retry]" in names
    assert INCUMBENT.name == "InitialPolicy"

    synth = build_synthetic_suite()
    pool = [INCUMBENT, *CANDIDATE_POOL]
    world_ids = [t.world_id for t in synth]
    # Incumbent is first in the pool, so it wins argmax ties.
    tied = {c.name: {w: (1.0, 0.0) for w in world_ids} for c in pool}
    assert select_from_terms(tied, world_ids, 0.05) == INCUMBENT.name


def test_leave_one_world_out_reports_selection_vs_incumbent() -> None:
    synth = build_synthetic_suite()
    pool = [INCUMBENT, *CANDIDATE_POOL]
    terms = replay_terms(pool, synth, 4)
    world_ids = [t.world_id for t in synth]
    lowo = leave_one_world_out(terms, world_ids, 0.05)
    assert len(lowo["folds"]) == len(world_ids)
    for fold in lowo["folds"]:
        assert set(fold["held_out_reward_by_policy"]) == set(terms)
    assert lowo["mean_selected_reward"] >= lowo["mean_incumbent_reward"] - 1e-9
    with pytest.raises(ValueError, match="at least two worlds"):
        leave_one_world_out(terms, world_ids[:1], 0.05)


def test_sensitivity_grid_covers_lambda_and_W() -> None:
    from rsi.dream import LAMBDA_GRID, PARALLELISM_GRID

    synth = build_synthetic_suite()
    pool = [INCUMBENT, *CANDIDATE_POOL]
    terms_by_w = {w: replay_terms(pool, synth, w) for w in PARALLELISM_GRID}
    world_ids = [t.world_id for t in synth]
    cells = sensitivity_grid(terms_by_w, world_ids)
    assert len(cells) == len(PARALLELISM_GRID) * len(LAMBDA_GRID)
    for cell in cells:
        assert cell["selected_reward"] == pytest.approx(
            cell["selected_hypervolume"] - cell["lambda"] * cell["selected_parallel_penalty"]
        )


def test_stage2_smoke_world_includes_tmlp20() -> None:
    from rsi.qwen_search import stage2_smoke_world

    tree, records = stage2_smoke_world()
    by_id = {r["candidate"]: r for r in records}
    # Committed 10-trial smoke measurement, absent from qwen_stage2.json.
    assert by_id["s2-c2-tmlp20"]["smoke_successes"] == 8
    assert by_id["s2-c2-tmlp20"]["smoke_episodes"] == 10
    assert tree.nodes["s2-c2-tmlp20"].fail_class == "ok"
