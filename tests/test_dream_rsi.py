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
    assert "real_smoke_coarse" in payload["held_out"]
    assert "real_smoke_fine" in payload["held_out"]
    assert "live_rounds" in payload["held_out"]
    assert "synthetic" in payload["held_out"]
    assert "qwen_compression_development" not in payload["held_out"]
    for suite_data in payload["held_out"].values():
        for row in suite_data.values():
            assert "folds" in row
            assert set(row["mean_held_out_reward_by_policy"]) >= {payload["incumbent"]}
    # Multi-seed stability: one winner-per-seed record per (lambda, W) cell.
    assert set(payload["stability_seeds"]) == {0, 1, 2, 3, 4}
    for stab in payload["selection_stability"].values():
        assert 0.0 <= stab["unanimous_fraction"] <= 1.0
        assert stab["cells"]
        for cell in stab["cells"]:
            assert set(cell["winners_by_seed"]) == {"0", "1", "2", "3", "4"}
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
    assert any("EpsilonGreedy" in n for n in names)
    assert any("ShallowFirst" in n for n in names)
    assert any("TwoStageGate" in n for n in names)
    assert any("IncumbentWithExceptions" in n for n in names)
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
    from rsi.stage2 import stage2_smoke_world

    tree, records = stage2_smoke_world()
    by_id = {r["candidate"]: r for r in records}
    # Committed 10-trial smoke measurement, absent from qwen_stage2.json.
    assert by_id["s2-c2-tmlp20"]["smoke_successes"] == 8
    assert by_id["s2-c2-tmlp20"]["smoke_episodes"] == 10
    assert tree.nodes["s2-c2-tmlp20"].fail_class == "ok"


def test_pool_for_seed_isolates_stochastic_draws() -> None:
    from rsi.dream import STABILITY_SEEDS, pool_for_seed

    assert tuple(STABILITY_SEEDS) == (0, 1, 2, 3, 4)
    for seed in STABILITY_SEEDS:
        pool = pool_for_seed(seed)
        names = [c.name for c in pool]
        assert names[0] == INCUMBENT.name
        assert f"RandomPolicy[seed={seed}]" in names
        assert f"EpsilonGreedyPolicy[eps=0.2,seed={seed}]" in names
        assert sum("RandomPolicy" in n for n in names) == 1
        assert sum("EpsilonGreedy" in n for n in names) == 1


def test_split_cycle_branches_partitions_disjointly() -> None:
    from rsi.tree import partition_attempts_by_root_cycle, split_cycle_branches

    cycles = partition_attempts_by_root_cycle("studies/dream-rsi/attempts.jsonl")
    branches, fragments = split_cycle_branches(cycles)
    assert branches and fragments
    by_world: dict[str, int] = {}
    for t in [*branches, *fragments]:
        for cid in t.order:
            by_world[cid] = by_world.get(cid, 0) + 1
    assert set(by_world) == {cid for t in cycles for cid in t.order}
    assert set(by_world.values()) == {1}
    for t in branches:
        assert t.num_nonroot >= 2


def test_compression_smoke_families_are_disjoint_chains() -> None:
    from rsi.tree import compression_smoke_by_family, from_compression_summary

    full = from_compression_summary("studies/compression/summary.csv")[0]
    families = compression_smoke_by_family("studies/compression/summary.csv")
    assert len(families) >= 10
    seen: dict[str, int] = {}
    for t in families:
        assert t.num_nonroot >= 4
        for cid in t.order:
            seen[cid] = seen.get(cid, 0) + 1
    assert set(seen.values()) == {1}
    assert set(seen) <= set(full.order)


def test_live_round_worlds_are_smoke_only_and_chained() -> None:
    import json as _json

    from rsi.tree import from_live_records

    cases = (("live_round1", "qwen-round1", 5), ("live_round2", "qwen-round2", 2))
    for rnd, world_id, n in cases:
        with open(f"studies/dream-rsi/{rnd}.json") as f:
            recs = _json.load(f)["world_records"]
        tree = from_live_records(recs, world_id, provenance=rnd)
        assert tree.world_id == world_id
        assert tree.num_nonroot == n
        assert all(tree.nodes[c].episodes == 10 for c in tree.order)
        for c in tree.order:
            parent = tree.nodes[c].parent_id
            assert parent == "root" or parent in tree.nodes


def test_new_policies_solve_legally_and_seed_is_deterministic() -> None:
    from rsi.dream import build_synthetic_suite
    from rsi.policies import (
        EpsilonGreedyPolicy,
        IncumbentWithExceptions,
        ShallowFirstPolicy,
        TwoStageGatePolicy,
    )
    from rsi.tree import ReplayQuestion

    synth = build_synthetic_suite()
    for cls in (ShallowFirstPolicy, TwoStageGatePolicy, IncumbentWithExceptions):
        res = cls({}).solve(ReplayQuestion(synth[0], max_parallelism=4))
        assert res.total_probes > 0
    r1 = EpsilonGreedyPolicy({"epsilon": 0.2, "seed": 3}).solve(
        ReplayQuestion(synth[1], max_parallelism=4)
    )
    r2 = EpsilonGreedyPolicy({"epsilon": 0.2, "seed": 3}).solve(
        ReplayQuestion(synth[1], max_parallelism=4)
    )
    assert r1.total_probes == r2.total_probes > 0
    assert r1.best_score == r2.best_score


def test_selection_stability_flags_seed_luck() -> None:
    from rsi.dream import (
        STABILITY_SEEDS,
        build_synthetic_suite,
        pool_for_seed,
        replay_terms,
        selection_stability,
    )

    synth = build_synthetic_suite()
    world_ids = [t.world_id for t in synth]
    by_seed = {s: {4: replay_terms(pool_for_seed(s), synth, 4)} for s in STABILITY_SEEDS}
    stab = selection_stability(by_seed, world_ids)
    assert 0.0 <= stab["unanimous_fraction"] <= 1.0
    assert len(stab["cells"]) == 5  # 5 lambdas x 1 W level
    for cell in stab["cells"]:
        assert set(cell["winners_by_seed"]) == {str(s) for s in STABILITY_SEEDS}


def test_wilson_interval_covers_reported_dev_rates() -> None:
    from rsi.gate import wilson_interval

    assert wilson_interval(8, 10) == [0.4902, 0.9433]
    assert wilson_interval(88, 100) == [0.8019, 0.93]


def test_gate_binomial_rates_are_exact() -> None:
    from rsi.gate import binomial_pmf, combined_oc, legacy_oc, prob_at_least, sequential_oc

    assert binomial_pmf(10, 8, 0.5) == pytest.approx(45 / 1024)
    assert prob_at_least(10, 8, 0.5) == pytest.approx(56 / 1024)
    # Legacy 8/10 screen kills a good (p=0.87) branch with ~13% probability.
    assert legacy_oc(0.87)["p_reject_overall"] == pytest.approx(0.13076421)
    assert legacy_oc(0.87)["expected_trials"] == pytest.approx(10.0)
    # Near-miss confirm (7/10 -> disjoint 10-trial screen) cuts that to ~4.4%.
    seq = sequential_oc(0.87)
    assert seq["p_screen_pass"] == pytest.approx(0.86923579)
    assert seq["p_near_miss"] == pytest.approx(0.09945945)
    assert seq["p_reject_overall"] == pytest.approx(0.04431049)
    assert seq["p_reject_overall"] < legacy_oc(0.87)["p_reject_overall"]
    assert seq["expected_trials"] == pytest.approx(10.99459454)
    # Confirm helps bad branches slightly: more passes at p=0.5, still rare.
    assert sequential_oc(0.5)["p_pass_overall"] > legacy_oc(0.5)["p_pass_overall"]
    assert sequential_oc(0.5)["p_pass_overall"] == pytest.approx(0.06109619)
    # Full C0 -> screen -> confirm gate and its bad-branch counterpart.
    assert combined_oc(0.87)["p_reject_overall"] == pytest.approx(0.07525183)
    assert combined_oc(0.87)["expected_trials"] == pytest.approx(16.63863424)
    with pytest.raises(ValueError, match="probability"):
        legacy_oc(1.5)


def test_gate_decisions_and_phase_separation() -> None:
    from rsi.gate import (
        assert_disjoint_seeds,
        c0_decision,
        c0_oc,
        confirm_decision,
        screen_decision,
    )

    assert screen_decision(8) == "pass"
    assert screen_decision(10) == "pass"
    assert screen_decision(7) == "confirm"
    assert screen_decision(6) == "reject"
    assert screen_decision(0) == "reject"
    assert confirm_decision(8) == "pass"
    assert confirm_decision(7) == "reject"
    assert c0_decision(3) == "reject"
    assert c0_decision(4) == "continue"
    assert c0_oc(0.87)["p_reject"] == pytest.approx(0.03237594)
    with pytest.raises(ValueError, match="successes"):
        screen_decision(11)
    assert_disjoint_seeds(("screen", [0, 1]), ("confirm", [10, 11]), ("dev", [20]))
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_seeds(("screen", [0, 1]), ("confirm", [1, 2]))
