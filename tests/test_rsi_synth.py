"""Synthetic calibration worlds: determinism, family rules, splits, and the replay upper bound."""

from __future__ import annotations

from itertools import pairwise

import pytest

from rsi.api import GATE_FAILED, GATE_NEAR_MISS
from rsi.objective import evaluate_policy_on_tree
from rsi.policies.baselines import GreedySerialPolicy, RandomPolicy, RoundRobinPolicy
from rsi.synth import (
    FAMILIES,
    calibration_table,
    make_world,
    scenario_suite,
    upper_bound,
)
from rsi.tree import make_synthetic_tree

BASELINES = {
    "RandomPolicy": RandomPolicy,
    "RoundRobinPolicy": RoundRobinPolicy,
    "GreedySerialPolicy": GreedySerialPolicy,
}
SEEDS = range(5)


def _signature(tree):
    return [(n.cell_id, n.score, n.fail_class) for n in tree.nodes.values()]


def _chains(tree):
    chains: dict[int, list] = {}
    for node in tree.nodes.values():
        chains.setdefault(node.branch, []).append(node)
    return [sorted(chain, key=lambda n: n.attempt) for _, chain in sorted(chains.items())]


def _best_branch(tree):
    best = max(n.score for n in tree.nodes.values() if n.fail_class == "ok")
    return next(c for c in _chains(tree) if any(n.score == best for n in c))


def test_make_world_is_deterministic_and_seed_sensitive():
    for family in FAMILIES:
        assert _signature(make_world(family, 3)) == _signature(make_world(family, 3))
        assert _signature(make_world(family, 3)) != _signature(make_world(family, 4))


def test_shapes_are_within_declared_ranges():
    for family in FAMILIES:
        for seed in SEEDS:
            chains = _chains(make_world(family, seed))
            assert 3 <= len(chains) <= 6
            assert all(2 <= len(c) <= 4 for c in chains)


def test_deep_winner_rises_to_global_max_at_last_attempt():
    for seed in SEEDS:
        tree = make_world("deep_winner", seed)
        winner = _best_branch(tree)
        scores = [n.score for n in winner]
        assert all(n.fail_class == "ok" for n in winner)
        assert all(a < b for a, b in pairwise(scores))
        for chain in _chains(tree):
            if chain[0].branch != winner[0].branch:
                assert len({n.score for n in chain}) == 1
                assert max(n.score for n in chain) < scores[-1]


def test_wide_flat_roots_agree_and_children_never_beat_roots():
    for seed in SEEDS:
        tree = make_world("wide_flat", seed)
        roots = [c[0].score for c in _chains(tree)]
        assert max(roots) - min(roots) <= 0.02
        for chain in _chains(tree):
            assert all(n.fail_class == "ok" for n in chain)
            assert all(child.score <= chain[0].score for child in chain[1:])


def test_late_bloomer_winner_starts_with_near_miss_and_recovers():
    for seed in SEEDS:
        tree = make_world("late_bloomer", seed)
        winner = _best_branch(tree)
        assert winner[0].fail_class == GATE_NEAR_MISS and winner[0].score == 0.0
        recovered = [n.score for n in winner[1:]]
        assert all(n.fail_class == "ok" for n in winner[1:])
        assert all(a < b for a, b in pairwise(recovered))


def test_deceptive_start_best_root_regresses_and_second_root_wins():
    for seed in SEEDS:
        tree = make_world("deceptive_start", seed)
        chains = _chains(tree)
        best_root = max(chains, key=lambda c: c[0].score)
        assert best_root[0].score == 0.6
        for child in best_root[1:]:
            assert child.fail_class == GATE_FAILED or child.score < best_root[0].score
        second = sorted(chains, key=lambda c: c[0].score)[-2]
        assert second[0].score == 0.3
        assert second[-1].score == tree.best_score


def test_cliff_rises_then_fails_on_every_branch():
    for seed in SEEDS:
        tree = make_world("cliff", seed)
        for chain in _chains(tree):
            assert chain[-1].fail_class == GATE_FAILED and chain[-1].score == 0.0
            before = [n.score for n in chain[:-1]]
            assert all(n.fail_class == "ok" for n in chain[:-1])
            assert all(a < b for a, b in pairwise(before))
            assert max(before) <= tree.best_score
        peaks = [chain[-2].score for chain in _chains(tree)]
        assert tree.best_score == max(peaks)


def test_all_fail_has_no_passing_node_and_mixes_failure_kinds():
    for seed in SEEDS:
        tree = make_world("all_fail", seed)
        assert all(n.fail_class != "ok" for n in tree.nodes.values())
        kinds = {n.fail_class for n in tree.nodes.values()}
        assert {GATE_FAILED, GATE_NEAR_MISS, "runtime_failure"} <= kinds


def test_noisy_plateau_all_pass_near_0_3():
    for seed in SEEDS:
        tree = make_world("noisy_plateau", seed)
        assert all(n.fail_class == "ok" for n in tree.nodes.values())
        assert all(0.27 <= n.score <= 0.33 for n in tree.nodes.values())


def test_runtime_minefield_failing_roots_have_failing_children():
    for seed in SEEDS:
        tree = make_world("runtime_minefield", seed)
        chains = _chains(tree)
        failing = [c for c in chains if c[0].fail_class != "ok"]
        assert failing, "expected failing roots"
        for chain in failing:
            assert all(n.fail_class in ("runtime_failure", "implementation_failure") for n in chain)
        clean = [c for c in chains if c[0].fail_class == "ok"]
        assert clean and tree.best_score == max(c[-1].score for c in clean)
        assert _best_branch(tree)[0].branch == 0


def test_train_and_heldout_are_disjoint():
    train = scenario_suite("train")
    heldout = scenario_suite("heldout")
    assert len(train) == 24 and len(heldout) == 24
    assert not {w.world_id for w in train} & {w.world_id for w in heldout}
    train_sigs = {tuple(_signature(w)) for w in train}
    assert not train_sigs & {tuple(_signature(w)) for w in heldout}


@pytest.mark.parametrize("max_parallelism", [1, 2, 4])
def test_baseline_rewards_never_exceed_upper_bound(max_parallelism):
    lam = 0.05
    for world in scenario_suite("train") + scenario_suite("heldout"):
        bound = upper_bound(world, max_parallelism, lam)
        for policy_cls in BASELINES.values():
            reward = evaluate_policy_on_tree(
                policy_cls,
                world,
                max_parallelism=max_parallelism,
                lambda_parallel=lam,
            ).reward
            assert reward <= bound + 1e-9, (world.world_id, policy_cls.__name__)


def test_upper_bound_handles_shallow_partial_gain_counterexample():
    tree = make_synthetic_tree(
        "bound-case",
        branches=[[(0.5, "ok")], [(0.0, GATE_FAILED), (1.0, "ok")]],
        baseline_score=0.0,
    )
    # B = 3; F(1) = 0.5, F(2) = 1 -> HV bound = (0.5 * 2 + 0.5 * 1) / 3 = 0.5.
    assert upper_bound(tree, max_parallelism=2, lam=0.05) == pytest.approx(0.5 - 0.005)


def test_upper_bound_is_zero_without_a_passing_node():
    tree = make_synthetic_tree("none", branches=[[(0.0, GATE_FAILED)]], baseline_score=0.0)
    assert upper_bound(tree, max_parallelism=2, lam=0.05) == 0.0


def test_calibration_table_has_one_row_per_policy_and_family():
    worlds = [make_world(f, s) for f in ("deep_winner", "cliff") for s in (0, 1)]
    rows = calibration_table(BASELINES, worlds)
    keys = {(r["policy"], r["family"]) for r in rows}
    assert len(rows) == len(BASELINES) * 2
    assert keys == {(p, f) for p in BASELINES for f in ("deep_winner", "cliff")}
    for row in rows:
        assert set(row) == {"policy", "family", "mean_v", "mean_bound", "regret"}
        assert row["regret"] == pytest.approx(row["mean_bound"] - row["mean_v"])


def test_calibration_table_rejects_unknown_world_family():
    stray = make_synthetic_tree("mystery-7", branches=[[(0.2, "ok")]], baseline_score=0.0)
    with pytest.raises(ValueError):
        calibration_table(BASELINES, [stray])
