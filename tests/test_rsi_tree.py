"""Replay question: prefix reveal rules, metadata visibility and determinism."""

from __future__ import annotations

import pytest

from rsi.tree import DiscoveryTree, ReplayQuestion, TreeNode, make_synthetic_tree


def _node(cid: str, parent: str | None, tags: tuple[str, ...] = ()) -> TreeNode:
    return TreeNode(
        cell_id=cid,
        branch=0 if parent is None else 1,
        attempt=0 if parent is None else 1,
        parent_id=parent,
        seq=0,
        score=0.5,
        evaluated=True,
        valid=True,
        fail_class="ok",
        error=None,
        mechanism_family="fam",
        summary=cid,
        cycle=0,
        tags=tags,
    )


def _fan_tree() -> DiscoveryTree:
    # r1 has children x then y (stored order); r2 is a second root; z hangs off x.
    nodes = [
        _node("r1", None),
        _node("x", "r1", tags=("secret-relation",)),
        _node("y", "r1"),
        _node("r2", None),
        _node("z", "x"),
    ]
    return DiscoveryTree("fan", 0.0, nodes)


def test_reveal_exposes_only_stored_children_in_order() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    assert q.legal_actions() == ["r1", "r2"]
    q.probe_batch(["r1"])
    assert q.legal_actions() == ["r2", "x", "y"]
    assert "z" not in q.legal_actions()
    q.probe_batch(["x"])
    assert q.legal_actions() == ["r2", "y", "z"]


def test_duplicate_batch_rejected() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    with pytest.raises(ValueError, match="duplicate"):
        q.probe_batch(["r1", "r1"])


def test_over_width_batch_rejected() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=1)
    with pytest.raises(ValueError, match="max_parallelism"):
        q.probe_batch(["r1", "r2"])


def test_illegal_batch_rejected() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    with pytest.raises(ValueError, match="illegal"):
        q.probe_batch(["x"])


def test_meta_of_unrevealed_illegal_cell_raises() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    with pytest.raises(KeyError):
        q.meta("z")
    with pytest.raises(KeyError):
        q.meta("missing")


def test_meta_of_legal_cell_has_no_tags() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    q.probe_batch(["r1"])
    meta = q.meta("x")
    assert meta.tags == ()
    assert (meta.branch, meta.attempt, meta.parent_id) == (1, 1, "r1")


def test_meta_of_revealed_cell_keeps_tags() -> None:
    q = ReplayQuestion(_fan_tree(), max_parallelism=4)
    q.probe_batch(["r1"])
    q.probe_batch(["x"])
    assert q.meta("x").tags == ("secret-relation",)


def _trace(tree: DiscoveryTree) -> tuple:
    q = ReplayQuestion(tree, max_parallelism=2)
    seen = []
    for batch in (["b0_a0", "b1_a0"], ["b0_a1"]):
        seen.extend(o.cell_id for o in q.probe_batch(batch))
    return (
        tuple(seen),
        tuple(q.legal_actions()),
        q.budget_spent,
        q.decision_rounds,
        q.effective_sequential_rounds,
        q.best_so_far,
    )


def test_replay_twice_gives_identical_trace() -> None:
    tree = make_synthetic_tree("det", [[(0.2, "ok"), (0.4, "ok")], [(0.1, "ok")]])
    assert _trace(tree) == _trace(tree)
