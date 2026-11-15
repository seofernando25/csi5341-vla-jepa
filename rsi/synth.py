"""Synthetic calibration worlds with known structure for Dream-RSI policy development."""

from __future__ import annotations

import random
from collections.abc import Callable

from rsi.api import GATE_FAILED, GATE_NEAR_MISS
from rsi.objective import DEFAULT_BETA_GRID, evaluate_policy_over_history
from rsi.tree import DiscoveryTree, make_synthetic_tree

OK = "ok"
RUNTIME = "runtime_failure"
IMPLEMENTATION = "implementation_failure"
Chain = list[tuple[float, str]]

FAMILIES: tuple[str, ...] = (
    "deep_winner",
    "wide_flat",
    "late_bloomer",
    "deceptive_start",
    "cliff",
    "all_fail",
    "noisy_plateau",
    "runtime_minefield",
)
_SPLITS: dict[str, range] = {"train": range(3), "heldout": range(100, 103)}


def _branch_count(rng: random.Random) -> int:
    return rng.randint(3, 6)


def _length(rng: random.Random) -> int:
    return rng.randint(2, 4)


def _rising(rng: random.Random, start: float, length: int) -> list[float]:
    """Strictly increasing ok scores starting at `start`, steps in [0.05, 0.15], capped at 0.9."""
    scores = [start]
    for _ in range(length - 1):
        scores.append(min(0.9, scores[-1] + rng.uniform(0.05, 0.15)))
    return [round(s, 3) for s in scores]


def _deep_winner(rng: random.Random) -> list[Chain]:
    """One branch rises at every attempt to the global max; other branches are flat and low."""
    nb = _branch_count(rng)
    winner = rng.randrange(nb)
    chains: list[Chain] = []
    for b in range(nb):
        length = _length(rng)
        if b == winner:
            chains.append([(s, OK) for s in _rising(rng, 0.2, length)])
        else:
            flat = round(rng.uniform(0.02, 0.1), 3)
            chains.append([(flat, OK)] * length)
    return chains


def _wide_flat(rng: random.Random) -> list[Chain]:
    """Every root passes within 0.02 of the others; children never beat their root."""
    nb = _branch_count(rng)
    base = rng.uniform(0.3, 0.5)
    chains: list[Chain] = []
    for _ in range(nb):
        root = round(base + rng.uniform(-0.008, 0.008), 3)
        chain: Chain = [(root, OK)]
        for _ in range(_length(rng) - 1):
            chain.append((round(root - rng.uniform(0.0, 0.02), 3), OK))
        chains.append(chain)
    return chains


def _late_bloomer(rng: random.Random) -> list[Chain]:
    """The global-max branch opens with a gate near miss (score 0) and recovers from attempt 1."""
    nb = _branch_count(rng)
    winner = rng.randrange(nb)
    chains: list[Chain] = []
    for b in range(nb):
        length = _length(rng)
        if b == winner:
            ok_scores = _rising(rng, 0.3, length - 1)
            chains.append([(0.0, GATE_NEAR_MISS)] + [(s, OK) for s in ok_scores])
        else:
            flat = round(rng.uniform(0.05, 0.25), 3)
            chains.append([(flat, OK)] * length)
    return chains


def _deceptive_start(rng: random.Random) -> list[Chain]:
    """Best root's children regress or fail; second-best root's chain ends at the global max."""
    nb = _branch_count(rng)
    deceptive: Chain = [(0.6, OK)]
    for _ in range(_length(rng) - 1):
        deceptive.append((0.4, OK) if rng.random() < 0.5 else (0.0, GATE_FAILED))
    climber: Chain = [(s, OK) for s in (round(x, 3) for x in _linspace(0.3, 0.8, _length(rng)))]
    chains: list[Chain] = [deceptive, climber]
    for _ in range(nb - 2):
        flat = round(rng.uniform(0.05, 0.2), 3)
        chains.append([(flat, OK)] * _length(rng))
    return chains


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _cliff(rng: random.Random) -> list[Chain]:
    """Each branch rises to a peak one step before a GATE_FAILED final attempt."""
    chains: list[Chain] = []
    for _ in range(_branch_count(rng)):
        length = _length(rng)
        base = rng.uniform(0.1, 0.3)
        step = rng.uniform(0.05, 0.15)
        chain: Chain = [(round(base + k * step, 3), OK) for k in range(length - 1)]
        chains.append(chain + [(0.0, GATE_FAILED)])
    return chains


def _all_fail(rng: random.Random) -> list[Chain]:
    """No node passes; root classes cycle through the three failure kinds."""
    kinds = (GATE_FAILED, GATE_NEAR_MISS, RUNTIME)
    chains: list[Chain] = []
    for b in range(_branch_count(rng)):
        chain: Chain = [(0.0, kinds[b % len(kinds)])]
        chain += [(0.0, rng.choice(kinds)) for _ in range(_length(rng) - 1)]
        chains.append(chain)
    return chains


def _noisy_plateau(rng: random.Random) -> list[Chain]:
    """All nodes pass with scores 0.3 plus uniform noise in [-0.03, 0.03]."""
    return [
        [(round(0.3 + rng.uniform(-0.03, 0.03), 3), OK) for _ in range(_length(rng))]
        for _ in range(_branch_count(rng))
    ]


def _runtime_minefield(rng: random.Random) -> list[Chain]:
    """Roughly half the roots fail at runtime with failing children; branch 0 holds the max."""
    chains: list[Chain] = []
    for b in range(_branch_count(rng)):
        length = _length(rng)
        if b == 0:
            chains.append([(s, OK) for s in (round(x, 3) for x in _linspace(0.2, 0.7, length))])
        elif b % 2 == 1:
            kind = RUNTIME if rng.random() < 0.5 else IMPLEMENTATION
            chains.append([(0.0, kind)] * length)
        else:
            chains.append([(0.1, OK)] * length)
    return chains


_GENERATORS: dict[str, Callable[[random.Random], list[Chain]]] = {
    "deep_winner": _deep_winner,
    "wide_flat": _wide_flat,
    "late_bloomer": _late_bloomer,
    "deceptive_start": _deceptive_start,
    "cliff": _cliff,
    "all_fail": _all_fail,
    "noisy_plateau": _noisy_plateau,
    "runtime_minefield": _runtime_minefield,
}


def make_world(family: str, seed: int) -> DiscoveryTree:
    """Build one world of `family`, drawn deterministically from `random.Random(f"{family}:{seed}")`."""
    if family not in _GENERATORS:
        raise ValueError(f"unknown family {family!r}")
    rng = random.Random(f"{family}:{seed}")
    chains = _GENERATORS[family](rng)
    return make_synthetic_tree(f"synth-{family}-{seed}", branches=chains, baseline_score=0.0)


def scenario_suite(split: str) -> list[DiscoveryTree]:
    """Eight families x three seeds for `split` in {"train", "heldout"}."""
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [make_world(family, seed) for family in FAMILIES for seed in _SPLITS[split]]


def _depths(tree: DiscoveryTree) -> dict[str, int]:
    """Number of probes on the root-to-node path (roots have depth 1)."""
    depth: dict[str, int] = {}

    def visit(cid: str) -> int:
        if cid not in depth:
            parent = tree.nodes[cid].parent_id
            depth[cid] = 1 if parent in (None, "root") else 1 + visit(parent)
        return depth[cid]

    for cid in tree.nodes:
        visit(cid)
    return depth


def upper_bound(tree: DiscoveryTree, max_parallelism: int, lam: float) -> float:
    """Upper bound on Eq. (1) reward V for any prefix-only policy on `tree`.

    Hypervolume: a policy with n probes reveals only nodes of depth <= n (prefixes are
    prefix-closed), so its attainment is at most F(n), the best attainment at depth <= n.
    Substituting F for the frontier gives HV <= (1/B) sum_n (F(n) - F(n-1)) (B - n), B = num_nonroot.
    This is tighter than the single-depth form (B-d)/B, which is not valid: a shallow partial
    gain can beat it (B=3, root 0.5, depth-2 child 1.0 gives HV 0.5 > 1/3).
    Penalty: any beta with a positive attainment spends >= 1 probe, and every round costs
    ceil(k/W) >= k/W, so each such beta contributes >= 1/W to the beta-mean penalty; with a
    positive HV at least one beta does, giving penalty >= 1 / (|beta grid| * W). If HV is zero,
    V = -lam * penalty <= 0. Hence V <= max(0, HV_bound - lam / (|grid| * W)).
    Returns 0.0 when no node improves on the baseline.
    """
    gain = tree.max_gain
    if gain <= 0.0:
        return 0.0
    depth = _depths(tree)
    best_at: dict[int, float] = {}
    for cid, node in tree.nodes.items():
        if node.fail_class == OK and node.score is not None:
            attainment = min(1.0, max(0.0, (node.score - tree.baseline_score) / gain))
            best_at[depth[cid]] = max(best_at.get(depth[cid], 0.0), attainment)
    num = tree.num_nonroot
    prefix_best = 0.0
    hypervolume = 0.0
    for n in range(1, max(depth.values()) + 1):
        new_best = max(prefix_best, best_at.get(n, 0.0))
        hypervolume += (new_best - prefix_best) * (num - n)
        prefix_best = new_best
    hypervolume /= float(num)
    penalty_floor = lam / (len(DEFAULT_BETA_GRID) * max_parallelism)
    return max(0.0, hypervolume - penalty_floor)


def calibration_table(
    policies: dict[str, type],
    worlds: list[DiscoveryTree],
    *,
    max_parallelism: int = 2,
    lam: float = 0.05,
) -> list[dict]:
    """One row per (policy, family): mean replay V, mean upper bound and regret = bound - V."""
    unknown = {w.world_id for w in worlds if _family_of(w.world_id) not in FAMILIES}
    if unknown:
        raise ValueError(f"worlds with unknown family: {sorted(unknown)}")
    present = {_family_of(w.world_id) for w in worlds}
    rows: list[dict] = []
    for family in (f for f in FAMILIES if f in present):
        members = [w for w in worlds if _family_of(w.world_id) == family]
        mean_bound = sum(upper_bound(w, max_parallelism, lam) for w in members) / len(members)
        for name, policy_cls in policies.items():
            result = evaluate_policy_over_history(
                policy_cls,
                members,
                max_parallelism=max_parallelism,
                lambda_parallel=lam,
            )
            mean_v = float(result["pareto"]["reward"])
            rows.append(
                {
                    "policy": name,
                    "family": family,
                    "mean_v": mean_v,
                    "mean_bound": mean_bound,
                    "regret": mean_bound - mean_v,
                }
            )
    return rows


def _family_of(world_id: str) -> str:
    """Recover the family from a world id of the form `synth-<family>-<seed>`."""
    return world_id.removeprefix("synth-").rsplit("-", 1)[0]
