"""Policy-development agent: write M archived policy revisions per cycle from replay trajectories.

The paper's "policy improvement" step needs an agent that reads replay trajectories and
proposes new policy versions. This module does that deterministically and CPU-only: it
replays the base pool on the current history, derives a tuned default beta (best mean
attainment over the beta sweep), counts near-miss failures, and archives M=4
`OptimalPolicy` revisions with provenance under `rsi/policies/revisions/`.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.api import GATE_NEAR_MISS
from rsi.dream import CANDIDATE_POOL, INCUMBENT, Candidate, leave_one_world_out, replay_terms
from rsi.objective import DEFAULT_BETA_GRID, DEFAULT_LAMBDA_PARALLEL, evaluate_policy_over_history
from rsi.tree import DiscoveryTree

REVISIONS_DIR = Path("rsi/policies/revisions")


def _code_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def trajectory_stats(history: list[DiscoveryTree]) -> dict[str, Any]:
    """Count fail classes and best-beta attainment signals across replay worlds."""
    fail_counts: dict[str, int] = {}
    for tree in history:
        for node in tree.nodes.values():
            fail_counts[node.fail_class] = fail_counts.get(node.fail_class, 0) + 1
    total = sum(fail_counts.values())
    return {
        "worlds": [t.world_id for t in history],
        "nodes": total,
        "fail_counts": fail_counts,
        "near_miss": fail_counts.get(GATE_NEAR_MISS, 0),
    }


def best_beta_for(
    cls: Any,
    history: list[DiscoveryTree],
    extra_config: dict[str, Any] | None = None,
    max_parallelism: int = 4,
) -> float:
    """Beta with the best mean attainment over history (ties resolve to the middle)."""
    ev = evaluate_policy_over_history(
        cls, history, max_parallelism=max_parallelism, extra_config=extra_config
    )
    scored = [(p["mean_attainment"], -abs(p["beta"] - 0.5), p["beta"]) for p in ev["per_beta"]]
    return float(max(scored)[2])


def develop_policies(
    history: list[DiscoveryTree],
    round_id: str,
    m: int = 4,
    base_pool: list[Candidate] | None = None,
    max_parallelism: int = 4,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    revisions_dir: str | Path = REVISIONS_DIR,
) -> dict[str, Any]:
    """Derive M revisions from replay trajectories and archive them with provenance.

    Selection is argmax including the incumbent on held-out (leave-one-world-out) V.
    Returns the report plus loadable `candidates`.
    """
    if m != 4:
        raise ValueError("the protocol archives M=4 revisions per cycle")
    if len(history) < 2:
        raise ValueError("held-out selection needs at least two worlds")
    pool = [INCUMBENT, *(base_pool if base_pool is not None else CANDIDATE_POOL)]
    world_ids = [t.world_id for t in history]
    terms = replay_terms(pool, history, max_parallelism)
    lowo = leave_one_world_out(terms, world_ids, lambda_parallel)
    mean_held = lowo["mean_held_out_reward_by_policy"]
    order = [c.name for c in pool]
    selected = max(mean_held, key=lambda n: (mean_held[n], -order.index(n)))
    sel_cand = next(c for c in pool if c.name == selected)
    stats = trajectory_stats(history)
    tuned_beta = best_beta_for(sel_cand.cls, history, sel_cand.config, max_parallelism)
    keep_retry = stats["near_miss"] > 0

    serial_beta = 0.25 if tuned_beta != 0.25 else 0.5
    wide_beta = 1.0 if tuned_beta != 1.0 else 0.75
    specs = [
        (
            "tuned",
            {"beta": tuned_beta, "gate_retry": keep_retry},
            f"selected-class best mean attainment beta={tuned_beta}",
        ),
        (
            "serial",
            {"beta": serial_beta, "gate_retry": keep_retry},
            "low-beta serial exploitation arm",
        ),
        (
            "wide",
            {"beta": wide_beta, "gate_retry": keep_retry},
            "high-beta parallel exploration arm",
        ),
        (
            "retry-flip",
            {"beta": tuned_beta, "gate_retry": not keep_retry},
            "gate-retry ablated against the tuned arm",
        ),
    ]
    out_dir = Path(revisions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = _code_head()
    stamped = datetime.now(UTC).isoformat()
    candidates: list[Candidate] = []
    records: list[dict[str, Any]] = []
    for tag, config, rationale in specs:
        name = f"OptimalPolicy[{round_id}-{tag}]"
        record = {
            "name": name,
            "round_id": round_id,
            "base": "OptimalPolicy",
            "config": config,
            "rationale": rationale,
            "parent_selection": selected,
            "parent_config": sel_cand.config,
            "held_out_mean_V": {n: round(mean_held[n], 6) for n in order},
            "trajectory_stats": stats,
            "beta_grid": list(DEFAULT_BETA_GRID),
            "lambda_parallel": lambda_parallel,
            "max_parallelism": max_parallelism,
            "code_head": head,
            "host": {"python": platform.python_version(), "machine": platform.machine()},
            "created_utc": stamped,
        }
        (out_dir / f"{round_id}-{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        candidates.append(Candidate(name, sel_cand.cls, dict(config)))
    return {
        "round_id": round_id,
        "selected_parent": selected,
        "tuned_beta": tuned_beta,
        "trajectory_stats": stats,
        "mean_held_out_V": {n: round(mean_held[n], 6) for n in order},
        "revisions": [r["name"] for r in records],
        "records": records,
        "candidates": candidates,
    }


def load_revision_candidates(
    round_id: str, revisions_dir: str | Path = REVISIONS_DIR
) -> list[Candidate]:
    """Load archived revision candidates for one round (deterministic filename order)."""
    from rsi.policies.optimal import OptimalPolicy

    out: list[Candidate] = []
    for path in sorted(Path(revisions_dir).glob(f"{round_id}-*.json")):
        record = json.loads(path.read_text())
        out.append(Candidate(record["name"], OptimalPolicy, dict(record["config"])))
    return out
