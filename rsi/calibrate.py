"""Calibrate the policy-update instructions on synthetic worlds (CPU/API only, no project evidence)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rsi.agent_backends import AgentSpec
from rsi.objective import DEFAULT_LAMBDA_PARALLEL
from rsi.policies.baselines import GreedySerialPolicy, RandomPolicy, RoundRobinPolicy
from rsi.policy_audit import load_policy_class
from rsi.policy_dev import INCUMBENT_TEMPLATE, POLICY_DEV_PROMPT, develop_policies_llm
from rsi.sandbox import REPO_ROOT
from rsi.synth import FAMILIES, calibration_table, make_world, scenario_suite
from rsi.tree import DiscoveryTree

DEFAULT_OUT_ROOT = "outputs/rsi-calibration"
RULE_KEYS = ("audit", "world_literal", "timeout", "output", "replay")
_AUDIT_PREFIXES = (
    "Forbidden",
    "Prefix-only",
    "Syntax error",
    "OptimalPolicy must",
    "Policy file must",
    "Failed importing",
    "Unable to create",
)
_OUTPUT_MARKERS = (
    "cannot open",
    "invalid output name",
    "is not a regular file",
    "exceeds",
    "is not valid UTF-8",
    "Agent exited",
)
_BASELINES = {
    "random": RandomPolicy,
    "round_robin": RoundRobinPolicy,
    "greedy_serial": GreedySerialPolicy,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_out_root(out_root: Path) -> Path:
    """Resolve `out_root`; refuse locations under `studies/` (calibration output is not evidence)."""
    resolved = Path(out_root).resolve()
    studies = (REPO_ROOT / "studies").resolve()
    if resolved == studies or studies in resolved.parents:
        raise SystemExit(
            f"--out-root {resolved} is under studies/; calibration output is not evidence"
        )
    return resolved


def classify_rule_break(error: str) -> str:
    """Map a rejected step's error message to one of RULE_KEYS (messages are the only record)."""
    if "timed out" in error:
        return "timeout"
    if "copies a recorded" in error or "node score" in error:
        return "world_literal"
    if any(marker in error for marker in _OUTPUT_MARKERS):
        return "output"
    if error.startswith(_AUDIT_PREFIXES):
        return "audit"
    return "replay"


def _train_worlds(families: list[str], seeds_per_family: int) -> list[DiscoveryTree]:
    return [make_world(f, seed) for f in families for seed in range(seeds_per_family)]


def _steps(summary: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for v in summary["variants"]:
        if v["step"] == 0:
            continue
        status = v["status"]
        if status in ("rejected", "timeout"):
            reason = v["error"]
        elif status == "timeout_salvaged":
            better = (v["delta_vs_best"] or 0.0) > 0
            reason = (
                "scored after timeout; mean_v above best so far"
                if better
                else "scored after timeout; mean_v not above best so far"
            )
        elif status == "accepted":
            reason = "mean_v above best so far"
        else:
            reason = "mean_v not above best so far"
        steps.append(
            {
                "step": v["step"],
                "status": status,
                "reason": reason,
                "mean_v": v["mean_v"],
                "delta_vs_best": v["delta_vs_best"],
                "repair_used": v["attempts"] > 1,
                "agent_seconds": None,
            }
        )
    return steps


def _rule_breaks(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(RULE_KEYS, 0)
    for s in steps:
        if s["status"] in ("rejected", "timeout"):
            counts[classify_rule_break(s["reason"] or "")] += 1
    return counts


def _verdict(selected_slot: str, train_delta: float, heldout_delta: float) -> str:
    if selected_slot == "m0_incumbent":
        return "no_change"
    if train_delta > 0 and heldout_delta >= 0:
        return "improved_and_generalised"
    if train_delta > 0 and heldout_delta < 0:
        return "overfit"
    return "no_change"


def run_calibration(
    *,
    tag: str,
    prompt_path: Path | None = None,
    m: int = 4,
    max_parallelism: int = 2,
    seeds_per_family: int = 1,
    families: list[str] | None = None,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    timeout: float = 300.0,
    out_root: Path = Path(DEFAULT_OUT_ROOT),
    use_bwrap: bool = True,
) -> dict[str, Any]:
    """Run one policy-update step on train worlds, score on held-out worlds, write calibration.json."""
    started = time.monotonic()
    fams = list(families) if families else list(FAMILIES)
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {unknown}")
    root = check_out_root(out_root)
    run_dir = root / tag
    history = _train_worlds(fams, seeds_per_family)
    prompt = Path(prompt_path) if prompt_path is not None else POLICY_DEV_PROMPT

    summary = develop_policies_llm(
        history,
        cycle_id=tag,
        output_dir=run_dir,
        m=m,
        max_parallelism=max_parallelism,
        agent_cmd=agent_cmd,
        agent=agent,
        timeout=timeout,
        use_bwrap=use_bwrap,
        prompt_path=prompt,
    )

    selected_path = Path(summary["selected_policy_path"])
    inc_mean = summary["in_sample_V_by_slot"]["m0_incumbent"]
    sel_mean = summary["selected_in_sample_V"]
    train_delta = round(sel_mean - inc_mean, 6)
    heldout_means = summary["heldout"]["mean_V_by_slot"]
    heldout_inc = heldout_means["m0_incumbent"]
    heldout_sel = heldout_means[summary["selected_slot"]]
    heldout_delta = round(heldout_sel - heldout_inc, 6)

    policies = {
        "incumbent": load_policy_class(INCUMBENT_TEMPLATE, unique_tag=f"{tag}_inc"),
        "selected": load_policy_class(selected_path, unique_tag=f"{tag}_sel"),
        **_BASELINES,
    }
    heldout_worlds = scenario_suite("heldout")
    rows = calibration_table(
        policies, heldout_worlds, max_parallelism=max_parallelism, lam=DEFAULT_LAMBDA_PARALLEL
    )
    steps = _steps(summary)
    payload = {
        "tag": tag,
        "prompt_path": str(prompt),
        "prompt_sha256": _sha256_file(prompt),
        "model": summary["agent"]["model"],
        "agent": summary["agent"],
        "m": m,
        "max_parallelism": max_parallelism,
        "lambda": DEFAULT_LAMBDA_PARALLEL,
        "train_worlds": [t.world_id for t in history],
        "steps": steps,
        "train": {
            "incumbent_mean_v": inc_mean,
            "selected_mean_v": sel_mean,
            "delta": train_delta,
        },
        "heldout": {
            "incumbent_mean_v": heldout_inc,
            "selected_mean_v": heldout_sel,
            "delta": heldout_delta,
        },
        "by_family": rows,
        "rule_breaks": _rule_breaks(steps),
        "verdict": _verdict(summary["selected_slot"], train_delta, heldout_delta),
        "selected_slot": summary["selected_slot"],
        "selected_sha256": _sha256_file(selected_path),
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    (run_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def format_summary(payload: dict[str, Any], path: Path) -> str:
    """Compact text table of steps, deltas and verdict."""
    lines = [f"calibration {payload['tag']}: verdict={payload['verdict']}"]
    lines.append(f"{'step':>4} {'status':<10} {'mean_v':>9} {'delta':>9}  reason")
    for s in payload["steps"]:
        mv = "-" if s["mean_v"] is None else f"{s['mean_v']:.4f}"
        dv = "-" if s["delta_vs_best"] is None else f"{s['delta_vs_best']:+.4f}"
        lines.append(f"{s['step']:>4} {s['status']:<10} {mv:>9} {dv:>9}  {s['reason']}")
    lines.append(f"train delta={payload['train']['delta']:+.4f}")
    lines.append(f"heldout delta={payload['heldout']['delta']:+.4f}")
    lines.append(f"rule_breaks={payload['rule_breaks']}")
    lines.append(f"calibration.json: {path}")
    return "\n".join(lines)
