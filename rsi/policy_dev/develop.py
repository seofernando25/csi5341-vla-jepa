"""Sequential LLM development of `OptimalPolicy` with replay-gated, monotonic selection (Dream-RSI App. D.2)."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.agents import (
    AgentSpec,
    agent_record,
    load_agent_spec,
)
from rsi.dream import leave_one_world_out
from rsi.objective import DEFAULT_LAMBDA_PARALLEL
from rsi.paths import (
    RSI_DIR,
    STUDY_DIR,
)
from rsi.policy_dev.agent_env import POLICY_DEV_PROMPT
from rsi.policy_dev.replay import replay_policy
from rsi.policy_dev.states import SCORED, STEP, StepStatus
from rsi.policy_dev.step import StepRun, check_plan_grid, plan_context, run_agent_step
from rsi.provenance import sha256_file
from rsi.replay_worker import save_tree_json
from rsi.synth import scenario_suite
from rsi.tree import DiscoveryTree

INCUMBENT_TEMPLATE = RSI_DIR / "policies" / "optimal.py"
DEFAULT_POLICY_TIMEOUT_S = 1200.0


def _recorded_path(path: Path | None, record_root: Path | None) -> str | None:
    """A path as recorded in evidence: relative to `record_root` when it is under it."""
    if path is None:
        return None
    if record_root is None:
        return str(path)
    try:
        return Path(path).resolve().relative_to(Path(record_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_history(evidence: Path, rec: dict[str, Any], verdict: dict[str, Any]) -> None:
    d = evidence / "history" / f"step_{rec['step']:02d}"
    d.mkdir(parents=True, exist_ok=True)
    if rec["text"] is not None:
        (d / "method.py").write_text(rec["text"], encoding="utf-8")
    if rec["result"] is not None:
        (d / "scaling_curve.json").write_text(
            json.dumps(rec["result"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (d / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")


def _write_best_evidence(evidence: Path, best: dict[str, Any]) -> None:
    (evidence / "beta_sweep.json").write_text(
        json.dumps(best["result"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "policy_execution_traces.jsonl").write_text(best["traces"], encoding="utf-8")


def _verdict(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": rec["step"],
        "slot": rec["slot"],
        "status": rec["status"],
        "mean_v": rec.get("mean_v"),
        "delta_vs_best": rec.get("delta_vs_best"),
        "error": rec["error"],
        "attempts": rec["attempts"],
        "grid_plan": rec.get("plan"),
    }


def _summary_variant(rec: dict[str, Any], record_root: Path | None = None) -> dict[str, Any]:
    return {
        "slot": rec["slot"],
        "step": rec["step"],
        "status": rec["status"],
        "mean_v": rec.get("mean_v"),
        "delta_vs_best": rec.get("delta_vs_best"),
        "error": rec["error"],
        "attempts": rec["attempts"],
        "grid_plan": rec.get("plan"),
        "policy_path": _recorded_path(rec["path"], record_root),
        "policy_sha256": sha256_file(rec["path"]) if rec["path"] is not None else None,
    }


def develop_policies_llm(
    history: list[DiscoveryTree],
    *,
    cycle_id: str,
    incumbent_path: Path | None = None,
    output_dir: Path,
    live_manifests: list[dict[str, Any]] | None = None,
    m: int = 4,
    max_parallelism: int = 4,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    agent_cmd: str | list[str] | None = None,
    agent: AgentSpec | None = None,
    timeout: float = DEFAULT_POLICY_TIMEOUT_S,
    use_bwrap: bool = True,
    prompt_path: Path | None = None,
    record_root: Path | None = None,
) -> dict[str, Any]:
    """Develop m-1 sequential rewrites of the incumbent; select argmax mean V on history.

    `record_root` is the directory recorded policy paths are made relative to (the study dir);
    without it they stay absolute, as calibration output does.
    """
    if not history:
        raise ValueError("Policy development requires at least 1 frozen DiscoveryTree in history")
    if m < 2:
        raise ValueError("m must be >= 2 (incumbent + at least 1 rewritten candidate)")

    inc_src = Path(incumbent_path) if incumbent_path is not None else INCUMBENT_TEMPLATE
    out_root = Path(output_dir)
    evidence = out_root / "evidence"
    manifests = list(live_manifests or ())
    world_paths = [
        save_tree_json(t, evidence / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(history)
    ]
    (evidence / "live_cycle_manifests.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "reference").mkdir(parents=True, exist_ok=True)
    for name in ("api.py", "tree.py", "objective.py"):
        shutil.copyfile(RSI_DIR / name, evidence / "reference" / name)

    spec = agent or load_agent_spec(record_root or STUDY_DIR)
    run = StepRun(
        cycle_id=cycle_id,
        history=history,
        world_paths=world_paths,
        evidence=evidence,
        out_root=out_root,
        steps=m - 1,
        max_parallelism=max_parallelism,
        lambda_parallel=lambda_parallel,
        prompt_path=Path(prompt_path) if prompt_path is not None else POLICY_DEV_PROMPT,
        agent_cmd=agent_cmd,
        agent=spec,
        timeout=timeout,
        use_bwrap=use_bwrap,
        plan_context=plan_context(manifests, max_parallelism),
        study_dir=Path(record_root) if record_root is not None else None,
    )

    inc_text = inc_src.read_text(encoding="utf-8")
    inc_dir = out_root / "step_00"
    inc_dir.mkdir(parents=True, exist_ok=True)
    (inc_dir / "method.py").write_text(inc_text, encoding="utf-8")
    shutil.copyfile(inc_src, inc_dir / "candidate.py")
    inc_plan = check_plan_grid(run, inc_dir / "candidate.py")
    inc_result, inc_traces = run.replay(
        inc_dir / "candidate.py", inc_dir / "replay", world_paths, evidence
    )
    best: dict[str, Any] = {
        "step": 0,
        "slot": "m0_incumbent",
        "status": StepStatus.INCUMBENT.value,
        "error": None,
        "attempts": 0,
        "text": inc_text,
        "result": inc_result,
        "traces": inc_traces,
        "plan": inc_plan,
        "path": inc_dir / "candidate.py",
        "mean_v": inc_result["mean_v"],
        "delta_vs_best": 0.0,
    }
    records: list[dict[str, Any]] = [best]
    _write_history(evidence, best, _verdict(best))

    for step in range(1, m):
        _write_best_evidence(evidence, best)
        rec = run_agent_step(run, step, best["text"], best["mean_v"])
        if rec["status"] == StepStatus.VALID:
            better = rec["mean_v"] > best["mean_v"]
            STEP.advance(rec, StepStatus.ACCEPTED if better else StepStatus.NOT_BETTER)
            if better:
                best = rec
        elif rec["status"] == StepStatus.TIMEOUT_SALVAGED and rec["mean_v"] > best["mean_v"]:
            best = rec
        _write_history(evidence, rec, _verdict(rec))
        records.append(rec)

    valid = [r for r in records if r["status"] in SCORED]
    world_ids = [t.world_id for t in history]
    mean_by_slot = {r["slot"]: r["mean_v"] for r in valid}
    selected = max(valid, key=lambda r: (r["mean_v"], -r["step"]))
    winner_path = out_root / "selected_policy.py"
    shutil.copyfile(selected["path"], winner_path)

    terms = {
        r["slot"]: {
            wid: (w["hypervolume"], w["penalty"]) for wid, w in r["result"]["worlds"].items()
        }
        for r in valid
    }
    lowo_report = (
        leave_one_world_out(terms, world_ids, lambda_parallel) if len(history) >= 2 else None
    )

    heldout_dir = out_root / "heldout"
    heldout_paths = [
        save_tree_json(t, heldout_dir / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(scenario_suite("heldout"))
    ]
    heldout_mean: dict[str, float] = {}
    for r in valid:
        res, _ = replay_policy(
            r["path"],
            heldout_paths,
            ws_dir=heldout_dir / "replay" / r["slot"],
            evidence_dir=heldout_dir,
            max_parallelism=max_parallelism,
            lambda_parallel=lambda_parallel,
            use_bwrap=use_bwrap,
        )
        heldout_mean[r["slot"]] = round(float(res["mean_v"]), 6)
    incumbent_heldout = heldout_mean["m0_incumbent"]

    summary = {
        "protocol": "dream-rsi-policy-dev",
        "cycle_id": cycle_id,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "num_worlds": len(history),
        "world_ids": world_ids,
        "max_parallelism": max_parallelism,
        "lambda_parallel": lambda_parallel,
        "selected_slot": selected["slot"],
        "selected_policy_path": _recorded_path(winner_path, record_root),
        "incumbent_in_sample_V": round(float(mean_by_slot["m0_incumbent"]), 6),
        "selected_in_sample_V": round(float(selected["mean_v"]), 6),
        "in_sample_V_by_slot": {k: round(float(v), 6) for k, v in mean_by_slot.items()},
        "heldout": {
            "num_worlds": len(heldout_paths),
            "mean_V_by_slot": heldout_mean,
        },
        "heldout_delta_vs_incumbent": round(heldout_mean[selected["slot"]] - incumbent_heldout, 6),
        "lowo": (
            {
                "mean_selected_reward": round(float(lowo_report["mean_selected_reward"]), 6),
                "mean_incumbent_reward": round(float(lowo_report["mean_incumbent_reward"]), 6),
                "folds_selected_beats_incumbent": int(
                    lowo_report["folds_selected_beats_incumbent"]
                ),
                "folds_selected_below_incumbent": int(
                    lowo_report["folds_selected_below_incumbent"]
                ),
                "mean_held_out_reward_by_policy": {
                    k: round(float(v), 6)
                    for k, v in lowo_report["mean_held_out_reward_by_policy"].items()
                },
            }
            if lowo_report is not None
            else None
        ),
        "variants": [_summary_variant(r, record_root) for r in records],
        "agent": agent_record(None if agent_cmd is not None else spec, run.results),
    }
    (out_root / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
