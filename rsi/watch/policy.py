"""Policy-development blocks for the watch page: per-step verdicts under one cycle's policy root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi.policy_dev.states import RECORDED, StepStatus
from rsi.watch.files import STEP_RE, as_int, as_num, as_str, read_json

# Verdicts of steps 1 and up; anything else shows as rejected.
STEP_VERDICTS = RECORDED - {StepStatus.INCUMBENT}


def _dev_block(
    selection: dict[str, Any],
    steps: list[dict[str, Any]],
    selected_step: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "selected_step": selected_step,
        "incumbent_v": as_num(selection.get("incumbent_in_sample_V")),
        "selected_v": as_num(selection.get("selected_in_sample_V")),
        "heldout_delta": as_num(selection.get("heldout_delta_vs_incumbent")),
        "steps": steps,
    }


def policy_dev_block(study: Path, root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Policy development under one root: `step_NN/` dirs plus `evidence/history/step_NN`."""
    selection = read_json(study, root / "selection.json") or {}
    steps: list[dict[str, Any]] = []
    selected_step = None
    if root.is_dir():
        for sdir in sorted(root.iterdir()):
            match = STEP_RE.match(sdir.name)
            if not match or not sdir.is_dir() or int(match.group(1)) < 1:
                continue
            verdict = read_json(study, root / "evidence" / "history" / sdir.name / "verdict.json")
            step = int(match.group(1))
            if verdict is None:
                status, mean_v, delta, reason = "running", None, None, None
            else:
                status = str(verdict.get("status"))
                status = status if status in STEP_VERDICTS else StepStatus.REJECTED.value
                mean_v, delta = as_num(verdict.get("mean_v")), as_num(verdict.get("delta_vs_best"))
                reason = as_str(verdict.get("error"))
                if (
                    selection.get("selected_slot")
                    and verdict.get("slot") == selection["selected_slot"]
                ):
                    selected_step = step
            steps.append(
                {
                    "step": step,
                    "status": status,
                    "mean_v": mean_v,
                    "delta_vs_best": delta,
                    "reason": reason,
                    "attempts": as_int((verdict or {}).get("attempts")),
                    "slot": as_str((verdict or {}).get("slot")),
                }
            )
    pd_status = raw.get("status")
    return _dev_block(
        selection,
        steps,
        selected_step,
        pd_status if pd_status in ("pending", "running", "done", "skipped") else "pending",
    )


def variant_dev_block(study: Path) -> dict[str, Any] | None:
    """A finished policy-development root with no loop state: variants are the steps."""
    selection = read_json(study, study / "selection.json")
    if selection is None or not (study / "evidence" / "history").is_dir():
        return None
    steps = []
    selected_step = None
    for item in selection.get("variants") or []:
        if not isinstance(item, dict):
            continue
        step = as_int(item.get("step"))
        if selection.get("selected_slot") and item.get("slot") == selection["selected_slot"]:
            selected_step = step
        steps.append(
            {
                "step": step,
                "status": as_str(item.get("status")) or "running",
                "mean_v": as_num(item.get("mean_v")),
                "delta_vs_best": as_num(item.get("delta_vs_best")),
                "reason": as_str(item.get("error")),
                "attempts": as_int(item.get("attempts")),
                "slot": as_str(item.get("slot")),
            }
        )
    return _dev_block(selection, steps, selected_step, "done")
