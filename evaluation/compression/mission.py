"""Resume the sequential component study through development selection.

Reserved confirmation trials require a separately frozen selection. This
runner never reads or uses final outcomes to select or tune configurations.
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.compression.queue import ledger

STUDY = ROOT / "studies/compression"
RUNTIME = ROOT / "outputs/compression"


def event(stage, **fields):
    row = {"at": datetime.now(UTC).isoformat(), "stage": stage, **fields}
    with (RUNTIME / "mission-journal.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    write_json(RUNTIME / "mission-state.json", row)
    print(json.dumps(row), flush=True)


def result(candidate, phase):
    path = STUDY / "measurements" / candidate / phase / "run.json"
    return read_json(path) if path.exists() else None


def stopped():
    return (RUNTIME / "STOP").exists()


def queue(phase, candidates):
    if not candidates or stopped():
        return
    event(phase, candidates=candidates)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.compression.queue",
            "--phase",
            phase,
            "--candidates",
            *candidates,
        ],
        cwd=ROOT,
        check=True,
    )


def smoke(candidate):
    queue("screen", [candidate])
    screen = result(candidate, "screen")
    if screen and screen["status"] == "completed":
        queue("smoke", [candidate])


def additional_smokes(plan, maximum=None):
    groups = defaultdict(list)
    for candidate in plan["candidates"]:
        screen = result(candidate, "screen")
        if not screen or screen["status"] != "completed" or result(candidate, "smoke"):
            continue
        changes = screen["candidate"]["changes"]
        if len(changes) != 1:
            continue
        change = changes[0]
        groups[
            (
                change["method"],
                change.get("scope", ""),
                change.get("bits", 0),
                change.get("sparse_kernel", False),
            )
        ].append(screen)
    selected = []
    for key, screens in sorted(groups.items()):
        # Probe the stronger endpoint of each family; screen errors are only
        # used for scheduling, never as a robot-success estimate.
        chosen = min(
            screens,
            key=lambda r: (
                r["summary"]["peak_allocated_bytes"],
                -r["candidate"]["changes"][0].get("fraction", 0),
                -r["candidate"]["changes"][0].get("count", 0),
            ),
        )
        selected.append(chosen["candidate"]["id"])
    return selected if maximum is None else selected[:maximum]


def development_selection(execution):
    path = STUDY / "development_selection.json"
    if path.exists():
        return read_json(path)["candidates"]
    baseline = result("B16", "smoke")
    if not baseline or baseline["status"] != "completed":
        raise RuntimeError("Matched baseline smoke trials must complete before selection")
    minimum = baseline["summary"]["successes"] - execution["development_smoke_tolerance_episodes"]
    rows = ledger()
    eligible = []
    for row in rows:
        screen, tested = row["phases"].get("screen", {}), row["phases"].get("smoke", {})
        if (
            row["candidate"] == "B16"
            or screen.get("status") != "completed"
            or tested.get("status") != "completed"
        ):
            continue
        if tested["summary"]["successes"] >= minimum:
            eligible.append(row)
    sort_key = lambda row: (
        row["phases"]["screen"]["summary"]["peak_allocated_bytes"],
        row["phases"]["screen"]["summary"]["pipeline_median_ms"],
    )
    selected = sorted([r for r in eligible if r["family"] == "joint compression"], key=sort_key)[:8]
    conservative = "deploy-nf4-text-bf16-embedding"
    if conservative not in [r["candidate"] for r in selected]:
        selected.extend(r for r in eligible if r["candidate"] == conservative)
    structural = defaultdict(list)
    for row in eligible:
        if len(row["changes"]) == 1 and row["changes"][0]["method"] in {
            "layers",
            "heads",
            "channels",
        }:
            change = row["changes"][0]
            structural[(change["method"], change["scope"])].append(row)
    for _, candidates in sorted(structural.items()):
        selected.append(min(candidates, key=sort_key))
    identities = [r["candidate"] for r in selected]
    for row in sorted(eligible, key=sort_key):
        if row["candidate"] not in identities:
            identities.append(row["candidate"])
    identities = identities[: execution["development_maximum_candidates"]]
    # Outcome evidence used for this decision is copied into the immutable
    # selection manifest. Future resumes preserve it rather than reselecting.
    evidence = {
        r["candidate"]: {
            "smoke": r["phases"]["smoke"]["summary"],
            "screen": r["phases"]["screen"]["summary"],
        }
        for r in eligible
    }
    write_json(
        path,
        {
            "candidates": ["B16", *identities],
            "minimum_smoke_successes": minimum,
            "baseline_smoke": baseline["summary"],
            "eligible_evidence": evidence,
            "execution_plan_sha256": file_hash(STUDY / "execution_plan.json"),
            "final_outcomes_used": False,
        },
    )
    return ["B16", *identities]


def main():
    execution = read_json(STUDY / "execution_plan.json")
    plan = read_json(STUDY / "plan.json")
    event(
        "started",
        plan_sha256=file_hash(STUDY / "plan.json"),
        execution_plan_sha256=file_hash(STUDY / "execution_plan.json"),
    )
    for candidate in execution["priority_component_rollouts"]:
        if stopped():
            event("stopped_at_candidate_boundary")
            return
        smoke(candidate)
    queue("screen", plan["candidates"])
    if stopped():
        event("stopped_at_candidate_boundary")
        return
    extra = STUDY / "additional_smoke_selection.json"
    if not extra.exists():
        write_json(extra, {"candidates": additional_smokes(plan), "final_outcomes_used": False})
    queue("smoke", read_json(extra)["candidates"])
    if stopped():
        event("stopped_at_candidate_boundary")
        return
    event("structural_boundary_probes")
    subprocess.run(
        [sys.executable, "-m", "evaluation.compression.boundaries"], cwd=ROOT, check=True
    )
    if stopped():
        event("stopped_at_candidate_boundary")
        return
    selected = development_selection(execution)
    queue("development", selected)
    completed = [
        candidate
        for candidate in selected
        if result(candidate, "development")
        and result(candidate, "development")["status"] == "completed"
    ]
    queue("size", completed)
    ledger()
    if stopped():
        event("stopped_at_candidate_boundary")
    else:
        event(
            "ready_for_final_selection",
            note="Review development evidence and freeze at most three configurations before reserved confirmation",
        )


if __name__ == "__main__":
    main()
