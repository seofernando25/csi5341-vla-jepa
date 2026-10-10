"""Probe registered structural strengths between observed smoke outcomes.

This narrows tested brackets, not a proof of monotonic robot performance or a
global compression optimum. It never reads reserved confirmation outcomes.
"""

import subprocess
import sys
from collections import defaultdict

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.compression.queue import ledger

STUDY = ROOT / "studies/compression"


def strength(row):
    change = row["changes"][0]
    return change.get("fraction", change.get("count", 0))


def select(rows, minimum_successes):
    groups = defaultdict(list)
    for row in rows:
        if (
            len(row["changes"]) != 1
            or row["changes"][0]["method"] not in {"channels", "heads", "layers"}
            or row["phases"].get("screen", {}).get("status") != "completed"
        ):
            continue
        change = row["changes"][0]
        groups[(change["method"], change["scope"], change.get("criterion", "default"))].append(row)
    selected, evidence = [], {}
    for key, group in sorted(groups.items()):
        completed = [
            row for row in group if row["phases"].get("smoke", {}).get("status") == "completed"
        ]
        pending = [row for row in group if "smoke" not in row["phases"]]
        passing = [
            row
            for row in completed
            if row["phases"]["smoke"]["summary"]["successes"] >= minimum_successes
        ]
        failing = [row for row in completed if row not in passing]
        evidence["/".join(key)] = {
            "completed_smokes": {
                row["candidate"]: {
                    "strength": strength(row),
                    "successes": row["phases"]["smoke"]["summary"]["successes"],
                }
                for row in completed
            },
            "nonmonotonic_observed_outcomes": any(
                strength(failed) < strength(passed) for failed in failing for passed in passing
            ),
        }
        if not pending:
            continue
        if not passing:
            # Try an untested milder setting when all completed settings failed.
            choices = [
                row for row in pending if not failing or strength(row) < min(map(strength, failing))
            ]
            if choices:
                selected.append(min(choices, key=strength)["candidate"])
            continue
        lower = max(map(strength, passing))
        upper = min(
            [strength(row) for row in failing if strength(row) > lower],
            default=max(map(strength, group)),
        )
        choices = [row for row in pending if lower < strength(row) <= upper]
        if choices:
            midpoint = (lower + upper) / 2
            selected.append(
                min(choices, key=lambda row: (abs(strength(row) - midpoint), -strength(row)))[
                    "candidate"
                ]
            )
    return selected, evidence


def main():
    execution = read_json(STUDY / "execution_plan.json")
    baseline = read_json(STUDY / "measurements/B16/smoke/run.json")
    if baseline["status"] != "completed":
        raise ValueError("Complete the matched baseline smoke cohort first")
    minimum = baseline["summary"]["successes"] - execution["development_smoke_tolerance_episodes"]
    for index in range(execution["structural_boundary_rounds"]):
        if (ROOT / "outputs/compression/STOP").exists():
            return
        path = STUDY / f"boundary_round_{index + 1}.json"
        if path.exists():
            recorded = read_json(path)
        else:
            selected, evidence = select(ledger(), minimum)
            recorded = {
                "candidates": selected,
                "minimum_smoke_successes": minimum,
                "smoke_evidence": evidence,
                "execution_plan_sha256": file_hash(STUDY / "execution_plan.json"),
                "final_outcomes_used": False,
                "interpretation": "Development smoke brackets; no monotonicity or final-success claim",
            }
            write_json(path, recorded)
        if not recorded["candidates"]:
            break
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.compression.queue",
                "--phase",
                "smoke",
                "--candidates",
                *recorded["candidates"],
            ],
            cwd=ROOT,
            check=True,
        )
    # Joint refinements registered from preliminary evidence run before the
    # parent creates its development selection, including when that parent
    # began before the refinement was registered.
    from evaluation.compression.mission import smoke
    from evaluation.compression.refinements import register

    candidates = [
        *execution.get("post_boundary_joint_candidates", []),
        *register(execution),
        *execution.get("supplemental_deployment_endpoints", []),
    ]
    for candidate in candidates:
        if (ROOT / "outputs/compression/STOP").exists():
            return
        smoke(candidate)


if __name__ == "__main__":
    main()
