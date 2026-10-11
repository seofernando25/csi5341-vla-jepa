"""Run the registered component screens sequentially on the local GPU."""

import argparse
import fcntl
import os
import subprocess
import sys

from evaluation.common import ROOT, read_json, write_json

DOMAINS = [
    "language MLP",
    "language attention",
    "language heads",
    "language depth",
    "vision MLP",
    "vision attention",
    "vision heads",
    "vision depth",
    "main/DeepStack mergers",
    "embeddings",
    "conditioning tokens",
    "visual tokens",
    "action MLP",
    "action attention",
    "action heads",
    "action depth",
    "action interfaces",
    "solver steps",
    "generated action horizon",
    "replanning horizon",
    "world-model residency",
]


def domains(recipe):
    found = set()
    for change in recipe["changes"]:
        method, scope = change["method"], change.get("scope")
        if method == "layers":
            found.add(
                {"text": "language depth", "vision": "vision depth", "action": "action depth"}[
                    scope
                ]
            )
        elif method == "heads":
            found.add(
                {
                    "text_attention": "language heads",
                    "vision_attention": "vision heads",
                    "action_attention": "action heads",
                }[scope]
            )
        elif method in {"int8_embedding", "compact_embedding"}:
            found.add("embeddings")
        elif method in {
            "conditioning_tokens",
            "visual_tokens",
            "solver_steps",
            "action_horizon",
            "replanning",
            "world_remove",
        }:
            found.add(
                {
                    "conditioning_tokens": "conditioning tokens",
                    "visual_tokens": "visual tokens",
                    "solver_steps": "solver steps",
                    "action_horizon": "generated action horizon",
                    "replanning": "replanning horizon",
                    "world_remove": "world-model residency",
                }[method]
            )
        elif method == "attention_matching":
            if not change.get("control"):
                found.update(["action attention", "action MLP", "action interfaces"])
        else:
            found.update(
                {
                    "text": ["language MLP", "language attention"],
                    "text_mlp": ["language MLP"],
                    "text_attention": ["language attention"],
                    "vision": ["vision MLP", "vision attention", "main/DeepStack mergers"],
                    "vision_blocks": ["vision MLP", "vision attention"],
                    "vision_mlp": ["vision MLP"],
                    "vision_attention": ["vision attention"],
                    "merger": ["main/DeepStack mergers"],
                    "action": ["action MLP", "action attention", "action interfaces"],
                    "action_mlp": ["action MLP"],
                    "action_attention": ["action attention"],
                    "action_interfaces": ["action interfaces"],
                }.get(scope, [])
            )
    return sorted(found)


def ledger():
    plan = read_json(ROOT / "studies/compression/plan.json")
    rows = []
    for candidate in plan["candidates"]:
        recipe = read_json(ROOT / f"studies/compression/recipes/{candidate}.json")
        phases = {}
        for phase in ["screen", "smoke", "development", "size", "final", "benchmark"]:
            path = ROOT / f"studies/compression/measurements/{candidate}/{phase}/run.json"
            if path.exists():
                result = read_json(path)
                phases[phase] = {
                    "status": result["status"],
                    "summary": result.get("summary"),
                    "error": result.get("error"),
                }
        rows.append(
            {
                "candidate": candidate,
                "family": recipe["family"],
                "changes": recipe["changes"],
                "domains": domains(recipe),
                "phases": phases,
            }
        )
    coverage = {}
    for domain in DOMAINS:
        candidates = [r for r in rows if domain in r["domains"]]
        coverage[domain] = {
            "registered": len(candidates),
            "screened": sum(
                r["phases"].get("screen", {}).get("status") == "completed" for r in candidates
            ),
            "rollout_tested": sum(
                any(
                    r["phases"].get(p, {}).get("status") == "completed"
                    for p in ["smoke", "development", "final"]
                )
                for r in candidates
            ),
            "development_100": sum(
                r["phases"].get("development", {}).get("status") == "completed" for r in candidates
            ),
            "confirmed_500": sum(
                r["phases"].get("final", {}).get("status") == "completed" for r in candidates
            ),
        }
    write_json(
        ROOT / "studies/compression/coverage.json",
        {
            "domains": coverage,
            "major_domains_with_rollout_evidence": sum(
                v["rollout_tested"] > 0 for v in coverage.values()
            ),
            "major_domains_total": len(DOMAINS),
            "interpretation": "Screened action agreement is not robot success. Smoke=10, development=100, final=500 trials.",
            "candidates": rows,
        },
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["screen", "smoke", "development", "size", "final", "benchmark"],
        default="screen",
    )
    parser.add_argument("--candidates", nargs="*")
    args = parser.parse_args()
    lock = (ROOT / "outputs/compression/gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    plan = read_json(ROOT / "studies/compression/plan.json")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONPATH="src:.",
        HF_HUB_OFFLINE="1",
        MUJOCO_GL="egl",
    )
    for candidate in args.candidates or plan["candidates"]:
        path = ROOT / f"studies/compression/measurements/{candidate}/{args.phase}/run.json"
        if path.exists():
            print(f"Preserving {candidate}/{args.phase}: {read_json(path)['status']}", flush=True)
            continue
        if (ROOT / "outputs/compression/STOP").exists():
            print("Stop requested at candidate boundary", flush=True)
            break
        log = ROOT / f"outputs/compression/{candidate}-{args.phase}.log"
        print(f"Starting {candidate}/{args.phase}", flush=True)
        with log.open("w") as stream:
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "evaluation.compression.study",
                        "run",
                        "--recipe",
                        f"studies/compression/recipes/{candidate}.json",
                        "--phase",
                        args.phase,
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=14400,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                if path.exists():
                    record = read_json(path)
                    record.update(
                        status="failed",
                        error_type="TimeoutExpired",
                        error="Candidate exceeded the four-hour runtime limit",
                    )
                    write_json(path, record)
        print(f"Finished {candidate}/{args.phase}: exit {returncode}", flush=True)
        ledger()
    ledger()


if __name__ == "__main__":
    main()
