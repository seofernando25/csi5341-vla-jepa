"""Freeze reviewed development choices before any reserved final trial."""

import argparse
import math
import zipfile
from datetime import UTC, datetime

from evaluation.analysis import paired_difference
from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.compression.protocol import (
    STUDY,
    matched_controls,
    scientific_sources,
    validated_episodes,
    verify_state_assets,
)
from evaluation.compression.queue import ledger


def freeze(candidates):
    path = STUDY / "final_selection.json"
    if path.exists():
        frozen = read_json(path)
        if frozen["candidates"] != candidates:
            raise ValueError("Preserve the frozen selection; no retuning from final outcomes")
        return frozen
    if (
        not 1 <= len(candidates) <= 3
        or len(set(candidates)) != len(candidates)
        or "B16" in candidates
    ):
        raise ValueError(
            "Select one to three distinct compressed candidates; B16 is added as control"
        )
    if any((STUDY / "measurements").glob("*/final/run.json")):
        raise ValueError("Reserved outcomes already exist; cannot create a new final selection")
    rows = ledger()
    coverage = read_json(STUDY / "coverage.json")
    if coverage["major_domains_with_rollout_evidence"] < math.ceil(
        0.8 * coverage["major_domains_total"]
    ):
        raise ValueError("At least 80% of major component domains need rollout evidence")
    if any(
        row["phases"].get("screen", {}).get("status") not in {"completed", "failed"} for row in rows
    ):
        raise ValueError("Finish registered component screens before final selection")
    baseline, baseline_rows = validated_episodes(STUDY / "measurements/B16/development")
    evidence, recipes, tensor_hashes = {}, {}, {}
    for candidate in ["B16", *candidates]:
        record, episodes = validated_episodes(STUDY / "measurements" / candidate / "development")
        matched_controls(baseline, record)
        if (
            candidate != "B16"
            and record["summary"]["successes"] < baseline["summary"]["successes"] - 5
        ):
            raise ValueError(f"{candidate} loses more than five development successes versus B16")
        size_path = STUDY / "measurements" / candidate / "size/run.json"
        size = read_json(size_path)
        if size["status"] != "completed":
            raise ValueError("Measure actual packed archive size before final selection")
        recipe_path = STUDY / "recipes" / f"{candidate}.json"
        recipes[candidate] = file_hash(recipe_path)
        if (
            record["recipe_sha256"] != recipes[candidate]
            or size["recipe_sha256"] != recipes[candidate]
        ):
            raise ValueError("Recipe changed after development or size measurement")
        tensor_hashes[candidate] = record["deployed_tensor_fingerprint"]["sha256"]
        if tensor_hashes[candidate] != size["deployed_tensor_fingerprint"]["sha256"]:
            raise ValueError("Development and size measurement contain different model tensors")
        evidence[candidate] = {
            "development_summary": record["summary"],
            "size_summary": size["summary"],
            "paired_development_change": paired_difference(baseline_rows, episodes),
            "development_record_sha256": file_hash(
                STUDY / "measurements" / candidate / "development/run.json"
            ),
            "development_episodes_sha256": file_hash(
                STUDY / "measurements" / candidate / "development/episodes.csv"
            ),
            "size_record_sha256": file_hash(size_path),
        }
    frozen = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "candidates": candidates,
        "control": "B16",
        "target_successes": 450,
        "final_trials_per_candidate": 500,
        "recipe_sha256": recipes,
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "deployed_tensor_sha256": tensor_hashes,
        "scientific_sources": scientific_sources(),
        "development_evidence": evidence,
        "coverage_at_selection": coverage["domains"],
        "final_outcomes_used": False,
        "input_sha256": {
            name: file_hash(ROOT / name)
            for name in [
                "outputs/compression/calibration.pt",
                "outputs/compression/references.pt",
                "outputs/evaluation/observations.pt",
                "studies/evaluation/initial_states.json",
                "studies/compression/plan.json",
                "studies/compression/execution_plan.json",
                "studies/compression/cka.json",
            ]
        },
        "state_assets": verify_state_assets(),
    }
    with zipfile.ZipFile(
        STUDY / "final-source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in frozen["scientific_sources"]:
            if relative.startswith("installed/"):
                continue  # Native package file hashes are recorded; package pins remain in provenance.
            if (ROOT / relative).suffix == ".py":
                archive.write(ROOT / relative, relative)
    frozen["source_archive_sha256"] = file_hash(STUDY / "final-source.zip")
    write_json(path, frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    frozen = freeze(args.candidates)
    print(f"Frozen final candidates: {frozen['candidates']}; matched B16 control; 500 trials each")


if __name__ == "__main__":
    main()
