"""Register joint pruning tests from matched preliminary component outcomes."""

from copy import deepcopy
from datetime import UTC, datetime

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.compression.protocol import matched_controls, validated_episodes
from evaluation.compression.queue import ledger

STUDY = ROOT / "studies/compression"


def choose(rows, minimum):
    groups = {}
    for row in rows:
        if len(row["changes"]) != 1:
            continue
        change = row["changes"][0]
        if change["method"] not in {"channels", "heads", "layers"}:
            continue
        screen = row["phases"].get("screen", {})
        smoke = row["phases"].get("smoke", {})
        if (
            screen.get("status") != "completed"
            or smoke.get("status") != "completed"
            or smoke["summary"]["successes"] < minimum
        ):
            continue
        key = change["method"], change["scope"]
        rank = (
            -change.get("fraction", change.get("count", 0)),
            -smoke["summary"]["successes"],
            screen["summary"]["peak_allocated_bytes"],
            row["candidate"],
        )
        if key not in groups or rank < groups[key][0]:
            groups[key] = rank, row
    return {key: row for key, (_, row) in groups.items()}


def register(execution):
    if not execution.get("joint_pruning_boundary_refinements", False):
        return []
    path = STUDY / "joint_boundary_selection.json"
    plan = read_json(STUDY / "plan.json")
    if path.exists():
        frozen = read_json(path)
        for candidate, digest in frozen["recipe_sha256"].items():
            if (
                candidate not in plan["candidates"]
                or file_hash(STUDY / "recipes" / f"{candidate}.json") != digest
            ):
                raise ValueError("Preserve registered joint-boundary recipes and selection")
        return frozen["candidates"]
    if (STUDY / "development_selection.json").exists() or any(
        (STUDY / "measurements").glob("*/final/run.json")
    ):
        raise ValueError("Register joint boundaries before development selection or final outcomes")
    baseline, _ = validated_episodes(STUDY / "measurements/B16/smoke")
    minimum = baseline["summary"]["successes"] - execution["development_smoke_tolerance_episodes"]
    selected = choose(ledger(), minimum)
    evidence = {}
    for row in selected.values():
        candidate = row["candidate"]
        record, _ = validated_episodes(STUDY / "measurements" / candidate / "smoke")
        matched_controls(baseline, record)
        evidence[candidate] = {
            "change": row["changes"][0],
            "successes": record["summary"]["successes"],
            "screen_record_sha256": file_hash(STUDY / "measurements" / candidate / "screen/run.json"),
            "smoke_record_sha256": file_hash(STUDY / "measurements" / candidate / "smoke/run.json"),
            "episodes_sha256": file_hash(STUDY / "measurements" / candidate / "smoke/episodes.csv"),
        }
    families = {
        "text-channel": [row for key, row in selected.items() if key == ("channels", "text_mlp")],
        "channel": [row for key, row in selected.items() if key[0] == "channels"],
        "head": [row for key, row in selected.items() if key[0] == "heads"],
        "depth": [row for key, row in selected.items() if key[0] == "layers"],
    }
    base = read_json(STUDY / "recipes/deploy-nf4-all-restricted.json")
    assert base["changes"][:2] == [{"method": "world_remove"}, {"method": "compact_embedding"}]
    recipes, sources = {}, {}
    for family, rows in families.items():
        if not rows:
            continue
        recipe = deepcopy(base)
        candidate = f"deploy-nf4-{family}-boundaries-restricted"
        recipe["id"] = candidate
        changes = [deepcopy(row["changes"][0]) for row in sorted(rows, key=lambda r: r["candidate"])]
        recipe["changes"] = base["changes"][:2] + changes + base["changes"][2:]
        recipe_path = STUDY / "recipes" / f"{candidate}.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("Preserve existing joint-boundary recipe")
        if not recipe_path.exists():
            write_json(recipe_path, recipe)
        if candidate not in plan["candidates"]:
            plan["candidates"].append(candidate)
        recipes[candidate] = file_hash(recipe_path)
        sources[candidate] = [row["candidate"] for row in rows]
    write_json(STUDY / "plan.json", plan)
    frozen = {
        "at": datetime.now(UTC).isoformat(),
        "candidates": list(recipes),
        "recipe_sha256": recipes,
        "component_sources": sources,
        "component_evidence": evidence,
        "minimum_smoke_successes": minimum,
        "execution_plan_sha256": file_hash(STUDY / "execution_plan.json"),
        "final_outcomes_used": False,
        "interpretation": "Strongest passing component settings; joint success must be measured, not inferred from isolated outcomes",
    }
    write_json(path, frozen)
    return frozen["candidates"]
