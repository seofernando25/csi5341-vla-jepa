"""Contract checks for the Story demo fixture and the demo adapter that serves it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "rsi" / "watch_static"
STORY = STATIC / "fixture_story.json"
STATE = STATIC / "fixture.json"
DEMO = STATIC / "js" / "demo.js"

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _fields(names: str) -> set[str]:
    return set(names.split())


PAYLOAD_FIELDS = _fields(
    "cycle status recorded_utc protocol worlds tree candidates curves rounds heldout verdict agent"
)
FIXTURE_FIELDS = (PAYLOAD_FIELDS - {"tree"}) | {"trees", "_note"}
WORLD_FIELDS = _fields("id nodes baseline_score max_gain degenerate")
TREE_FIELDS = {"world_id", "nodes"}
NODE_FIELDS = _fields("cell_id parent_id branch attempt seq score fail_class evaluated summary")
CANDIDATE_FIELDS = _fields("slot step status mean_v delta_vs_best attempts error sha256")
POINT_FIELDS = _fields("beta probes attainment rounds")
ROUND_FIELDS = _fields("round batch prefix_size outcomes")
OUTCOME_FIELDS = {"cell_id", "score", "fail_class"}
CANDIDATE_STATUSES = {"incumbent", "accepted", "not_better"}
FORBIDDEN = ("0.7957", "5532276502", "cleanroom", "synth-", "/home/", "/tmp/", "/Users/")


def _story() -> dict:
    return json.loads(STORY.read_text(encoding="utf-8"))


def _check(obj: object, required: set[str], where: str) -> None:
    assert isinstance(obj, dict), f"{where} is not an object"
    assert set(obj) == required, f"{where} keys {sorted(set(obj) ^ required)}"


def _slots(story: dict) -> list[str]:
    return [candidate["slot"] for candidate in story["candidates"]]


def _trees(story: dict) -> dict[str, dict[str, dict]]:
    return {
        world: {node["cell_id"]: node for node in tree["nodes"]}
        for world, tree in story["trees"].items()
    }


def _max_gain(tree: dict, baseline: float) -> float:
    gains = [
        max(0.0, node["score"] - baseline)
        for node in tree["nodes"]
        if node["score"] is not None and node["fail_class"] == "ok"
    ]
    return round(max(gains), 6) if gains else 0.0


def _reward(world: dict, points: list[dict], lam: float) -> float:
    """Eq. (1) hypervolume minus the parallel penalty, from one world's beta sweep."""
    total = world["nodes"]
    hv = 0.0
    prev = 0.0
    for probes, attainment in sorted({(p["probes"], p["attainment"]) for p in points}):
        if attainment > prev:
            hv += (attainment - prev) * (total - probes)
            prev = attainment
    hv /= float(total)
    penalty = sum(p["rounds"] / p["probes"] for p in points) / len(points)
    return hv - lam * penalty


def test_story_fixture_has_the_payload_fields() -> None:
    story = _story()
    _check(story, FIXTURE_FIELDS, "story fixture")
    assert "synthetic" in story["_note"].lower(), "the fixture does not say it is synthetic"
    assert story["status"] == "ready"
    assert isinstance(story["cycle"], str) and story["cycle"]
    assert isinstance(story["recorded_utc"], str) and story["recorded_utc"].endswith("Z")
    _check(
        story["protocol"],
        {"m", "lambda_parallel", "beta_grid", "max_parallelism"},
        "story protocol",
    )
    assert story["protocol"]["beta_grid"] == list(BETAS)
    assert story["protocol"]["m"] == len(story["candidates"])
    assert set(story["trees"]) == {world["id"] for world in story["worlds"]}


def test_story_fixture_worlds_and_trees() -> None:
    story = _story()
    worlds = story["worlds"]
    assert len(worlds) == 2, "the story needs two worlds"
    assert len({world["id"] for world in worlds}) == 2
    degenerate = [world for world in worlds if world["degenerate"]]
    assert len(degenerate) == 1, "exactly one world must be degenerate"
    for world in worlds:
        where = f"world {world['id']}"
        _check(world, WORLD_FIELDS, where)
        tree = story["trees"][world["id"]]
        _check(tree, TREE_FIELDS, f"{where} tree")
        assert tree["world_id"] == world["id"]
        assert 6 <= world["nodes"] <= 8, f"{where} has {world['nodes']} nodes"
        assert len(tree["nodes"]) == world["nodes"]
        assert world["max_gain"] == _max_gain(tree, world["baseline_score"]), where
        assert world["degenerate"] == (world["max_gain"] <= 0.0), where
        assert [node["seq"] for node in tree["nodes"]] == list(range(1, world["nodes"] + 1))
        assert len({node["cell_id"] for node in tree["nodes"]}) == world["nodes"]
        seen: set[str] = set()
        for node in tree["nodes"]:
            _check(node, NODE_FIELDS, f"{where} node {node.get('cell_id')}")
            assert node["parent_id"] is None or node["parent_id"] in seen, "parent is not earlier"
            assert isinstance(node["branch"], int) and isinstance(node["attempt"], int)
            assert isinstance(node["evaluated"], bool)
            assert isinstance(node["summary"], str) and node["summary"]
            assert node["score"] is None or 0.0 <= node["score"] <= 1.0
            assert (node["score"] is None) == (node["fail_class"] != "ok")
            seen.add(node["cell_id"])


def test_story_fixture_candidates_and_verdict() -> None:
    story = _story()
    candidates = story["candidates"]
    assert len(candidates) == 3, "one incumbent and two rewrites"
    assert len({candidate["slot"] for candidate in candidates}) == 3
    statuses = [candidate["status"] for candidate in candidates]
    assert statuses[0] == "incumbent" and set(statuses) <= CANDIDATE_STATUSES
    assert statuses.count("accepted") == 1 and statuses.count("not_better") == 1
    best = None
    for candidate in candidates:
        _check(candidate, CANDIDATE_FIELDS, f"candidate {candidate['slot']}")
        assert isinstance(candidate["step"], int) and candidate["step"] >= 0
        assert isinstance(candidate["attempts"], int) and candidate["attempts"] >= 0
        assert candidate["error"] is None
        assert isinstance(candidate["mean_v"], float) and 0.0 <= candidate["mean_v"] <= 1.0
        assert len(candidate["sha256"]) == 64 and candidate["sha256"] != "0" * 64
        if best is None:
            assert candidate["delta_vs_best"] is None
        else:
            assert candidate["delta_vs_best"] == pytest.approx(candidate["mean_v"] - best, abs=1e-3)
        best = candidate["mean_v"] if best is None else max(best, candidate["mean_v"])
    assert len({candidate["sha256"] for candidate in candidates}) == 3

    verdict = story["verdict"]
    _check(verdict, {"selected_slot", "incumbent_in_sample_V", "selected_in_sample_V"}, "verdict")
    by_slot = {candidate["slot"]: candidate for candidate in candidates}
    accepted = next(c for c in candidates if c["status"] == "accepted")
    assert verdict["selected_slot"] == accepted["slot"]
    assert verdict["selected_in_sample_V"] == accepted["mean_v"]
    assert verdict["incumbent_in_sample_V"] == by_slot["m0_incumbent"]["mean_v"]

    heldout = story["heldout"]
    _check(heldout, {"num_worlds", "mean_V_by_slot", "delta_vs_incumbent"}, "heldout")
    assert isinstance(heldout["num_worlds"], int) and heldout["num_worlds"] > 0
    assert set(heldout["mean_V_by_slot"]) == set(by_slot)
    delta = heldout["mean_V_by_slot"][accepted["slot"]]
    delta -= heldout["mean_V_by_slot"]["m0_incumbent"]
    assert heldout["delta_vs_incumbent"] == pytest.approx(delta, abs=1e-3)

    agent = story["agent"]
    _check(agent, {"backend", "model", "effort", "calls", "num_turns", "cost_usd"}, "agent")
    assert "synthetic" in agent["model"]
    assert isinstance(agent["calls"], int) and isinstance(agent["num_turns"], int)
    assert isinstance(agent["cost_usd"], float)


def test_story_fixture_curves_cover_every_slot_world_and_beta() -> None:
    story = _story()
    assert set(story["curves"]) == set(_slots(story))
    world_ids = [world["id"] for world in story["worlds"]]
    for slot, by_world in story["curves"].items():
        assert set(by_world) == set(world_ids), f"{slot} curves miss a world"
        for world_id, points in by_world.items():
            assert [point["beta"] for point in points] == list(BETAS), f"{slot}/{world_id}"
            for point in points:
                _check(point, POINT_FIELDS, f"curve {slot}/{world_id}/{point.get('beta')}")
                assert isinstance(point["probes"], int) and point["probes"] >= 1
                assert isinstance(point["rounds"], int) and 3 <= point["rounds"] <= 5
                assert 0.0 <= point["attainment"] <= 1.0
    for world in story["worlds"]:
        if not world["degenerate"]:
            continue
        for by_world in story["curves"].values():
            assert all(point["attainment"] == 0.0 for point in by_world[world["id"]])


def test_story_fixture_rounds_match_their_curves() -> None:
    story = _story()
    trees = _trees(story)
    worlds = {world["id"]: world for world in story["worlds"]}
    assert set(story["rounds"]) == set(_slots(story))
    for slot, by_world in story["rounds"].items():
        assert set(by_world) == set(trees), f"{slot} rounds miss a world"
        for world_id, by_beta in by_world.items():
            assert set(by_beta) == {f"{beta:g}" for beta in BETAS}, f"{slot}/{world_id}"
            points = {f"{point['beta']:g}": point for point in story["curves"][slot][world_id]}
            for key, rounds in by_beta.items():
                where = f"{slot}/{world_id}/{key}"
                nodes = trees[world_id]
                assert 3 <= len(rounds) <= 5, f"{where} has {len(rounds)} rounds"
                seen: set[str] = set()
                spent = 0
                best = None
                for index, entry in enumerate(rounds):
                    _check(entry, ROUND_FIELDS, f"round {where}/{index}")
                    assert entry["round"] == index + 1
                    assert 1 <= len(entry["batch"]) <= 2, f"{where} batch too large"
                    assert entry["prefix_size"] == spent
                    assert [o["cell_id"] for o in entry["outcomes"]] == entry["batch"]
                    for outcome in entry["outcomes"]:
                        _check(outcome, OUTCOME_FIELDS, f"outcome {where}/{outcome.get('cell_id')}")
                        node = nodes[outcome["cell_id"]]
                        assert outcome["cell_id"] not in seen, "a cell is probed twice"
                        assert node["parent_id"] is None or node["parent_id"] in seen
                        assert outcome["score"] == node["score"]
                        assert outcome["fail_class"] == node["fail_class"]
                        seen.add(outcome["cell_id"])
                        if outcome["score"] is not None:
                            best = outcome["score"] if best is None else max(best, outcome["score"])
                    spent += len(entry["batch"])
                point = points[key]
                assert point["probes"] == spent, f"{where} probes disagree with its rounds"
                assert point["rounds"] == len(rounds)
                assert point["attainment"] == _attainment(worlds[world_id], best), where


def _attainment(world: dict, best: float | None) -> float:
    if best is None or world["max_gain"] <= 0.0:
        return 0.0
    gain = best - world["baseline_score"]
    return round(min(1.0, max(0.0, gain / world["max_gain"])), 3)


def test_story_fixture_mean_v_follows_eq1() -> None:
    story = _story()
    worlds = story["worlds"]
    live = [world for world in worlds if not world["degenerate"]] or worlds
    lam = story["protocol"]["lambda_parallel"]
    means = {}
    for slot, by_world in story["curves"].items():
        rewards = [_reward(world, by_world[world["id"]], lam) for world in live]
        means[slot] = round(sum(rewards) / len(rewards), 3)
    by_slot = {candidate["slot"]: candidate["mean_v"] for candidate in story["candidates"]}
    for slot, mean in means.items():
        assert by_slot[slot] == pytest.approx(mean, abs=1e-3), f"{slot} mean_v is not Eq. (1)"
    assert story["verdict"]["selected_slot"] == max(by_slot, key=lambda slot: by_slot[slot])


def test_story_fixture_is_synthetic() -> None:
    text = STORY.read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in text, f"the story fixture quotes a real run value: {token}"
    assert not Path(json.loads(text)["cycle"]).is_absolute()


def test_state_fixture_carries_the_story_fields() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    baseline = state["loop"]["baseline"]
    assert baseline["score"] is None, "the baseline success rate is not recorded"
    assert isinstance(baseline["resident_bytes"], int) and baseline["resident_bytes"] > 0
    nodes = [node for cycle in state["cycles"] for node in cycle["nodes"]]
    detailed = [node for node in nodes if "per_task" in node]
    assert len(detailed) == 1, "exactly one node carries per-task results"
    node = detailed[0]
    assert isinstance(node["baseline_mb"], float) and node["baseline_mb"] > node["resident_mb"]
    reduction = 1.0 - node["resident_mb"] / node["baseline_mb"]
    assert reduction == pytest.approx(node["size_reduction"], abs=1e-3)
    per_task = node["per_task"]
    assert len(per_task) == 10, "the task suite has ten tasks"
    assert sum(counts["episodes"] for counts in per_task.values()) == node["episodes"]
    assert sum(counts["successes"] for counts in per_task.values()) == node["successes"]
    assert node["successes"] == 8


def test_demo_module_parses() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=DEMO.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"},
    )
    assert result.returncode == 0, result.stderr


HARNESS = """
globalThis.location = { search: "" };
const { readFile } = await import("node:fs/promises");
const { pathToFileURL } = await import("node:url");
const [demoPath, root] = process.argv.slice(2);
globalThis.fetch = async (url) => {
  const text = await readFile(`${root}/${url}`, "utf8");
  return { ok: true, status: 200, json: async () => JSON.parse(text) };
};
const { createDemo } = await import(pathToFileURL(demoPath).href);
const demo = await createDemo();
const out = {
  ready: await demo.dream("cycle-1", "w1"),
  other: await demo.dream("cycle-1", "w0"),
  pending: await demo.dream("cycle-2", "w0"),
  missing: await demo.dream("cycle-0", "w0"),
  unknown: await demo.dream("cycle-9", "w0"),
  noWorld: await demo.dream("cycle-1"),
  unknownWorld: await demo.dream("cycle-1", "w9"),
  state: await demo.state(),
};
process.stdout.write(JSON.stringify(out));
"""


def test_demo_adapter_serves_one_world(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "dream.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run(
        ["node", str(harness), str(DEMO), str(STATIC)],
        text=True,
        capture_output=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "NODE_OPTIONS"},
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    ready = out["ready"]
    assert set(ready) == PAYLOAD_FIELDS, "the adapter does not return the /api/dream shape"
    assert ready["cycle"] == "cycle-1" and ready["status"] == "ready"
    assert [world["id"] for world in ready["worlds"]] == ["w0", "w1"]
    assert ready["tree"]["world_id"] == "w1"
    assert set(ready["rounds"]) == set(_slots(_story()))
    for by_world in ready["rounds"].values():
        assert set(by_world) == {"w1"}, "rounds leak another world"
        assert set(next(iter(by_world.values()))) == {f"{beta:g}" for beta in BETAS}
    assert out["other"]["tree"]["world_id"] == "w0"
    assert out["noWorld"]["tree"]["world_id"] == "w0", "no world means the first live one"
    assert out["noWorld"]["rounds"] is None
    assert out["unknownWorld"]["tree"] is None, "an unknown world must not invent a tree"
    assert out["unknownWorld"]["rounds"] is None
    assert out["pending"]["status"] == "pending" and out["pending"]["tree"] is None
    assert out["pending"]["rounds"] is None and out["pending"]["cycle"] == "cycle-2"
    assert out["missing"]["status"] == "missing" and out["missing"]["cycle"] == "cycle-0"
    assert out["unknown"] is None, "an unknown cycle must resolve null"
    assert out["state"]["loop"]["baseline"]["score"] is None
