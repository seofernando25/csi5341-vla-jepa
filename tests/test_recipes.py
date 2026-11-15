"""Every committed recipe validates through the code `vla-compress run` uses."""

import pytest

from evaluation.common import ROOT, read_json
from evaluation.compression.recipes import validate

RECIPES = ROOT / "studies/program-v1/recipes"
PLAN = read_json(ROOT / "studies/program-v1/precision_surface_plan.json")


def recipe_paths():
    return sorted(RECIPES.glob("*.json"))


def family(name):
    return read_json(RECIPES / f"{name}.json")["family"]


@pytest.mark.parametrize("path", recipe_paths(), ids=lambda path: path.stem)
def test_every_recipe_validates(path):
    recipe = validate(read_json(path), path)
    assert recipe["id"] == path.stem
    assert recipe["family"]


def test_every_plan_arm_has_a_recipe():
    assert [arm for arm in PLAN["arms"] if not (RECIPES / f"{arm}.json").exists()] == []


def test_uncompressed_reference_is_a_no_op():
    recipe = validate(read_json(RECIPES / "B16.json"))
    assert recipe["changes"] == []
    assert recipe["family"] == "baseline"


def test_joint_cell_quantizes_only_vision_and_action():
    """H4 needs the factorial cell: vision HQQ2 plus action HQQ2, language BF16."""
    recipe = read_json(RECIPES / "precision-joint-va-hqq2.json")
    changes = {(c["method"], c["scope"], c["bits"]) for c in recipe["changes"]}
    assert changes == {("hqq", "vision_blocks", 2), ("hqq", "action_mlp", 2)}


def test_precision_ladder_covers_every_planned_module_and_width():
    for module in ("lang", "vision", "action"):
        assert {family(f"precision-{module}-{fmt}") for fmt in ("int8", "nf4", "hqq3", "hqq2")} == {
            f"precision-{'language' if module == 'lang' else module}"
        }


def test_format_curve_covers_the_planned_decoder_formats():
    formats = {path.stem for path in recipe_paths() if family(path.stem) == "format-language"}
    assert formats == {"format-lang-hqq4", "format-lang-gptq", "format-lang-awq"}


def test_execution_ladder_arms_exist():
    ladder = {path.stem for path in recipe_paths() if family(path.stem) == "execution-ladder"}
    assert ladder == {
        "ladder-b16-graph-4step",
        "ladder-b16-graph-1step",
        "ladder-nf4-graph-4step",
        "ladder-nf4-graph-1step-dense",
        "ladder-nf4-graph-1step-packed",
    }


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ({"family": "f", "changes": []}, "missing 'id'"),
        ({"id": "x", "changes": []}, "missing 'family'"),
        ({"id": "x", "family": "f"}, "missing 'changes'"),
        ({"id": "x", "family": "f", "changes": [{"method": "nope"}]}, "unknown compression"),
        ({"id": "x", "family": "f", "changes": [{"method": "bnb", "bits": 4}]}, "needs a scope"),
        (
            {"id": "x", "family": "f", "changes": [{"method": "bnb", "scope": "text", "bits": 3}]},
            "bits must be",
        ),
        (
            {"id": "x", "family": "f", "changes": [{"method": "hqq", "scope": "nowhere"}]},
            "unknown scope",
        ),
        ({"id": "x", "family": "f", "changes": [], "engine": "magic"}, "unknown engine"),
        ({"id": "x", "family": "f", "changes": [], "engine_options": {"x": 1}}, "engine options"),
        ({"id": "x", "family": "f", "changes": [{"method": "solver_steps"}]}, "needs a count"),
    ],
)
def test_validator_rejects_defects(recipe, message):
    with pytest.raises(ValueError, match=message):
        validate(recipe)


def test_validator_rejects_an_id_that_disagrees_with_the_file(tmp_path):
    path = tmp_path / "B16.json"
    path.write_text('{"id": "other", "family": "baseline", "changes": []}')
    with pytest.raises(ValueError, match="file stem"):
        validate(read_json(path), path)
