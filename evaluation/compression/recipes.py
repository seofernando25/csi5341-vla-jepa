"""Apply a compression recipe: an ordered list of changes, each handled by one method.

Recipes live in studies/program-v1/recipes/<id>.json as {"id", "family", "changes": [...]}.
A change names its method and, depending on it, a scope, fraction, bits, count or criterion.
Changes are applied in memory to a loaded policy; published weights are never edited.
"""

from pathlib import Path

from evaluation.compression import execution, pruning, quantization
from evaluation.compression.attention_matching import matching
from evaluation.compression.compiled import compile_linears
from evaluation.compression.lowbit import hqq_quantize
from evaluation.compression.second_order import second_order
from evaluation.compression.tokens import visual_token_pruning
from evaluation.compression.topology import ROOT_NAMES, SCOPES

METHODS = {
    "bnb": quantization.bitsandbytes,
    "awq_diagonal": quantization.activation_scaled_int4,
    "torchao": quantization.torchao_quantize,
    "int8_embedding": quantization.embedding,
    "compact_embedding": quantization.embedding,
    "hqq": hqq_quantize,
    "gptq": second_order,
    "sparsegpt": second_order,
    "attention_matching": matching,
    "compile_linears": compile_linears,
    "magnitude": pruning.sparsity,
    "wanda": pruning.sparsity,
    "channels": pruning.channels,
    "heads": pruning.heads,
    "layers": pruning.layers,
    "visual_tokens": visual_token_pruning,
    "world_remove": execution.remove_world_model,
    "conditioning_tokens": execution.conditioning_tokens,
    "solver_steps": execution.solver_steps,
    "action_horizon": execution.action_horizon,
    "replanning": execution.replanning,
}

SCOPED = {
    "bnb",
    "awq_diagonal",
    "torchao",
    "hqq",
    "gptq",
    "sparsegpt",
    "attention_matching",
    "compile_linears",
    "magnitude",
    "wanda",
    "channels",
    "heads",
    "layers",
}
BITS = {"bnb": {4, 8}, "torchao": {4, 8}, "hqq": {2, 3, 4}}
POSITIVE = {"count", "group_size"}
ENGINE_OPTIONS = {"use_compile", "dense"}


def validate(recipe, path=None):
    """Reject a recipe the run path cannot apply, before a GPU is touched.

    `path` is the recipe file: its stem must equal `id`, so measurements and figures
    can address a cell by plan id.
    """
    where = Path(path).name if path is not None else "recipe"
    for key in ("id", "family", "changes"):
        if key not in recipe:
            raise ValueError(f"{where}: missing {key!r}")
    if path is not None and recipe["id"] != Path(path).stem:
        raise ValueError(f"{where}: id {recipe['id']!r} does not match the file stem")
    if not isinstance(recipe["changes"], list):
        raise TypeError(f"{where}: changes must be a list")
    if recipe.get("engine") not in (None, "graph"):
        raise ValueError(f"{where}: unknown engine {recipe['engine']!r}")
    unknown = set(recipe.get("engine_options") or {}) - ENGINE_OPTIONS
    if unknown:
        raise ValueError(f"{where}: unknown engine options {sorted(unknown)}")
    for index, change in enumerate(recipe["changes"]):
        _validate_change(change, f"{where}[{index}]")
    return recipe


def _validate_change(change, where):
    method = change.get("method")
    if method not in METHODS:
        raise ValueError(f"{where}: unknown compression method {method!r}")
    if method in SCOPED:
        if "scope" not in change:
            raise ValueError(f"{where}: {method} needs a scope")
        allowed = set(SCOPES) | set(ROOT_NAMES) | {"all"}
        if change["scope"] not in allowed:
            raise ValueError(f"{where}: unknown scope {change['scope']!r}")
    if method in BITS and change.get("bits") not in BITS[method]:
        raise ValueError(f"{where}: {method} bits must be one of {sorted(BITS[method])}")
    if "axis" in change and change["axis"] not in (0, 1):
        raise ValueError(f"{where}: axis must be 0 or 1")
    if "backend" in change and change["backend"] not in ("pytorch", "gemlite"):
        raise ValueError(f"{where}: unknown backend {change['backend']!r}")
    if change.get("backend") == "gemlite" and (
        change.get("axis") != 1 or change["bits"] not in {2, 4}
    ):
        raise ValueError(f"{where}: gemlite supports axis 1 at 2 or 4 bits")
    for key in POSITIVE:
        if key in change and (not isinstance(change[key], int) or change[key] < 1):
            raise ValueError(f"{where}: {key} must be a positive integer")
    if method in {"solver_steps", "action_horizon", "replanning"} and "count" not in change:
        raise ValueError(f"{where}: {method} needs a count")
    if (
        method == "conditioning_tokens"
        and "count" not in change
        and change.get("scope") != "latent"
    ):
        raise ValueError(f"{where}: conditioning_tokens needs a count or scope 'latent'")
    if method == "visual_tokens" and not 0 <= change.get("fraction", 0) <= 1:
        raise ValueError(f"{where}: visual_tokens needs a fraction in [0, 1]")
    if "quantizer" in change:
        _validate_change(change["quantizer"], f"{where}.quantizer")


def apply(policy, recipe, calibration=None):
    """Apply every change in order; return the concatenated event records."""
    calibration = calibration or {}
    events = []
    for change in recipe.get("changes", []):
        handler = METHODS.get(change["method"])
        if handler is None:
            raise ValueError(f"Unknown compression method: {change['method']}")
        events.extend(handler(policy, change, calibration))
    return events
