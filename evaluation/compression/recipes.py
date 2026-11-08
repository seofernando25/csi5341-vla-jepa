"""Apply a compression recipe: an ordered list of changes, each handled by one method.

Recipes live in studies/compression/recipes/<id>.json as {"id", "family", "changes": [...]}.
A change names its method and, depending on it, a scope, fraction, bits, count or criterion.
Changes are applied in memory to a loaded policy; published weights are never edited.
"""

from evaluation.compression import execution, pruning, quantization
from evaluation.compression.attention_matching import matching
from evaluation.compression.compiled import compile_linears
from evaluation.compression.lowbit import hqq_quantize
from evaluation.compression.second_order import second_order
from evaluation.compression.tokens import visual_token_pruning

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
