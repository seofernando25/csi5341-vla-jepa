"""Where VLA-JEPA's compressible modules live, and in-place module surgery helpers."""

from torch import nn

ROOT_NAMES = ["text", "vision", "action"]

SCOPES = {
    "text_mlp": ("text", lambda name: ".mlp." in name),
    "text_attention": ("text", lambda name: ".self_attn." in name),
    "vision_blocks": ("vision", lambda name: name.startswith("blocks.")),
    "merger": ("vision", lambda name: "merger" in name),
    "vision_mlp": ("vision", lambda name: ".mlp." in name),
    "vision_attention": ("vision", lambda name: ".attn." in name),
    "action_mlp": ("action", lambda name: ".ff." in name),
    "action_attention": ("action", lambda name: ".attn1." in name),
    "action_interfaces": (
        "action",
        lambda name: not name.startswith("model.transformer_blocks."),
    ),
}


class SkipBlock(nn.Module):
    """Identity block that keeps layer indices and DeepStack injection positions."""

    def forward(self, hidden_states=None, *args, **kwargs):
        return hidden_states if hidden_states is not None else args[0]


def roots(policy):
    inner = policy.model.qwen.model.model
    return {
        "text": inner.language_model,
        "vision": inner.visual,
        "action": policy.model.action_model,
    }


def blocks(policy, root):
    """Transformer blocks of one root, in order."""
    r = roots(policy)
    return {
        "text": r["text"].layers,
        "vision": r["vision"].blocks,
        "action": r["action"].model.transformer_blocks,
    }[root]


def selected_linears(policy, scope):
    """{"<root>.<path>": (root module, path, Linear)} for a root name, a SCOPES key, or "all"."""
    result = {}
    for root_name, root in roots(policy).items():
        for name, module in root.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if scope in (root_name, "all"):
                include = True
            elif scope in SCOPES:
                wanted, predicate = SCOPES[scope]
                include = root_name == wanted and predicate(name)
            else:
                include = False
            if include:
                result[f"{root_name}.{name}"] = (root, name, module)
    if not result:
        raise ValueError(f"No Linear modules selected for {scope}")
    return result


def replace(root, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, module)


def slice_linear(linear, rows=None, columns=None):
    """Frozen copy of `linear` keeping only the given output rows and input columns."""
    weight = linear.weight.detach()
    if rows is not None:
        weight = weight[rows]
    if columns is not None:
        weight = weight[:, columns]
    result = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=linear.bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    result.weight = nn.Parameter(weight.clone(), requires_grad=False)
    if linear.bias is not None:
        bias = linear.bias.detach() if rows is None else linear.bias.detach()[rows]
        result.bias = nn.Parameter(bias.clone(), requires_grad=False)
    return result.eval()
