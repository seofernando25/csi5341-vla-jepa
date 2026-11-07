"""Visual-token compaction preserving Qwen's original multimodal positions."""

import math
from itertools import pairwise

import torch


def visual_token_pruning(policy, fraction, criterion="magnitude"):
    language = policy.model.qwen.model.model.language_model
    current = {}

    def compact(module, args, kwargs):
        embeddings = kwargs["inputs_embeds"]
        mask = kwargs["visual_pos_masks"]
        if embeddings.shape[0] != 1 or mask is None:
            raise ValueError("Visual-token pruning requires a single two-view observation")
        visual = mask[0].nonzero().flatten()
        selected = []
        # Select separately in contiguous image spans, retaining both camera views.
        boundaries = (
            [0] + (visual[1:] - visual[:-1] > 1).nonzero().flatten().add(1).tolist() + [len(visual)]
        )
        for start, end in pairwise(boundaries):
            span = visual[start:end]
            count = max(1, math.ceil(len(span) * (1 - fraction)))
            if criterion == "magnitude":
                scores = embeddings[0, span].float().square().sum(-1)
                selected.append(span[scores.argsort(descending=True, stable=True)[:count]])
            else:
                selected.append(
                    span[torch.linspace(0, len(span) - 1, count, device=span.device).round().long()]
                )
        kept_visual = torch.cat(selected).sort().values
        keep_mask = ~mask[0].clone()
        keep_mask[kept_visual] = True
        keep = keep_mask.nonzero().flatten()
        visual_rows = torch.searchsorted(visual, kept_visual)
        current.update(keep=keep, length=embeddings.shape[1])
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = embeddings[:, keep]
        kwargs["visual_pos_masks"] = mask[:, keep]
        kwargs["position_ids"] = kwargs["position_ids"][..., keep]
        if kwargs.get("attention_mask") is not None:
            if kwargs["attention_mask"].ndim != 2:
                raise ValueError("Expected unexpanded attention mask before decoder")
            kwargs["attention_mask"] = kwargs["attention_mask"][:, keep]
        if kwargs.get("deepstack_visual_embeds") is not None:
            kwargs["deepstack_visual_embeds"] = [
                x[visual_rows] for x in kwargs["deepstack_visual_embeds"]
            ]
        return args, kwargs

    def restore(module, args, output):
        # VLA-JEPA gathers conditioning using original input_ids positions. Restore
        # that indexing after all decoder computation, leaving removed visual rows unused.
        result = output.new_zeros(output.shape[0], current["length"], output.shape[-1])
        result[:, current["keep"]] = output
        return result

    language.register_forward_pre_hook(compact, with_kwargs=True)
    language.layers[-1].register_forward_hook(restore)
    return {
        "method": "visual token compaction",
        "fraction_removed": fraction,
        "criterion": criterion,
        "original_mrope_positions_preserved": True,
        "selection_per_camera": True,
        "weight_storage_reduction": False,
        "vision_encoder_compute_reduction": False,
    }
