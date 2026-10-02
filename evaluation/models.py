"""Load pinned trained policies, then quantize their trained decoder weights in place."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn

from evaluation.common import ROOT, read_json

SOURCES = read_json(ROOT / "evaluation/sources.json")


class OncePerWarning(logging.Filter):
    """Keep the first backend warning without timing hundreds of duplicate log writes."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        message = record.getMessage()
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


def artifact(name):
    source = SOURCES["models"][name]
    return Path(
        snapshot_download(
            source["repo"],
            revision=source["revision"],
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
            ignore_patterns=["onnx/*", "original/*"],
        )
    )


def quantize_decoder(decoder, bits):
    """Convert only decoder Linear leaves after trained checkpoint restoration.

    Quantizing a freshly downloaded Qwen and then overwriting packed weights with
    policy weights would be incorrect. Parent vision/action/world modules are untouched.
    """
    import bitsandbytes as bnb

    logging.getLogger("bitsandbytes.autograd._functions").addFilter(OncePerWarning())

    if bits not in {4, 8}:
        raise ValueError("Only 4-bit and 8-bit variants are defined")
    converted = []
    for name, linear in list(decoder.named_modules()):
        if not isinstance(linear, nn.Linear):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = decoder.get_submodule(parent_name) if parent_name else decoder
        weight = linear.weight.detach().cpu()
        if bits == 8:
            replacement = bnb.nn.Linear8bitLt(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            replacement.weight = bnb.nn.Int8Params(
                weight, requires_grad=False, has_fp16_weights=False
            )
        else:
            replacement = bnb.nn.Linear4bit(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            replacement.weight = bnb.nn.Params4bit(
                weight, requires_grad=False, compress_statistics=True, quant_type="nf4"
            )
        if linear.bias is not None:
            replacement.bias = nn.Parameter(linear.bias.detach().cpu(), requires_grad=False)
        replacement = replacement.to(device=linear.weight.device).eval()
        setattr(parent, leaf, replacement)
        converted.append({"module": name, "shape": list(weight.shape), "bits": bits})
    if not converted:
        raise ValueError("No decoder layers converted; refusing mislabeled quantized variant")
    return converted


def restore_published_checkpoint(policy, path):
    """Audit the known published-checkpoint extras; reject every other mismatch."""
    from safetensors import safe_open
    from safetensors.torch import load_model

    duplicate = "model.qwen.model.model.language_model.embed_tokens.weight"
    unused = {
        "model.video_predictor.extrinsics_encoder.bias",
        "model.video_predictor.extrinsics_encoder.weight",
        "model.video_predictor.state_encoder.bias",
        "model.video_predictor.state_encoder.weight",
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not torch.equal(
            handle.get_tensor(duplicate), handle.get_tensor("model.qwen.model.lm_head.weight")
        ):
            raise ValueError("Published tied embedding entries differ")
    missing, unexpected = load_model(policy, str(path), strict=False, device="cpu")
    if missing or set(unexpected) - (unused | {duplicate}):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "missing_keys": list(missing),
        "ignored_keys": sorted(unexpected),
        "tied_embedding_equality_verified": True,
        "reason": "Identical tied alias and unused checkpoint-only state/extrinsics world-predictor modules",
    }


def cast_inference_policy(policy, device, buffer_precision="legacy"):
    """Preserve native RoPE values before casting, never upcast rounded values."""
    if buffer_precision not in {"legacy", "native_rope"}:
        raise ValueError("Unknown inference buffer precision")
    originals = {}
    if buffer_precision == "native_rope":
        for name, value in policy.named_buffers():
            if name.endswith(("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")):
                if value.dtype != torch.float32:
                    raise ValueError(f"Native rotary frequencies already rounded: {name}")
                originals[name] = value.detach().clone()
        if not originals:
            raise ValueError("No native rotary frequencies found")
    policy.to(device=device, dtype=torch.bfloat16)
    for name, original in originals.items():
        parent, _, leaf = name.rpartition(".")
        module = policy.get_submodule(parent) if parent else policy
        module._buffers[leaf] = original.to(device=device)
    policy.eval().requires_grad_(False)
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in originals.items()}


def load_policy(variant, checkpoint=None, device="cuda", buffer_precision="legacy"):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    if variant not in {"B16", "Q8", "Q4", "S500"}:
        raise ValueError("Unknown configuration")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The matched campaign requires CUDA; CPU fallback is forbidden")
    compatibility = {}
    if variant == "S500":
        from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig, VLAJEPASmolVLMPolicy

        if checkpoint is None:
            baseline = PreTrainedConfig.from_pretrained(artifact("baseline"))
            config = VLAJEPASmolVLMConfig(
                input_features=baseline.input_features,
                output_features=baseline.output_features,
                device=device,
                torch_dtype="bfloat16",
                vlm_model_name=str(artifact("smolvlm")),
                jepa_encoder_name=str(artifact("world_model")),
                init_from_vla_jepa=str(artifact("pretrain")),
                resize_images_to=baseline.resize_images_to,
                binarize_gripper_action=baseline.binarize_gripper_action,
                pre_snap_gripper_action=baseline.pre_snap_gripper_action,
            )
            policy = VLAJEPASmolVLMPolicy(config)
            status = "untrained_adapter_smoke_only"
        else:
            config = PreTrainedConfig.from_pretrained(checkpoint)
            config.device = device
            config.init_from_vla_jepa = None
            config.vlm_model_name = str(artifact("smolvlm"))
            config.jepa_encoder_name = str(artifact("world_model"))
            policy = VLAJEPASmolVLMPolicy.from_pretrained(checkpoint, config=config, strict=True)
            status = "adapted"
    else:
        config = PreTrainedConfig.from_pretrained(artifact("baseline"))
        config.device = device
        config.qwen_model_name = str(artifact("qwen"))
        config.jepa_encoder_name = str(artifact("world_model"))
        config.torch_dtype = "bfloat16"
        policy = VLAJEPAPolicy(config)
        compatibility = restore_published_checkpoint(
            policy, artifact("baseline") / "model.safetensors"
        )
        status = "published_trained_policy"
    preserved_buffers = cast_inference_policy(policy, device, buffer_precision)
    logical_parameters = sum(p.numel() for p in policy.parameters())
    converted = []
    if variant in {"Q8", "Q4"}:
        converted = quantize_decoder(policy.model.qwen.model.model.language_model, int(variant[1:]))
    torch.cuda.empty_cache()
    metadata = {
        "variant": variant,
        "status": status,
        "logical_parameters": logical_parameters,
        "quantized_modules": converted,
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in policy.parameters())),
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "num_inference_timesteps": config.num_inference_timesteps,
        "world_model_loaded": policy.model.video_encoder is not None,
        "sources": SOURCES,
        "checkpoint_compatibility": compatibility,
        "buffer_precision": buffer_precision,
        "preserved_rotary_buffers": preserved_buffers,
        "Q8_internal_activation_cast": "float16" if variant == "Q8" else None,
    }
    return policy, metadata
