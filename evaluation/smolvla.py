"""Pinned complete-policy SmolVLA comparison; no training or VLA-JEPA mutation."""

from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from evaluation.common import ROOT, file_hash, read_json

SOURCES = read_json(ROOT / "evaluation/smolvla_sources.json")


def artifact(name):
    source = SOURCES[name]
    return Path(snapshot_download(
        source["repo"], revision=source["revision"], local_files_only=True,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    ))


def load_policy():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA comparison requires CUDA")
    path = artifact("policy")
    config = SmolVLAConfig.from_pretrained(path)
    if (config.chunk_size, config.n_action_steps, config.num_steps) != (50, 1, 10):
        raise ValueError("Native SmolVLA execution settings changed")
    config.device = "cuda"
    config.vlm_model_name = str(artifact("backbone"))
    # Restore the complete policy strictly, preserving native FP32 rotary buffers.
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True).eval()
    policy.requires_grad_(False)
    files = {p.name: file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}
    metadata = {
        "variant": "SmolVLA", "status": "published_trained_policy",
        "architecture": "smolvla", "comparison": SOURCES["comparison"],
        "logical_parameters": sum(p.numel() for p in policy.parameters()),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in policy.parameters())),
        "chunk_size": config.chunk_size, "n_action_steps": config.n_action_steps,
        "num_inference_timesteps": config.num_steps, "world_model_loaded": False,
        "sources": SOURCES, "checkpoint_files": files,
        "checkpoint_compatibility": {"strict": True, "missing_keys": [], "unexpected_keys": []},
        "precision": SOURCES["precision"],
        "processor_overrides": {"tokenizer_processor": {"tokenizer_name": SOURCES["backbone"]}},
        "control_frequency_hz": 20, "control_mode": "relative",
    }
    return policy, metadata


def processors(policy):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    env_pre, _ = LiberoEnv(task="libero_spatial").get_env_processors()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=artifact("policy"),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(artifact("backbone"))},
        },
    )
    return env_pre, pre, post
