"""One native action-loss backward pass, without optimizer updates or saved weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from evaluation.common import file_hash, write_json
from evaluation.models import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier diagnostic evidence")
    sys.path.insert(0, str(args.architecture_source.resolve() / "src"))
    import lerobot_policy_vla_jepa_smolvlm
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    policy, _ = load_policy("S500", args.checkpoint)
    groups = {"hidden_adapter": policy.model.qwen.hidden_adapter,
              "decoder_adapter": policy.model.qwen.decoder_adapter,
              "action_head": policy.model.action_model}
    for module in groups.values():
        module.requires_grad_(True).float()
    policy.train()
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})
    metadata = LeRobotDatasetMetadata("local/libero_spatial", root=args.dataset_root)
    rename = {"observation.images.wrist_image": "observation.images.image2"}
    dataset = LeRobotDataset("local/libero_spatial", root=args.dataset_root, episodes=[0],
                            delta_timestamps=resolve_delta_timestamps(policy.config, metadata, rename),
                            revision="v3.0", video_backend="pyav", return_uint8=True)
    sample = dataset[30]
    batch = _preprocess_dataset_batch(torch.utils.data.default_collate([sample]),
                                      metadata.camera_keys, rename, pre)
    inputs = policy._prepare_model_inputs(batch, training=False)
    torch.manual_seed(92000)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embodied, _ = policy.model._encode_qwen(inputs["images"], inputs["instructions"], need_action_tokens=False)
        loss = policy.model._action_loss(embodied, inputs["actions"], inputs["state"],
                                        inputs.get("action_is_pad"))
    loss.backward()
    results = {}
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        results[name] = {"parameters_with_grad": len(grads),
                         "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
                         "all_finite": all(bool(torch.isfinite(g).all()) for g in grads),
                         "grad_l2": float(torch.sqrt(sum(g.float().square().sum() for g in grads)))}
    passed = all(r["parameters_with_grad"] and r["all_finite"] and r["grad_l2"] > 0
                 for r in results.values())
    result = {"checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "protocol": "One native action-loss backward pass on training episode 0/frame 30; seed 92000; no optimizer",
              "action_loss": float(loss.detach()), "groups": results, "passed": bool(passed),
              "special_token_embeddings_trainable": bool(policy.model.qwen.model.get_input_embeddings().weight.requires_grad),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "limitations": "Connectivity check on one batch; no proof of useful representations or training convergence."}
    write_json(args.output, result)
    print(result, flush=True)
    if not passed:
        raise RuntimeError("Gradient connectivity check failed")


if __name__ == "__main__":
    main()
