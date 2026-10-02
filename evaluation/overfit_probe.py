"""Small training-only overfit experiment, separate from frozen RSI confirmation.

This is a diagnostic intervention, not a full adaptation run or success result.
Keeps native action/world losses, preprocessing, initialization and semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.common import environment, file_hash, write_json
from evaluation.models import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic run directory")
    if not 1 <= args.steps <= 2000 or not 0 <= args.unfreeze_last_n <= 4:
        raise ValueError("This diagnostic is bounded to 2,000 updates and four decoder layers")
    sys.path.insert(0, str(args.architecture_source.resolve() / "src"))
    import lerobot_policy_vla_jepa_smolvlm
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    torch.manual_seed(42)
    policy, _ = load_policy("S500", args.checkpoint)
    policy.config.unfreeze_last_n = args.unfreeze_last_n
    policy.model.qwen._configure_trainability()
    policy.model.action_model.requires_grad_(True)
    policy.model.video_predictor.requires_grad_(True)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})
    metadata = LeRobotDatasetMetadata("local/libero_spatial", root=args.dataset_root)
    rename = {"observation.images.wrist_image": "observation.images.image2"}
    dataset = LeRobotDataset("local/libero_spatial", root=args.dataset_root, episodes=[0],
                            delta_timestamps=resolve_delta_timestamps(policy.config, metadata, rename),
                            revision="v3.0", video_backend="pyav", return_uint8=True)
    indices = np.linspace(0, len(dataset) - 9, 16, dtype=int).tolist()
    batches = [_preprocess_dataset_batch(torch.utils.data.default_collate([dataset[j] for j in indices[i:i+2]]),
                                         metadata.camera_keys, rename, pre) for i in range(0, 16, 2)]
    args.output.mkdir(parents=True)
    backbone, other = [], []
    for name, p in policy.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith("model.qwen.model.") else other).append(p)
    optimizer = torch.optim.AdamW([{"params": other, "lr": 1e-4}, {"params": backbone, "lr": 1e-5}],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8)
    record = {"status": "running", "purpose": "training-only diagnostic, not confirmation",
              "initial_checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
              "architecture_source_manifest": {
                  str(p.relative_to(args.architecture_source)): file_hash(p)
                  for p in sorted((args.architecture_source / "src").rglob("*.py"))},
              "environment": environment(), "episode": 0, "frame_indices": indices,
              "batch_size": 2, "steps": args.steps, "unfreeze_last_n": args.unfreeze_last_n,
              "optimizer": "fresh AdamW, no optimizer continuation",
              "learning_rates": {"backbone": 1e-5, "adapter_action_world": 1e-4},
              "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
              "gate": "At least 90% reduction in same-frame arm MSE and at most 5% gripper errors",
              "limitations": "Sixteen training frames from one trajectory; overfitting does not demonstrate held-out control."}
    write_json(args.output / "run.json", record)

    def evaluate(step):
        policy.eval()
        losses, errors, grip_errors = [], [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, batch in enumerate(batches):
                torch.manual_seed(95000 + i)
                loss, metrics = policy(batch)
                torch.manual_seed(96000 + i)
                pred = policy.predict_action_chunk(batch).float()
                physical = post(pred.clone()).to("cuda")
                expected = post(batch["action"].clone()).to("cuda")
                losses.append(float(loss))
                errors.append(float(((physical[..., :6] - expected[..., :6]) ** 2).mean()))
                grip_errors.append(float((physical[..., 6] != expected[..., 6]).float().mean()))
        result = {"phase": "probe", "step": step, "loss": float(np.mean(losses)),
                  "arm_mse": float(np.mean(errors)), "gripper_error": float(np.mean(grip_errors))}
        print(json.dumps(result), flush=True)
        policy.train()
        return result

    before = evaluate(0)
    intermediate = []
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("w") as stream:
        for step in range(1, args.steps + 1):
            # A reproducible cycling order; this intentionally reuses the same sixteen frames.
            batch = batches[(step - 1) % len(batches)]
            torch.manual_seed(97000 + step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite diagnostic gradient")
            optimizer.step()
            row = {"phase": "training", "step": step, "loss": float(loss.detach()),
                   "grad_norm": float(norm), **{k: float(v) for k, v in metrics.items()}}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if step % 25 == 0:
                print(json.dumps(row), flush=True)
            if step % 250 == 0:
                intermediate.append(evaluate(step))
                latest = intermediate[-1]
                if latest["arm_mse"] < .1 * before["arm_mse"] and latest["gripper_error"] <= .05:
                    break
    after = intermediate[-1] if intermediate and intermediate[-1]["step"] == step else evaluate(step)
    record.update(status="completed", before=before, after=after,
                  intermediate=intermediate,
                  steps_completed=step,
                  elapsed_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gate_passed=after["arm_mse"] < 0.1 * before["arm_mse"] and after["gripper_error"] <= 0.05)
    # Preserve compact evidence only; these diagnostic weights are not a deliverable policy.
    write_json(args.output / "run.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
