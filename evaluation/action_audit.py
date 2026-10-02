"""Bounded offline action diagnostics; no optimizer updates or robot rollouts.

Uses the recorded train/held-out trajectories and native processing. Results
diagnose imitation errors, not closed-loop task success or a matched final test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation.common import ROOT, environment, file_hash, write_json
from evaluation.models import artifact, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["B16", "S500"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--architecture-source", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perturb", action="store_true", help="Diagnostic input ablations on the selected held-out frames")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Preserve earlier audit results; select a new output directory")
    if args.architecture_source:
        sys.path.insert(0, str(args.architecture_source.resolve() / "src"))
    if args.variant == "S500":
        import lerobot_policy_vla_jepa_smolvlm  # Register the selected snapshot configuration.
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    checkpoint = args.checkpoint or artifact("baseline")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    metadata = LeRobotDatasetMetadata("local/libero_spatial", root=args.dataset_root)
    rename = {"observation.images.wrist_image": "observation.images.image2"}
    deltas = resolve_delta_timestamps(config, metadata, rename)
    manifest = json.loads((ROOT / "studies/evaluation/training_split.json").read_text())
    datasets = {}
    # A few evenly spaced frames per task, away from padded episode ends.
    selection = []
    for split, key in [("train", "train_episodes"), ("heldout", "validation_episodes")]:
        dataset = LeRobotDataset("local/libero_spatial", root=args.dataset_root,
                                 episodes=manifest[key], delta_timestamps=deltas,
                                 revision="v3.0", video_backend="pyav", return_uint8=True)
        datasets[split] = dataset
        task_ids = np.asarray(dataset.hf_dataset.data.column("task_index").to_numpy())
        for task in sorted(set(task_ids.tolist())):
            rows = np.flatnonzero(task_ids == task)
            for fraction in (0.2, 0.5):
                selection.append((split, int(task), int(rows[int(fraction * (len(rows) - 1))])))

    policy, _ = load_policy(args.variant, args.checkpoint)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})
    mean_action = torch.tensor(np.asarray(datasets["train"].hf_dataset["action"]).mean(0),
                               device="cuda", dtype=torch.float32)
    records, probes = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (split, task, index) in enumerate(selection):
        dataset = datasets[split]
        sample = dataset[index]
        batch = torch.utils.data.default_collate([sample])
        raw_target = batch["action"].to("cuda")
        batch = _preprocess_dataset_batch(batch, metadata.camera_keys, rename, pre)
        normalized = batch["action"]
        mask = ~batch["action_is_pad"].bool()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(82000 + i)
            predicted = policy.predict_action_chunk(batch).float()
            physical = post(predicted.clone()).to("cuda").float()
            target_physical = post(normalized.clone()).to("cuda").float()
            changes = {}
            if args.perturb and split == "heldout":
                for name in ("black_images", "generic_instruction", "zero_state"):
                    altered = dict(batch)
                    if name == "black_images":
                        for key in policy.config.image_features:
                            altered[key] = torch.zeros_like(altered[key])
                    elif name == "generic_instruction":
                        altered["task"] = ["Execute the robot action."]
                    else:
                        altered["observation.state"] = torch.zeros_like(altered["observation.state"])
                    torch.manual_seed(82000 + i)
                    changed = policy.predict_action_chunk(altered).float()
                    changes[name] = float(((changed[..., :6] - predicted[..., :6]) ** 2).mean())
        # Dataset gripper is 0/1; simulator sign is +1 closed, -1 open.
        expected_physical = raw_target.clone()
        expected_physical[..., 6] = 1 - 2 * raw_target[..., 6]
        valid_target, valid_prediction = raw_target[mask], physical[mask]
        expected = expected_physical[mask]
        row = {"split": split, "task_id": task, "row": index,
               "episode": int(sample["episode_index"]), "frame": int(sample["frame_index"]),
               "valid_actions": int(mask.sum()),
               "arm_mse": float(((valid_prediction[..., :6] - expected[..., :6]) ** 2).mean()),
               "translation_mse": float(((valid_prediction[..., :3] - expected[..., :3]) ** 2).mean()),
               "mean_arm_mse": float(((mean_action[:6] - valid_target[..., :6]) ** 2).mean()),
               "normalized_mse": float(((predicted[mask] - normalized[mask]) ** 2).mean()),
               "gripper_error": float((valid_prediction[..., 6] != expected[..., 6]).float().mean()),
               "roundtrip_arm_error": float((target_physical[mask][..., :6] - expected[..., :6]).abs().max()),
               "roundtrip_gripper_error": float((target_physical[mask][..., 6] != expected[..., 6]).float().mean()),
               "gripper_prediction_mean": float(predicted[mask][..., 6].mean()),
               "target_gripper_mean": float(normalized[mask][..., 6].mean())}
        records.append(row)
        probes.append({"split": split, "task_id": task, "episode": row["episode"],
                       "frame": row["frame"], "instruction": sample["task"],
                       "predicted_normalized": predicted.cpu().tolist(),
                       "target_normalized": normalized.cpu().tolist(),
                       "predicted_physical": physical.cpu().tolist(),
                       "target_physical": expected_physical.cpu().tolist()})
        if changes:
            probes[-1]["normalized_arm_change_mse"] = changes
        print(json.dumps(row), flush=True)
    with (args.output / "actions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    summary = {"variant": args.variant, "checkpoint_sha256": file_hash(checkpoint / "model.safetensors"),
               "environment": environment(),
               "source_manifest": ({str(p.relative_to(args.architecture_source)): file_hash(p)
                                    for p in sorted((args.architecture_source / "src").rglob("*.py"))}
                                   if args.architecture_source else None),
               "split_manifest_sha256": file_hash(ROOT / "studies/evaluation/training_split.json"),
               "protocol": "20 frames/split, two/task; native seven-action chunks; seed 82000+row; BF16",
               "limitations": "Selected diagnostic frames; Qwen training membership is unknown; no task-success inference. Normalized MSE includes a gripper-label convention mismatch for Qwen and must not rank the two models; compare simulator arm commands instead.",
               "splits": {s: {k: float(np.mean([r[k] for r in records if r["split"] == s]))
                              for k in records[0] if k not in {"split", "task_id", "row", "episode", "frame"}}
                          for s in datasets}}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "probes.json", probes)


if __name__ == "__main__":
    main()
