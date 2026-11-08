"""Reduced SmolVLM adaptation using the pinned native LeRobot training loop.

Only orchestration, trainability, held-out sampling, and telemetry are customized.
The inherited action/world losses and the native optimizer/resume path are retained.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.common import ROOT, environment, read_json, recorded_run, run_id, write_json
from evaluation.sources import SOURCES, artifact

RECIPE = read_json(ROOT / "evaluation/training_config.json")


def configuration(dataset_root, output, steps):
    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.accelerator import AcceleratorConfig
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig

    baseline = PreTrainedConfig.from_pretrained(artifact("baseline"))
    fields = {
        f.name: getattr(baseline, f.name) for f in dataclasses.fields(VLAJEPAConfig) if f.init
    }
    fields.update(
        device="cuda",
        torch_dtype="bfloat16",
        pretrained_path=None,
        push_to_hub=False,
        optimizer_lr=RECIPE["learning_rate"],
        optimizer_weight_decay=RECIPE["weight_decay"],
        optimizer_grad_clip_norm=RECIPE["gradient_clip"],
        scheduler_warmup_steps=RECIPE["warmup_steps"],
        scheduler_decay_steps=RECIPE["decay_steps"],
        scheduler_decay_lr=RECIPE["decay_learning_rate"],
    )
    policy = VLAJEPASmolVLMConfig(
        **fields,
        vlm_model_name=str(artifact("smolvlm")),
        init_from_vla_jepa=str(artifact("pretrain")),
    )
    policy.jepa_encoder_name = str(artifact("world_model"))
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/libero_spatial",
            root=str(dataset_root),
            revision="v3.0",
            video_backend="pyav",
            eval_split=RECIPE["validation_fraction"],
        ),
        policy=policy,
        output_dir=output,
        job_name="csi5341_s500",
        seed=RECIPE["seed"],
        batch_size=RECIPE["batch_size"],
        num_workers=RECIPE["num_workers"],
        steps=steps,
        eval_steps=RECIPE["eval_steps"],
        env_eval_freq=0,
        log_freq=10,
        save_checkpoint=True,
        save_freq=RECIPE["save_freq"],
        cudnn_deterministic=True,
        accelerator=AcceleratorConfig(mixed_precision="bf16"),
        rename_map={"observation.images.wrist_image": "observation.images.image2"},
    )


def split_manifest(train, heldout):
    """Record whole-trajectory membership of the train/validation split; no host paths."""
    train_episodes, validation_episodes = list(train.episodes), list(heldout.episodes)
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("Training and validation trajectories overlap")
    return {
        "dataset": "LIBERO-Spatial local no-noops LeRobot conversion",
        "codebase_version": train.meta.info["codebase_version"],
        "split_rule": "Last ceil(10% of episodes) per task, pinned native LeRobot split",
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "train_frames": len(train),
        "validation_frames": len(heldout),
    }


def validation_indices(dataset):
    task_ids = np.asarray(dataset.hf_dataset.data.column("task_index").to_numpy())
    selected = []
    for task in sorted(set(task_ids.tolist())):
        rows = np.flatnonzero(task_ids == task)
        n = min(RECIPE["validation_samples_per_task"], len(rows))
        selected.extend(rows[np.linspace(0, len(rows) - 1, n, dtype=int)].tolist())
    return selected


def freeze_or_check(path, value, message):
    """Write `value` on first use; later runs must reproduce it exactly."""
    if path.exists() and read_json(path) != value:
        raise ValueError(message)
    write_json(path, value)


def verify_initialization(policy, evidence_dir):
    """Every transferred action/predictor tensor must equal the pretrained checkpoint."""
    from safetensors import safe_open

    transferred = []
    with safe_open(
        artifact("pretrain") / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        source_keys = set(source.keys())
        for name, tensor in policy.state_dict().items():
            if name.startswith(("model.action_model.", "model.video_predictor.")):
                if name not in source_keys or not torch.equal(
                    tensor.cpu(), source.get_tensor(name)
                ):
                    raise ValueError(f"Incomplete or modified pretrained initialization: {name}")
                transferred.append(name)
    write_json(
        evidence_dir / "initialization.json",
        {"source": SOURCES["models"]["pretrain"], "verified_tensors": transferred},
    )


def trainability(policy):
    groups = {}
    for name, parameter in policy.named_parameters():
        entry = groups.setdefault(".".join(name.split(".")[:3]), {"total": 0, "trainable": 0})
        entry["total"] += parameter.numel()
        entry["trainable"] += parameter.numel() if parameter.requires_grad else 0
    return groups


class NativeTrainingHooks:
    """Replacements for LeRobot's lerobot_train factories that add this study's constraints.

    They freeze the split and validation sample, freeze SmolVLM, verify initialization, keep
    the baseline normalization, fix the learning-rate schedule to the full 10k-step budget,
    seed validation batches, and log every update and validation batch to metrics.jsonl.
    """

    def __init__(self, native, args, evidence_dir, record):
        self.native, self.args, self.evidence_dir, self.record = native, args, evidence_dir, record
        self.original = {
            name: getattr(native, name)
            for name in [
                "make_train_eval_datasets",
                "make_dataloaders",
                "make_policy",
                "make_pre_post_processors",
                "update_policy",
                "make_optimizer_and_scheduler",
            ]
        }
        self.step, self.validation_batch, self.training_output = 0, 0, None

    def install(self):
        self.native.make_train_eval_datasets = self.datasets
        self.native.make_dataloaders = self.loaders
        self.native.make_policy = self.make_policy
        self.native.make_pre_post_processors = self.processors
        self.native.update_policy = self.update
        self.native.make_optimizer_and_scheduler = self.optimizer_and_scheduler

    def emit(self, row):
        with (self.evidence_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def datasets(self, config):
        if Path(config.dataset.root).resolve() != self.args.dataset_root.resolve():
            raise ValueError("Resume dataset path differs; use the original dataset location")
        if config.batch_size != RECIPE["batch_size"] or config.seed != RECIPE["seed"]:
            raise ValueError("Resume batch size or seed differs from the frozen recipe")
        self.training_output = Path(config.output_dir)
        train, heldout = self.original["make_train_eval_datasets"](config)
        if heldout is None:
            raise ValueError("Validation split is required")
        freeze_or_check(
            ROOT / "studies/evaluation/training_split.json",
            split_manifest(train, heldout),
            "Dataset or frozen split changed",
        )
        return train, heldout

    def loaders(self, config, dataset, eval_dataset, step, parallel_dims):
        train, validation = self.original["make_dataloaders"](
            config, dataset, eval_dataset, step, parallel_dims
        )
        self.step = step
        selected = validation_indices(eval_dataset)
        rows = eval_dataset.hf_dataset.select(selected)
        selection = {
            "heldout_row_indices": selected,
            "episode_indices": list(rows["episode_index"]),
            "frame_indices": list(rows["frame_index"]),
            "task_indices": list(rows["task_index"]),
        }
        # HF datasets may return scalar torch tensors via their format adapter.
        selection = {k: [int(x) for x in v] for k, v in selection.items()}
        freeze_or_check(
            ROOT / "studies/evaluation/validation_samples.json",
            selection,
            "Frozen validation sample membership changed",
        )
        validation = torch.utils.data.DataLoader(
            torch.utils.data.Subset(eval_dataset, selected),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=validation.collate_fn,
            multiprocessing_context="spawn" if config.num_workers else None,
        )
        return train, validation

    def make_policy(self, *args, **kwargs):
        policy = self.original["make_policy"](*args, **kwargs)
        policy.model.video_encoder.requires_grad_(False)
        if any(p.requires_grad for p in policy.model.qwen.model.parameters()):
            raise ValueError("The adaptation recipe requires a frozen SmolVLM")
        if not self.args.resume:
            verify_initialization(policy, self.evidence_dir)
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        write_json(self.evidence_dir / "trainability.json", trainability(policy))
        policy.register_forward_pre_hook(self.before_forward)
        policy.register_forward_hook(self.after_forward, always_call=True)
        return policy

    def before_forward(self, module, inputs):
        """Validation batches use a fixed noise seed without disturbing the training RNG."""
        if not module.training:
            self.rng = torch.random.fork_rng(devices=[torch.cuda.current_device()])
            self.rng.__enter__()
            torch.manual_seed(RECIPE["validation_seed"] + self.validation_batch)

    def after_forward(self, module, inputs, output):
        if module.training:
            return
        rng, self.rng = getattr(self, "rng", None), None
        if rng:
            rng.__exit__(None, None, None)
        if output is not None:
            loss, metrics = output
            self.emit(
                {
                    "phase": "validation",
                    "step": self.step,
                    "batch": self.validation_batch,
                    "samples": inputs[0]["action"].shape[0],
                    "loss": float(loss.detach()),
                    **{k: float(v) for k, v in metrics.items()},
                }
            )
            self.validation_batch += 1

    def processors(self, *args, **kwargs):
        """Baseline normalization statistics, never recomputed over held-out trajectories."""
        from lerobot_policy_vla_jepa_smolvlm.processor_vla_jepa_smolvlm import (
            DropImagePadMasksProcessorStep,
        )

        pre, post = self.original["make_pre_post_processors"](
            kwargs["policy_cfg"],
            pretrained_path=artifact("baseline"),
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        pre.steps.insert(3, DropImagePadMasksProcessorStep())
        return pre, post

    def update(self, *args, **kwargs):
        self.validation_batch = 0
        started = time.perf_counter()
        result = self.original["update_policy"](*args, **kwargs)
        torch.cuda.synchronize()
        self.step += 1
        tracker, metrics = result
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[3]
        self.emit(
            {
                "phase": "training",
                "step": self.step,
                "update_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                **tracker.to_dict(),
                **(metrics or {}),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": tracker.grad_norm.val,
            }
        )
        return result

    def optimizer_and_scheduler(self, config, policy):
        """Cosine schedule over the full budget even when a stage stops early."""
        fixed = dataclasses.replace(config, steps=RECIPE["decay_steps"])
        result = self.original["make_optimizer_and_scheduler"](fixed, policy)
        write_json(
            self.evidence_dir / "schedule.json",
            {
                "warmup_steps": RECIPE["warmup_steps"],
                "decay_steps": RECIPE["decay_steps"],
                "stage_stop_step": config.steps,
            },
        )
        return result

    def checkpoints(self):
        found = []
        pattern = "checkpoints/*/pretrained_model/model.safetensors"
        for weights in sorted(self.training_output.glob(pattern)):
            step_name = weights.parents[1].name
            if step_name.isdigit():
                found.append(
                    {
                        "step": int(step_name),
                        "bytes": weights.stat().st_size,
                        "artifact": str(weights.parent.relative_to(ROOT)),
                    }
                )
        return found


def train(native, args, cfg, runtime):
    if args.resume:
        sys.argv = [
            sys.argv[0],
            f"--config_path={args.resume / 'train_config.json'}",
            "--resume=true",
            f"--steps={args.steps}",
        ]
        native.main()
        return
    # The native trainer renames cameras only when loading a policy checkpoint, so the
    # seeded backbone-swap initializer is saved first and loaded like any checkpoint.
    from lerobot.utils.random_utils import set_seed

    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMPolicy

    sys.argv = [sys.argv[0]]
    set_seed(RECIPE["seed"])
    initial = VLAJEPASmolVLMPolicy(cfg.policy)
    initial_dir = runtime / "initial_policy"
    initial.config.init_from_vla_jepa = None
    initial.save_pretrained(initial_dir)
    cfg.policy.pretrained_path = initial_dir
    del initial
    native.train(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resume", type=Path, help="Native checkpoint pretrained_model directory")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    from lerobot.scripts import lerobot_train as native

    name = run_id("S500-train")
    evidence_dir = ROOT / "studies/evaluation/training" / name
    runtime = ROOT / "outputs/evaluation/training" / name
    evidence_dir.mkdir(parents=True)
    runtime.mkdir(parents=True)
    record = {
        "run_id": name,
        "variant": "S500",
        "recipe": RECIPE,
        "sources": SOURCES,
        "environment": environment(),
        "requested_steps": args.steps,
        "resume_checkpoint": str(args.resume) if args.resume else None,
    }
    cfg = configuration(args.dataset_root.resolve(), runtime / "train", args.steps)
    cfg.policy.image_processor_backend = "torchvision"
    hooks = NativeTrainingHooks(native, args, evidence_dir, record)
    hooks.install()
    with recorded_run(evidence_dir / "run.json", record):
        if args.prepare_only:
            _, heldout = hooks.datasets(cfg)
            record.update(status="prepared", validation_samples=len(validation_indices(heldout)))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required")
            train(native, args, cfg, runtime)
            record.update(completed_steps=hooks.step, checkpoints=hooks.checkpoints())
    print(json.dumps({"run_id": name, "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
