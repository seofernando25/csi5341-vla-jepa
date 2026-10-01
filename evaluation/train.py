"""Reduced SmolVLM adaptation using the pinned native LeRobot training loop.

Only orchestration, trainability, held-out sampling, and telemetry are customized.
The inherited action/world losses and the native optimizer/resume path are retained.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import SOURCES, artifact

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


def split_manifest(dataset_root, train, heldout):
    """Hash local data and record whole-trajectory membership; no host paths."""
    files = sorted(
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".parquet", ".mp4"}
    )
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
        "files": {str(p.relative_to(dataset_root)): file_hash(p) for p in files},
    }


def validation_indices(dataset):
    task_ids = np.asarray(dataset.hf_dataset.data.column("task_index").to_numpy())
    selected = []
    for task in sorted(set(task_ids.tolist())):
        rows = np.flatnonzero(task_ids == task)
        n = min(RECIPE["validation_samples_per_task"], len(rows))
        selected.extend(rows[np.linspace(0, len(rows) - 1, n, dtype=int)].tolist())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resume", type=Path, help="Native checkpoint pretrained_model directory")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-label", default="S500")
    parser.add_argument("--architecture-source", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.run_label):
        parser.error("Invalid run label")
    architecture = None
    if args.architecture_source:
        source = args.architecture_source.resolve() / 'src'
        sys.path.insert(0, str(source))
        import lerobot_policy_vla_jepa_smolvlm as plugin
        if not Path(plugin.__file__).resolve().is_relative_to(source):
            raise ValueError('Selected architecture was not imported')
        architecture = {str(p.relative_to(source)): file_hash(p)
                        for p in sorted(source.rglob('*.py'))}
    from lerobot.scripts import lerobot_train as native

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"-{args.run_label}-train"
    evidence_dir = ROOT / "studies/evaluation/training" / run_id
    runtime = ROOT / "outputs/evaluation/training" / run_id
    evidence_dir.mkdir(parents=True)
    runtime.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "variant": args.run_label,
        "architecture_source_manifest": architecture,
        "status": "preparing",
        "recipe": RECIPE,
        "sources": SOURCES,
        "environment": environment(),
        "requested_steps": args.steps,
        "purpose": "engineering_preflight" if args.steps < 500 else "adaptation",
        "resume_checkpoint_sha256": file_hash(args.resume / "model.safetensors")
        if args.resume
        else None,
    }
    write_json(evidence_dir / "run.json", record)
    cfg = configuration(args.dataset_root.resolve(), runtime / "train", args.steps)
    cfg.policy.image_processor_backend = 'torchvision'
    original_datasets = native.make_train_eval_datasets
    original_loaders = native.make_dataloaders
    original_policy = native.make_policy
    original_processors = native.make_pre_post_processors
    original_update = native.update_policy
    original_optimizer = native.make_optimizer_and_scheduler
    state = {"step": 0, "validation_batch": 0}

    def emit(row):
        with (evidence_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def datasets(config):
        if Path(config.dataset.root).resolve() != args.dataset_root.resolve():
            raise ValueError("Resume dataset path differs; use the original dataset location")
        if config.batch_size != RECIPE["batch_size"] or config.seed != RECIPE["seed"]:
            raise ValueError("Resume batch size or seed differs from the frozen recipe")
        state["training_output"] = Path(config.output_dir)
        train, heldout = original_datasets(config)
        if heldout is None:
            raise ValueError("Validation split is required")
        manifest = split_manifest(args.dataset_root.resolve(), train, heldout)
        path = ROOT / "studies/evaluation/training_split.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError("Dataset or frozen split changed")
        write_json(path, manifest)
        record["training_split_sha256"] = file_hash(path)
        write_json(evidence_dir / "run.json", record)
        return train, heldout

    def loaders(config, dataset, eval_dataset, step, parallel_dims):
        train, validation = original_loaders(config, dataset, eval_dataset, step, parallel_dims)
        state["step"] = step
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
        path = ROOT / "studies/evaluation/validation_samples.json"
        if path.exists() and read_json(path) != selection:
            raise ValueError("Frozen validation sample membership changed")
        write_json(path, selection)
        validation = torch.utils.data.DataLoader(
            torch.utils.data.Subset(eval_dataset, selected),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=validation.collate_fn,
            multiprocessing_context="spawn" if config.num_workers else None,
        )
        return train, validation

    def make_policy(*a, **kw):
        policy = original_policy(*a, **kw)
        policy.model.video_encoder.requires_grad_(False)
        if any(p.requires_grad for p in policy.model.qwen.model.parameters()):
            raise ValueError("The initial adaptation recipe requires a frozen SmolVLM")
        if not args.resume:
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
                            raise ValueError(
                                f"Incomplete or modified pretrained initialization: {name}"
                            )
                        transferred.append(name)
            write_json(
                evidence_dir / "initialization.json",
                {"source": SOURCES["models"]["pretrain"], "verified_tensors": transferred},
            )
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        groups = {}
        for name, parameter in policy.named_parameters():
            group = ".".join(name.split(".")[:3])
            entry = groups.setdefault(group, {"total": 0, "trainable": 0})
            entry["total"] += parameter.numel()
            entry["trainable"] += parameter.numel() if parameter.requires_grad else 0
        write_json(evidence_dir / "trainability.json", groups)

        def before(module, inputs):
            if not module.training:
                context = torch.random.fork_rng(devices=[torch.cuda.current_device()])
                context.__enter__()
                state["rng_context"] = context
                torch.manual_seed(RECIPE["validation_seed"] + state["validation_batch"])

        def after(module, inputs, output):
            if not module.training:
                context = state.pop("rng_context", None)
                if context:
                    context.__exit__(None, None, None)
                if output is not None:
                    loss, metrics = output
                    emit(
                        {
                            "phase": "validation",
                            "step": state["step"],
                            "batch": state["validation_batch"],
                            "samples": inputs[0]["action"].shape[0],
                            "loss": float(loss.detach()),
                            **{k: float(v) for k, v in metrics.items()},
                        }
                    )
                    state["validation_batch"] += 1

        policy.register_forward_pre_hook(before)
        policy.register_forward_hook(after, always_call=True)
        return policy

    def processors(*a, **kw):
        from lerobot_policy_vla_jepa_smolvlm.processor_vla_jepa_smolvlm import (
            DropImagePadMasksProcessorStep,
        )

        # Freeze baseline action/state statistics instead of recomputing statistics
        # over a dataset containing held-out trajectories.
        policy_cfg = kw["policy_cfg"]
        pre, post = original_processors(
            policy_cfg,
            pretrained_path=artifact("baseline"),
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        pre.steps.insert(3, DropImagePadMasksProcessorStep())
        return pre, post

    def update(*a, **kw):
        state["validation_batch"] = 0
        started = time.perf_counter()
        result = original_update(*a, **kw)
        torch.cuda.synchronize()
        state["step"] += 1
        tracker, metrics = result
        optimizer = kw.get("optimizer") if "optimizer" in kw else a[3]
        emit(
            {
                "phase": "training",
                "step": state["step"],
                "update_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                **tracker.to_dict(),
                **(metrics or {}),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": tracker.grad_norm.val,
            }
        )
        return result

    def optimizer_and_scheduler(config, policy):
        # Native cosine scheduling auto-compresses its horizon to cfg.steps.
        # Keep a fixed 10k-step trajectory while stopping/resuming at milestones.
        fixed = dataclasses.replace(config, steps=RECIPE["decay_steps"])
        result = original_optimizer(fixed, policy)
        write_json(
            evidence_dir / "schedule.json",
            {
                "warmup_steps": RECIPE["warmup_steps"],
                "decay_steps": RECIPE["decay_steps"],
                "stage_stop_step": config.steps,
                "budget_independent": True,
            },
        )
        record["fixed_schedule"] = True
        write_json(evidence_dir / "run.json", record)
        return result

    native.make_train_eval_datasets = datasets
    native.make_dataloaders = loaders
    native.make_policy = make_policy
    native.make_pre_post_processors = processors
    native.update_policy = update
    native.make_optimizer_and_scheduler = optimizer_and_scheduler
    started = time.perf_counter()
    try:
        if args.prepare_only:
            _train, heldout = datasets(cfg)
            selected = validation_indices(heldout)
            record.update(status="prepared", validation_samples=len(selected))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required")
            record["status"] = "running"
            write_json(evidence_dir / "run.json", record)
            if args.resume:
                sys.argv = [
                    sys.argv[0],
                    f"--config_path={args.resume / 'train_config.json'}",
                    "--resume=true",
                    f"--steps={args.steps}",
                ]
                native.main()
            else:
                sys.argv = [sys.argv[0]]
                # The native trainer permits camera renaming only when a policy
                # checkpoint exists. Materialize the seeded backbone-swap initializer
                # first, then let native loading/renaming handle training and resume.
                from lerobot.utils.random_utils import set_seed

                from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMPolicy

                set_seed(RECIPE["seed"])
                initial = VLAJEPASmolVLMPolicy(cfg.policy)
                initial_dir = runtime / "initial_policy"
                initial.config.init_from_vla_jepa = None
                initial.save_pretrained(initial_dir)
                cfg.policy.pretrained_path = initial_dir
                record["initializer_sha256"] = file_hash(initial_dir / "model.safetensors")
                write_json(evidence_dir / "run.json", record)
                del initial
                native.train(cfg)
            checkpoints = []
            for weights in sorted(
                state["training_output"].glob("checkpoints/*/pretrained_model/model.safetensors")
            ):
                step_name = weights.parents[1].name
                if step_name.isdigit():
                    checkpoints.append(
                        {
                            "step": int(step_name),
                            "sha256": file_hash(weights),
                            "bytes": weights.stat().st_size,
                            "artifact": str(weights.parent.relative_to(ROOT)),
                        }
                    )
            record.update(
                status="completed", completed_steps=state["step"], checkpoints=checkpoints
            )
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        write_json(evidence_dir / "run.json", record)
    print(json.dumps({"run_id": run_id, "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
