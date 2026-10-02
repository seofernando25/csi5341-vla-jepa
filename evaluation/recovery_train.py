"""Registered, resumable native recovery study; never modifies the RSI harness."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import OncePerWarning, artifact
from evaluation.train import split_manifest, validation_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Verified legacy 10k warm start')
    parser.add_argument('--resume', type=Path, help='Native recovery checkpoint pretrained_model directory')
    parser.add_argument('--steps', type=int, required=True, help='Absolute recovery update at which to stop')
    parser.add_argument('--preflight', action='store_true', help='At most 20 updates; exercise validation at the stop step')
    parser.add_argument('--recipe', type=Path, default=ROOT / 'evaluation/recovery_config.json')
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe = read_json(recipe_path)
    if not 1 <= args.steps <= recipe['decay_steps']:
        parser.error('Stop step exceeds the registered schedule')
    if args.preflight and args.steps > 20:
        parser.error('Engineering validation preflight is bounded to 20 updates')
    if recipe.get('engineering_only') and not args.preflight:
        parser.error('Engineering recipes cannot launch production training')
    if recipe['batch_size'] != 8 or recipe['validation_samples_per_task'] != 20:
        parser.error('Registered physical-action evaluation requires 25 batches of eight')
    if recipe.get('required_gpu') and torch.cuda.get_device_name() != recipe['required_gpu']:
        raise ValueError('This engineering gate must run on the registered intended GPU')
    if recipe.get('deterministic_training'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    logging.getLogger('transformers.processing_utils').addFilter(OncePerWarning())
    source = args.architecture_source.resolve()
    amendment = read_json(source / 'amendment.json')
    query_enabled = recipe.get('query_token_adaptation', 'none') == 'input_residual'
    expected_path = ('query_source_amendment.json' if query_enabled else 'image_pipeline_amendment.json')
    expected = read_json(ROOT / 'studies/recovery/diagnostics' / expected_path)
    if query_enabled and (not recipe.get('smol_gradient_checkpointing')
                          or recipe.get('inference_buffer_precision') != 'native_rope'):
        raise ValueError('Query study requires registered checkpointing and native inference rotary buffers')
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != expected['source_manifest'] or amendment['source_manifest'] != manifest:
        raise ValueError('Recovery source differs from the registered input amendment')
    initial_hash = file_hash(args.checkpoint / 'model.safetensors')
    if initial_hash != recipe['initial_checkpoint_sha256']:
        raise ValueError('Warm-start checkpoint does not match the registered weights')
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Selected recovery source was not imported')
    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.accelerator import AcceleratorConfig
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as native
    from safetensors import safe_open
    from safetensors.torch import load_model

    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + ('-Query-recovery-train' if query_enabled else '-RGB-recovery-train')
    evidence = ROOT / 'studies/recovery/training' / run_id
    runtime = ROOT / 'outputs/recovery/training' / run_id
    evidence.mkdir(parents=True)
    runtime.mkdir(parents=True)
    record = {'run_id': run_id, 'study': recipe['study'], 'status': 'preparing',
              'recipe': recipe, 'recipe_sha256': file_hash(recipe_path),
              'source_manifest': manifest, 'initial_checkpoint_sha256': initial_hash,
              'resume_checkpoint_sha256': file_hash(args.resume / 'model.safetensors') if args.resume else None,
              'requested_stop_step': args.steps, 'environment': environment(),
              'purpose': 'engineering_preflight' if args.steps < 500 else 'recovery_adaptation',
              'validation_timing_override': args.steps if args.preflight else None,
              'allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'isolated_worker_rng': recipe.get('isolate_data_loader_rng', False),
              'limitations': 'Recovery updates are additional to the legacy 10k. Offline accuracy is not task success.'}
    write_json(evidence / 'run.json', record)
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = args.checkpoint.resolve()
    policy_cfg.vlm_model_name = str(artifact('smolvlm'))
    policy_cfg.jepa_encoder_name = str(artifact('world_model'))
    policy_cfg.init_from_vla_jepa = None
    policy_cfg.unfreeze_last_n = recipe['unfreeze_last_n']
    if query_enabled:
        policy_cfg.query_token_adaptation = 'input_residual'
        policy_cfg.smol_gradient_checkpointing = True
    policy_cfg.device = 'cuda'
    policy_cfg.image_processor_backend = 'torchvision'
    policy_cfg.optimizer_lr = recipe['learning_rate']
    policy_cfg.optimizer_weight_decay = recipe['weight_decay']
    policy_cfg.optimizer_grad_clip_norm = recipe['gradient_clip']
    policy_cfg.scheduler_warmup_steps = recipe['warmup_steps']
    policy_cfg.scheduler_decay_steps = recipe['decay_steps']
    policy_cfg.scheduler_decay_lr = recipe['decay_learning_rate']
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='local/libero_spatial', root=str(args.dataset_root.resolve()),
                              revision='v3.0', video_backend='pyav', eval_split=recipe['validation_fraction']),
        policy=policy_cfg, output_dir=runtime / 'train', job_name=recipe['study'],
        seed=recipe['seed'], batch_size=recipe['batch_size'], num_workers=recipe['num_workers'],
        steps=args.steps, eval_steps=recipe['eval_steps'], env_eval_freq=0, log_freq=10,
        save_checkpoint=True, save_freq=recipe['save_freq'], cudnn_deterministic=True,
        accelerator=AcceleratorConfig(mixed_precision='bf16'),
        rename_map={'observation.images.wrist_image': 'observation.images.image2'})
    state = {'step': 0, 'validation_batch': 0, 'action_rows': [], 'best_step': None,
             'best_mse': math.inf, 'checkpoints': {}}
    originals = {name: getattr(native, name) for name in ('make_train_eval_datasets', 'make_dataloaders',
                 'make_policy', 'make_pre_post_processors', 'make_optimizer_and_scheduler',
                 'update_policy', 'save_checkpoint')}

    def emit(row):
        with (evidence / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')

    def datasets(config):
        if args.preflight:
            # This engineering-only override is recorded above; production uses
            # the registered 500-update evaluation interval.
            config.eval_steps = args.steps
        elif config.eval_steps != recipe['eval_steps']:
            raise ValueError('Resume evaluation interval differs from registration')
        if config.batch_size != recipe['batch_size'] or config.seed != recipe['seed']:
            raise ValueError('Resume batch or seed differs from registration')
        if Path(config.dataset.root).resolve() != args.dataset_root.resolve():
            raise ValueError('Resume dataset differs')
        if config.policy.unfreeze_last_n != recipe['unfreeze_last_n']:
            raise ValueError('Resume trainability differs')
        if query_enabled and (config.policy.query_token_adaptation != 'input_residual'
                              or not config.policy.smol_gradient_checkpointing):
            raise ValueError('Resume changed registered query adaptation')
        protected = ('chunk_size', 'n_action_steps', 'normalization_mapping', 'conditioning_dim',
                     'decoder_adaptation', 'adapter_type', 'train_multimodal_projector', 'freeze_vision_tower',
                     'image_processor_backend', 'torch_dtype', 'action_hidden_size', 'action_model_type',
                     'action_num_layers', 'action_num_heads', 'action_dropout', 'repeated_diffusion_steps',
                     'num_inference_timesteps', 'enable_world_model', 'num_video_frames', 'predictor_depth',
                     'world_model_loss_weight', 'causal_world_model_context', 'use_relative_actions',
                     'binarize_gripper_action', 'pre_snap_gripper_action', 'clip_normalized_actions',
                     'gripper_threshold', 'resize_images_to', 'optimizer_betas', 'optimizer_eps',
                     'optimizer_weight_decay', 'optimizer_grad_clip_norm', 'scheduler_warmup_steps',
                     'scheduler_decay_lr')
        if any(getattr(config.policy, k) != getattr(policy_cfg, k) for k in protected):
            raise ValueError('Resume changed a protected policy setting')
        if (config.policy.vlm_model_name != policy_cfg.vlm_model_name
                or config.policy.jepa_encoder_name != policy_cfg.jepa_encoder_name):
            raise ValueError('Resume model locations differ; explicitly relocate the exported recovery config first')
        state['training_output'] = Path(config.output_dir)
        journal = state['training_output'] / 'recovery_checkpoints.json'
        if config.resume and journal.exists():
            prior = read_json(journal)
            state['checkpoints'] = {r['step']: r for r in prior}
            scored = [r for r in prior if 'heldout_arm_mse' in r and r['retained']]
            if scored:
                best = min(scored, key=lambda r: (r['heldout_arm_mse'], r['step']))
                state['best_step'], state['best_mse'] = best['step'], best['heldout_arm_mse']
        train, heldout = originals['make_train_eval_datasets'](config)
        observed = split_manifest(args.dataset_root.resolve(), train, heldout)
        if observed != read_json(ROOT / 'studies/evaluation/training_split.json'):
            raise ValueError('Data bytes or trajectory split changed')
        record['split_sha256'] = file_hash(ROOT / 'studies/evaluation/training_split.json')
        return train, heldout

    def loaders(config, dataset, heldout, step, parallel_dims):
        train, validation = originals['make_dataloaders'](config, dataset, heldout, step, parallel_dims)
        if recipe.get('isolate_data_loader_rng'):
            # Worker base seeds must not consume the CPU RNG used by flow times,
            # especially when an iterator is reconstructed during resume.
            train.generator = torch.Generator().manual_seed(recipe['worker_seed'])
        state['step'] = step
        indices = validation_indices(heldout)
        rows = heldout.hf_dataset.select(indices)
        selection = {'heldout_row_indices': indices, 'episode_indices': rows['episode_index'],
                     'frame_indices': rows['frame_index'], 'task_indices': rows['task_index']}
        selection = {k: [int(x) for x in v] for k, v in selection.items()}
        if selection != read_json(ROOT / 'studies/evaluation/validation_samples.json'):
            raise ValueError('Held-out sample membership changed')
        validation = torch.utils.data.DataLoader(torch.utils.data.Subset(heldout, indices),
                        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
                        collate_fn=validation.collate_fn,
                        generator=(torch.Generator().manual_seed(recipe['validation_seed'])
                                   if recipe.get('isolate_data_loader_rng') else None),
                        multiprocessing_context='spawn' if config.num_workers else None)
        return train, validation

    def make_policy(*a, **kw):
        # Construct first, cast trainable masters, THEN load. Loading into BF16 first
        # would silently round a resumed decoder's saved FP32 optimizer parameters.
        kw['defer_weight_load'] = True
        policy = originals['make_policy'](*a, **kw)
        policy.model.video_encoder.requires_grad_(False)
        for p in policy.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        checkpoint = Path(kw['cfg'].pretrained_path)
        if query_enabled and not args.resume:
            missing, unexpected = load_model(policy, checkpoint / 'model.safetensors', strict=False, device='cpu')
            query_key = 'model.qwen.query_adapter.delta'
            if set(missing) != {query_key} or unexpected:
                raise ValueError('Only the declared new query residual may lack initial weights')
            if torch.count_nonzero(policy.state_dict()[query_key]):
                raise ValueError('New query residual must initialize to zero')
        else:
            load_model(policy, checkpoint / 'model.safetensors', strict=True, device='cpu')
        counts = Counter()
        with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
            for name, tensor in policy.state_dict().items():
                if name in saved.keys():
                    if not torch.equal(tensor.detach().cpu(), saved.get_tensor(name)):
                        raise ValueError(f'Warm-start tensor precision changed: {name}')
                    counts[str(tensor.dtype)] += tensor.numel()
        write_json(evidence / 'initialization.json', {'exact_restoration': True,
                   'checkpoint_sha256': file_hash(checkpoint / 'model.safetensors'), 'parameter_dtypes': counts,
                   'new_zero_query_parameters': 3840 if query_enabled and not args.resume else 0})
        backbone, other = [], []
        trainability = {}
        for name, p in policy.named_parameters():
            if p.requires_grad:
                (backbone if name.startswith('model.qwen.model.') else other).append(p)
                trainability[name] = {'elements': p.numel(), 'dtype': str(p.dtype)}
        if not backbone or not other:
            raise ValueError('Expected trainable decoder layers and adapter/action/world parameters')
        write_json(evidence / 'trainability.json', trainability)
        record['trainable_parameters'] = sum(p.numel() for p in backbone + other)
        policy.get_optim_params = lambda: [{'params': other, 'lr': recipe['learning_rate'], 'name': 'adapter_action_world'},
                                         {'params': backbone, 'lr': recipe['backbone_learning_rate'], 'name': 'smol_decoder'}]

        def before(module, inputs):
            if not module.training:
                context = torch.random.fork_rng(devices=[torch.cuda.current_device()])
                context.__enter__()
                state['rng_context'] = context
                torch.manual_seed(recipe['validation_seed'] + state['validation_batch'])

        def after(module, inputs, output):
            if module.training:
                return
            context = state.pop('rng_context', None)
            try:
                if output is None:
                    return
                batch = inputs[0]
                loss, metrics = output
                emit({'phase': 'validation', 'step': state['step'], 'batch': state['validation_batch'],
                      'samples': batch['action'].shape[0], 'loss': float(loss),
                      **{k: float(v) for k, v in metrics.items()}})
                torch.manual_seed(recipe['action_validation_seed'] + state['validation_batch'])
                with torch.no_grad():
                    pred = state['post'](policy.predict_action_chunk(batch).float()).to('cuda')
                    target = state['post'](batch['action'].clone()).to('cuda')
                    valid = ~batch['action_is_pad'].bool()
                    count = int(valid.sum())
                    row = {'phase': 'validation_action', 'step': state['step'],
                           'batch': state['validation_batch'], 'valid_actions': count,
                           'arm_squared_error_sum': float(((pred[valid][..., :6] - target[valid][..., :6]) ** 2).sum()),
                           'gripper_errors': int((pred[valid][..., 6] != target[valid][..., 6]).sum())}
                    state['action_rows'].append(row)
                    emit(row)
                state['validation_batch'] += 1
            finally:
                if context:
                    context.__exit__(None, None, None)

        policy.register_forward_pre_hook(before)
        policy.register_forward_hook(after, always_call=True)
        return policy

    def processors(*a, **kw):
        # Saved processors retain the original baseline statistics and native semantics.
        pre, post = originals['make_pre_post_processors'](kw['policy_cfg'],
            pretrained_path=args.resume or args.checkpoint,
            preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                   'rename_observations_processor': {'rename_map': {}}})
        state['post'] = post
        return pre, post

    def optimizer(config, policy):
        if (config.policy.optimizer_lr != recipe['learning_rate']
                or config.policy.scheduler_decay_steps != recipe['decay_steps']):
            raise ValueError('Resume optimizer or schedule differs')
        return originals['make_optimizer_and_scheduler'](dataclasses.replace(config, steps=recipe['decay_steps']), policy)

    def update(*a, **kw):
        state['validation_batch'] = 0
        state['action_rows'] = []
        started = time.perf_counter()
        result = originals['update_policy'](*a, **kw)
        torch.cuda.synchronize()
        state['step'] += 1
        tracker, metrics = result
        active_optimizer = kw.get('optimizer', a[3] if len(a) > 3 else None)
        if not math.isfinite(tracker.loss.val) or not math.isfinite(tracker.grad_norm.val):
            raise RuntimeError('Non-finite recovery loss or gradient; stop before another update')
        emit({'phase': 'training', 'step': state['step'], 'loss': tracker.loss.val,
              'grad_norm': tracker.grad_norm.val, 'update_seconds': time.perf_counter() - started,
              'data_seconds': tracker.dataloading_s.val, 'preprocessing_seconds': tracker.preprocessing_s.val,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'learning_rates': {g['name']: g['lr'] for g in active_optimizer.param_groups}, **(metrics or {})})
        return result

    def save(*a, **kw):
        result = originals['save_checkpoint'](*a, **kw)
        directory = Path(kw['checkpoint_dir'])
        step = kw['step']
        write_json(directory / 'pretrained_model/recovery_registration.json',
                   {'study': recipe['study'], 'recipe_sha256': record['recipe_sha256'], 'source_manifest': manifest})
        model = directory / 'pretrained_model/model.safetensors'
        entry = {'step': step, 'checkpoint_sha256': file_hash(model), 'model_bytes': model.stat().st_size,
                 'retained': True}
        rows = state['action_rows']
        if rows:
            n = sum(r['valid_actions'] for r in rows)
            entry.update(heldout_arm_mse=sum(r['arm_squared_error_sum'] for r in rows) / (6 * n),
                         heldout_gripper_error=sum(r['gripper_errors'] for r in rows) / n,
                         heldout_valid_actions=n)
            if entry['heldout_arm_mse'] < state['best_mse']:
                state['best_step'], state['best_mse'] = step, entry['heldout_arm_mse']
        state['checkpoints'][step] = entry
        # Retention does not delete anything outside this native recovery output.
        keep = set(sorted(state['checkpoints'])[-2:]) | {state['best_step']}
        for old in directory.parent.iterdir():
            if old.is_dir() and old.name.isdigit() and int(old.name) not in keep:
                old_step = int(old.name)
                if old_step not in state['checkpoints']:
                    continue  # Resume history belongs to an earlier record; preserve it.
                shutil.rmtree(old)
                state['checkpoints'][old_step]['retained'] = False
        write_json(evidence / 'checkpoints.json', list(state['checkpoints'].values()))
        write_json(state['training_output'] / 'recovery_checkpoints.json', list(state['checkpoints'].values()))
        return result

    for name, replacement in [('make_train_eval_datasets', datasets), ('make_dataloaders', loaders),
                              ('make_policy', make_policy), ('make_pre_post_processors', processors),
                              ('make_optimizer_and_scheduler', optimizer), ('update_policy', update),
                              ('save_checkpoint', save)]:
        setattr(native, name, replacement)
    started = time.perf_counter()
    try:
        record['status'] = 'running'
        write_json(evidence / 'run.json', record)
        if args.resume:
            registration = read_json(args.resume / 'recovery_registration.json')
            if registration['recipe_sha256'] != record['recipe_sha256'] or registration['source_manifest'] != manifest:
                raise ValueError('Resume is not from this registered recovery study')
            # Paths may be relocated across hosts; artifact revisions, bytes and
            # scientific settings remain verified above and by the native loader.
            sys.argv = [sys.argv[0], f'--config_path={args.resume / "train_config.json"}', '--resume=true',
                        f'--steps={args.steps}', f'--dataset.root={args.dataset_root.resolve()}',
                        f'--policy.vlm_model_name={policy_cfg.vlm_model_name}',
                        f'--policy.jepa_encoder_name={policy_cfg.jepa_encoder_name}']
            native.main()
        else:
            sys.argv = [sys.argv[0]]
            native.train(cfg)
        record.update(status='completed', completed_steps=state['step'])
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record['wall_seconds'] = time.perf_counter() - started
        write_json(evidence / 'run.json', record)
        for weights in state.get('training_output', runtime / 'train').glob('checkpoints/*/pretrained_model/model.safetensors'):
            if weights.parent.parent.name.isdigit():
                write_json(weights.parent / 'recovery_registration.json',
                           {'study': recipe['study'], 'recipe_sha256': record['recipe_sha256'], 'source_manifest': manifest})
    print(json.dumps({'run_id': run_id, 'status': record['status']}), flush=True)


if __name__ == '__main__':
    main()
