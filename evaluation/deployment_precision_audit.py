"""Registered forward-only comparison of native masters and BF16 deployment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import artifact, cast_inference_policy
from evaluation.query_milestone import context, selected_entry, validation_score, verify_counters, verify_files


MODES = ('native_masters', 'bf16_deployment')
ROPE_SUFFIXES = ('rotary_emb.inv_freq', 'rotary_emb.original_inv_freq')


def config_identity(selected, native):
    # This recorded warm-start path changes across registered stage resumes.
    selected, native = selected.copy(), native.copy()
    selected.pop('pretrained_path', None)
    native.pop('pretrained_path', None)
    if selected != native:
        raise ValueError('Selected policy configuration differs from final native architecture/semantics')


def selection_context(milestone, journal, registration):
    """Bind the final study's existing selection rule before seeing audit outcomes."""
    expected = {'purpose': 'registered_query_verified_milestone',
                'study': registration['query_study'], 'recipe_sha256': registration['query_recipe_sha256'],
                'training_source_manifest': registration['source_manifest'], 'native_step': 10000,
                'scheduler_last_epoch': 10000, 'selected_local_weights_verified': True,
                'native_export_verified_files': 13, 'native_optimizer_parameter_states': 693,
                'native_optimizer_registered_parameters': 694, 'buffer_precision': 'native_rope',
                'loader_sha256': registration['loader_sha256']}
    if any(milestone.get(key) != value for key, value in expected.items()):
        raise ValueError('Require the verified final10k query milestone and unchanged loader/source')
    selected = selected_entry(journal, 10000, {'checkpoint_sha256': milestone['checkpoint_sha256']})
    if (selected['step'] != milestone['selected_step']
            or selected['heldout_arm_mse'] != milestone['heldout_arm_mse']
            or selected['heldout_gripper_error'] != milestone['heldout_gripper_error']):
        raise ValueError('Milestone differs from the existing held-out selection rule')
    return selected


def summarize(rows):
    if (len(rows) != 25 or [r['batch'] for r in rows] != list(range(25))
            or any(r['samples'] != 8 for r in rows)
            or sum(r['valid_actions'] for r in rows) != 1303):
        raise ValueError('Require all registered200 frames and1303 valid actions')
    for row in rows:
        for name in ('loss', 'action_loss', 'wm_loss', 'arm_squared_error_sum'):
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError('Invalid passive validation measurement')
        if (not 0 < row['valid_actions'] <= 56
                or not 0 <= row['gripper_errors'] <= row['valid_actions']
                or not math.isclose(row['loss'], row['action_loss'] + row['wm_loss'], rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError('Invalid physical counts or loss decomposition')
    return {'frames': 200, 'valid_actions': 1303,
            'arm_mse': sum(r['arm_squared_error_sum'] for r in rows) / (6 * 1303),
            'gripper_error': sum(r['gripper_errors'] for r in rows) / 1303,
            **{key: sum(r[key] * r['samples'] for r in rows) / 200
               for key in ('loss', 'action_loss', 'wm_loss')}}


def load_case(policy, checkpoint, trainability, mode):
    """Promote before loading; never reconstruct FP32 masters from rounded BF16."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model

    if mode not in MODES:
        raise ValueError('Unknown registered precision case')
    parameters = dict(policy.named_parameters())
    if {name for name, p in parameters.items() if p.requires_grad} != set(trainability):
        raise ValueError('Registered trainability differs from constructed source')
    # Only parameters vary between cases. Reject an unregistered buffer difference.
    buffers = {name: tensor.detach().clone() for name, tensor in policy.named_buffers()}
    floating = {name for name, tensor in buffers.items() if tensor.is_floating_point()}
    if len(floating) != 2 or any(not name.endswith(ROPE_SUFFIXES) for name in floating):
        raise ValueError('Unexpected floating buffers; explicitly register their treatment before measuring')
    if any(buffers[name].dtype != torch.float32 for name in floating):
        raise ValueError('Native rotary buffers already rounded')
    for name, parameter in parameters.items():
        parameter.data = parameter.data.to(torch.float32 if mode == MODES[0] and name in trainability else torch.bfloat16)
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
        masters = {name for name in saved.keys() if saved.get_slice(name).get_dtype() == 'F32'}
        if masters != set(trainability):
            raise ValueError('Native FP32 checkpoint membership differs from registered trainability')
        if any(math.prod(saved.get_slice(name).get_shape()) != trainability[name]['elements']
               for name in masters):
            raise ValueError('Native master shape changed')
    load_model(policy, checkpoint / 'model.safetensors', strict=True, device='cpu')
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
        checked = 0
        for name, tensor in policy.state_dict().items():
            if name not in saved.keys():
                raise ValueError('State missing from native checkpoint')
            native = saved.get_tensor(name)
            expected = native.to(tensor.dtype)
            if not torch.equal(tensor.detach().cpu(), expected):
                raise ValueError('Loaded weights differ from declared native or BF16 values')
            checked += 1
    if mode == MODES[1]:
        # Exercise the protected deployment cast, including its native-RoPE restoration.
        cast_inference_policy(policy, 'cpu', 'native_rope')
    policy.eval().requires_grad_(False)
    for name, tensor in policy.named_buffers():
        if tensor.dtype != buffers[name].dtype or not torch.equal(tensor.cpu(), buffers[name]):
            raise ValueError('A buffer changed across the declared parameter-only comparison')
    return {'checked_state_tensors': checked, 'declared_weight_values_verified': True,
            'fp32_master_tensors': len(masters) if mode == MODES[0] else 0,
            'native_rotary_buffers': sorted(floating),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in policy.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Held-out-selected pretrained_model directory')
    parser.add_argument('--native-checkpoint', type=Path, required=True, help='Final10000 native resume directory')
    parser.add_argument('--parent-anchor', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier measurements')
    registration_path = ROOT / 'studies/recovery/deployment_precision_registration.json'
    registration = read_json(registration_path)
    for name, digest in registration['implementation_manifest'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Registered diagnostic or evaluation dependency changed')
    source = args.architecture_source.resolve()
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest']:
        raise ValueError('Wrong registered query source')
    split = read_json(ROOT / 'studies/evaluation/training_split.json')
    for name, digest in registration['data_manifest'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Registered held-out membership changed')
    for name, digest in split['files'].items():
        if file_hash(args.dataset_root / name) != digest:
            raise ValueError('Frozen dataset bytes changed')
    milestone_path = ROOT / 'studies/recovery/diagnostics/query_10000_milestone.json'
    milestone = read_json(milestone_path)
    history = ROOT / 'studies/recovery/training' / milestone['training_run_id']
    journal = read_json(history / 'checkpoints.json')
    selected = selection_context(milestone, journal, registration)
    query_registration = read_json(ROOT / 'studies/recovery/query_recovery_registration.json')
    recipe = read_json(ROOT / 'evaluation/query_recovery_config.json')
    if file_hash(ROOT / 'evaluation/query_recovery_config.json') != registration['query_recipe_sha256']:
        raise ValueError('Frozen query recipe changed')
    context(read_json(history / 'run.json'), query_registration, recipe)
    validation_history = ROOT / 'studies/recovery/training' / milestone['selected_validation_run_id']
    context(read_json(validation_history / 'run.json'), query_registration, recipe)
    stream = (validation_history / 'metrics.jsonl').read_bytes()
    validation_score([json.loads(line) for line in stream.splitlines()], selected)
    if int(args.native_checkpoint.name) != 10000:
        raise ValueError('Require final10k native resume directory')
    verify_files(args.native_checkpoint, milestone['native_export_files'])
    trainability_path = history / 'trainability.json'
    trainability = read_json(trainability_path)
    verify_counters(args.native_checkpoint, 10000, trainability)
    if file_hash(args.checkpoint / 'model.safetensors') != selected['checkpoint_sha256']:
        raise ValueError('Selected local weights differ from the final journal')
    checkpoint_files = {}
    for name in milestone['native_export_files']:
        if name.startswith('pretrained_model/') and not name.endswith('model.safetensors'):
            leaf = Path(name).name
            checkpoint_files[leaf] = file_hash(args.checkpoint / leaf)
            if leaf == 'config.json':
                config_identity(read_json(args.checkpoint / leaf),
                                read_json(args.native_checkpoint / 'pretrained_model' / leaf))
            elif leaf != 'train_config.json' and checkpoint_files[leaf] != milestone['native_export_files'][name]['sha256']:
                raise ValueError('Selected config/processors differ from final native package')
    pair = read_json(ROOT / 'outputs/recovery/local-lr-pair/job.json')
    if (pair.get('status') != 'completed' or pair.get('child_pid') is not None
            or set(pair.get('branches', {})) != {'high', 'low'}):
        raise ValueError('Wait until both local rate branches finish')
    from evaluation.local_lr_pair_analysis import anchor_context
    pair_registration = read_json(ROOT / 'studies/recovery/local_lr_pair_registration.json')
    anchor_registration = read_json(ROOT / 'studies/recovery/parent_native_selection_registration.json')
    flags = {'deterministic_algorithms': True, 'matmul_allow_tf32': True, 'cudnn_allow_tf32': True,
             'cudnn_benchmark': False, 'cudnn_deterministic': True}
    anchor_context(read_json(args.parent_anchor), pair_registration, anchor_registration, flags)
    clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if clients.strip():
        raise ValueError('GPU is occupied; do not overlap registered measurements')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_name() != registration['gpu']:
        raise ValueError('Use the registered idle local GPU')
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Wrong plugin imported')
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    selection = read_json(ROOT / 'studies/evaluation/validation_samples.json')
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    output = {'purpose': registration['purpose'], 'status': 'running', 'cases': [],
              'registration_sha256': file_hash(registration_path), 'milestone_sha256': file_hash(milestone_path),
              'source_manifest': manifest, 'checkpoint_sha256': selected['checkpoint_sha256'],
              'selected_step': selected['step'], 'selection_uses_development': False,
              'checkpoint_package_hashes': checkpoint_files, 'trainability_sha256': file_hash(trainability_path),
              'environment': environment(), 'numerical_flags': {
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                  'cudnn_deterministic': torch.backends.cudnn.deterministic},
              'limitations': registration['limitations']}
    write_json(args.output, output)
    reference_predictions = []
    try:
        for mode in MODES:
            cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
            cfg.vlm_model_name, cfg.jepa_encoder_name = str(artifact('smolvlm')), str(artifact('world_model'))
            cfg.init_from_vla_jepa, cfg.device = None, 'cuda'
            policy = plugin.VLAJEPASmolVLMPolicy(cfg)
            policy.model.video_encoder.requires_grad_(False)
            proof = load_case(policy, args.checkpoint, trainability, mode)
            policy.to('cuda')
            dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root,
                episodes=split['validation_episodes'],
                delta_timestamps=resolve_delta_timestamps(cfg, metadata, rename), revision='v3.0',
                video_backend='pyav', return_uint8=True)
            frames = dataset.hf_dataset.select(selection['heldout_row_indices'])
            for key, field in [('episode_indices', 'episode_index'), ('frame_indices', 'frame_index'),
                               ('task_indices', 'task_index')]:
                if selection[key] != [int(x) for x in frames[field]]:
                    raise ValueError('Observed held-out identity changed')
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selection['heldout_row_indices']),
                batch_size=8, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(73000))
            pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint,
                preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                        'rename_observations_processor': {'rename_map': {}}})
            torch.cuda.reset_peak_memory_stats()
            rows = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for index, raw in enumerate(loader):
                    batch = _preprocess_dataset_batch(raw, metadata.camera_keys, rename, pre)
                    torch.manual_seed(registration['validation_seed'] + index)
                    loss, metrics = policy(batch)
                    torch.manual_seed(registration['action_validation_seed'] + index)
                    pred = post(policy.predict_action_chunk(batch).float()).to('cuda')
                    target = post(batch['action'].clone()).to('cuda')
                    valid = ~batch['action_is_pad'].bool()
                    if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                        raise ValueError('Nonfinite passive prediction')
                    physical = pred[valid].detach().cpu()
                    row = {'batch': index, 'samples': int(batch['action'].shape[0]),
                           'valid_actions': int(valid.sum()), 'loss': float(loss),
                           **{key: float(value) for key, value in metrics.items()},
                           'arm_squared_error_sum': float(((pred[valid][..., :6] - target[valid][..., :6]) ** 2).sum()),
                           'gripper_errors': int((pred[valid][..., 6] != target[valid][..., 6]).sum())}
                    if mode == MODES[0]:
                        reference_predictions.append(physical)
                    else:
                        reference = reference_predictions[index]
                        row.update(arm_cast_difference_squared_sum=float(((physical[..., :6] - reference[..., :6]) ** 2).sum()),
                                   gripper_cast_disagreements=int((physical[..., 6] != reference[..., 6]).sum()))
                    rows.append(row)
                    print(mode, index + 1, '/25', flush=True)
            output['cases'].append({'mode': mode, **proof, 'batches': rows, 'summary': summarize(rows),
                                    'validation_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
            write_json(args.output, output)
            del policy, dataset, frames, loader, pre, post, batch, raw, pred, target, loss, metrics
            gc.collect()
            torch.cuda.empty_cache()
        native, deployed = [case['summary'] for case in output['cases']]
        rows = output['cases'][1]['batches']
        output['comparison'] = {
            'bf16_minus_native_arm_mse': deployed['arm_mse'] - native['arm_mse'],
            'bf16_relative_arm_mse_change': deployed['arm_mse'] / native['arm_mse'] - 1 if native['arm_mse'] else None,
            'bf16_minus_native_gripper_error': deployed['gripper_error'] - native['gripper_error'],
            'prediction_cast_arm_rmse': math.sqrt(sum(r['arm_cast_difference_squared_sum'] for r in rows) / (6 * 1303)),
            'prediction_cast_gripper_disagreement': sum(r['gripper_cast_disagreements'] for r in rows) / 1303}
        output['comparison']['registered_follow_up_trigger'] = (
            (output['comparison']['bf16_relative_arm_mse_change'] or 0) >= registration['follow_up_trigger']['relative_arm_mse_degradation']
            or output['comparison']['bf16_minus_native_gripper_error'] >= registration['follow_up_trigger']['gripper_error_increase'])
        output['status'] = 'completed'
        print(output['comparison'], flush=True)
    except BaseException as exc:
        output.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output, output)


if __name__ == '__main__':
    main()
