"""Forward-only registered comparison before and after initial query learning."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--query500', type=Path)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration', type=Path,
                        default=ROOT / 'studies/recovery/query_initial_validation_registration.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier measurements')
    registration_path = args.registration
    registration = read_json(registration_path)
    source = args.architecture_source.resolve()
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest']:
        raise ValueError('Use only the registered query source')
    checkpoints = [(args.parent, 'parent_checkpoint_sha256')]
    if 'query500_native_all32' in registration['modes']:
        if args.query500 is None:
            parser.error('Registered query500 comparison requires its checkpoint')
        checkpoints.append((args.query500, 'query500_checkpoint_sha256'))
    for checkpoint, key in checkpoints:
        if file_hash(checkpoint / 'model.safetensors') != registration[key]:
            raise ValueError('Checkpoint differs from registration')
    if registration.get('diagnostic_sha256') and file_hash(__file__) != registration['diagnostic_sha256']:
        raise ValueError('Registered diagnostic changed')
    for name, expected in registration.get('parent_files', {}).items():
        if file_hash(args.parent / name) != expected:
            raise ValueError('Parent config or processors differ')
    for name, key in [('validation_samples.json', 'validation_samples_sha256'),
                      ('training_split.json', 'split_sha256')]:
        if file_hash(ROOT / 'studies/evaluation' / name) != registration[key]:
            raise ValueError('Validation membership differs from registration')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    matmul_tf32 = registration.get('matmul_allow_tf32', False)
    if not isinstance(matmul_tf32, bool):
        raise ValueError('Register TF32 as an explicit Boolean')
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    if torch.cuda.get_device_name() != registration['gpu']:
        raise ValueError('Use the registered local GPU')
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
    split = read_json(ROOT / 'studies/evaluation/training_split.json')
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    output = {'purpose': registration['purpose'], 'registration_sha256': file_hash(registration_path),
              'diagnostic_sha256': file_hash(__file__), 'source_manifest': manifest,
              'environment': environment(), 'status': 'running', 'cases': [],
              'numerical_flags': {
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                  'cudnn_deterministic': torch.backends.cudnn.deterministic,
              },
              'limitations': registration['limitations']}
    write_json(args.output, output)
    try:
        for mode in registration['modes']:
            query = mode != 'parent_native_last4'
            checkpoint = args.query500 if mode == 'query500_native_all32' else args.parent
            cfg = PreTrainedConfig.from_pretrained(checkpoint)
            cfg.vlm_model_name, cfg.jepa_encoder_name = str(artifact('smolvlm')), str(artifact('world_model'))
            cfg.init_from_vla_jepa, cfg.device = None, 'cuda'
            cfg.query_token_adaptation = 'input_residual' if query else 'none'
            cfg.smol_gradient_checkpointing = query
            cfg.unfreeze_last_n = 32 if query else 4
            policy = plugin.VLAJEPASmolVLMPolicy(cfg).to('cuda')
            policy.model.video_encoder.requires_grad_(False)
            for p in policy.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            missing, unexpected = load_model(policy, checkpoint / 'model.safetensors', strict=False, device='cpu')
            zero_query = mode == 'parent_zero_query_native_all32'
            if set(missing) != ({'model.qwen.query_adapter.delta'} if zero_query else set()) or unexpected:
                raise ValueError('Unexpected missing or extra weights')
            if zero_query and torch.count_nonzero(policy.model.qwen.query_adapter.delta):
                raise ValueError('Untrained query residual is not zero')
            with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
                for key, value in policy.state_dict().items():
                    if key in saved.keys() and not torch.equal(value.detach().cpu(), saved.get_tensor(key)):
                        raise ValueError('Inherited weight values or precision changed')
            if any(value.dtype != torch.float32 for name, value in policy.named_buffers()
                   if name.endswith(('rotary_emb.inv_freq', 'rotary_emb.original_inv_freq'))):
                raise ValueError('Native rotary buffers were rounded')
            dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root,
                episodes=split['validation_episodes'],
                delta_timestamps=resolve_delta_timestamps(cfg, metadata, rename), revision='v3.0',
                video_backend='pyav', return_uint8=True)
            selected = dataset.hf_dataset.select(selection['heldout_row_indices'])
            for key, field in [('episode_indices', 'episode_index'), ('frame_indices', 'frame_index'),
                               ('task_indices', 'task_index')]:
                if selection[key] != [int(x) for x in selected[field]]:
                    raise ValueError('Observed held-out frame identity changed')
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, selection['heldout_row_indices']), batch_size=8,
                shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(73000))
            pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint,
                preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                        'rename_observations_processor': {'rename_map': {}}})
            policy.eval()
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
                    rows.append({'batch': index, 'samples': int(batch['action'].shape[0]),
                                 'valid_actions': int(valid.sum()), 'loss': float(loss),
                                 **{key: float(value) for key, value in metrics.items()},
                                 'arm_squared_error_sum': float(((pred[valid][..., :6] - target[valid][..., :6]) ** 2).sum()),
                                 'gripper_errors': int((pred[valid][..., 6] != target[valid][..., 6]).sum())})
                    print(mode, index + 1, '/25', flush=True)
            count = sum(row['valid_actions'] for row in rows)
            if len(rows) != 25 or sum(row['samples'] for row in rows) != 200 or count != 1303:
                raise ValueError('Fixed full validation membership/count failed')
            summary = {'frames': 200, 'valid_actions': count,
                       'arm_mse': sum(row['arm_squared_error_sum'] for row in rows) / (6 * count),
                       'gripper_error': sum(row['gripper_errors'] for row in rows) / count,
                       'loss': sum(row['loss'] * row['samples'] for row in rows) / 200}
            output['cases'].append({'mode': mode, 'checkpoint_sha256': file_hash(checkpoint / 'model.safetensors'),
                                    'exact_inherited_values': True, 'zero_queries': zero_query,
                                    'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
                                    'batches': rows, 'summary': summary})
            write_json(args.output, output)
            print(mode, summary, flush=True)
            del policy, pre, post, dataset, loader, selected, batch, raw, pred, target, loss, metrics
            gc.collect()
            torch.cuda.empty_cache()
        output['status'] = 'completed'
    except BaseException as exc:
        output.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output, output)


if __name__ == '__main__':
    main()
