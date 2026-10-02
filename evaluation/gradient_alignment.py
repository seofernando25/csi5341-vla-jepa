"""One native backward diagnostic; no optimizer, training run or saved weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from evaluation.common import ROOT, file_hash, read_json, write_json


def alignment(left, right):
    paired = [(a.float(), b.float()) for a, b in zip(left, right) if a is not None and b is not None]
    left_norm = sum(float(g.float().square().sum()) for g in left if g is not None) ** .5
    right_norm = sum(float(g.float().square().sum()) for g in right if g is not None) ** .5
    dot = sum(float((a * b).sum()) for a, b in paired)
    return {'action_gradient_l2': left_norm, 'weighted_world_gradient_l2': right_norm,
            'cosine': dot / (left_norm * right_norm) if left_norm and right_norm else None,
            'action_parameters_with_grad': sum(g is not None for g in left),
            'world_parameters_with_grad': sum(g is not None for g in right),
            'finite': all(bool(torch.isfinite(g).all()) for g in (*left, *right) if g is not None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostic results')
    if args.episode not in read_json(ROOT / 'studies/evaluation/training_split.json')['train_episodes']:
        parser.error('Gradient diagnostics use training episodes only')
    if args.frame < 0:
        parser.error('Frame index must be nonnegative')
    source = args.architecture_source.resolve()
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Wrong architecture snapshot')
    from evaluation.models import load_policy
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from safetensors.torch import load_model

    policy, _ = load_policy('S500', args.checkpoint)
    policy.requires_grad_(True)
    policy.model.qwen._configure_trainability()
    policy.model.video_encoder.requires_grad_(False)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    # Restore saved FP32 masters after the ordinary BF16 inference loader.
    load_model(policy, args.checkpoint / 'model.safetensors', strict=True, device='cpu')
    policy.train()
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root, episodes=[args.episode],
                            delta_timestamps=resolve_delta_timestamps(policy.config, metadata, rename),
                            revision='v3.0', video_backend='pyav', return_uint8=True)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': {}}})
    frame_indices = [int(value) for value in dataset.hf_dataset['frame_index']]
    if args.frame not in frame_indices:
        parser.error('Requested frame is absent from the selected training episode')
    sample = dataset[frame_indices.index(args.frame)]
    batch = _preprocess_dataset_batch(torch.utils.data.default_collate([sample]),
                                      metadata.camera_keys, rename, pre)
    groups = {'hidden_adapter': list(policy.model.qwen.hidden_adapter.parameters()),
              'decoder_adapter': list(policy.model.qwen.decoder_adapter.parameters())}
    layers = policy.model.qwen.model.model.text_model.layers
    for index in range(len(layers) - policy.config.unfreeze_last_n, len(layers)):
        groups[f'decoder_layer_{index}'] = list(layers[index].parameters())
    parameters = [p for values in groups.values() for p in values if p.requires_grad]
    # Public policy metrics are detached floats. Request the same native
    # objectives directly, from one shared forward graph.
    torch.manual_seed(92000)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = policy._prepare_model_inputs(batch, training=True)
        losses = policy.model(**inputs)
        action, world = losses['action_loss'], losses['wm_loss']
    left = torch.autograd.grad(action, parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(world, parameters, allow_unused=True)
    results, offset = {}, 0
    for name, values in groups.items():
        size = sum(p.requires_grad for p in values)
        results[name] = alignment(left[offset:offset + size], right[offset:offset + size])
        offset += size
    result = {'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
              'source_manifest': {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))},
              'episode': int(sample['episode_index']), 'frame': int(sample['frame_index']), 'seed': 92000,
              'task_id': int(sample['task_index']),
              'native_action_loss': float(action.detach()), 'native_weighted_world_loss': float(world.detach()),
              'world_loss_weight': policy.config.world_model_loss_weight, 'groups': results,
              'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': 'One training frame, one stochastic draw, batch one. Gradient alignment is local and does not prove objective conflict causes control failure. No optimizer update or weight export.'}
    write_json(args.output, result)
    print(result)
    if not all(r['finite'] for r in results.values()):
        raise RuntimeError('Nonfinite diagnostic gradients')


if __name__ == '__main__':
    main()
