"""Isolate floating-buffer casts with fixed native weights, data and random draws."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_model

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parameter-cast', action='store_true',
                        help='Hold native buffers fixed and isolate BF16 deployment of FP32 trainable masters')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostic results')
    source = args.architecture_source.resolve()
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Wrong source snapshot')
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.vlm_model_name, cfg.jepa_encoder_name = str(artifact('smolvlm')), str(artifact('world_model'))
    cfg.init_from_vla_jepa, cfg.device = None, 'cuda'
    policy = plugin.VLAJEPASmolVLMPolicy(cfg).to('cuda')
    policy.model.video_encoder.requires_grad_(False)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    load_model(policy, args.checkpoint / 'model.safetensors', strict=True, device='cpu')
    buffers = {name: value.detach().clone() for name, value in policy.named_buffers() if value.is_floating_point()}
    masters = {name: p.detach().clone() for name, p in policy.named_parameters()
               if args.parameter_cast and p.requires_grad}
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    if 53 not in read_json(ROOT / 'studies/evaluation/training_split.json')['train_episodes']:
        raise ValueError('Use the registered training frame only')
    dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root, episodes=[53],
        delta_timestamps=resolve_delta_timestamps(cfg, metadata, rename), revision='v3.0',
        video_backend='pyav', return_uint8=True)
    index = [int(x) for x in dataset.hf_dataset['frame_index']].index(40)
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': {}}})
    batch = _preprocess_dataset_batch(torch.utils.data.default_collate([dataset[index]]),
                                      metadata.camera_keys, rename, pre)
    policy.train()
    results = {}
    for mode in ('native', 'bf16_masters' if args.parameter_cast else 'bf16_buffers', 'native_restored'):
        for name, original in buffers.items():
            parent, _, leaf = name.rpartition('.')
            module = policy.get_submodule(parent) if parent else policy
            module._buffers[leaf] = original.to(torch.bfloat16) if mode == 'bf16_buffers' else original.clone()
        for name, p in policy.named_parameters():
            if name in masters:
                p.data = masters[name].to(torch.bfloat16) if mode == 'bf16_masters' else masters[name].clone()
        torch.manual_seed(92000)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = policy.model(**policy._prepare_model_inputs(batch, training=True))
        results[mode] = {k: float(v.detach()) for k, v in losses.items()}
        del losses
    if results['native'] != results['native_restored']:
        raise ValueError('Buffer restoration did not restore the native result')
    write_json(args.output, {'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
        'source_manifest': {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))},
        'diagnostic_source_sha256': file_hash(__file__), 'episode': 53, 'frame': 40, 'seed': 92000,
        'intervention': 'trainable FP32 masters to BF16; native buffers fixed' if args.parameter_cast
                        else 'floating buffers to BF16; parameters fixed',
        'buffers': {name: {'dtype': str(v.dtype), 'shape': list(v.shape),
                    'max_bf16_rounding_error': float((v - v.to(torch.bfloat16).to(v.dtype)).abs().max())}
                    for name, v in buffers.items()}, 'native_losses': results,
        'limitations': 'One native training-frame forward per precision mode, no optimizer update or rollout. '
        'Only the declared precision intervention is applied and restored. No task-success inference.'})
    print(results)


if __name__ == '__main__':
    main()
