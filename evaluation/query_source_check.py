"""Bounded native query-source check; forward/backward only, no optimizer update."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_model

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=[1, 8], default=1)
    parser.add_argument('--reserve-adam-moments', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier source checks')
    source = args.architecture_source.resolve()
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != read_json(source / 'amendment.json')['source_manifest']:
        raise ValueError('Source snapshot changed after preparation')
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
    cfg.vlm_model_name = str(artifact('smolvlm'))
    cfg.jepa_encoder_name = str(artifact('world_model'))
    cfg.init_from_vla_jepa = None
    cfg.query_token_adaptation = 'input_residual'
    cfg.smol_gradient_checkpointing = True
    cfg.device = 'cuda'
    policy = plugin.VLAJEPASmolVLMPolicy(cfg).to('cuda')
    policy.model.video_encoder.requires_grad_(False)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    missing, unexpected = load_model(policy, args.checkpoint / 'model.safetensors', strict=False, device='cpu')
    query_key = 'model.qwen.query_adapter.delta'
    if set(missing) != {query_key} or unexpected:
        raise ValueError('Only the declared new query residual may lack warm-start weights')
    state = policy.state_dict()
    with safe_open(args.checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
        for key in saved.keys():
            if not torch.equal(state[key].detach().cpu(), saved.get_tensor(key)):
                raise ValueError(f'Warm-start precision or tensor changed: {key}')
    if torch.count_nonzero(state[query_key]):
        raise ValueError('New queries must initialize to zero')
    config_folder = args.output.parent / (args.output.stem + '-config')
    cfg.save_pretrained(config_folder)
    restored = PreTrainedConfig.from_pretrained(config_folder)
    if restored.query_token_adaptation != cfg.query_token_adaptation or not restored.smol_gradient_checkpointing:
        raise ValueError('Native config serialization dropped query flags')
    query = policy.model.qwen.query_adapter
    ids_before = query.token_ids.clone()
    hooks_before = len(policy.model.qwen.model.get_input_embeddings()._forward_hooks)
    policy.model.qwen.expand_tokenizer()
    if not torch.equal(ids_before, query.token_ids) or hooks_before != len(policy.model.qwen.model.get_input_embeddings()._forward_hooks):
        raise ValueError('Repeated tokenizer expansion changed query IDs or duplicated hooks')
    buffers = []
    if args.reserve_adam_moments:
        buffers = [torch.zeros_like(p) for p in policy.parameters() if p.requires_grad for _ in range(2)]
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    if 53 not in read_json(ROOT / 'studies/evaluation/training_split.json')['train_episodes']:
        raise ValueError('This bounded check uses only registered training data')
    dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root, episodes=[53],
        delta_timestamps=resolve_delta_timestamps(cfg, metadata, rename), revision='v3.0',
        video_backend='pyav', return_uint8=True)
    frame_indices = [int(x) for x in dataset.hf_dataset['frame_index']]
    frames = list(range(40, 40 + args.batch_size))
    samples = [dataset[frame_indices.index(frame)] for frame in frames]
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': {}}})
    batch = _preprocess_dataset_batch(torch.utils.data.default_collate(samples), metadata.camera_keys, rename, pre)
    policy.train()
    torch.manual_seed(92000)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = policy.model(**policy._prepare_model_inputs(batch, training=True))
        total = losses['action_loss'] + losses['wm_loss']
    if args.batch_size == 1:
        reference = read_json(ROOT / 'studies/recovery/diagnostics/rotary_buffer_precision.json')
        if file_hash(args.checkpoint / 'model.safetensors') != reference['checkpoint_sha256']:
            raise ValueError('Batch-one reference uses the verified 5k checkpoint')
        for output_key in ('action_loss', 'wm_loss'):
            expected = reference['native_losses']['native'][output_key]
            if abs(float(losses[output_key].detach()) - expected) > 1e-6:
                raise ValueError(f'Native loss mismatch: {output_key}={float(losses[output_key].detach())}, '
                                 f'reference={expected}')
    total.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite checkpointed native gradients')
    norm = float(torch.linalg.vector_norm(query.delta.grad))
    if norm <= 0 or torch.count_nonzero(query.delta):
        raise ValueError('Query gradients must connect without changing weights')
    write_json(args.output, {'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
        'source_manifest': manifest, 'diagnostic_source_sha256': file_hash(__file__),
        'episode': 53, 'frames': frames, 'seed': 92000, 'batch_size': args.batch_size,
        'exact_warm_tensor_restoration': True, 'missing_initial_tensor': query_key,
        'query_parameters': query.delta.numel(), 'query_token_ids': query.token_ids.tolist(),
        'config_round_trip': True, 'repeated_expansion_hook_count_stable': True,
        'buffer_precision': 'native_fp32_rotary_frequencies',
        'nonreentrant_checkpointing': policy.model.qwen.model.is_gradient_checkpointing,
        'native_action_loss': float(losses['action_loss'].detach()),
        'native_weighted_world_loss': float(losses['wm_loss'].detach()),
        'query_joint_gradient_l2': norm, 'finite_gradient_tensors': len(gradients),
        'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
        'reserved_adam_moment_bytes': sum(b.numel() * b.element_size() for b in buffers),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'limitations': 'Native forward/backward only, no optimizer update or policy export. '
        'Optional synthetic AdamW moments exclude optimizer-step workspace; this is not a completed training preflight or task-performance result.'})
    print('Native query source passed; no optimizer update.')


if __name__ == '__main__':
    main()
