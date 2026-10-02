"""Separate native action error into bias and inference-noise variance.

Five draws on recorded demonstration frames; no training, changed solver,
deployed ensemble, robot rollout or timing measurement.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.models import artifact, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['S500', 'B16'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--architecture-source', type=Path)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--buffer-precision', choices=['legacy', 'native_rope'], default='legacy')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier measurements')
    if args.variant != 'S500' and args.buffer_precision != 'legacy':
        parser.error('Qwen retains the pinned baseline inference protocol')
    if args.variant == 'S500':
        if not args.architecture_source or not args.checkpoint:
            parser.error('S500 requires checkpoint and source snapshot')
        source = args.architecture_source.resolve()
        sys.path.insert(0, str(source / 'src'))
        import lerobot_policy_vla_jepa_smolvlm as plugin
        if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
            raise ValueError('Wrong source snapshot')
    elif args.checkpoint or args.architecture_source:
        parser.error('B16 uses the pinned baseline without overrides')
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    checkpoint = args.checkpoint or artifact('baseline')
    policy, _ = load_policy(args.variant, args.checkpoint, buffer_precision=args.buffer_precision)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': {}}})
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    manifest = read_json(ROOT / 'studies/evaluation/training_split.json')
    selected = list(csv.DictReader(args.cohort.open()))
    datasets = {split: LeRobotDataset('local/libero_spatial', root=args.dataset_root,
        episodes=manifest[key], delta_timestamps=resolve_delta_timestamps(policy.config, metadata, rename),
        revision='v3.0', video_backend='pyav', return_uint8=True)
        for split, key in [('train', 'train_episodes'), ('heldout', 'validation_episodes')]}
    records = []
    for index, identity in enumerate(selected):
        sample = datasets[identity['split']][int(identity['row'])]
        if (int(sample['episode_index']), int(sample['frame_index']), int(sample['task_index'])) != (
                int(identity['episode']), int(identity['frame']), int(identity['task_id'])):
            raise ValueError('Dataset frame membership changed')
        batch = torch.utils.data.default_collate([sample])
        raw = batch['action'].to('cuda')
        target = raw.clone()
        target[..., 6] = 1 - 2 * raw[..., 6]
        batch = _preprocess_dataset_batch(batch, metadata.camera_keys, rename, pre)
        mask = ~batch['action_is_pad'].bool()
        if int(mask.sum()) != int(identity['valid_actions']):
            raise ValueError('Action padding membership changed')
        draws, seeds = [], [82000 + index + 100000 * draw for draw in range(5)]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for seed in seeds:
                torch.manual_seed(seed)
                policy.reset()
                physical = post(policy.predict_action_chunk(batch).float()).to(mask.device)
                draws.append(physical[mask].float().cpu().numpy())
        draws, target = np.stack(draws), target[mask].cpu().numpy()
        mean = draws[..., :6].mean(0)
        single = float(np.mean((draws[..., :6] - target[None, :, :6]) ** 2))
        bias = float(np.mean((mean - target[:, :6]) ** 2))
        variance = float(np.mean((draws[..., :6] - mean[None]) ** 2))
        if not np.isfinite(draws).all() or not np.isclose(single, bias + variance, rtol=1e-5, atol=1e-8):
            raise ValueError('Nonfinite prediction or failed error decomposition')
        grip = draws[..., 6] > 0
        target_grip = target[:, 6] > 0
        records.append({**{k: int(identity[k]) for k in ('task_id', 'row', 'episode', 'frame', 'valid_actions')},
            'split': identity['split'], 'seeds': seeds,
            'single_draw_arm_mse_mean': single, 'mean_prediction_arm_mse': bias,
            'within_draw_arm_variance': variance,
            'gripper_error_mean': float(np.mean(grip != target_grip[None])),
            'gripper_majority_error': float(np.mean((grip.sum(0) >= 3) != target_grip)),
            'gripper_any_disagreement_fraction': float(np.mean(np.any(grip != grip[0], axis=0)))})
    summary = {}
    metrics = [k for k in records[0] if k.endswith(('_mean', '_mse', '_variance', '_error', '_fraction'))]
    for split in datasets:
        rows = [r for r in records if r['split'] == split]
        summary[split] = {k: float(np.mean([r[k] for r in rows])) for k in metrics}
        summary[split]['frames'] = len(rows)
    write_json(args.output, {'variant': args.variant, 'checkpoint_sha256': file_hash(checkpoint / 'model.safetensors'),
        'environment_observed_at_completion': {
            'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark},
        'cohort_sha256': file_hash(args.cohort), 'diagnostic_source_sha256': file_hash(__file__),
        'buffer_precision': args.buffer_precision, 'loader_sha256': file_hash(ROOT / 'evaluation/models.py'),
        'source_manifest': {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
        if args.variant == 'S500' else None,
        'native_inference_timesteps': policy.config.num_inference_timesteps,
        'draws_per_frame': 5, 'frames': records, 'summary': summary,
        'limitations': 'Finite-draw bias/variance decomposition, equally weighted frames. '
        'Mean predictions and majority grippers are passive diagnostics, not a deployed ensemble or success estimate. '
        'Qwen training membership is unknown.'})
    print(summary)


if __name__ == '__main__':
    main()
