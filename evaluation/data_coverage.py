"""CPU audit of frozen demonstration membership; never load a policy or alter data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from evaluation.common import ROOT, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier coverage evidence')
    split_path = ROOT / 'studies/evaluation/training_split.json'
    sample_path = ROOT / 'studies/evaluation/validation_samples.json'
    mapping_path = ROOT / 'studies/recovery/diagnostics/task_instruction_mapping.json'
    states_path = ROOT / 'studies/evaluation/initial_states.json'
    split, sample, mapping, states = [read_json(p) for p in (split_path, sample_path, mapping_path, states_path)]
    for name, expected in split['files'].items():
        if file_hash(args.dataset_root / name) != expected:
            raise ValueError('Dataset bytes differ from the frozen split')
    episodes = pd.concat([pd.read_parquet(args.dataset_root / name) for name in split['files']
                          if name.startswith('meta/episodes/')], ignore_index=True)
    frames = pd.concat([pd.read_parquet(args.dataset_root / name,
                        columns=['index', 'episode_index', 'frame_index', 'task_index'])
                        for name in split['files'] if name.startswith('data/')], ignore_index=True)
    frames = frames.sort_values('index').reset_index(drop=True)
    tasks = pd.read_parquet(args.dataset_root / 'meta/tasks.parquet')
    train, held = [set(split[k]) for k in ('train_episodes', 'validation_episodes')]
    if (len(train) != len(split['train_episodes']) or len(held) != len(split['validation_episodes'])
            or train & held or train | held != set(episodes['episode_index'])
            or not episodes['episode_index'].is_unique or frames['index'].tolist() != list(range(len(frames)))):
        raise ValueError('Episode partition or frame coverage is incomplete')
    for _, episode in episodes.iterrows():
        rows = frames[frames['episode_index'] == episode['episode_index']]
        if (len(rows) != int(episode['length'])
                or rows['frame_index'].tolist() != list(range(int(episode['length'])))
                or rows['index'].tolist() != list(range(int(episode['dataset_from_index']), int(episode['dataset_to_index'])))
                or rows['task_index'].nunique() != 1
                or int(rows['task_index'].iloc[0]) != int(episode['stats/task_index/min'].item())
                or int(episode['stats/task_index/min'].item()) != int(episode['stats/task_index/max'].item())
                or tasks.loc[episode['tasks'][0], 'task_index'] != int(rows['task_index'].iloc[0])):
            raise ValueError('Episode lengths, task instructions or frame ranges differ')
    train_frames, held_frames = [frames[frames['episode_index'].isin(ids)] for ids in (train, held)]
    if len(train_frames) != split['train_frames'] or len(held_frames) != split['validation_frames']:
        raise ValueError('Frozen split frame totals differ')
    if (not mapping['instruction_sets_exactly_equal'] or len(mapping['mapping']) != 10
            or sorted(m['dataset_task_id'] for m in mapping['mapping']) != list(range(10))
            or sorted(m['libero_task_id'] for m in mapping['mapping']) != list(range(10))):
        raise ValueError('Require the verified instruction bijection')
    picked = held_frames.iloc[sample['heldout_row_indices']]
    if (len(picked) != 200 or len(set(sample['heldout_row_indices'])) != 200
            or picked['episode_index'].tolist() != sample['episode_indices']
            or picked['frame_index'].tolist() != sample['frame_indices']
            or picked['task_index'].tolist() != sample['task_indices']):
        raise ValueError('Actual held-out row membership differs')
    coverage = []
    for link in sorted(mapping['mapping'], key=lambda m: m['libero_task_id']):
        data_id, simulator_id = link['dataset_task_id'], link['libero_task_id']
        if (tasks.loc[link['instruction'], 'task_index'] != data_id
                or states['tasks'][simulator_id]['instruction'] != link['instruction']):
            raise ValueError('Canonical task instruction mapping differs')
        group = episodes[episodes['stats/task_index/min'].map(lambda value: int(value.item())) == data_id]
        task_train, task_held = [group[group['episode_index'].isin(ids)] for ids in (train, held)]
        expected_held = group['episode_index'].tolist()[-math.ceil(.1 * len(group)):]
        if task_held['episode_index'].tolist() != expected_held or (picked['task_index'] == data_id).sum() != 20:
            raise ValueError('Per-task split rule or fixed sample coverage differs')
        coverage.append({**link, 'train_episodes': len(task_train), 'heldout_episodes': len(task_held),
                         'train_frames': int(task_train['length'].sum()),
                         'heldout_frames': int(task_held['length'].sum()), 'selected_heldout_frames': 20,
                         'training_frame_fraction': int(task_train['length'].sum()) / len(train_frames)})
    recipe_path = ROOT / 'evaluation/recovery_config.json'
    completion_path = ROOT / 'studies/recovery/diagnostics/completed_r1.json'
    recipe, completion = read_json(recipe_path), read_json(completion_path)
    if (completion['status'] != 'completed_native_export_verified'
            or completion['recipe_sha256'] != file_hash(recipe_path)
            or max(row['stage'] for row in completion['milestones']) != 20000 or recipe['batch_size'] != 8):
        raise ValueError('Completed corrected recovery recipe differs')
    prefix = next(name.removesuffix('training_state/training_step.json') for name in completion['export_manifest']
                  if name.endswith('/020000/training_state/training_step.json'))
    native_paths = [prefix + 'training_state/' + name for name in ('training_step.json', 'optimizer_state.safetensors')]
    for name in native_paths:
        proof, path = completion['export_manifest'][name], ROOT / name
        if path.stat().st_size != proof['bytes'] or file_hash(path) != proof['sha256']:
            raise ValueError('Actual completed native state differs from export proof')
    topology = read_json(ROOT / native_paths[0])
    if any(topology[key] != value for key, value in {'step': 20000, 'batch_size': 8,
                                                   'grad_accum_steps': 1, 'dp_world_size': 1}.items()):
        raise ValueError('Completed native exposure topology differs')
    from safetensors import safe_open
    with safe_open(ROOT / native_paths[1], framework='numpy') as saved:
        counters = [saved.get_tensor(key).item() for key in saved.keys() if key.endswith('/step')]
    if not counters or any(value != 20000 for value in counters):
        raise ValueError('Observed optimizer counters do not establish 20k completed updates')
    write_json(args.output, {'status': 'verified', 'purpose': 'frozen_demonstration_coverage_audit',
        'implementation_sha256': file_hash(__file__), 'split_sha256': file_hash(split_path),
        'validation_samples_sha256': file_hash(sample_path), 'task_mapping_sha256': file_hash(mapping_path),
        'initial_states_manifest_sha256': file_hash(states_path), 'dataset_files': split['files'],
        'train_episodes': len(train), 'heldout_episodes': len(held), 'train_frames': len(train_frames),
        'heldout_frames': len(held_frames), 'tasks': coverage,
        'corrected_r1_exposure': {'recipe_sha256': file_hash(recipe_path),
            'completion_sha256': file_hash(completion_path), 'optimizer_updates': 20000, 'batch_size': 8,
            'populated_native_optimizer_states': len(counters),
            'native_exposure_proofs': {Path(name).name: completion['export_manifest'][name] for name in native_paths},
            'anchor_frame_presentations': 160000, 'train_frame_count_equivalent': 160000 / len(train_frames)},
        'limitations': 'Every registered task is represented; this does not establish adequate coverage or convergence. Sample presentations are repeated anchor frames, not independent demonstrations, compute or robot success. Legacy incorrect-input training is not pooled with corrected exposure. Dataset IDs differ from simulator IDs and are linked by exact instruction. No prediction, policy, split or sampler change occurred.'})
    print('Verified ten-task coverage:', len(train), 'training episodes,', len(train_frames), 'training frames.')


if __name__ == '__main__':
    main()
