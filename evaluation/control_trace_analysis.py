"""Summarize paired development traces without inferring architecture causality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.common import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs=2, required=True)
    parser.add_argument('--labels', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measured, snapshots = [], []
    for folder, label in zip(args.traces, args.labels):
        record = read_json(folder / 'run.json')
        episode = list(csv.DictReader((folder / 'episodes.csv').open()))
        if record['status'] != 'completed' or len(episode) != 1:
            raise ValueError('Use one completed development episode per trace')
        task_folder = folder / f"task-{record['task_id']}"
        commands = task_folder / 'commands.jsonl'
        rows = [json.loads(line) for line in commands.read_text().splitlines()]
        action = np.asarray([r['action'] for r in rows])
        position = np.asarray([rows[0]['state_before']['eef']['pos']] +
                              [r['state_after']['eef']['pos'] for r in rows])
        measured.append({'label': label, 'checkpoint_sha256': record['checkpoint_sha256'],
                         'trace_implementation_sha256': record.get('trace_implementation_sha256'),
                         'source_manifest': record.get('source_manifest'),
                         'commands_sha256': file_hash(commands), 'task_id': record['task_id'],
                         'initial_state_hash': episode[0]['initial_state_hash'], 'seed': int(episode[0]['seed']),
                         'steps': len(rows), 'success': bool(int(episode[0]['success'])),
                         'eef_path_length': float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
                         'eef_net_displacement': float(np.linalg.norm(position[-1] - position[0])),
                         'gripper_command_switches': int(np.count_nonzero(np.diff(action[:, 6]))),
                         'closed_command_fraction': float(np.mean(action[:, 6] > 0)),
                         'arm_command_magnitude_ge_one_fraction': float(np.mean(np.abs(action[:, :6]) >= 1)),
                         'position': position, 'action': action})
        files = sorted(task_folder.glob('frames-*.npz'))
        snapshots.append([files[index] for index in np.linspace(0, len(files) - 1, 5).round().astype(int)])
    if len({(r['initial_state_hash'], r['seed'], r['task_id']) for r in measured}) != 1:
        raise ValueError('Trace comparison requires identical task, state and seed')
    with np.load(snapshots[0][0]) as left, np.load(snapshots[1][0]) as right:
        if not all(np.array_equal(left[k], right[k]) for k in left.files):
            raise ValueError('Initial camera observations differ')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 5, figsize=(9.5, 4.8))
    for i, (result, frames) in enumerate(zip(measured, snapshots)):
        for j, path in enumerate(frames):
            with np.load(path) as image:
                axes[i, j].imshow(np.rot90(image['image'], 2))
            axes[i, j].axis('off')
            axes[i, j].set_title(f"{result['label']} · step {int(path.stem.split('-')[1])}", fontsize=9)
    figure.text(.045, .018, 'Same development starting state · each row spans its own episode · last sparse frame may precede termination',
                fontsize=8, color='#596776')
    figure.tight_layout(rect=[0, .04, 1, 1])
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output / f'F9_control_case.{suffix}', dpi=190)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for result, color in zip(measured, ['#1f5f94', '#bd5e3b']):
        xy = result['position'][:, :2]
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=1.4, label=result['label'])
        axes[0].scatter(*xy[0], color=color, s=18)
        axes[0].scatter(*xy[-1], color=color, marker='x', s=30)
        axes[1].step(np.arange(result['steps']), result['action'][:, 6], where='post',
                     color=color, lw=1, label=result['label'])
    axes[0].set(xlabel='End-effector x (simulator units)', ylabel='End-effector y (simulator units)')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[1].set(xlabel='Control step', ylabel='Gripper command', yticks=[-1, 1])
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='#dce1e7', lw=.6)
    figure.tight_layout()
    for suffix in ('pdf', 'png', 'svg'):
        path = args.output / f'F9_control_commands.{suffix}'
        figure.savefig(path, dpi=190)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(figure)
    for result in measured:
        result.pop('position')
        result.pop('action')
    write_json(args.output / 'summary.json', {'traces': measured, 'identical_initial_cameras': True,
               'limitations': 'One development case; different model/training/input histories. Trajectories diverge, so per-step command differences are not expert imitation errors. No architecture causality or success-rate estimate.'})
    print(json.dumps([{'label': r['label'], 'success': r['success'], 'steps': r['steps']} for r in measured]))


if __name__ == '__main__':
    main()
