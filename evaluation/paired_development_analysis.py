"""Paired diagnostic outcomes on identical development tasks, states and seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.common import file_hash, read_json, write_json


def analyze(records, episodes):
    if len(records) != 2 or len(episodes) != 2:
        raise ValueError('Require exactly two development records')
    identity = ('task_id', 'task_name', 'trial', 'initial_state_hash', 'seed')
    indexed = []
    for record, rows in zip(records, episodes):
        if (record.get('status') != 'completed' or record.get('phase') != 'development'
                or record.get('experiment') != 'rollout' or len(rows) != 10):
            raise ValueError('Require completed ten-episode development rollouts')
        if record.get('model', {}).get('buffer_precision') != 'native_rope':
            raise ValueError('Require observed native rotary-buffer inference')
        if {key: record['model'].get(key) for key in
                ('num_inference_timesteps', 'n_action_steps', 'chunk_size', 'world_model_loaded')} != {
                'num_inference_timesteps': 4, 'n_action_steps': 7, 'chunk_size': 7, 'world_model_loaded': True}:
            raise ValueError('Require the fixed solver, action horizon and loaded world model')
        if not record.get('checkpoint_sha256'):
            raise ValueError('Checkpoint identity is missing')
        tasks = [int(row['task_id']) for row in rows]
        if sorted(tasks) != list(range(10)):
            raise ValueError('Each task must occur exactly once')
        for row in rows:
            if (row.get('status') != 'completed' or row.get('phase') != 'development'
                    or row['variant'] != record['variant'] or int(row['trial']) != 0
                    or int(row['success']) not in (0, 1) or int(row['steps']) < 1):
                raise ValueError('Invalid development episode')
        summary = record['summary']
        if (summary.get('phase') != 'development' or summary['episodes'] != 10 or sorted(summary['task_ids']) != list(range(10))
                or summary['successes'] != sum(int(row['success']) for row in rows)):
            raise ValueError('Episode outcomes disagree with the recorded summary')
        indexed.append({int(row['task_id']): row for row in rows})
    left, right = records
    for key in ('protocol', 'initial_states_manifest_sha256'):
        if not left.get(key) or left.get(key) != right.get(key):
            raise ValueError('Recorded protocol or initial-state manifest differs')
    for key in ('gpu', 'packages', 'torch_cuda', 'driver'):
        if not left['environment'].get(key) or left['environment'][key] != right['environment'].get(key):
            raise ValueError('Recorded GPU/software environment differs')
    for generic, implementation in [('loader_sha256', 'loader_implementation_sha256'),
                                    ('evaluator_sha256', 'evaluator_implementation_sha256')]:
        a, b = [record.get(generic, record.get(implementation)) for record in records]
        if not a or a != b:
            raise ValueError('Recorded loader/evaluator differs')
    rows = []
    counts = dict(both_success=0, gained_success=0, lost_success=0, both_failure=0)
    for task in range(10):
        a, b = indexed[0][task], indexed[1][task]
        if any(a[key] != b[key] for key in identity):
            raise ValueError('Task, state or seed membership differs')
        first, second = int(a['success']), int(b['success'])
        outcome = ('both_success' if first and second else 'lost_success' if first else
                   'gained_success' if second else 'both_failure')
        counts[outcome] += 1
        rows.append({**{key: a[key] for key in identity}, 'left_success': first,
                     'right_success': second, 'left_steps': int(a['steps']),
                     'right_steps': int(b['steps']), 'transition': outcome})
    return {'episodes': rows, 'transitions': counts,
            'left_successes': sum(row['left_success'] for row in rows),
            'right_successes': sum(row['right_success'] for row in rows)}


def figure(result, labels, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(7.1, 2.2))
    for y, key in enumerate(('left_success', 'right_success')):
        for row in result['episodes']:
            x, success = int(row['task_id']), row[key]
            ax.scatter(x, y, s=270, marker='s', color='#1f5f94' if success else '#e4e9ee',
                       edgecolors='none', zorder=2)
            ax.text(x, y, 'S' if success else 'F', ha='center', va='center',
                    color='white' if success else '#596776', fontsize=9, zorder=3)
    ax.set(xticks=range(10), xlabel='LIBERO task ID', yticks=[0, 1],
           yticklabels=[f'{labels[0]} ({result["left_successes"]}/10)',
                        f'{labels[1]} ({result["right_successes"]}/10)'],
           xlim=(-.55, 9.55), ylim=(1.6, -.6))
    ax.tick_params(length=0, pad=7)
    ax.spines[:].set_visible(False)
    fig.text(.22, .025, 'S: success · F: failure · same first development state and seed per task · diagnostic trials',
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .07, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        path = output.with_suffix('.' + suffix)
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--labels', nargs=2)
    args = parser.parse_args()
    records = [read_json(path / 'run.json') for path in args.runs]
    episodes = [list(csv.DictReader((path / 'episodes.csv').open())) for path in args.runs]
    result = analyze(records, episodes)
    result.update(purpose='paired_saved_development_outcomes',
                  input_records=[{'run_id': record['run_id'], 'variant': record['variant'],
                                  'status': record['status'], 'phase': record['phase'],
                                  'experiment': record['experiment'], 'summary': record['summary'],
                                  'checkpoint_sha256': record['checkpoint_sha256'],
                                  'source_manifest': record.get('source_manifest'),
                                  'loader_sha256': record.get('loader_sha256', record.get('loader_implementation_sha256')),
                                  'evaluator_sha256': record.get('evaluator_sha256', record.get('evaluator_implementation_sha256')),
                                  'model': {key: record['model'][key] for key in
                                            ('buffer_precision', 'num_inference_timesteps', 'n_action_steps',
                                             'chunk_size', 'world_model_loaded')},
                                  'run_record_sha256': file_hash(path / 'run.json'),
                                  'episodes_csv_sha256': file_hash(path / 'episodes.csv')}
                                 for record, path in zip(records, args.runs)],
                  environment={key: records[0]['environment'][key]
                               for key in ('gpu', 'packages', 'torch_cuda', 'driver')},
                  buffer_precision='native_rope', analysis_sha256=file_hash(__file__),
                  protocol=records[0]['protocol'],
                  initial_states_manifest_sha256=records[0]['initial_states_manifest_sha256'],
                  limitations='Ten repeated development states, one per task, not representative final trials. '
                  'Source manifests are included only when independently recorded by the rollout; '
                  'null does not imply verified plugin-source identity. Recorded environment equality '
                  'does not establish identical unrecorded numerical state. No significance, '
                  'isolated architecture causality, timing comparison or checkpoint selection.')
    write_json(args.output, result)
    if args.figure:
        figure(result, args.labels or [record['variant'] for record in records], args.figure)
    print(result['transitions'])


if __name__ == '__main__':
    main()
