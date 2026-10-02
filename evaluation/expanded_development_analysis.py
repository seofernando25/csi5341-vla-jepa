"""Analyze the registered100-state local parent/B16 development comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json


def analyze(records, episodes, registration, states, protocol):
    if len(records) != 2 or len(episodes) != 2:
        raise ValueError('Require B16 and parent records in that order')
    expected = {(task, trial) for task in range(10) for trial in range(10)}
    indexed = []
    state_tasks = {task['task_id']: task for task in states['tasks']}
    for name, record, rows in zip(('B16', 'RGB-parent20k'), records, episodes):
        case = registration['variants'][name]
        if (record.get('status') != 'completed' or record.get('phase') != 'development'
                or record.get('experiment') != 'rollout' or record.get('variant') != name + '-dev100'
                or len(rows) != 100):
            raise ValueError('Require completed registered100-episode development records')
        if (record.get('checkpoint_sha256') != case['checkpoint_sha256']
                or record.get('source_manifest') != case['source_manifest']
                or record.get('protocol') != protocol
                or record.get('initial_states_manifest_sha256') != registration['initial_states_sha256']
                or record.get('loader_sha256') != registration['loader_sha256']
                or record.get('evaluator_sha256') != registration['evaluator_sha256']):
            raise ValueError('Registered model/source/protocol/implementation differs')
        if (record['environment'].get('gpu') != registration['gpu']
                or record['model'].get('buffer_precision') != case['buffer_precision']
                or {key: record['model'].get(key) for key in
                    ('chunk_size', 'n_action_steps', 'num_inference_timesteps', 'world_model_loaded')} != {
                        'chunk_size': 7, 'n_action_steps': 7, 'num_inference_timesteps': 4, 'world_model_loaded': True}):
            raise ValueError('Registered GPU/deployment settings differ')
        mapping = {(int(row['task_id']), int(row['trial'])): row for row in rows}
        if set(mapping) != expected:
            raise ValueError('Task/trial membership is incomplete or duplicated')
        for (task, trial), row in mapping.items():
            target = state_tasks[task]
            if (row.get('status') != 'completed' or row.get('phase') != 'development'
                    or row.get('variant') != record['variant'] or row['task_name'] != target['name']
                    or row['initial_state_hash'] != target['development_hashes'][trial]
                    or int(row['seed']) != protocol['seed'] + task * 1000 + trial
                    or int(row['success']) not in (0, 1) or not 1 <= int(row['steps']) <= 280):
                raise ValueError('Episode differs from the exact registered development membership')
        summary = record['summary']
        successes = sum(int(row['success']) for row in rows)
        if (summary.get('phase') != 'development' or summary.get('episodes') != 100
                or sorted(summary['task_ids']) != list(range(10)) or summary['successes'] != successes
                or not math.isclose(summary['task_macro_success'], successes / 100, abs_tol=1e-12)):
            raise ValueError('Episode outcomes disagree with summary')
        indexed.append(mapping)
    for key in ('packages', 'torch_cuda', 'driver', 'gpu_total_bytes'):
        if (not records[0]['environment'].get(key)
                or records[0]['environment'][key] != records[1]['environment'].get(key)):
            raise ValueError('Recorded hardware/software environment differs')
    for key in ('numerical_flags', 'wrapper_sha256', 'registration_sha256'):
        if not records[0].get(key) or records[0][key] != records[1].get(key):
            raise ValueError('Recorded numerical flags/wrapper/registration differs')
    rows, task_rows = [], []
    counts = dict(both_success=0, parent_only_success=0, baseline_only_success=0, both_failure=0)
    for task, trial in sorted(expected):
        baseline, parent = [mapping[task, trial] for mapping in indexed]
        a, b = int(baseline['success']), int(parent['success'])
        category = ('both_success' if a and b else 'baseline_only_success' if a else
                    'parent_only_success' if b else 'both_failure')
        counts[category] += 1
        rows.append({'task_id': task, 'trial': trial, 'task_name': baseline['task_name'],
                     'initial_state_hash': baseline['initial_state_hash'], 'seed': int(baseline['seed']),
                     'baseline_success': a, 'parent_success': b, 'transition': category})
    for task in range(10):
        subset = [row for row in rows if row['task_id'] == task]
        task_rows.append({'task_id': task, 'task_name': state_tasks[task]['name'], 'episodes': 10,
                          'baseline_successes': sum(row['baseline_success'] for row in subset),
                          'parent_successes': sum(row['parent_success'] for row in subset)})
    return {'episodes_per_model': 100, 'baseline_successes': sum(row['baseline_success'] for row in rows),
            'parent_successes': sum(row['parent_success'] for row in rows), 'paired_transitions': counts,
            'per_task': task_rows, 'episodes': rows}


def figure(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.analysis import wilson
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (overall, tasks) = plt.subplots(1, 2, figsize=(7.1, 3.8), gridspec_kw={'width_ratios': [1, 1.5]})
    colors = ['#526273', '#1f5f94']
    for y, key in enumerate(('baseline_successes', 'parent_successes')):
        value = result[key]
        low, high = [100 * v for v in wilson(value, 100)]
        overall.errorbar(value, y, xerr=[[value - low], [high - value]], fmt='o',
                         color=colors[y], capsize=3, markersize=6)
        overall.annotate(f'{value}/100', (value, y), xytext=(0, 10), textcoords='offset points', ha='center')
    overall.set(xlim=(-4, 104), ylim=(1.6, -.6), yticks=[0, 1],
                yticklabels=['B16', 'Smol parent20k'], xlabel='Success (%)', title='Overall · 95% Wilson intervals')
    for row in result['per_task']:
        task = row['task_id']
        a, b = [10 * row[key] for key in ('baseline_successes', 'parent_successes')]
        tasks.plot([a, b], [task, task], color='#c9d1d9', linewidth=1.1, zorder=1)
        tasks.scatter(a, task, color=colors[0], s=23, zorder=3)
        tasks.scatter(b, task, color=colors[1], s=28, zorder=3)
    tasks.set(xlim=(-4, 104), ylim=(9.6, -.6), yticks=range(10), ylabel='LIBERO task ID',
              xlabel='Success (%)', title='Per task · 10 paired states')
    for ax in (overall, tasks):
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e6ebf0', linewidth=.7)
        ax.set_axisbelow(True)
    tasks.scatter([], [], color=colors[0], label='B16', s=23)
    tasks.scatter([], [], color=colors[1], label='Smol parent20k', s=28)
    handles, labels = tasks.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(.65, .995), ncol=2, fontsize=8)
    fig.text(.03, .025, 'Same RTX3090 and 100 original development states · fixed checkpoints · diagnostic cohort, not final acceptance',
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .07, 1, .91])
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        path = output.with_suffix('.' + suffix)
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=2, required=True, help='B16 then parent')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    registration_path = ROOT / 'studies/recovery/expanded_development_registration.json'
    registration = read_json(registration_path)
    if (file_hash(ROOT / 'evaluation/protocol.json') != registration['protocol_sha256']
            or file_hash(ROOT / 'studies/evaluation/initial_states.json') != registration['initial_states_sha256']):
        raise ValueError('Registered state/protocol files changed')
    records = [read_json(path / 'run.json') for path in args.runs]
    for record in records:
        if (record['registration_sha256'] != file_hash(registration_path)
                or record['wrapper_sha256'] != file_hash(ROOT / 'evaluation/expanded_development.py')):
            raise ValueError('Run used a different registration')
    episodes = []
    for path in args.runs:
        with (path / 'episodes.csv').open() as handle:
            episodes.append(list(csv.DictReader(handle)))
    result = analyze(records, episodes, registration, read_json(ROOT / 'studies/evaluation/initial_states.json'),
                     read_json(ROOT / 'evaluation/protocol.json'))
    result.update(purpose=registration['purpose'], registration_sha256=file_hash(registration_path),
        analysis_sha256=file_hash(__file__),
        input_records=[{**{key: record[key] for key in ('run_id', 'variant', 'status', 'phase', 'checkpoint_sha256',
                         'source_manifest', 'loader_sha256', 'evaluator_sha256', 'wrapper_sha256', 'numerical_flags')},
                        'run_record_sha256': file_hash(path / 'run.json'),
                        'episodes_csv_sha256': file_hash(path / 'episodes.csv')}
                       for record, path in zip(records, args.runs)],
        environment={key: records[0]['environment'][key] for key in ('gpu', 'gpu_total_bytes', 'packages', 'torch_cuda', 'driver')},
        limitations=registration['limits'] + ' ' + registration['scope'] +
            ' This100-state cohort includes the previously inspected first state/task; do not pool it with those ten trials. '
            'Wilson intervals describe trial counts, not uncertainty across new tasks or evidence of final acceptance.')
    write_json(args.output, result)
    if args.figure:
        figure(result, args.figure)
    print({k: result[k] for k in ('baseline_successes', 'parent_successes', 'paired_transitions')})


if __name__ == '__main__':
    main()
