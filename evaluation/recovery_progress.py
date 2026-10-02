"""Measured recovery curves; engineering preflights and legacy runs are excluded."""

from __future__ import annotations

import csv
import argparse
import json
import hashlib
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.common import ROOT, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('r1', 'query'), default='r1')
    args = parser.parse_args()
    query = args.study == 'query'
    root = ROOT / 'studies/recovery'
    recipe_hash = file_hash(ROOT / 'evaluation' / ('query_recovery_config.json' if query else 'recovery_config.json'))
    registration = read_json(root / 'query_recovery_registration.json') if query else None
    if query and registration['recipe_sha256'] != recipe_hash:
        raise ValueError('Query recipe differs from its registration')
    curve_name = 'query_recovery_curve' if query else 'recovery_curve'
    figure_name = 'F13_query_recovery_progress' if query else 'F8_recovery_progress'
    training, validation, action = {}, defaultdict(list), defaultdict(list)
    provenance = []
    for folder in sorted((root / 'training').glob('*')):
        record_path, metrics = folder / 'run.json', folder / 'metrics.jsonl'
        if not record_path.exists() or not metrics.exists():
            continue
        record = read_json(record_path)
        if record.get('purpose') != 'recovery_adaptation' or record.get('recipe_sha256') != recipe_hash:
            continue
        if query and (record.get('study') != registration['study']
                      or record.get('source_manifest') != registration['source_manifest']
                      or record.get('initial_checkpoint_sha256') != registration['initial_checkpoint_sha256']):
            raise ValueError(f'Query provenance differs from registration: {record["run_id"]}')
        metric_bytes = metrics.read_bytes()
        provenance.append({'run_id': record['run_id'], 'metrics_sha256': hashlib.sha256(metric_bytes).hexdigest(),
                           'metrics_snapshot_bytes': len(metric_bytes),
                           'source_manifest': record['source_manifest']})
        # The exporter atomically replaces metadata files. Ignore a final
        # incomplete line if reading a directly written local stream.
        for line in metric_bytes.decode().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase, step = row['phase'], row['step']
            if phase == 'training':
                training[step] = row
            elif phase == 'validation':
                validation[step].append(row)
            elif phase == 'validation_action':
                action[step].append(row)
    rows = []
    for step in sorted(action):
        loss_rows, action_rows = validation[step], action[step]
        # Production uses exactly 200 frames in batches of eight. A live
        # incomplete evaluation must not become a plotted measurement.
        if len(action_rows) != 25 or len(loss_rows) != 25 or sum(r['samples'] for r in loss_rows) != 200:
            continue
        count = sum(r['valid_actions'] for r in action_rows)
        rows.append({'step': step,
                     'native_loss': sum(r['loss'] * r['samples'] for r in loss_rows) / 200,
                     'arm_rmse': np.sqrt(sum(r['arm_squared_error_sum'] for r in action_rows) / (6 * count)),
                     'gripper_error_percent': 100 * sum(r['gripper_errors'] for r in action_rows) / count,
                     'frames': 200, 'valid_actions': count})
    if not rows:
        raise ValueError('No complete registered recovery validation measurements')
    if len({r['valid_actions'] for r in rows}) != 1:
        raise ValueError('Validation action membership changed across checkpoints')
    development = []
    job_path = ROOT / 'outputs/recovery/cloud/remote-job.json'
    job = read_json(job_path) if job_path.exists() else {}
    stages = job.get('stages', []) if job.get('recipe_sha256') == recipe_hash else []
    completion_path = root / 'diagnostics' / (f'{curve_name}.json' if query else 'completed_r1.json')
    if not stages and completion_path.exists():
        completion = read_json(completion_path)
        if completion.get('recipe_sha256') == recipe_hash:
            stages = [{'completed_step': r['stage'], 'selected_step': r['selected_step'],
                       'selected_checkpoint_sha256': r['checkpoint_sha256']}
                      for r in completion['development' if query else 'milestones']]
    for stage in stages:
        for path in sorted((ROOT / 'studies/evaluation/runs').glob('*/run.json')):
            run = read_json(path)
            summary = run.get('summary', {})
            if (run.get('variant') == f"{'Query' if query else 'RGB'}-n0008-{stage['completed_step']}"
                    and run.get('status') == 'completed' and run.get('phase') == 'development'
                    and run.get('experiment') == 'rollout' and summary.get('episodes') == 10
                    and sorted(summary.get('task_ids', [])) == list(range(10))
                    and run.get('checkpoint_sha256') == stage['selected_checkpoint_sha256']):
                if query and run.get('arguments', {}).get('buffer_precision') != 'native_rope':
                    raise ValueError('Query development used a different inference-buffer protocol')
                development.append({'stage': stage['completed_step'], 'selected_step': stage['selected_step'],
                                    'checkpoint_sha256': run['checkpoint_sha256'], 'run_id': run['run_id'],
                                    'episodes_csv_sha256': file_hash(path.parent / 'episodes.csv'),
                                    **({'buffer_precision': 'native_rope',
                                        'run_record_sha256': file_hash(path),
                                        'initial_states_manifest_sha256': run['initial_states_manifest_sha256'],
                                        'loader_sha256': run['loader_implementation_sha256'],
                                        'evaluator_sha256': run['evaluator_implementation_sha256']} if query else {}),
                                    'successes': summary['successes'], 'episodes': summary['episodes']})
                break
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.05), sharex=True)
    steps = [r['step'] / 1000 for r in rows]
    panels = [('native_loss', 'Native validation loss', '#1f5f94'),
              ('arm_rmse', 'Arm-command RMSE', '#bd5e3b'),
              ('gripper_error_percent', 'Gripper errors (%)', '#65528c')]
    for ax, (key, label, color) in zip(axes, panels):
        ax.plot(steps, [r[key] for r in rows], 'o-', ms=4.5, lw=1.5, color=color)
        ax.set_ylabel(label)
        ax.set_xlabel('Query recovery updates (thousands)' if query else 'Additional recovery updates (thousands)')
        ax.grid(axis='y', color='#dce1e7', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#bac4ce')
        ax.tick_params(length=3, color='#bac4ce')
        ax.set_ylim(bottom=0)
    if training:
        grouped = defaultdict(list)
        for step, row in training.items():
            grouped[(step - 1) // 100].append(row)
        bins = [grouped[key] for key in sorted(grouped) if len(grouped[key]) == 100]
        axes[0].plot([np.mean([r['step'] for r in group]) / 1000 for group in bins],
                     [np.median([r['loss'] for r in group]) for group in bins],
                     color='#94a1ad', lw=1, alpha=.7, label='Train: median / 100 updates')
        axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    if query:
        for ax in axes:
            maximum = max(float(np.max(line.get_ydata())) for line in ax.lines)
            ax.set_ylim(0, maximum * 1.12)
    footer = ('Full decoder + input queries · native rotary buffers · 200 fixed held-out frames · offline errors, not task success'
              if query else 'Corrected RGB · n0008 · 200 fixed held-out frames · inference errors exclude padded actions · no task-success claim')
    fig.text(.09, .018, footer,
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .065, 1, 1])
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        path = figures / f'{figure_name}.{suffix}'
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    compact = root / 'diagnostics'
    with (compact / f'{curve_name}.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(compact / f'{curve_name}.json', {'recipe_sha256': recipe_hash, 'runs': provenance,
               **({'study': registration['study'],
                   'initial_checkpoint_sha256': registration['initial_checkpoint_sha256'],
                   'registration_sha256': file_hash(root / 'query_recovery_registration.json')} if query else {}),
               'latest_training_step': max(training, default=0), 'validation': rows,
               'development': development,
               'limitations': ('Ongoing, single-seed coupled query/decoder/numerical amendment. Offline errors are not LIBERO success. '
                               'Development is ten diagnostic episodes per stage, not final acceptance. '
                               'Rollout metadata does not independently record the declared training plugin source manifest. '
                               'No matched retrained baseline.' if query else
                               'Ongoing, single-seed amended study. Offline action errors are not LIBERO success. No matched retrained baseline.')})
    print(json.dumps({'complete_validations': len(rows), 'latest_training_step': max(training, default=0)}))


if __name__ == '__main__':
    main()
