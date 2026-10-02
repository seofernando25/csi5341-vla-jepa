"""CPU reanalysis of saved, matched endpoint action-error diagnostics.

No new predictions, policy changes, checkpoint selection or robot trials.
"""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.common import ROOT, file_hash, read_json, write_json


def analyze(baseline, endpoint, mapping):
    if (not mapping['instruction_sets_exactly_equal']
            or len({r['instruction'] for r in mapping['mapping']}) != 10):
        raise ValueError('Instruction identity has not been established')
    identities = lambda d: [(r['split'], r['task_id'], r['row'], r['episode'],
                              r['frame'], r['valid_actions'], r['seeds']) for r in d['frames']]
    if (identities(baseline) != identities(endpoint)
            or baseline['cohort_sha256'] != endpoint['cohort_sha256']
            or any(d['draws_per_frame'] != 5 or d['native_inference_timesteps'] != 4
                   for d in (baseline, endpoint))):
        raise ValueError('Frame, action, seed or solver membership differs')
    dataset_ids = [r['dataset_task_id'] for r in mapping['mapping']]
    simulator_ids = [r['libero_task_id'] for r in mapping['mapping']]
    if sorted(dataset_ids) != list(range(10)) or sorted(simulator_ids) != list(range(10)):
        raise ValueError('Instruction mapping must be a task bijection')
    grouped = []
    for data in (baseline, endpoint):
        groups = defaultdict(list)
        for row in data['frames']:
            total, mean, variance = [row[k] for k in
                ('single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance')]
            if (not all(math.isfinite(v) and v >= 0 for v in (total, mean, variance))
                    or not math.isclose(total, mean + variance, rel_tol=1e-5, abs_tol=1e-8)):
                raise ValueError('Saved finite-draw decomposition is inconsistent')
            if not all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in
                       ('gripper_error_mean', 'gripper_any_disagreement_fraction')):
                raise ValueError('Saved gripper fractions are invalid')
            if row['split'] == 'heldout':
                groups[row['task_id']].append(row)
        if sorted(groups) != list(range(10)) or any(len(v) != 2 for v in groups.values()):
            raise ValueError('Require the existing two held-out frames per task')
        grouped.append(groups)
    keys = ['single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance',
            'gripper_error_mean', 'gripper_any_disagreement_fraction']
    tasks = []
    for link in sorted(mapping['mapping'], key=lambda r: r['libero_task_id']):
        task = dict(link)
        for label, groups in zip(('baseline', 'endpoint'), grouped):
            rows = groups[link['dataset_task_id']]
            values = {k: float(np.mean([r[k] for r in rows])) for k in keys}
            values['variance_fraction'] = values['within_draw_arm_variance'] / values['single_draw_arm_mse_mean']
            values['frame_single_draw_mse'] = [r['single_draw_arm_mse_mean'] for r in rows]
            values['frames'] = [{'episode': r['episode'], 'frame': r['frame'],
                                'valid_actions': r['valid_actions']} for r in rows]
            task[label] = values
        task['mean_error_ratio_to_b16'] = (task['endpoint']['mean_prediction_arm_mse']
                                          / task['baseline']['mean_prediction_arm_mse'])
        tasks.append(task)
    return tasks


def main():
    root = ROOT / 'studies/recovery'
    paths = [root / 'diagnostics' / n for n in
             ('b16_stochasticity.json', 'rgb_20k_native_stochasticity.json', 'task_instruction_mapping.json')]
    baseline, endpoint, mapping = [read_json(p) for p in paths]
    completed = read_json(root / 'diagnostics/completed_r1.json')
    if (endpoint.get('buffer_precision') != 'native_rope'
            or endpoint['checkpoint_sha256'] != completed['selected_checkpoint_sha256']):
        raise ValueError('Require the recorded native-frequency selected r1 endpoint')
    tasks = analyze(baseline, endpoint, mapping)
    ratios = [r['mean_error_ratio_to_b16'] for r in tasks]
    compact = {'purpose': 'saved_matched_endpoint_task_error_analysis',
               'input_sha256': {p.name: file_hash(p) for p in paths},
               'analysis_sha256': file_hash(__file__),
               'baseline_checkpoint_sha256': baseline['checkpoint_sha256'],
               'endpoint_checkpoint_sha256': endpoint['checkpoint_sha256'],
               'endpoint_buffer_precision': endpoint['buffer_precision'],
               'cohort_sha256': endpoint['cohort_sha256'], 'draws_per_frame': 5,
               'tasks': tasks, 'summary': {
                   'tasks_with_larger_endpoint_mean_error': sum(r > 1 for r in ratios),
                   'mean_error_ratio_range': [min(ratios), max(ratios)],
                   'endpoint_variance_fraction_range': [min(r['endpoint']['variance_fraction'] for r in tasks),
                                                        max(r['endpoint']['variance_fraction'] for r in tasks)],
                   'tasks_with_endpoint_gripper_errors': sum(r['endpoint']['gripper_error_mean'] > 0 for r in tasks)},
               'limitations': 'Two demonstration frames per task and five seeded draws per frame; descriptive '
               'equal-frame averages, no confidence intervals or representative task estimates. Finite-draw '
               'mean prediction error is not statistical bias. Qwen training membership is unknown. Differing '
               'training histories and buffer protocols prevent isolated backbone-size inference. No deployed '
               'ensemble, paired final benchmark, closed-loop causal conclusion or new model evaluation.'}
    write_json(root / 'diagnostics/endpoint_task_errors.json', compact)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.25))
    x = np.arange(10)
    colors = {'baseline': '#1f5f94', 'endpoint': '#65528c'}
    for label, offset, legend in [('baseline', -.1, 'Qwen B16'), ('endpoint', .1, 'SmolVLM r1-20k')]:
        vals = [r[label] for r in tasks]
        axes[0].plot(x + offset, [r['single_draw_arm_mse_mean'] for r in vals],
                     'o', color=colors[label], ms=4, label=legend)
        axes[1].plot(x + offset, [100 * r['variance_fraction'] for r in vals],
                     'o', color=colors[label], ms=4)
        axes[2].plot(x + offset, [100 * r['gripper_error_mean'] for r in vals],
                     'o', color=colors[label], ms=4)
    axes[0].plot(x + .1, [r['endpoint']['mean_prediction_arm_mse'] for r in tasks],
                 '_', color='#bd5e3b', ms=8, label='r1 five-draw mean (passive)')
    axes[0].set(yscale='log', ylabel='Arm-command MSE', title='(a) Prediction error')
    axes[1].set(ylabel='Fraction of arm MSE (%)', ylim=(0, 100), title='(b) Within-draw variance')
    axes[2].set(ylabel='Wrong gripper commands (%)', title='(c) Gripper errors', ylim=(-.7, None))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel('LIBERO task ID (instruction-matched)')
        ax.grid(axis='y', color='#dce1e7', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#bac4ce')
        ax.tick_params(length=3, color='#bac4ce')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', ncol=3, fontsize=9)
    fig.text(.075, .015, '2 matched held-out demonstration frames / task · 5 seeded draws · equal-frame means · no task-success inference',
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .075, 1, .89])
    for suffix in ('pdf', 'svg', 'png'):
        path = root / 'figures' / f'F11_endpoint_task_errors.{suffix}'
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    print(compact['summary'])


if __name__ == '__main__':
    main()
