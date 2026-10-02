"""Validate and plot the registered passive initialization comparison."""

from __future__ import annotations

import math

from evaluation.common import ROOT, file_hash, read_json, write_json


def compare(data, registration):
    if (data.get('status') != 'completed' or data.get('source_manifest') != registration['source_manifest']
            or [case['mode'] for case in data['cases']] != registration['modes']):
        raise ValueError('Require completed registered cases and source')
    for index, case in enumerate(data['cases']):
        expected = registration['query500_checkpoint_sha256' if index == 2 else 'parent_checkpoint_sha256']
        if case['checkpoint_sha256'] != expected or not case['exact_inherited_values']:
            raise ValueError('Checkpoint restoration differs from registration')
        rows = case['batches']
        if (len(rows) != 25 or [row['batch'] for row in rows] != list(range(25))
                or any(row['samples'] != 8 for row in rows)
                or sum(row['valid_actions'] for row in rows) != 1303):
            raise ValueError('Full fixed validation membership is incomplete')
        if any(any(not math.isfinite(row[key]) or row[key] < 0 for key in
                   ('loss', 'action_loss', 'wm_loss', 'arm_squared_error_sum'))
               or not 0 <= row['gripper_errors'] <= row['valid_actions']
               or not math.isclose(row['loss'], row['action_loss'] + row['wm_loss'], rel_tol=1e-5, abs_tol=1e-8)
               for row in rows):
            raise ValueError('Invalid per-batch native loss or action measurement')
        measured = {'frames': 200, 'valid_actions': 1303,
                    'arm_mse': sum(row['arm_squared_error_sum'] for row in rows) / (6 * 1303),
                    'gripper_error': sum(row['gripper_errors'] for row in rows) / 1303,
                    'loss': sum(row['loss'] * row['samples'] for row in rows) / 200}
        if any(not math.isfinite(value) for value in measured.values()) or measured != case['summary']:
            raise ValueError('Saved summary differs from measured batches')
    parent, initial, trained = data['cases']
    return {'initialization_batch_scores_exactly_equal': parent['batches'] == initial['batches'],
            'tested_batches': 25,
            'physical_arm_mse_increase_at500': trained['summary']['arm_mse'] - parent['summary']['arm_mse'],
            'physical_arm_mse_ratio_at500': trained['summary']['arm_mse'] / parent['summary']['arm_mse'],
            'gripper_error_increase_pp_at500': 100 * (trained['summary']['gripper_error'] - parent['summary']['gripper_error']),
            'interpretation': 'Exact equality refers to all25 saved batch scores, not independently saved prediction tensors. '
            'Learning interventions remain coupled. This is passive validation, not robot success or general equivalence.'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = ROOT / 'studies/recovery'
    path = root / 'diagnostics/query_initial_validation.json'
    registration = read_json(root / 'query_initial_validation_registration.json')
    data = read_json(path)
    if data['registration_sha256'] != file_hash(root / 'query_initial_validation_registration.json'):
        raise ValueError('Measurement registration changed')
    comparison = compare(data, registration)
    write_json(root / 'diagnostics/query_initial_validation_comparison.json', {
        'measurement_sha256': file_hash(path), 'analysis_sha256': file_hash(__file__),
        'registration_sha256': data['registration_sha256'], 'comparison': comparison})
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    labels = ['Parent\n4 trainable', 'Zero residuals\n32 trainable', '500 updates\n32 trainable']
    for ax, key, scale, label in zip(axes, ['arm_mse', 'gripper_error'], [1, 100],
                                    ['Arm-command MSE', 'Gripper errors (%)']):
        values = [case['summary'][key] * scale for case in data['cases']]
        ax.bar(range(3), values, width=.55, color=['#596776', '#aab6c1', '#65528c'], zorder=2)
        for x, value in enumerate(values):
            ax.text(x, value, f'{value:.4f}' if scale == 1 else f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set(xticks=range(3), xticklabels=labels, ylabel=label, ylim=(0, max(values) * 1.25))
        ax.grid(axis='y', color='#dce1e7', lw=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#bac4ce')
        ax.tick_params(length=3, color='#bac4ce', axis='y')
        ax.tick_params(length=0, axis='x')
    fig.text(.085, .018, 'RTX3090 · native precision · trainable decoder counts shown · 200 held-out frames / 1,303 actions · offline only',
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .075, 1, 1])
    for suffix in ('pdf', 'svg', 'png'):
        output = root / 'figures' / f'F15_query_initialization.{suffix}'
        fig.savefig(output, dpi=200)
        if suffix == 'svg':
            output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
    plt.close(fig)
    print(comparison)


if __name__ == '__main__':
    main()
