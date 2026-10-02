"""Measured inference-noise figure on the existing matched action cohort."""

from __future__ import annotations

import argparse
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

from evaluation.common import ROOT, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=['legacy', 'endpoint-query'], default='legacy')
    args = parser.parse_args()
    root = ROOT / 'studies/recovery'
    names = (('b16_stochasticity.json', 'rgb_5k_stochasticity.json') if args.comparison == 'legacy'
             else ('rgb_20k_native_noise_environment.json', 'query_500_native_noise_environment.json'))
    labels = ['Qwen B16', 'RGB-5k'] if args.comparison == 'legacy' else ['r1 parent: 20k', 'Query study: 500']
    data = [read_json(root / 'diagnostics' / name) for name in names]
    identity = lambda d: [(r['split'], r['task_id'], r['row'], r['episode'], r['frame'], r['valid_actions'], r['seeds'])
                          for r in d['frames']]
    if identity(data[0]) != identity(data[1]) or any(d['draws_per_frame'] != 5 for d in data):
        raise ValueError('Models require identical frame/action/seed membership')
    if any(d['native_inference_timesteps'] != 4 for d in data):
        raise ValueError('Use the recorded native four-step prediction protocol')
    if args.comparison == 'endpoint-query':
        parent = read_json(root / 'diagnostics/completed_r1.json')
        query = read_json(root / 'diagnostics/query_500_milestone.json')
        if (data[0]['checkpoint_sha256'] != parent['selected_checkpoint_sha256']
                or data[1]['checkpoint_sha256'] != query['checkpoint_sha256']
                or data[0]['source_manifest'] != parent['source_manifest']
                or data[1]['source_manifest'] != query['training_source_manifest']):
            raise ValueError('Require the registered parent and first query milestone')
        if (data[0]['environment_observed_at_completion'] != data[1]['environment_observed_at_completion']
                or any(d.get('buffer_precision') != 'native_rope' for d in data)
                or data[0]['cohort_sha256'] != data[1]['cohort_sha256']):
            raise ValueError('Require matched observed hardware, numerical flags, buffers and cohort')
        for d in data:
            for r in d['frames']:
                total, mean, variance = [r[k] for k in
                    ('single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance')]
                if not np.isclose(total, mean + variance, rtol=1e-5, atol=1e-8):
                    raise ValueError('Finite-draw decomposition failed')
        before, after = [d['summary']['heldout'] for d in data]
        change = {k: after[k] - before[k] for k in
                  ('single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance')}
        if not np.isclose(change['single_draw_arm_mse_mean'],
                          change['mean_prediction_arm_mse'] + change['within_draw_arm_variance'],
                          rtol=1e-5, atol=1e-8):
            raise ValueError('Change decomposition failed')
        write_json(root / 'diagnostics/query_500_noise_comparison.json', {
            'purpose': 'matched_saved_endpoint_query_noise_comparison',
            'input_sha256': {n: file_hash(root / 'diagnostics' / n) for n in names},
            'analysis_sha256': file_hash(__file__),
            'checkpoint_sha256': {label: d['checkpoint_sha256'] for label, d in zip(labels, data)},
            'source_manifest': {label: d['source_manifest'] for label, d in zip(labels, data)},
            'environment': data[0]['environment_observed_at_completion'],
            'cohort_sha256': data[0]['cohort_sha256'], 'heldout': dict(zip(labels, (before, after))),
            'heldout_change': change,
            'mean_prediction_fraction_of_mse_increase': change['mean_prediction_arm_mse'] / change['single_draw_arm_mse_mean'],
            'limitations': '20 fixed held-out demonstration frames and five draws each; descriptive equal-frame '
            'averages, not independent trials or statistical bias. Query adaptation, decoder trainability, '
            'fresh optimizer/schedule and training numerical amendments are coupled. No isolated cause, '
            'deployed ensemble, generalization or robot-success claim.'})
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.45))
    keys = ['single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance']
    grip_keys = ['gripper_error_mean', 'gripper_majority_error', 'gripper_any_disagreement_fraction']
    for index, (record, label, color) in enumerate(zip(data, labels, ['#1f5f94', '#65528c'])):
        row = record['summary']['heldout']
        x = np.arange(3) + (index - .5) * .16
        values = [row[k] for k in keys]
        axes[0].scatter(x, values, s=40, color=color, label=label, zorder=3)
        for px, value in zip(x, values):
            axes[0].annotate(f'{value:.2g}', (px, value), xytext=(0, 8 if index else -13),
                             textcoords='offset points', ha='center', fontsize=8, color=color)
        axes[1].scatter(x, [100 * row[k] for k in grip_keys], s=40, color=color, zorder=3)
    axes[0].set(yscale='log', ylabel='Arm-command MSE (simulator inputs)',
                xticks=range(3), xticklabels=['Single-draw\nmean error', 'Five-draw\nmean error', 'Within-draw\nvariance'])
    axes[0].set_ylim(5e-6, .075) if args.comparison == 'legacy' else axes[0].set_ylim(.0015, .08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', ncol=2)
    axes[1].set(ylabel='Gripper actions (%)', xticks=range(3),
                xticklabels=['Single-draw\nerrors', 'Majority-vote\nerrors', 'Any draw\ndisagreement'], ylim=(-1, 22))
    if args.comparison == 'endpoint-query':
        axes[0].set_ylabel('Arm-command MSE (log scale)')
        axes[0].yaxis.set_major_locator(FixedLocator([.002, .005, .01, .02, .05]))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        axes[0].yaxis.set_minor_formatter(NullFormatter())
        axes[1].set_ylim(-1, 12)
    for ax in axes:
        ax.grid(axis='y', color='#dce1e7', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#bac4ce')
        ax.tick_params(length=3, color='#bac4ce')
        ax.set_xlim(-.5, 2.5)
    if args.comparison == 'legacy':
        fig.text(.085, .02, '20 matched held-out frames · 5 native seeded draws · passive averages, no deployed ensemble or task-success claim',
                 fontsize=8, color='#596776')
    else:
        fig.text(.085, .04, '20 matched held-out frames · RTX3090 · BF16 parameters / native rotary buffers · 4 solver steps · 5 draws',
                 fontsize=7.5, color='#596776')
        fig.text(.085, .012, 'Simulator command units · passive averages/votes · no deployed ensemble or task-success inference',
                 fontsize=7.5, color='#596776')
    fig.tight_layout(rect=[0, .08 if args.comparison == 'endpoint-query' else .055, 1, .91])
    for suffix in ('pdf', 'svg', 'png'):
        prefix = 'F10_inference_noise' if args.comparison == 'legacy' else 'F12_query_noise_comparison'
        path = root / 'figures' / f'{prefix}.{suffix}'
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
