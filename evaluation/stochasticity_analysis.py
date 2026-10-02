"""Measured inference-noise figure on the existing matched action cohort."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.common import ROOT, read_json


def main():
    root = ROOT / 'studies/recovery'
    data = [read_json(root / 'diagnostics' / name) for name in
            ('b16_stochasticity.json', 'rgb_5k_stochasticity.json')]
    identity = lambda d: [(r['split'], r['row'], r['episode'], r['frame'], r['valid_actions'], r['seeds'])
                          for r in d['frames']]
    if identity(data[0]) != identity(data[1]) or any(d['draws_per_frame'] != 5 for d in data):
        raise ValueError('Models require identical frame/action/seed membership')
    if any(d['native_inference_timesteps'] != 4 for d in data):
        raise ValueError('Use the recorded native four-step prediction protocol')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.45))
    keys = ['single_draw_arm_mse_mean', 'mean_prediction_arm_mse', 'within_draw_arm_variance']
    grip_keys = ['gripper_error_mean', 'gripper_majority_error', 'gripper_any_disagreement_fraction']
    for index, (record, label, color) in enumerate(zip(data, ['Qwen B16', 'RGB-5k'], ['#1f5f94', '#65528c'])):
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
    axes[0].set_ylim(5e-6, .075)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', ncol=2)
    axes[1].set(ylabel='Gripper actions (%)', xticks=range(3),
                xticklabels=['Single-draw\nerrors', 'Majority-vote\nerrors', 'Any draw\ndisagreement'], ylim=(-1, 22))
    for ax in axes:
        ax.grid(axis='y', color='#dce1e7', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#bac4ce')
        ax.tick_params(length=3, color='#bac4ce')
        ax.set_xlim(-.5, 2.5)
    fig.text(.085, .02, '20 matched held-out frames · 5 native seeded draws · passive averages, no deployed ensemble or task-success claim',
             fontsize=8, color='#596776')
    fig.tight_layout(rect=[0, .055, 1, .91])
    for suffix in ('pdf', 'svg', 'png'):
        path = root / 'figures' / f'F10_inference_noise.{suffix}'
        fig.savefig(path, dpi=200)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
