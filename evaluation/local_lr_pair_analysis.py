"""Offline evidence checks and figures for the registered local learning-rate pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.query_milestone import validation_score, verify_counters, verify_files


def score(rows, step, entry):
    validation_score(rows, dict(entry, step=step))
    losses = [r for r in rows if r.get('phase') == 'validation' and r['step'] == step]
    actions = [r for r in rows if r.get('phase') == 'validation_action' and r['step'] == step]
    return {'optimizer_updates': step // 2, 'native_microsteps': step,
            'arm_mse': sum(r['arm_squared_error_sum'] for r in actions) / (6 * 1303),
            'gripper_error': sum(r['gripper_errors'] for r in actions) / 1303,
            'native_loss': sum(r['loss'] * r['samples'] for r in losses) / 200,
            'frames': 200, 'valid_actions': 1303, 'checkpoint_sha256': entry['checkpoint_sha256']}


def record_context(record, registration, recipe, branch, registration_hash):
    expected = {'purpose': 'recovery_adaptation', 'study': recipe['study'], 'recipe': recipe,
                'recipe_sha256': registration['recipe_sha256'][branch],
                'source_manifest': registration['source_manifest'],
                'initial_checkpoint_sha256': registration['initial_checkpoint_sha256'],
                'local_lr_pair_registration_sha256': registration_hash,
                'split_sha256': registration['data_manifest_sha256']['studies/evaluation/training_split.json'],
                'native_step_unit': 'microbatch', 'gradient_accumulation_steps': 2,
                'effective_batch_size': 8, 'deterministic_algorithms': True, 'isolated_worker_rng': True,
                'requested_stop_step': 1000}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Branch training provenance or native topology differs')
    if record['environment']['gpu'] != registration['gpu']:
        raise ValueError('Branch GPU differs')


def merge_streams(streams):
    """A resumed stream supersedes the earlier uncommitted tail, including validation."""
    observations = {}
    for rows in streams:
        starts = [r['step'] for r in rows if r.get('phase') == 'training']
        if not starts:
            raise ValueError('Require observed training in every selected history')
        start = min(starts)
        observations = {key: row for key, row in observations.items() if key[1] < start}
        seen = set()
        for row in rows:
            if row.get('phase') not in {'training', 'validation', 'validation_action'}:
                continue
            step = row.get('step')
            if type(step) is not int or not 0 < step <= 1000:
                raise ValueError('Invalid native microstep')
            key = (row['phase'], step, row.get('batch'))
            if key in seen:
                raise ValueError('Duplicate observation within a native stream')
            seen.add(key)
            observations[key] = row
    return list(observations.values())


def anchor_context(anchor, registration, anchor_registration, numerical_flags):
    if (anchor.get('status') != 'completed' or anchor.get('source_manifest') != registration['source_manifest']
            or anchor['environment']['gpu'] != registration['gpu']
            or anchor.get('numerical_flags') != numerical_flags
            or numerical_flags != {'deterministic_algorithms': True, 'matmul_allow_tf32': True,
                                   'cudnn_allow_tf32': True, 'cudnn_benchmark': False,
                                   'cudnn_deterministic': True}
            or anchor.get('diagnostic_sha256') != anchor_registration['diagnostic_sha256']):
        raise ValueError('Matched parent source, GPU or observed numerical flags differ')
    if len(anchor.get('cases', [])) != 1:
        raise ValueError('Require only the registered native parent anchor')
    case = anchor['cases'][0]
    if (case['mode'] != 'parent_zero_query_native_all32' or case['checkpoint_sha256'] != registration['initial_checkpoint_sha256']
            or case.get('exact_inherited_values') is not True or case.get('zero_queries') is not True):
        raise ValueError('Parent weights or initializer differ')
    rows = []
    for row in case['batches']:
        rows.extend([dict(row, phase=phase, step=0) for phase in ('validation', 'validation_action')])
    entry = dict(checkpoint_sha256=case['checkpoint_sha256'], heldout_arm_mse=case['summary']['arm_mse'],
                 heldout_gripper_error=case['summary']['gripper_error'])
    result = score(rows, 0, entry)
    if (case['summary']['frames'] != 200 or case['summary']['valid_actions'] != 1303
            or not math.isclose(result['native_loss'], case['summary']['loss'], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Parent summary differs from its complete batch observations')
    return result


def select_policy(parent, branches):
    candidates = [dict(parent, branch='parent')]
    candidates.extend(dict(row, branch=branch) for branch, rows in branches.items() for row in rows)
    return min(candidates, key=lambda row: (row['arm_mse'], row['optimizer_updates'], row['branch']))


def analyze(root, job_path, source, parent_checkpoint, anchor_path):
    registration_path = root / 'studies/recovery/local_lr_pair_registration.json'
    registration, job = read_json(registration_path), read_json(job_path)
    registration_hash = file_hash(registration_path)
    if (job.get('status') != 'completed' or job.get('active_branch') is not None or job.get('child_pid') is not None
            or job.get('study') != registration['study'] or job.get('registration_sha256') != registration_hash
            or set(job.get('branches', {})) != {'high', 'low'}):
        raise ValueError('Require both registered terminal branches; do not analyze a live or partial pair')
    for name, path in [('harness_sha256', root / 'evaluation/recovery_train.py'),
                       ('supervisor_sha256', root / 'scripts/local_lr_pair.py'),
                       ('counter_verifier_sha256', root / 'evaluation/query_milestone.py')]:
        if file_hash(path) != registration[name]:
            raise ValueError('Registered training implementation changed')
    for name, expected in registration['data_manifest_sha256'].items():
        if file_hash(root / name) != expected:
            raise ValueError('Registered data membership differs')
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest']:
        raise ValueError('Actual architecture source differs')
    for name, expected in registration['parent_files'].items():
        if file_hash(parent_checkpoint / name) != expected:
            raise ValueError('Actual parent bytes differ')
    recipes = {branch: read_json(root / f'evaluation/local_lr_{branch}_config.json') for branch in ('high', 'low')}
    changed = {'study', 'learning_rate', 'backbone_learning_rate', 'decay_learning_rate'}
    if ({k: v for k, v in recipes['high'].items() if k not in changed} !=
            {k: v for k, v in recipes['low'].items() if k not in changed}
            or any(not math.isclose(recipes['high'][k], 10 * recipes['low'][k], rel_tol=1e-12)
                   for k in changed - {'study'})):
        raise ValueError('Branches differ beyond the registered tenfold rate scale')
    branches, proofs, flags, packages = {}, {}, None, None
    for branch, recipe in recipes.items():
        if file_hash(root / f'evaluation/local_lr_{branch}_config.json') != registration['recipe_sha256'][branch]:
            raise ValueError('Registered branch recipe changed')
        histories = []
        for path in sorted((root / 'studies/recovery/training').glob('*/run.json')):
            record = read_json(path)
            if record.get('recipe_sha256') == registration['recipe_sha256'][branch]:
                record_context(record, registration, recipe, branch, registration_hash)
                histories.append((path, record))
        if (not histories or histories[-1][1].get('status') != 'completed'
                or histories[-1][1].get('completed_steps') != 1000
                or sum(record.get('resume_checkpoint_sha256') is None for _, record in histories) != 1):
            raise ValueError('Require one completed native branch lineage')
        final = histories[-1][1]
        if flags is None:
            flags, packages = final['numerical_flags'], final['environment']['packages']
        if final['numerical_flags'] != flags or final['environment']['packages'] != packages:
            raise ValueError('Branch numerical profile or package environment differs')
        rows = merge_streams([[json.loads(line) for line in (p.parent / 'metrics.jsonl').read_text().splitlines()]
                              for p, _ in histories])
        training = [r for r in rows if r['phase'] == 'training']
        if (sorted(r['step'] for r in training) != list(range(1, 1001))
                or any(r.get('optimizer_updates') != r['step'] // 2 for r in training)
                or any(not math.isfinite(r[k]) or r[k] < 0 for r in training for k in ('loss', 'action_loss', 'wm_loss'))):
            raise ValueError('Training exposure or optimizer-update observations are incomplete')
        journal_path = histories[-1][0].parent / 'checkpoints.json'
        journal = read_json(journal_path)
        if (sorted(r['step'] for r in journal) != list(range(100, 1001, 100))
                or any(r.get('native_microsteps') != r['step'] or r.get('optimizer_updates') != r['step'] // 2
                       for r in journal)):
            raise ValueError('Native save journal does not cover the registered branch')
        scored = [r for r in journal if 'heldout_arm_mse' in r]
        if (sorted(r['step'] for r in scored) != [500, 1000]
                or any(r.get('heldout_valid_actions') != 1303 for r in scored)):
            raise ValueError('Require both registered full held-out checks')
        branches[branch] = [score(rows, row['step'], row) for row in scored]
        native = job['branches'][branch]
        if (native['recipe_sha256'] != registration['recipe_sha256'][branch]
                or native['completed_microsteps'] != 1000 or native['optimizer_updates'] != 500):
            raise ValueError('Supervisor completion record differs')
        directory = (root / native['latest_native_checkpoint']).resolve()
        if not directory.is_relative_to(root.resolve()) or directory.name != '001000':
            raise ValueError('Native checkpoint must belong to this local study')
        verify_files(directory, native['latest_native_files'])
        expected_native = {'study': recipe['study'], 'recipe_sha256': registration['recipe_sha256'][branch],
                           'source_manifest': registration['source_manifest']}
        if read_json(directory / 'pretrained_model/recovery_registration.json') != expected_native:
            raise ValueError('Native checkpoint registration differs')
        trainability_path = histories[-1][0].parent / 'trainability.json'
        counters = verify_counters(directory, 1000, read_json(trainability_path), batch_size=4, accumulation_steps=2)
        selected = min(scored, key=lambda row: (row['heldout_arm_mse'], row['step']))
        selected_path = (root / native['selected_model']).resolve()
        if (not selected_path.is_relative_to(directory.parent) or not selected['retained']
                or selected_path.parent.parent.name != f"{selected['step']:06d}"
                or selected != native['selected'] or file_hash(selected_path) != selected['checkpoint_sha256']
                or native['latest_native_files']['pretrained_model/model.safetensors']['sha256'] != journal[-1]['checkpoint_sha256']):
            raise ValueError('Selected policy or final native weights differ from the score journal')
        proofs[branch] = {'training': [{'run_id': record['run_id'], 'run_sha256': file_hash(path),
                                       'metrics_sha256': file_hash(path.parent / 'metrics.jsonl')}
                                      for path, record in histories],
                          'checkpoint_journal_sha256': file_hash(journal_path),
                          'trainability_sha256': file_hash(trainability_path),
                          'latest_native_files': native['latest_native_files'],
                          'native_scheduler_microsteps': counters[0], 'populated_optimizer_states': counters[1],
                          'optimizer_updates': 500, 'peak_training_allocated_gib': max(r['peak_allocated_bytes'] for r in training) / 2**30,
                          'selected_checkpoint_sha256': selected['checkpoint_sha256']}
    anchor_registration_path = root / 'studies/recovery/parent_native_selection_registration.json'
    anchor_registration, anchor = read_json(anchor_registration_path), read_json(anchor_path)
    if (anchor_registration['pair_registration_sha256'] != registration_hash
            or anchor_registration['source_manifest'] != registration['source_manifest']
            or anchor_registration['parent_checkpoint_sha256'] != registration['initial_checkpoint_sha256']
            or anchor_registration['matmul_allow_tf32'] is not True
            or anchor_registration['modes'] != ['parent_zero_query_native_all32']
            or anchor_registration['validation_samples_sha256'] != registration['data_manifest_sha256']['studies/evaluation/validation_samples.json']
            or anchor_registration['split_sha256'] != registration['data_manifest_sha256']['studies/evaluation/training_split.json']
            or anchor.get('registration_sha256') != file_hash(anchor_registration_path)
            or file_hash(root / 'evaluation/query_initial_validation.py') != anchor_registration['diagnostic_sha256']):
        raise ValueError('Matched parent diagnostic registration differs')
    parent = anchor_context(anchor, registration, anchor_registration, flags)
    if anchor['environment']['packages'] != packages:
        raise ValueError('Parent package environment differs from training')
    selected = select_policy(parent, branches)
    return {'status': 'completed', 'purpose': 'registered_local_learning_rate_comparison',
            'study': registration['study'], 'registration_sha256': registration_hash,
            'analysis_sha256': file_hash(__file__), 'source_manifest': manifest, 'gpu': registration['gpu'],
            'numerical_flags': flags, 'parent_anchor_sha256': file_hash(anchor_path),
            'parent_registration_sha256': file_hash(anchor_registration_path),
            'parent': parent, 'branches': branches, 'native_provenance': proofs, 'heldout_selected': selected,
            'provisional_controller_choice': job.get('heldout_selected_branch'),
            'controller_choice_matches_native_anchor': job.get('heldout_selected_branch') == selected['branch'],
            'limitations': 'Single seed; all learning-rate scales change together. Fixed200 held-out frames/1303 correlated actions, not robot success. Physical batch8 cloud results are not controls. No uninterrupted-value equivalence for this accumulated pair. Selection includes the measured numerical-profile-matched parent and never uses development or final rollouts.'}


def figure(result, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.0))
    panels = [('arm_mse', 'Arm-command RMSE', math.sqrt),
              ('gripper_error', 'Gripper errors (%)', lambda x: 100 * x),
              ('native_loss', 'Native validation loss', lambda x: x)]
    for ax, (key, label, transform) in zip(axes, panels):
        ax.axhline(transform(result['parent'][key]), color='#5d6875', ls='--', lw=1.2, label='Unchanged parent')
        for branch, color, marker in [('high', '#bd5e3b', 'o'), ('low', '#1f5f94', 's')]:
            rows = result['branches'][branch]
            ax.plot([r['optimizer_updates'] for r in rows], [transform(r[key]) for r in rows],
                    marker=marker, ms=5, lw=1.4, color=color, label=f'{branch.capitalize()} rate')
        ax.set(xlabel='Additional optimizer updates', ylabel=label, xticks=[250, 500], xlim=(225, 525))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dce2e7', lw=.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(prefix.with_suffix('.' + extension), bbox_inches='tight', dpi=180, metadata={'Creator': 'CSI5341'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=ROOT / 'outputs/recovery/local-lr-pair/job.json')
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--parent-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure-prefix', type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.figure_prefix and any(args.figure_prefix.with_suffix('.' + ext).exists()
                                                           for ext in ('pdf', 'svg', 'png'))):
        raise FileExistsError('Preserve existing comparison evidence and figures')
    result = analyze(ROOT, args.job, args.architecture_source.resolve(), args.parent_checkpoint.resolve(), args.parent_validation)
    write_json(args.output, result)
    if args.figure_prefix:
        figure(result, args.figure_prefix)
    print(json.dumps({'study': result['study'], 'heldout_selected': result['heldout_selected']}))


if __name__ == '__main__':
    main()
