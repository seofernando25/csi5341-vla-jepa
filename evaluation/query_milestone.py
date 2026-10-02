"""Export a verified query milestone without conflating selected and resume weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.paired_development_analysis import analyze as check_development


NATIVE_FILES = {
    'pretrained_model/' + name for name in (
        'config.json', 'model.safetensors', 'train_config.json', 'recovery_registration.json',
        'policy_preprocessor.json', 'policy_postprocessor.json',
        'policy_preprocessor_step_4_normalizer_processor.safetensors',
        'policy_postprocessor_step_2_unnormalizer_processor.safetensors')
} | {'training_state/' + name for name in (
    'training_step.json', 'rng_state.safetensors', 'scheduler_state.json',
    'optimizer_state.safetensors', 'optimizer_param_groups.json')}


def context(record, registration, recipe):
    if (record.get('status') != 'completed' or record.get('purpose') != 'recovery_adaptation'
            or record.get('study') != registration['study']
            or record.get('recipe_sha256') != registration['recipe_sha256']
            or record.get('recipe') != recipe
            or record.get('source_manifest') != registration['source_manifest']
            or record.get('initial_checkpoint_sha256') != registration['initial_checkpoint_sha256']):
        raise ValueError('Require completed registered production training evidence')
    if record.get('split_sha256') != file_hash(ROOT / 'studies/evaluation/training_split.json'):
        raise ValueError('Training split identity differs')


def selected_entry(journal, stop, development):
    if (not journal or len({row['step'] for row in journal}) != len(journal)
            or max(row['step'] for row in journal) != stop):
        raise ValueError('Journal does not uniquely reach the completed stage')
    for row in journal:
        if (row['step'] <= 0 or row['step'] % 500 or row['heldout_valid_actions'] != 1303
                or not math.isfinite(row['heldout_arm_mse']) or row['heldout_arm_mse'] < 0
                or not 0 <= row['heldout_gripper_error'] <= 1):
            raise ValueError('Invalid registered validation score')
    if sorted(row['step'] for row in journal) != list(range(500, stop + 1, 500)):
        raise ValueError('Validation journal is incomplete')
    selected = min(journal, key=lambda row: (row['heldout_arm_mse'], row['step']))
    if not selected['retained'] or development['checkpoint_sha256'] != selected['checkpoint_sha256']:
        raise ValueError('Development did not evaluate the held-out-selected policy')
    return selected


def validation_score(rows, selected):
    step = selected['step']
    losses = [row for row in rows if row.get('phase') == 'validation' and row['step'] == step]
    actions = [row for row in rows if row.get('phase') == 'validation_action' and row['step'] == step]
    if (sorted(row['batch'] for row in losses) != list(range(25))
            or sorted(row['batch'] for row in actions) != list(range(25))
            or any(row['samples'] != 8 for row in losses)
            or sum(row['valid_actions'] for row in actions) != 1303):
        raise ValueError('Selected validation membership/counts are incomplete')
    for row in losses:
        if (not all(math.isfinite(row[key]) and row[key] >= 0 for key in ('loss', 'action_loss', 'wm_loss'))
                or not math.isclose(row['loss'], row['action_loss'] + row['wm_loss'], rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError('Invalid native validation loss')
    for row in actions:
        if (not math.isfinite(row['arm_squared_error_sum']) or row['arm_squared_error_sum'] < 0
                or not 0 < row['valid_actions'] <= 56
                or not 0 <= row['gripper_errors'] <= row['valid_actions']):
            raise ValueError('Invalid physical validation errors')
    arm = sum(row['arm_squared_error_sum'] for row in actions) / (1303 * 6)
    grip = sum(row['gripper_errors'] for row in actions) / 1303
    if (not math.isclose(arm, selected['heldout_arm_mse'], rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(grip, selected['heldout_gripper_error'], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Journal score differs from actual validation rows')


def verify_files(directory, expected):
    if set(expected) != NATIVE_FILES:
        raise ValueError('Require all thirteen native export files')
    for name, proof in expected.items():
        path = directory / name
        if (not path.is_file() or path.stat().st_size != proof['bytes']
                or file_hash(path) != proof['sha256']):
            raise ValueError('Local native export bytes differ from backup proof')


def verify_counters(directory, stop, trainability):
    from safetensors import safe_open
    state = directory / 'training_state'
    topology = read_json(state / 'training_step.json')
    if {key: topology.get(key) for key in ('step', 'batch_size', 'dp_world_size', 'grad_accum_steps')} != {
            'step': stop, 'batch_size': 8, 'dp_world_size': 1, 'grad_accum_steps': 1}:
        raise ValueError('Native resume step or topology differs')
    scheduler = read_json(state / 'scheduler_state.json')
    if scheduler['last_epoch'] != stop or scheduler['_step_count'] != stop + 1:
        raise ValueError('Native scheduler counter differs')
    groups = read_json(state / 'optimizer_param_groups.json')
    parameters = [str(p) for group in groups for p in group['params']]
    # The inherited last-block capture bypasses the subsequent text norm. Its
    # registered parameter has no gradient/moments; every other trainable does.
    ordered_names = [name for name in trainability if not name.startswith('model.qwen.model.')]
    ordered_names += [name for name in trainability if name.startswith('model.qwen.model.')]
    if (len(parameters) != 694 or len(set(parameters)) != 694 or len(ordered_names) != 694
            or [g['name'] for g in groups] != ['adapter_action_world', 'smol_decoder']):
        raise ValueError('Native optimizer parameter membership differs')
    mapping = dict(zip(parameters, ordered_names, strict=True))
    unused = 'model.qwen.model.model.text_model.norm.weight'
    populated = [p for p in parameters if mapping[p] != unused]
    if len(populated) != 693:
        raise ValueError('Declared unused final norm differs')
    with safe_open(state / 'optimizer_state.safetensors', framework='pt', device='cpu') as saved:
        if set(saved.keys()) != {f'state/{p}/{key}' for p in populated for key in ('step', 'exp_avg', 'exp_avg_sq')}:
            raise ValueError('Native optimizer state membership differs')
        for p in populated:
            if saved.get_tensor(f'state/{p}/step').item() != stop:
                raise ValueError('Native optimizer counter differs')
            mean, variance = [saved.get_slice(f'state/{p}/{k}') for k in ('exp_avg', 'exp_avg_sq')]
            if (mean.get_dtype() != 'F32' or variance.get_dtype() != 'F32' or mean.get_shape() != variance.get_shape()
                    or math.prod(mean.get_shape()) != trainability[mapping[p]]['elements']):
                raise ValueError('Native optimizer moments differ in precision or shape')
    return scheduler['last_epoch'], len(populated), [unused]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-evidence', type=Path, required=True)
    parser.add_argument('--validation-evidence', type=Path, required=True)
    parser.add_argument('--development-run', type=Path, required=True)
    parser.add_argument('--native-checkpoint', type=Path, required=True)
    parser.add_argument('--selected-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous milestone evidence')
    registration = read_json(ROOT / 'studies/recovery/query_recovery_registration.json')
    recipe = read_json(ROOT / 'evaluation/query_recovery_config.json')
    if file_hash(ROOT / 'evaluation/query_recovery_config.json') != registration['recipe_sha256']:
        raise ValueError('Registered recipe changed')
    training, validation = [read_json(path / 'run.json') for path in (args.training_evidence, args.validation_evidence)]
    for record in (training, validation):
        context(record, registration, recipe)
    stop = training['requested_stop_step']
    if stop not in recipe['stage_stop_steps'] or int(args.native_checkpoint.name) != stop:
        raise ValueError('Not a registered stage/native directory')
    development = read_json(args.development_run / 'run.json')
    with (args.development_run / 'episodes.csv').open() as handle:
        episodes = list(csv.DictReader(handle))
    check_development([development, development], [episodes, episodes])
    if (development['environment']['gpu'] != training['environment']['gpu']
            or development['variant'] != f'Query-n0008-{stop}'
            or development['protocol'] != read_json(ROOT / 'evaluation/protocol.json')
            or development['initial_states_manifest_sha256'] != file_hash(ROOT / 'studies/evaluation/initial_states.json')):
        raise ValueError('Development GPU/protocol/state manifest differs')
    journal = read_json(args.training_evidence / 'checkpoints.json')
    selected = selected_entry(journal, stop, development)
    validation_journal = read_json(args.validation_evidence / 'checkpoints.json')
    if not any(row['step'] == selected['step'] and row['checkpoint_sha256'] == selected['checkpoint_sha256']
               for row in validation_journal):
        raise ValueError('Selected validation evidence belongs to different weights')
    if file_hash(args.selected_model) != selected['checkpoint_sha256']:
        raise ValueError('Selected local policy bytes differ')
    validation_rows = [json.loads(line) for line in (args.validation_evidence / 'metrics.jsonl').read_text().splitlines()]
    validation_score(validation_rows, selected)
    directory = args.native_checkpoint.resolve().relative_to(ROOT)
    backups = [json.loads(line) for line in (ROOT / 'outputs/recovery/cloud/checkpoint-backups.jsonl').read_text().splitlines()]
    proof = next(row for row in reversed(backups) if row['directory'] == str(directory))
    expected = {str(Path(name).relative_to(directory)): value for name, value in proof['files'].items()}
    verify_files(args.native_checkpoint, expected)
    native_registration = read_json(args.native_checkpoint / 'pretrained_model/recovery_registration.json')
    if native_registration != {k: registration[k] for k in ('study', 'recipe_sha256', 'source_manifest')}:
        raise ValueError('Native checkpoint registration differs')
    scheduler, count, unused = verify_counters(args.native_checkpoint, stop, read_json(args.training_evidence / 'trainability.json'))
    native_sha = expected['pretrained_model/model.safetensors']['sha256']
    if native_sha != next(row['checkpoint_sha256'] for row in journal if row['step'] == stop):
        raise ValueError('Latest native weights differ from the completed journal entry')
    result = dict(purpose='registered_query_verified_milestone', study=registration['study'],
        recipe_sha256=registration['recipe_sha256'], training_run_id=training['run_id'],
        training_source_manifest=training['source_manifest'], initial_checkpoint_sha256=registration['initial_checkpoint_sha256'],
        checkpoint_sha256=selected['checkpoint_sha256'], selected_step=selected['step'],
        heldout_arm_mse=selected['heldout_arm_mse'], heldout_gripper_error=selected['heldout_gripper_error'],
        selected_validation_run_id=validation['run_id'], selected_local_weights_verified=True,
        development_run_id=development['run_id'], development_summary=development['summary'],
        gpu=development['environment']['gpu'], buffer_precision=development['model']['buffer_precision'],
        loader_sha256=development['loader_implementation_sha256'], evaluator_sha256=development['evaluator_implementation_sha256'],
        initial_states_manifest_sha256=development['initial_states_manifest_sha256'],
        episodes_csv_sha256=file_hash(args.development_run / 'episodes.csv'),
        native_step=stop, native_resume_checkpoint_sha256=native_sha,
        selected_is_latest_native=selected['step'] == stop,
        native_export_files=expected, native_export_verified_files=len(expected),
        native_optimizer_parameter_states=count, native_optimizer_registered_parameters=count + len(unused),
        native_unpopulated_parameter_names=unused, scheduler_last_epoch=scheduler,
        exporter_sha256=file_hash(__file__),
        limitations='Ten development states are diagnostic, not final acceptance. Selected policy and latest native resume state are independently identified; repeated selected weights are not new training outcomes. Training source is declared through registered checkpoint lineage; rollout plugin source is not independently logged. Coupled amendments do not isolate architecture or numerical causes.')
    write_json(args.output, result)
    print({k: result[k] for k in ('native_step', 'selected_step', 'selected_is_latest_native', 'development_summary')})


if __name__ == '__main__':
    main()
