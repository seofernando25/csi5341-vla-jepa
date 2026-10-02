"""Guard attribution when a completed stage evaluates an older selected policy."""

import copy
import json

import pytest

from evaluation.common import ROOT, file_hash, read_json
from evaluation.query_milestone import NATIVE_FILES, context, selected_entry, validation_score, verify_counters, verify_files


def journal():
    saved = read_json(ROOT / 'studies/recovery/diagnostics/query_2000_milestone.json')
    return [{'step': step, 'checkpoint_sha256': saved['checkpoint_sha256'] if step == 2000 else f'other-{step}',
             'heldout_arm_mse': saved['heldout_arm_mse'] if step == 2000 else .08,
             'heldout_gripper_error': saved['heldout_gripper_error'],
             'heldout_valid_actions': 1303, 'retained': step == 2000}
            for step in range(500, 5001, 500)]


def test_older_policy_can_be_selected_after_more_updates():
    rows = journal()
    development = {'checkpoint_sha256': rows[3]['checkpoint_sha256']}
    chosen = selected_entry(rows, 5000, development)
    assert chosen['step'] == 2000
    assert chosen['checkpoint_sha256'] != rows[-1]['checkpoint_sha256']


def test_development_cannot_substitute_latest_weights_for_selected_weights():
    rows = journal()
    with pytest.raises(ValueError, match='held-out-selected'):
        selected_entry(rows, 5000, {'checkpoint_sha256': rows[-1]['checkpoint_sha256']})


def test_equal_error_tie_selects_earlier_weights():
    rows = journal()
    rows[-1]['heldout_arm_mse'] = rows[3]['heldout_arm_mse']
    rows[-1]['retained'] = True
    assert selected_entry(rows, 5000, {'checkpoint_sha256': rows[3]['checkpoint_sha256']})['step'] == 2000


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_count', 'nonfinite'])
def test_invalid_selection_evidence_is_rejected(mutation):
    rows = journal()
    development = {'checkpoint_sha256': rows[3]['checkpoint_sha256']}
    if mutation == 'missing':
        rows.pop(0)
    elif mutation == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'wrong_count':
        rows[0]['heldout_valid_actions'] = 1302
    else:
        rows[0]['heldout_arm_mse'] = float('nan')
    with pytest.raises(ValueError):
        selected_entry(rows, 5000, development)


def production_context():
    registration = read_json(ROOT / 'studies/recovery/query_recovery_registration.json')
    recipe = read_json(ROOT / 'evaluation/query_recovery_config.json')
    record = {key: registration[key] for key in ('study', 'recipe_sha256', 'source_manifest', 'initial_checkpoint_sha256')}
    record.update(status='completed', purpose='recovery_adaptation', recipe=recipe,
                  split_sha256=file_hash(ROOT / 'studies/evaluation/training_split.json'))
    return record, registration, recipe


@pytest.mark.parametrize('field,value', [('status', 'running'), ('purpose', 'engineering'),
                                       ('source_manifest', {}), ('initial_checkpoint_sha256', 'wrong')])
def test_nonproduction_or_changed_lineage_rejected(field, value):
    record, registration, recipe = production_context()
    record[field] = value
    with pytest.raises(ValueError, match='production'):
        context(record, registration, recipe)


def validation_rows():
    selected = journal()[3]
    rows = []
    for batch in range(25):
        count = 52 if batch < 24 else 55
        rows.extend([{'phase': 'validation', 'step': 2000, 'batch': batch, 'samples': 8,
                      'loss': .2, 'action_loss': .08, 'wm_loss': .12},
                     {'phase': 'validation_action', 'step': 2000, 'batch': batch, 'valid_actions': count,
                      'arm_squared_error_sum': selected['heldout_arm_mse'] * 6 * count,
                      'gripper_errors': 0}])
    selected['heldout_gripper_error'] = 0
    return rows, selected


def test_registered_score_must_equal_sample_weighted_validation():
    rows, selected = validation_rows()
    validation_score(rows, selected)
    rows[1]['arm_squared_error_sum'] *= 2
    with pytest.raises(ValueError, match='actual validation'):
        validation_score(rows, selected)


def test_incomplete_validation_is_rejected_even_with_matching_summary():
    rows, selected = validation_rows()
    rows.pop()
    with pytest.raises(ValueError, match='incomplete'):
        validation_score(rows, selected)


def test_every_exported_file_is_rehashed_and_all_thirteen_required(tmp_path):
    expected = {}
    for name in NATIVE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
        expected[name] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    verify_files(tmp_path, expected)
    path = tmp_path / 'training_state/optimizer_state.safetensors'
    path.write_bytes(b'altered!')  # Same byte count, different content.
    with pytest.raises(ValueError, match='bytes differ'):
        verify_files(tmp_path, expected)
    expected.pop('training_state/optimizer_state.safetensors')
    with pytest.raises(ValueError, match='thirteen'):
        verify_files(tmp_path, expected)


def native_state(tmp_path):
    import torch
    from safetensors.torch import save_file
    directory = tmp_path / 'training_state'
    directory.mkdir()
    (directory / 'training_step.json').write_text(json.dumps(
        {'step': 5000, 'batch_size': 8, 'dp_world_size': 1, 'grad_accum_steps': 1}))
    (directory / 'scheduler_state.json').write_text(json.dumps({'last_epoch': 5000, '_step_count': 5001}))
    (directory / 'optimizer_param_groups.json').write_text(json.dumps([
        {'name': 'adapter_action_world', 'params': list(range(405))},
        {'name': 'smol_decoder', 'params': list(range(405, 694))}]))
    trainability = {f'model.adapter.{i}': {'elements': 1} for i in range(405)}
    trainability.update({f'model.qwen.model.layer.{i}': {'elements': 1} for i in range(288)})
    trainability['model.qwen.model.model.text_model.norm.weight'] = {'elements': 1}
    tensors = {f'state/{i}/{key}': torch.tensor([5000. if key == 'step' else .1])
               for i in range(693) for key in ('step', 'exp_avg', 'exp_avg_sq')}
    save_file(tensors, directory / 'optimizer_state.safetensors')
    return trainability, tensors


def test_unused_post_capture_norm_is_distinct_from_populated_optimizer_states(tmp_path):
    trainability, _ = native_state(tmp_path)
    epoch, populated, unused = verify_counters(tmp_path, 5000, trainability)
    assert (epoch, populated, unused) == (5000, 693, ['model.qwen.model.model.text_model.norm.weight'])


@pytest.mark.parametrize('failure', ['counter', 'missing', 'precision'])
def test_wrong_or_missing_optimizer_state_is_rejected(tmp_path, failure):
    import torch
    from safetensors.torch import save_file
    trainability, tensors = native_state(tmp_path)
    if failure == 'counter':
        tensors['state/405/step'] = torch.tensor([4999.])
    elif failure == 'missing':
        del tensors['state/405/exp_avg_sq']
    else:
        tensors['state/405/exp_avg'] = tensors['state/405/exp_avg'].bfloat16()
    save_file(tensors, tmp_path / 'training_state/optimizer_state.safetensors')
    with pytest.raises(ValueError, match='counter|membership|precision'):
        verify_counters(tmp_path, 5000, trainability)
