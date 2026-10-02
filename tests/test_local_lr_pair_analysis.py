"""Synthetic scientific-evidence guards; fixtures never become measured results."""

import copy

import pytest

from evaluation.local_lr_pair_analysis import anchor_context, merge_streams, record_context, score, select_policy


def validation_fixture(step=500, mse=.03):
    rows = []
    for batch in range(25):
        count = 52 if batch < 24 else 55
        rows.extend([{'phase': 'validation', 'step': step, 'batch': batch, 'samples': 8,
                      'loss': .2, 'action_loss': .1, 'wm_loss': .1},
                     {'phase': 'validation_action', 'step': step, 'batch': batch, 'valid_actions': count,
                      'arm_squared_error_sum': 6 * count * mse, 'gripper_errors': 0}])
    entry = {'checkpoint_sha256': 'synthetic-policy', 'heldout_arm_mse': mse, 'heldout_gripper_error': 0}
    return rows, entry


def test_native_microsteps_are_not_optimizer_updates():
    rows, entry = validation_fixture()
    result = score(rows, 500, entry)
    assert result['optimizer_updates'] == 250
    assert result['native_microsteps'] == 500
    assert result['valid_actions'] == 1303


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'wrong_score', 'wrong_membership', 'nonfinite', 'wrong_loss'])
def test_partial_or_unattributable_validation_is_rejected(change):
    rows, entry = validation_fixture()
    if change == 'partial':
        rows = rows[:-2]
    elif change == 'duplicate':
        rows.append(copy.deepcopy(rows[-1]))
    elif change == 'wrong_score':
        entry['heldout_arm_mse'] = .02
    elif change == 'wrong_membership':
        rows[-1]['valid_actions'] -= 1
    elif change == 'nonfinite':
        rows[-1]['arm_squared_error_sum'] = float('nan')
    else:
        rows[0]['loss'] = .3
    with pytest.raises(ValueError):
        score(rows, 500, entry)


def anchor_fixture():
    rows, entry = validation_fixture(step=0)
    batches = [dict(rows[i], **{k: v for k, v in rows[i + 1].items() if k not in {'phase', 'step'}})
               for i in range(0, len(rows), 2)]
    for row in batches:
        row.pop('phase')
        row.pop('step')
    flags = {'deterministic_algorithms': True, 'matmul_allow_tf32': True, 'cudnn_allow_tf32': True,
             'cudnn_benchmark': False, 'cudnn_deterministic': True}
    registration = {'source_manifest': {'synthetic.py': 'fixture'}, 'gpu': 'synthetic GPU',
                    'initial_checkpoint_sha256': entry['checkpoint_sha256']}
    anchor_reg = {'diagnostic_sha256': 'fixture-diagnostic'}
    anchor = {'status': 'completed', 'source_manifest': registration['source_manifest'],
              'environment': {'gpu': registration['gpu']}, 'numerical_flags': flags.copy(),
              'diagnostic_sha256': anchor_reg['diagnostic_sha256'],
              'cases': [{'mode': 'parent_zero_query_native_all32', 'checkpoint_sha256': entry['checkpoint_sha256'],
                         'exact_inherited_values': True, 'zero_queries': True, 'batches': batches,
                         'summary': {'arm_mse': .03, 'gripper_error': 0, 'loss': .2,
                                     'frames': 200, 'valid_actions': 1303}}]}
    return anchor, registration, anchor_reg, flags


def test_parent_anchor_uses_the_same_complete_validation():
    anchor, registration, anchor_reg, flags = anchor_fixture()
    result = anchor_context(anchor, registration, anchor_reg, flags)
    assert result['optimizer_updates'] == 0
    assert result['arm_mse'] == pytest.approx(.03)


@pytest.mark.parametrize('change', ['tf32', 'source', 'partial', 'checkpoint', 'nonzero_query', 'loss', 'failed'])
def test_unmatched_parent_is_not_a_selection_comparator(change):
    anchor, registration, anchor_reg, flags = anchor_fixture()
    if change == 'tf32':
        anchor['numerical_flags']['matmul_allow_tf32'] = False
    elif change == 'source':
        anchor['source_manifest'] = {'other.py': 'fixture'}
    elif change == 'partial':
        anchor['cases'][0]['batches'].pop()
    elif change == 'checkpoint':
        anchor['cases'][0]['checkpoint_sha256'] = 'other-policy'
    elif change == 'nonzero_query':
        anchor['cases'][0]['zero_queries'] = False
    elif change == 'loss':
        anchor['cases'][0]['summary']['loss'] = .25
    else:
        anchor['status'] = 'failed'
    with pytest.raises(ValueError):
        anchor_context(anchor, registration, anchor_reg, flags)


def test_parent_stays_selected_when_training_regresses():
    parent = dict(arm_mse=.02, optimizer_updates=0)
    branches = {'high': [dict(arm_mse=.04, optimizer_updates=250)],
                'low': [dict(arm_mse=.03, optimizer_updates=500)]}
    assert select_policy(parent, branches)['branch'] == 'parent'


def test_selection_uses_arm_error_then_exposure_then_branch_not_gripper():
    parent = dict(arm_mse=.03, optimizer_updates=0)
    branches = {'low': [dict(arm_mse=.02, optimizer_updates=500, gripper_error=0)],
                'high': [dict(arm_mse=.02, optimizer_updates=250, gripper_error=.9),
                         dict(arm_mse=.02, optimizer_updates=500, gripper_error=0)]}
    assert select_policy(parent, branches)['branch'] == 'high'
    branches['low'].append(dict(arm_mse=.02, optimizer_updates=250))
    assert select_policy(parent, branches)['branch'] == 'high'
    parent['arm_mse'] = .02
    assert select_policy(parent, branches)['branch'] == 'parent'


def test_resumed_stream_discards_old_uncommitted_training_and_validation():
    before = [{'phase': 'training', 'step': step, 'loss': 1} for step in (1, 2, 3, 4)]
    before += [{'phase': 'validation', 'step': 4, 'batch': 0, 'loss': 99}]
    resumed = [{'phase': 'training', 'step': step, 'loss': 2} for step in (3, 4, 5, 6)]
    result = merge_streams([before, resumed])
    assert [r['step'] for r in result] == list(range(1, 7))
    assert result[2]['loss'] == 2


def test_duplicates_inside_one_stream_are_not_silently_overwritten():
    with pytest.raises(ValueError, match='Duplicate'):
        merge_streams([[dict(phase='training', step=1), dict(phase='training', step=1)]])


def record_fixture():
    registration = {'recipe_sha256': {'high': 'recipe-fixture'}, 'source_manifest': {'synthetic.py': 'fixture'},
                    'initial_checkpoint_sha256': 'parent-fixture', 'gpu': 'synthetic GPU',
                    'data_manifest_sha256': {'studies/evaluation/training_split.json': 'split-fixture'}}
    recipe = {'study': 'synthetic-high'}
    record = {'purpose': 'recovery_adaptation', 'study': recipe['study'], 'recipe': recipe,
              'recipe_sha256': 'recipe-fixture', 'source_manifest': registration['source_manifest'],
              'initial_checkpoint_sha256': 'parent-fixture', 'local_lr_pair_registration_sha256': 'pair-fixture',
              'split_sha256': 'split-fixture', 'native_step_unit': 'microbatch', 'gradient_accumulation_steps': 2,
              'effective_batch_size': 8, 'deterministic_algorithms': True, 'isolated_worker_rng': True,
              'requested_stop_step': 1000, 'environment': {'gpu': 'synthetic GPU'}}
    return record, registration, recipe


@pytest.mark.parametrize('key,value', [('purpose', 'engineering'), ('gradient_accumulation_steps', 1),
                                     ('native_step_unit', 'optimizer update'), ('split_sha256', 'other'),
                                     ('source_manifest', {}), ('initial_checkpoint_sha256', 'other'),
                                     ('local_lr_pair_registration_sha256', 'other')])
def test_other_studies_and_topologies_do_not_enter_the_pair(key, value):
    record, registration, recipe = record_fixture()
    record_context(record, registration, recipe, 'high', 'pair-fixture')
    record[key] = value
    with pytest.raises(ValueError):
        record_context(record, registration, recipe, 'high', 'pair-fixture')
