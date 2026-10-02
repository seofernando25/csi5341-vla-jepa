"""Scientific evidence guards using the saved real diagnostic cohort."""

import copy

import pytest

from evaluation.common import ROOT, read_json
from evaluation.endpoint_task_analysis import analyze


def records():
    root = ROOT / 'studies/recovery/diagnostics'
    return [read_json(root / n) for n in ('b16_stochasticity.json',
            'rgb_20k_native_stochasticity.json', 'task_instruction_mapping.json')]


def test_task_ids_are_joined_by_instruction_mapping():
    baseline, endpoint, mapping = records()
    tasks = analyze(baseline, endpoint, mapping)
    task6 = tasks[6]
    # Dataset task0 is simulator task6, not simulator task0.
    assert task6['dataset_task_id'] == 0
    expected = [r for r in endpoint['frames'] if r['split'] == 'heldout' and r['task_id'] == 0]
    assert task6['endpoint']['frames'] == [
        {k: r[k] for k in ('episode', 'frame', 'valid_actions')} for r in expected]
    assert len(tasks) == 10


@pytest.mark.parametrize('field,value', [('task_id', 99), ('valid_actions', 999), ('seeds', [1])])
def test_mismatched_frame_membership_is_rejected(field, value):
    baseline, endpoint, mapping = records()
    endpoint = copy.deepcopy(endpoint)
    endpoint['frames'][0][field] = value
    with pytest.raises(ValueError, match='membership'):
        analyze(baseline, endpoint, mapping)


def test_invalid_saved_decomposition_is_rejected():
    baseline, endpoint, mapping = records()
    endpoint['frames'][0]['within_draw_arm_variance'] += 1
    with pytest.raises(ValueError, match='decomposition'):
        analyze(baseline, endpoint, mapping)


def test_nonbijective_task_mapping_is_rejected():
    baseline, endpoint, mapping = records()
    mapping['mapping'][0]['libero_task_id'] = mapping['mapping'][1]['libero_task_id']
    with pytest.raises(ValueError, match='bijection'):
        analyze(baseline, endpoint, mapping)
