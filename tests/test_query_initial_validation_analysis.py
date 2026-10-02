"""Evidence guards for the actual full-set initializer comparison."""

import copy

import pytest

from evaluation.common import ROOT, read_json
from evaluation.query_initial_validation_analysis import compare


def context():
    root = ROOT / 'studies/recovery'
    return (read_json(root / 'diagnostics/query_initial_validation.json'),
            read_json(root / 'query_initial_validation_registration.json'))


def test_measured_initializer_scores_match_and_learning_regresses():
    result = compare(*context())
    assert result['initialization_batch_scores_exactly_equal']
    assert result['tested_batches'] == 25
    assert result['physical_arm_mse_ratio_at500'] == pytest.approx(1.6888762716248056)


def test_partial_validation_cannot_support_full_set_claim():
    data, registration = context()
    data['cases'][1]['batches'].pop()
    with pytest.raises(ValueError, match='incomplete'):
        compare(data, registration)


def test_wrong_checkpoint_is_rejected():
    data, registration = context()
    data['cases'][2]['checkpoint_sha256'] = registration['parent_checkpoint_sha256']
    with pytest.raises(ValueError, match='restoration'):
        compare(data, registration)


def test_summary_must_be_recomputed_from_batch_measurements():
    data, registration = context()
    data['cases'][2]['summary']['arm_mse'] = 0
    with pytest.raises(ValueError, match='summary'):
        compare(data, registration)


def test_equality_is_measured_instead_of_forced():
    data, registration = context()
    data['cases'][1] = copy.deepcopy(data['cases'][1])
    data['cases'][1]['batches'][0]['arm_squared_error_sum'] += .01
    data['cases'][1]['summary']['arm_mse'] = sum(
        row['arm_squared_error_sum'] for row in data['cases'][1]['batches']) / (6 * 1303)
    assert not compare(data, registration)['initialization_batch_scores_exactly_equal']
