"""Reject scientifically mismatched pairing using saved actual query outcomes."""

import copy

import pytest

from evaluation.common import ROOT, read_json
from evaluation.paired_development_analysis import analyze


def context():
    saved = read_json(ROOT / 'studies/recovery/diagnostics/query_500_2000_development_pair.json')
    records = copy.deepcopy(saved['input_records'])
    episodes = []
    for side, record in zip(('left', 'right'), records):
        record.update(environment=saved['environment'], protocol=saved['protocol'],
                      initial_states_manifest_sha256=saved['initial_states_manifest_sha256'])
        episodes.append([{**{key: row[key] for key in
                             ('task_id', 'task_name', 'trial', 'initial_state_hash', 'seed')},
                          'variant': record['variant'], 'phase': 'development', 'status': 'completed',
                          'success': row[side + '_success'], 'steps': row[side + '_steps']}
                         for row in saved['episodes']])
    return records, episodes


def test_actual_query_gains_preserve_both_prior_successes():
    result = analyze(*context())
    assert result['transitions'] == dict(both_success=2, gained_success=2, lost_success=0, both_failure=6)
    assert [int(row['task_id']) for row in result['episodes'] if row['transition'] == 'gained_success'] == [1, 3]


@pytest.mark.parametrize('field,value', [('seed', 42), ('initial_state_hash', 'different-state')])
def test_different_state_or_seed_cannot_be_paired(field, value):
    records, episodes = context()
    episodes[1][0][field] = value
    with pytest.raises(ValueError, match='membership'):
        analyze(records, episodes)


@pytest.mark.parametrize('field,value', [('buffer_precision', 'legacy'),
                                       ('num_inference_timesteps', 8), ('world_model_loaded', False)])
def test_changed_inference_semantics_are_rejected(field, value):
    records, episodes = context()
    records[1]['model'][field] = value
    with pytest.raises(ValueError, match='inference|solver'):
        analyze(records, episodes)


def test_different_gpu_is_not_a_matched_comparison():
    records, episodes = context()
    records[1]['environment'] = copy.deepcopy(records[1]['environment'])
    records[1]['environment']['gpu'] = 'other GPU'
    with pytest.raises(ValueError, match='environment'):
        analyze(records, episodes)


def test_duplicated_task_is_rejected():
    records, episodes = context()
    episodes[1][1]['task_id'] = episodes[1][0]['task_id']
    with pytest.raises(ValueError, match='exactly once'):
        analyze(records, episodes)


def test_summary_must_agree_with_episode_outcomes():
    records, episodes = context()
    records[1]['summary']['successes'] = 10
    with pytest.raises(ValueError, match='summary'):
        analyze(records, episodes)
