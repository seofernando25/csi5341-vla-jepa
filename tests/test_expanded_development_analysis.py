"""Synthetic guard fixtures only; no fixture represents measured success."""

import copy

import pytest

from evaluation.common import ROOT, read_json
from evaluation.expanded_development_analysis import analyze


def fixture():
    registration = read_json(ROOT / 'studies/recovery/expanded_development_registration.json')
    states = read_json(ROOT / 'studies/evaluation/initial_states.json')
    protocol = read_json(ROOT / 'evaluation/protocol.json')
    records, episodes = [], []
    for side, name in enumerate(('B16', 'RGB-parent20k')):
        case = registration['variants'][name]
        rows = [{'variant': name + '-dev100', 'phase': 'development', 'status': 'completed',
                 'task_id': task['task_id'], 'trial': trial, 'task_name': task['name'],
                 'initial_state_hash': task['development_hashes'][trial],
                 'seed': protocol['seed'] + task['task_id'] * 1000 + trial,
                 'success': int(trial < (9 if side == 0 else 3)), 'steps': 100}
                for task in states['tasks'] for trial in range(10)]
        successes = sum(row['success'] for row in rows)
        records.append({'variant': name + '-dev100', 'status': 'completed', 'phase': 'development',
                        'experiment': 'rollout', 'checkpoint_sha256': case['checkpoint_sha256'],
                        'source_manifest': case['source_manifest'], 'protocol': protocol,
                        'initial_states_manifest_sha256': registration['initial_states_sha256'],
                        'loader_sha256': registration['loader_sha256'],
                        'evaluator_sha256': registration['evaluator_sha256'],
                        'environment': {'gpu': registration['gpu'], 'packages': {'synthetic_fixture': 'only'},
                                        'torch_cuda': 'fixture', 'driver': 'fixture', 'gpu_total_bytes': 24 * 2**30},
                        'model': {'buffer_precision': case['buffer_precision'], 'chunk_size': 7,
                                  'n_action_steps': 7, 'num_inference_timesteps': 4, 'world_model_loaded': True},
                        'summary': {'phase': 'development', 'episodes': 100, 'task_ids': list(range(10)),
                                    'successes': successes, 'task_macro_success': successes / 100},
                        'numerical_flags': {'fixture': True}, 'wrapper_sha256': 'fixture',
                        'registration_sha256': 'fixture'})
        episodes.append(rows)
    return records, episodes, registration, states, protocol


def test_all_hundred_paired_trials_count_once():
    result = analyze(*fixture())
    assert result['baseline_successes'] == 90
    assert result['parent_successes'] == 30
    assert result['paired_transitions'] == dict(both_success=30, baseline_only_success=60,
                                              parent_only_success=0, both_failure=10)
    assert len(result['episodes']) == 100
    assert all(row['episodes'] == 10 for row in result['per_task'])


@pytest.mark.parametrize('field,value', [('seed', -1), ('initial_state_hash', 'incorrect'), ('task_name', 'different')])
def test_mutually_matching_but_wrong_membership_is_rejected(field, value):
    args = fixture()
    for rows in args[1]:
        rows[0][field] = value
    with pytest.raises(ValueError, match='exact registered'):
        analyze(*args)


@pytest.mark.parametrize('field,value', [('checkpoint_sha256', 'other-weights'), ('source_manifest', {}),
                                       ('loader_sha256', 'other-loader')])
def test_unregistered_policy_or_implementation_is_rejected(field, value):
    args = fixture()
    args[0][1][field] = value
    with pytest.raises(ValueError, match='Registered model'):
        analyze(*args)


def test_partial_running_cohort_cannot_be_summarized():
    args = fixture()
    args[0][1]['status'] = 'running'
    with pytest.raises(ValueError, match='completed'):
        analyze(*args)


def test_final_trials_cannot_substitute_for_development():
    args = fixture()
    args[0][1]['phase'] = 'final'
    with pytest.raises(ValueError, match='development'):
        analyze(*args)


def test_duplicate_trial_rejected_even_with_hundred_rows():
    args = fixture()
    args[1][1][1] = copy.deepcopy(args[1][1][0])
    with pytest.raises(ValueError, match='membership'):
        analyze(*args)


def test_changed_environment_or_numerical_flags_rejected():
    args = fixture()
    args[0][1]['environment']['driver'] = 'different'
    with pytest.raises(ValueError, match='environment'):
        analyze(*args)
    args = fixture()
    args[0][1]['numerical_flags'] = {'fixture': False}
    with pytest.raises(ValueError, match='numerical'):
        analyze(*args)


def test_summary_cannot_replace_actual_episode_outcomes():
    args = fixture()
    args[0][1]['summary']['successes'] += 1
    with pytest.raises(ValueError, match='summary'):
        analyze(*args)
