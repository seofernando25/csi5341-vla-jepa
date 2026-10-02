"""Synthetic supervisor evidence guards; these fixtures are never study results."""

import json

import pytest

from evaluation.common import file_hash
from evaluation.query_milestone import NATIVE_FILES
from scripts import local_lr_pair as supervisor


def completed_state(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    base = tmp_path / 'train'
    final = base / 'checkpoints/001000'
    for name in NATIVE_FILES:
        file = final / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'synthetic fixture')
    rows = [dict(step=500, heldout_arm_mse=.04, retained=False, optimizer_updates=250),
            dict(step=1000, heldout_arm_mse=.03, retained=True, optimizer_updates=500,
                 checkpoint_sha256=file_hash(final / 'pretrained_model/model.safetensors'))]
    (base / 'recovery_checkpoints.json').write_text(json.dumps(rows))
    return base, final, rows


def test_pruned_nonselected_validation_remains_in_comparison(tmp_path, monkeypatch):
    _, final, _ = completed_state(tmp_path, monkeypatch)
    result = supervisor.complete_branch('recipe-fixture', {'stop_microsteps': [1000]}, [(1000, final)])
    assert result['selected']['step'] == 1000
    assert result['optimizer_updates'] == 500
    assert len(result['latest_native_files']) == 13


def test_pruned_best_weights_cannot_be_substituted(tmp_path, monkeypatch):
    base, final, rows = completed_state(tmp_path, monkeypatch)
    rows[0]['heldout_arm_mse'] = .01
    (base / 'recovery_checkpoints.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='incorrectly pruned'):
        supervisor.complete_branch('recipe-fixture', {'stop_microsteps': [1000]}, [(1000, final)])


def test_partial_native_stop_is_not_completed(tmp_path):
    assert supervisor.complete_branch('recipe-fixture', {'stop_microsteps': [1000]}, [(900, tmp_path)]) is None


def test_selected_policy_hash_must_match_journal(tmp_path, monkeypatch):
    _, final, _ = completed_state(tmp_path, monkeypatch)
    (final / 'pretrained_model/model.safetensors').write_bytes(b'changed same size')
    with pytest.raises(ValueError, match='policy bytes'):
        supervisor.complete_branch('recipe-fixture', {'stop_microsteps': [1000]}, [(1000, final)])
