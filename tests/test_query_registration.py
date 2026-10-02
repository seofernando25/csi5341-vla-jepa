import json

import pytest

from evaluation.common import ROOT, file_hash
from evaluation.query_registration import require_completed_parent
from evaluation import query_registration


@pytest.mark.parametrize('status', ['training', 'development', 'benchmark', 'failed'])
def test_live_or_failed_parent_cannot_initialize_new_study(status):
    with pytest.raises(ValueError, match='completed original'):
        require_completed_parent({'status': status, 'study': 'n0008-rgb-recovery-r1'}, {})


def test_terminal_metadata_without_verified_native_export_is_insufficient():
    job = {'status': 'completed', 'study': 'n0008-rgb-recovery-r1',
           'recipe_sha256': file_hash(ROOT / 'evaluation/recovery_config.json'),
           'export_files': {'outputs/recovery/training/model.safetensors': {'bytes': 1, 'sha256': 'absent'}}}
    with pytest.raises(ValueError, match='locally verified'):
        require_completed_parent(job, {'status': 'monitoring'})


def test_export_manifest_cannot_escape_recovery_storage():
    job = {'status': 'completed', 'study': 'n0008-rgb-recovery-r1',
           'recipe_sha256': file_hash(ROOT / 'evaluation/recovery_config.json'),
           'export_files': {'../outside': {'bytes': 1, 'sha256': 'absent'}},
           'selected_checkpoint': 'outputs/recovery/training/x/000001/pretrained_model'}
    with pytest.raises(ValueError, match='escapes recovery'):
        require_completed_parent(job, {'status': 'verified_completion_review', 'verified_files': 1})


def test_native_export_claim_with_missing_weights_is_rejected():
    job = {'status': 'completed', 'study': 'n0008-rgb-recovery-r1',
           'recipe_sha256': file_hash(ROOT / 'evaluation/recovery_config.json'),
           'export_files': {'outputs/recovery/training/absent/model.safetensors': {'bytes': 1, 'sha256': 'absent'}},
           'selected_checkpoint': 'outputs/recovery/training/absent/pretrained_model'}
    with pytest.raises(ValueError, match='missing or differs'):
        require_completed_parent(job, {'status': 'verified_completion_review', 'verified_files': 1})


def test_registration_uses_heldout_selected_parent_and_requires_full_state(tmp_path, monkeypatch):
    monkeypatch.setattr(query_registration, 'ROOT', tmp_path)
    def write(relative, value):
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p
    recipe = file_hash(write('evaluation/recovery_config.json', {'study': 'n0008-rgb-recovery-r1'}))
    source = {'src/policy.py': 'source-hash'}
    write('studies/recovery/diagnostics/image_pipeline_amendment.json', {'source_manifest': source})
    base = 'outputs/recovery/training/parent/train/checkpoints'
    files, history = {}, []
    for step, mse in [(19500, .1), (20000, .2)]:
        checkpoint = f'{base}/{step:06d}'
        for name in query_registration.REQUIRED_NATIVE_FILES:
            value = ({'recipe_sha256': recipe, 'source_manifest': source}
                     if name.endswith('recovery_registration.json') else {'fixture': name})
            relative = f'{checkpoint}/{name}'
            p = write(relative, value)
            files[relative] = {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
        history.append({'step': step, 'retained': True, 'heldout_arm_mse': mse, 'heldout_valid_actions': 1303,
                        'checkpoint_sha256': files[f'{checkpoint}/pretrained_model/model.safetensors']['sha256']})
    evidence = 'studies/recovery/training/parent'
    write(f'{evidence}/run.json', {'status': 'completed', 'completed_steps': 20000, 'recipe_sha256': recipe})
    write(f'{evidence}/checkpoints.json', history)
    job = {'status': 'completed', 'study': 'n0008-rgb-recovery-r1', 'recipe_sha256': recipe,
           'export_files': files, 'selected_checkpoint': f'{base}/019500/pretrained_model',
           'latest_checkpoint': f'{base}/020000/pretrained_model', 'training_evidence': evidence,
           'stages': [{'completed_step': 20000, 'selected_step': 19500, 'heldout_arm_mse': .1,
                       'selected_checkpoint_sha256': history[0]['checkpoint_sha256']}]}
    export = {'status': 'verified_completion_review', 'verified_files': len(files)}
    parent, best = require_completed_parent(job, export)
    assert str(parent) == job['selected_checkpoint'] and best['step'] == 19500
    del files[f'{base}/019500/training_state/optimizer_state.safetensors']
    export['verified_files'] = len(files)
    with pytest.raises(ValueError, match='complete native resume state'):
        require_completed_parent(job, export)
