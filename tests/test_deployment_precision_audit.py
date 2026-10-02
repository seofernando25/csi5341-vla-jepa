"""CPU evidence guards; synthetic fixtures are never experimental results."""

import copy

import pytest
import torch
from safetensors.torch import save_model

from evaluation.deployment_precision_audit import config_identity, load_case, selection_context, summarize


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.master = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
        self.frozen = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.frozen.requires_grad_(False)
        self.rotary_emb = torch.nn.Module()
        frequencies = torch.tensor([1.0, .123456789], dtype=torch.float32)
        self.rotary_emb.register_buffer('inv_freq', frequencies.clone(), persistent=False)
        self.rotary_emb.register_buffer('original_inv_freq', frequencies.clone(), persistent=False)
        with torch.no_grad():
            self.master.weight.fill_(.123456789)


@pytest.mark.parametrize('mode', ['native_masters', 'bf16_deployment'])
def test_preserves_declared_weight_values_and_native_rope(tmp_path, mode):
    original = TinyPolicy()
    save_model(original, tmp_path / 'model.safetensors')
    policy = TinyPolicy()
    # Start with rounded construction weights, as the real BF16 backbone does.
    policy.master.to(torch.bfloat16)
    proof = load_case(policy, tmp_path, {'master.weight': {'elements': 4}}, mode)
    assert proof['declared_weight_values_verified']
    assert torch.equal(policy.rotary_emb.inv_freq, original.rotary_emb.inv_freq)
    assert all(not p.requires_grad for p in policy.parameters())
    if mode == 'native_masters':
        assert policy.master.weight.dtype == torch.float32
        assert torch.equal(policy.master.weight, original.master.weight)
        assert not torch.equal(policy.master.weight, original.master.weight.bfloat16().float())
        assert proof['fp32_master_tensors'] == 1
    else:
        assert policy.master.weight.dtype == torch.bfloat16
        assert torch.equal(policy.master.weight, original.master.weight.bfloat16())
        assert proof['fp32_master_tensors'] == 0


@pytest.mark.parametrize('fault', ['trainability', 'shape', 'extra_float_buffer', 'rounded_rope', 'rounded_checkpoint', 'mode'])
def test_unregistered_precision_or_membership_is_rejected(tmp_path, fault):
    original = TinyPolicy()
    policy = TinyPolicy()
    manifest = {'master.weight': {'elements': 4}}
    mode = 'native_masters'
    if fault == 'trainability':
        policy.frozen.requires_grad_(True)
    elif fault == 'shape':
        manifest['master.weight']['elements'] = 5
    elif fault == 'extra_float_buffer':
        policy.register_buffer('unexpected_buffer', torch.ones(2), persistent=False)
    elif fault == 'rounded_rope':
        policy.rotary_emb.inv_freq = policy.rotary_emb.inv_freq.bfloat16()
    elif fault == 'rounded_checkpoint':
        original.master.to(torch.bfloat16)
    else:
        mode = 'unknown'
    save_model(original, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError):
        load_case(policy, tmp_path, manifest, mode)


def fixture():
    registration = {'query_study': 'fixture', 'query_recipe_sha256': 'recipe',
                    'source_manifest': {'fixture.py': 'source'}, 'loader_sha256': 'loader'}
    journal = [{'step': step, 'checkpoint_sha256': f'policy-{step}', 'heldout_arm_mse': .03 + step / 1e6,
                'heldout_gripper_error': .1, 'heldout_valid_actions': 1303, 'retained': True}
               for step in range(500, 10001, 500)]
    milestone = {'purpose': 'registered_query_verified_milestone', 'study': 'fixture',
                 'recipe_sha256': 'recipe', 'training_source_manifest': registration['source_manifest'],
                 'native_step': 10000, 'scheduler_last_epoch': 10000,
                 'selected_local_weights_verified': True, 'native_export_verified_files': 13,
                 'native_optimizer_parameter_states': 693, 'native_optimizer_registered_parameters': 694,
                 'buffer_precision': 'native_rope', 'loader_sha256': 'loader',
                 'checkpoint_sha256': 'policy-500', 'selected_step': 500,
                 'heldout_arm_mse': .0305, 'heldout_gripper_error': .1}
    return milestone, journal, registration


def test_final_native_state_does_not_force_latest_policy_selection():
    milestone, journal, registration = fixture()
    assert selection_context(milestone, journal, registration)['step'] == 500
    milestone['development_summary'] = {'successes': 0}
    assert selection_context(milestone, journal, registration)['step'] == 500


@pytest.mark.parametrize('fault', ['partial_stage', 'partial_journal', 'different_source', 'loader', 'wrong_selection', 'score', 'unverified'])
def test_incomplete_or_reselected_study_is_rejected(fault):
    milestone, journal, registration = fixture()
    if fault == 'partial_stage':
        milestone['native_step'] = 5000
    elif fault == 'partial_journal':
        journal.pop(3)
    elif fault == 'different_source':
        milestone['training_source_manifest'] = {}
    elif fault == 'loader':
        milestone['loader_sha256'] = 'new-loader'
    elif fault == 'wrong_selection':
        milestone['checkpoint_sha256'] = 'policy-10000'
    elif fault == 'score':
        milestone['heldout_arm_mse'] = .01
    else:
        milestone['selected_local_weights_verified'] = False
    with pytest.raises(ValueError):
        selection_context(milestone, journal, registration)


def test_complete_counts_and_loss_are_required():
    rows = [{'batch': i, 'samples': 8, 'valid_actions': 52 if i < 24 else 55,
             'loss': .3, 'action_loss': .1, 'wm_loss': .2,
             'arm_squared_error_sum': 6 * (52 if i < 24 else 55) * .03, 'gripper_errors': 0}
            for i in range(25)]
    assert summarize(rows)['arm_mse'] == pytest.approx(.03)
    for fault in ('partial', 'duplicate', 'counts', 'nonfinite', 'loss'):
        broken = copy.deepcopy(rows)
        if fault == 'partial':
            broken.pop()
        elif fault == 'duplicate':
            broken[-1]['batch'] = 0
        elif fault == 'counts':
            broken[-1]['valid_actions'] = 54
        elif fault == 'nonfinite':
            broken[-1]['arm_squared_error_sum'] = float('nan')
        else:
            broken[-1]['loss'] = .4
        with pytest.raises(ValueError):
            summarize(broken)


def test_resume_path_is_allowed_but_prediction_config_is_protected():
    config_identity({'pretrained_path': 'older-stage', 'chunk_size': 7},
                    {'pretrained_path': 'later-stage', 'chunk_size': 7})
    with pytest.raises(ValueError):
        config_identity({'pretrained_path': 'older-stage', 'chunk_size': 7},
                        {'pretrained_path': 'later-stage', 'chunk_size': 8})
