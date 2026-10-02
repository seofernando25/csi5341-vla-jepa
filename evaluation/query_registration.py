"""Register endpoint precision tests and full-decoder adaptation after verified r1 export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json

REQUIRED_NATIVE_FILES = ('pretrained_model/model.safetensors', 'pretrained_model/config.json',
                        'pretrained_model/train_config.json', 'pretrained_model/recovery_registration.json',
                        'pretrained_model/policy_preprocessor.json', 'pretrained_model/policy_postprocessor.json',
                        'pretrained_model/policy_preprocessor_step_4_normalizer_processor.safetensors',
                        'pretrained_model/policy_postprocessor_step_2_unnormalizer_processor.safetensors',
                        'training_state/optimizer_state.safetensors', 'training_state/optimizer_param_groups.json',
                        'training_state/rng_state.safetensors', 'training_state/scheduler_state.json',
                        'training_state/training_step.json')


def require_completed_parent(job, export):
    if (job.get('status') != 'completed' or job.get('study') != 'n0008-rgb-recovery-r1'
            or job.get('recipe_sha256') != file_hash(ROOT / 'evaluation/recovery_config.json')):
        raise ValueError('Only the completed original recovery study can initialize this follow-up')
    files = job.get('export_files', {})
    if (not files or export.get('status') != 'verified_completion_review'
            or export.get('verified_files') != len(files)):
        raise ValueError('Complete parent native state must be locally verified before registration')
    parent = Path(job['selected_checkpoint'])
    if parent.is_absolute() or '..' in parent.parts or not parent.is_relative_to('outputs/recovery/training'):
        raise ValueError('Parent checkpoint must remain inside recovery runtime storage')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.is_relative_to('outputs/recovery/training'):
            raise ValueError('Native export path escapes recovery storage')
        p = ROOT / relative
        if not p.is_file() or p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
            raise ValueError('Parent export is missing or differs from its native hash manifest')
    if str(parent / 'model.safetensors') not in files:
        raise ValueError('Held-out-selected weights are absent from the verified export')
    latest = Path(job['latest_checkpoint'])
    if latest.parent.parent != parent.parent.parent or int(latest.parent.name) != 20000:
        raise ValueError('Latest native checkpoint must complete this same 20k study')
    source = read_json(ROOT / 'studies/recovery/diagnostics/image_pipeline_amendment.json')['source_manifest']
    for checkpoint in {parent.parent, latest.parent}:
        if any(str(checkpoint / name) not in files for name in REQUIRED_NATIVE_FILES):
            raise ValueError('Both selected and latest checkpoints require complete native resume state')
        registration = read_json(ROOT / checkpoint / 'pretrained_model/recovery_registration.json')
        if registration['recipe_sha256'] != job['recipe_sha256'] or registration['source_manifest'] != source:
            raise ValueError('Exported native checkpoint belongs to a different recipe or source')
    stage = job['stages'][-1]
    if stage['completed_step'] != 20000:
        raise ValueError('Complete the registered 20k endpoint first')
    evidence = Path(job['training_evidence'])
    if evidence.is_absolute() or '..' in evidence.parts or not evidence.is_relative_to('studies/recovery/training'):
        raise ValueError('Parent training evidence escapes the registered study')
    record = read_json(ROOT / evidence / 'run.json')
    if (record.get('status') != 'completed' or record.get('completed_steps') != 20000
            or record.get('recipe_sha256') != job['recipe_sha256']):
        raise ValueError('Completed parent training evidence is missing')
    history = read_json(ROOT / evidence / 'checkpoints.json')
    best = min((r for r in history if r.get('retained') and 'heldout_arm_mse' in r),
               key=lambda r: (r['heldout_arm_mse'], r['step']))
    if (best['step'] != stage['selected_step'] or best['heldout_valid_actions'] != 1303
            or best['checkpoint_sha256'] != stage['selected_checkpoint_sha256']
            or best['heldout_arm_mse'] != stage['heldout_arm_mse']
            or int(parent.parent.name) != best['step']
            or file_hash(ROOT / parent / 'model.safetensors') != best['checkpoint_sha256']):
        raise ValueError('Parent must be selected by the registered held-out error, never development success')
    return parent, best


def main():
    paths = [ROOT / 'studies/recovery/completed_r1_native_rope_registration.json',
             ROOT / 'evaluation/query_full_decoder_preflight_config.json',
             ROOT / 'evaluation/query_recovery_config.json',
             ROOT / 'studies/recovery/query_recovery_registration.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve existing registrations; inspect rather than overwrite')
    job = read_json(ROOT / 'outputs/recovery/cloud/remote-job.json')
    export = read_json(ROOT / 'outputs/recovery/cloud/export-status.json')
    parent, best = require_completed_parent(job, export)
    registered = datetime.now(UTC).isoformat()
    original = read_json(ROOT / 'studies/recovery/diagnostics/image_pipeline_amendment.json')['source_manifest']
    precision = {'study': 'n0008-completed-r1-inference-precision', 'registered_at': registered,
                 'status': 'registered_before_endpoint_precision_rollouts', 'training_changed': False,
                 'variant_prefix': 'RGB-n0008-endpoint', 'checkpoint_sha256': best['checkpoint_sha256'],
                 'parent_recipe_sha256': job['recipe_sha256'], 'selected_recovery_step': best['step'],
                 'loader_sha256': file_hash(ROOT / 'evaluation/models.py'),
                 'evaluator_sha256': file_hash(ROOT / 'evaluation/run.py'),
                 'source_manifest': original, 'protocol_sha256': file_hash(ROOT / 'evaluation/protocol.json'),
                 'initial_states_sha256': file_hash(ROOT / 'studies/evaluation/initial_states.json'),
                 'amendment': 'Only preserve original FP32 nonpersistent rotary frequencies before inference '
                              'BF16 cast. Compare both modes on the same local RTX3090 and ten first '
                              'development states. Parent chosen exclusively by fixed held-out arm MSE. '
                              'Do not pool original cloud rollouts or select weights by these outcomes.'}
    production = read_json(ROOT / 'evaluation/recovery_config.json')
    production.update(study='n0008-query-decoder-recovery-r1',
                      initial_checkpoint_sha256=best['checkpoint_sha256'], unfreeze_last_n=32,
                      query_token_adaptation='input_residual', smol_gradient_checkpointing=True,
                      inference_buffer_precision='native_rope', isolate_data_loader_rng=True,
                      worker_seed=42013, deterministic_training=True,
                      decay_steps=10000, stage_stop_steps=[500, 2000, 5000, 10000],
                      scope='Separate recovery after held-out selection of completed corrected-input r1. '
                            'Joint input-query residual and all32 decoder layers; initial inherited values '
                            'preserved exactly, newly trainable masters upcast losslessly to FP32. '
                            'Fresh AdamW and fixed10k schedule. Preserve image processing, data/split, '
                            'normalization, losses, action/world architecture and initialization lineage, '
                            'prediction semantics, fixed200-frame validation and held-out selection. '
                            'Private worker generators/deterministic algorithms are explicit amendments. '
                            'No isolated query or decoder causal claim, no changes to original RSI/r1.')
    engineering = dict(production, study='n0008-query-full-decoder-engineering-preflight',
                       engineering_only=True, stage_stop_steps=[2, 4],
                       required_gpu='NVIDIA GeForce RTX 5090',
                       scope='Engineering only, at most20 native updates; identical production warm '
                             'hash, source, trainability, schedule and numerical settings. Exercise batch8 '
                             'native optimizer/save/resume and fixed200-frame validation on intendedRTX5090. '
                             'Excluded from production curves and checkpoint selection.')
    query_source = read_json(ROOT / 'studies/recovery/diagnostics/query_source_amendment.json')['source_manifest']
    for path, value in zip(paths[:3], (precision, engineering, production), strict=True):
        write_json(path, value)
    write_json(paths[3], {'study': production['study'], 'registered_at': registered,
               'status': 'registered_before_intended_gpu_preflight_and_production',
               'recipe_sha256': file_hash(paths[2]), 'engineering_recipe_sha256': file_hash(paths[1]),
               'registration_implementation_sha256': file_hash(__file__),
               'parent_recipe_sha256': job['recipe_sha256'], 'parent_selected_step': best['step'],
               'initial_checkpoint_sha256': best['checkpoint_sha256'], 'source_manifest': query_source,
               'hypothesis': 'Four-layer adaptation may be too restrictive for the Qwen-to-Smol conditioning '
                             'transfer. Train query-specific input representations and all decoder layers '
                             'without changing action/world mechanisms or losses.',
               'production_gate': 'IntendedRTX5090 native batch8 optimizer/save/resume with exact inherited '
                                  'value restoration, finite FP32 query/decoder moments, full fixed validation '
                                  'and measured update speed/memory. Follow-up is sequential after verified '
                                  'r1 export; no extra rental; originalUSD14 cap and October2 deadlines remain.',
               'selection': production['selection'],
               'limitations': 'Coupled query/trainability/numerical amendments, one seed and reduced sample '
                              'exposure versus the paper. No guarantee of useful control. Ten development '
                              'episodes are diagnostics; final matched success/memory target unchanged.'})
    print({'parent_checkpoint': str(parent), 'selected_step': best['step'],
           'registered_study': production['study'], 'production_started': False})


if __name__ == '__main__':
    main()
