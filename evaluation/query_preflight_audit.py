"""Verify completed native query optimizer/save/resume engineering evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from evaluation.common import ROOT, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'evaluation/query_preflight_config.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous engineering audits')
    expected = read_json(ROOT / 'studies/recovery/diagnostics/query_source_amendment.json')['source_manifest']
    recipe_path = args.recipe
    recipe = read_json(recipe_path)
    expected_states = {4: 441, 32: 693}.get(recipe['unfreeze_last_n'])
    if not recipe.get('engineering_only') or expected_states is None:
        raise ValueError('This audit covers registered four-layer or full-decoder engineering gates')
    recipe_hash = file_hash(recipe_path)
    accumulation = recipe.get('gradient_accumulation_steps', 1)
    stops = [4, 8] if accumulation == 2 else [2, 4]
    if (accumulation not in (1, 2) or recipe['batch_size'] * accumulation != 8
            or recipe.get('validation_batch_size', 8) != 8):
        raise ValueError('Require effective batch eight and validation batches eight')
    results, previous_hash = [], None
    for index, run_id in enumerate(args.runs):
        folder = ROOT / 'studies/recovery/training' / run_id
        record, initialization = read_json(folder / 'run.json'), read_json(folder / 'initialization.json')
        if (record['status'] != 'completed' or record['purpose'] != 'engineering_preflight'
                or record['recipe_sha256'] != recipe_hash or record['source_manifest'] != expected
                or not initialization['exact_restoration']):
            raise ValueError('Run is not completed registered engineering evidence')
        if recipe.get('required_gpu') and record['environment']['gpu'] != recipe['required_gpu']:
            raise ValueError('Native preflight did not run on the registered intended GPU')
        stop = record['completed_steps']
        if stop != stops[index]:
            raise ValueError('Engineering stage does not match its registered microstep boundary')
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        training = [r for r in rows if r['phase'] == 'training']
        validation = [r for r in rows if r['phase'] == 'validation']
        action = [r for r in rows if r['phase'] == 'validation_action']
        start = 0 if index == 0 else stops[index - 1]
        if ([r['step'] for r in training] != list(range(start + 1, stop + 1))
                or len(validation) != 25 or sum(r['samples'] for r in validation) != 200
                or len(action) != 25 or sum(r['valid_actions'] for r in action) != 1303
                or not all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in training)):
            raise ValueError('Missing finite native updates or complete fixed evaluation')
        checkpoint = args.checkpoint_root / f'{stop:06d}'
        required = ['pretrained_model/model.safetensors', 'pretrained_model/config.json',
                    'pretrained_model/train_config.json', 'pretrained_model/recovery_registration.json',
                    'pretrained_model/policy_preprocessor.json', 'pretrained_model/policy_postprocessor.json',
                    'training_state/optimizer_state.safetensors', 'training_state/optimizer_param_groups.json',
                    'training_state/rng_state.safetensors', 'training_state/scheduler_state.json',
                    'training_state/training_step.json']
        if not all((checkpoint / name).is_file() for name in required):
            raise ValueError('Incomplete native checkpoint')
        config = read_json(checkpoint / 'pretrained_model/config.json')
        if (config['query_token_adaptation'] != 'input_residual' or not config['smol_gradient_checkpointing']
                or config['unfreeze_last_n'] != recipe['unfreeze_last_n']):
            raise ValueError('Serialized checkpoint lost query configuration')
        model_hash = file_hash(checkpoint / 'pretrained_model/model.safetensors')
        if index == 1 and (record['resume_checkpoint_sha256'] != previous_hash
                           or initialization['checkpoint_sha256'] != previous_hash):
            raise ValueError('Resume did not exactly restore the first native checkpoint')
        with safe_open(checkpoint / 'pretrained_model/model.safetensors', framework='pt') as saved:
            query = saved.get_tensor('model.qwen.query_adapter.delta')
        if query.dtype != torch.float32 or query.shape != (4, 960) or not torch.isfinite(query).all() or not torch.count_nonzero(query):
            raise ValueError('Query residual was not updated and serialized as FP32')
        with safe_open(checkpoint / 'training_state/optimizer_state.safetensors', framework='pt') as saved:
            counters = [float(saved.get_tensor(k)) for k in saved.keys() if k.endswith('/step')]
            query_moments = {k: saved.get_tensor(k) for k in saved.keys()
                             if tuple(saved.get_slice(k).get_shape()) == (4, 960)}
        optimizer_updates = stop // accumulation
        if len(counters) != expected_states or set(counters) != {float(optimizer_updates)} or len(query_moments) != 2:
            raise ValueError('Native optimizer counters or query moments were not resumed')
        if not all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) and bool(torch.count_nonzero(v))
                   for v in query_moments.values()):
            raise ValueError('Query AdamW moments must be finite nonzero FP32 tensors')
        topology = read_json(checkpoint / 'training_state/training_step.json')
        scheduler = read_json(checkpoint / 'training_state/scheduler_state.json')
        if (topology['step'] != stop or topology['batch_size'] != recipe['batch_size']
                or topology['grad_accum_steps'] != accumulation or scheduler['last_epoch'] != stop):
            raise ValueError('Native topology or scheduler state did not resume')
        files = {str(p.relative_to(checkpoint)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                 for p in sorted(checkpoint.rglob('*')) if p.is_file()}
        results.append({'run_id': run_id, 'completed_steps': stop, 'checkpoint_sha256': model_hash,
                        'initialization': initialization, 'metrics_sha256': file_hash(folder / 'metrics.jsonl'),
                        'training_steps': [r['step'] for r in training], 'query_delta_l2': float(query.norm()),
                        'query_optimizer_moment_keys': list(query_moments), 'optimizer_counters': optimizer_updates,
                        'native_microsteps': stop, 'optimizer_updates': optimizer_updates,
                        'scheduler_microsteps': scheduler['last_epoch'],
                        'optimizer_parameter_states': len(counters), 'validation_frames': 200,
                        'valid_actions': 1303, 'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in training),
                        'update_seconds': [r['update_seconds'] for r in training], 'files': files})
        previous_hash = model_hash
    write_json(args.output, {'purpose': 'native_query_engineering_save_resume_gate', 'status': 'passed',
               'recipe_sha256': recipe_hash, 'source_manifest': expected, 'audit_sha256': file_hash(__file__),
               'trainable_decoder_layers': recipe['unfreeze_last_n'],
               'gpu': record['environment']['gpu'], 'allocator_config': record['allocator_config'],
               'microbatch_size': recipe['batch_size'], 'gradient_accumulation_steps': accumulation,
               'effective_batch_size': 8, 'validation_batch_size': 8,
               'runs': results, 'limitations': 'Four engineering updates on the recorded GPU, excluded from production '
               'selection and curves. Exact inherited tensor restoration, optimizer counters/moments and scheduler '
               'resume verified. RNG is serialized and hash-verified; no uninterrupted-run equivalence claim. '
               'The native scheduler advances per microbatch, independently of optimizer accumulation; '
               'this is not numerical equivalence to physical batch-eight training. '
               'No longer-horizon, cross-hardware reproducibility or task-success claim.'})
    print('Native optimizer/save/resume gate passed; not a production control result.')


if __name__ == '__main__':
    main()
