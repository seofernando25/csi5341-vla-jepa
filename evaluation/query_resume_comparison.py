"""Compare registered engineering resume and uninterrupted native state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from evaluation.common import ROOT, file_hash, read_json, write_json


def compare_tensors(left: Path, right: Path):
    differences = []
    with safe_open(left, framework='pt') as a, safe_open(right, framework='pt') as b:
        if a.keys() != b.keys():
            raise ValueError('Checkpoint tensor membership differs')
        count = len(a.keys())
        for key in a.keys():
            x, y = a.get_tensor(key), b.get_tensor(key)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'Tensor topology differs: {key}')
            if not torch.equal(x, y):
                delta = (x.double() - y.double()).abs()
                differences.append({'tensor': key, 'max_abs_difference': float(delta.max()),
                                    'different_elements': int(torch.count_nonzero(delta)),
                                    'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all())})
    return {'tensor_count': count, 'exact_values': not differences,
            'different_tensors': len(differences),
            'max_abs_difference': max((r['max_abs_difference'] for r in differences), default=0),
            'largest_differences': sorted(differences, key=lambda r: r['max_abs_difference'], reverse=True)[:10]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--uninterrupted', type=Path, required=True)
    parser.add_argument('--resumed-run', required=True)
    parser.add_argument('--uninterrupted-run', required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous comparisons')
    recipe_hash = file_hash(args.recipe)
    if not read_json(args.recipe).get('engineering_only'):
        raise ValueError('Only bounded engineering evidence is accepted')
    records = [read_json(ROOT / 'studies/recovery/training' / run / 'run.json')
               for run in (args.resumed_run, args.uninterrupted_run)]
    for r in records:
        if (r['recipe_sha256'] != recipe_hash or r['status'] != 'completed'
                or r['purpose'] != 'engineering_preflight' or r['completed_steps'] != 4):
            raise ValueError('Comparison requires completed four-update registered runs')
    if records[0]['source_manifest'] != records[1]['source_manifest']:
        raise ValueError('Compared sources differ')
    results = {}
    for relative in ['pretrained_model/model.safetensors', 'training_state/optimizer_state.safetensors',
                     'training_state/rng_state.safetensors']:
        a, b = args.resumed / relative, args.uninterrupted / relative
        results[relative] = {**compare_tensors(a, b), 'resumed_sha256': file_hash(a),
                             'uninterrupted_sha256': file_hash(b)}
    for relative in ['training_state/scheduler_state.json', 'training_state/training_step.json']:
        results[relative] = {'equal_json': read_json(args.resumed / relative) == read_json(args.uninterrupted / relative)}
    metrics = []
    for run in (args.resumed_run, args.uninterrupted_run):
        rows = [json.loads(line) for line in (ROOT / 'studies/recovery/training' / run / 'metrics.jsonl').read_text().splitlines()]
        metrics.append([{k: r[k] for k in ('step', 'loss', 'grad_norm', 'action_loss', 'wm_loss')}
                        for r in rows if r['phase'] == 'training'])
    write_json(args.output, {'purpose': 'native_query_resume_equivalence_engineering',
               'recipe_sha256': recipe_hash, 'source_manifest': records[0]['source_manifest'],
               'audit_sha256': file_hash(__file__), 'gpu': records[0]['environment']['gpu'],
               'resumed_run': args.resumed_run, 'uninterrupted_run': args.uninterrupted_run,
               'deterministic_algorithms': records[0].get('deterministic_algorithms', False),
               'isolated_worker_rng': records[0].get('isolated_worker_rng', False),
               'state_comparison': results, 'training_metrics': metrics,
               'limitations': 'Four engineering updates on one RTX3090. No production, task-success, '
               'cross-hardware or longer-horizon reproducibility claim. Exact tensor comparisons '
               'are independent of safetensors header ordering.'})
    print(json.dumps({k: {q: v[q] for q in ('exact_values', 'equal_json') if q in v} for k, v in results.items()}))


if __name__ == '__main__':
    main()
