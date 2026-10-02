"""Registered paired development evaluation of Smol inference buffer precision."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.common import ROOT, environment, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--buffer-precision', choices=['legacy', 'native_rope'], required=True)
    args = parser.parse_args()
    registration_path = ROOT / 'studies/recovery/native_rope_registration.json'
    registration = read_json(registration_path)
    if file_hash(args.checkpoint / 'model.safetensors') != registration['checkpoint_sha256']:
        raise ValueError('Use only the preregistered fixed checkpoint')
    if file_hash(ROOT / 'evaluation/models.py') != registration['loader_sha256']:
        raise ValueError('Loader differs from registered amendment')
    source = args.architecture_source.resolve()
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest']:
        raise ValueError('Architecture differs from registered amendment')
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Wrong architecture imported')
    from evaluation import run
    from evaluation.models import load_policy

    if (file_hash(ROOT / 'evaluation/protocol.json') != registration['protocol_sha256']
            or file_hash(ROOT / 'studies/evaluation/initial_states.json') != registration['initial_states_sha256']):
        raise ValueError('Protocol or initial states changed')
    label = 'RGB-n0008-5k-' + ('NativeRoPE' if args.buffer_precision == 'native_rope' else 'LegacyRoPE')
    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + '-' + label + '-rollout'
    folder = ROOT / 'studies/evaluation/runs' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(variant=label, checkpoint=args.checkpoint,
                               tasks=list(range(10)), episodes=1, phase='development')
    record = {'run_id': run_id, 'variant': label, 'experiment': 'rollout',
              'phase': 'development', 'status': 'running', 'environment': environment(),
              'checkpoint_sha256': registration['checkpoint_sha256'],
              'source_manifest': manifest, 'protocol': run.PROTOCOL,
              'initial_states_manifest_sha256': registration['initial_states_sha256'],
              'registration_sha256': file_hash(registration_path),
              'evaluator_sha256': file_hash(ROOT / 'evaluation/run.py'),
              'wrapper_sha256': file_hash(__file__),
              'loader_sha256': registration['loader_sha256'],
              'arguments': {'buffer_precision': args.buffer_precision, 'episodes': 1,
                            'tasks': list(range(10)), 'phase': 'development'},
              'limitations': 'Ten development episodes; diagnostic of one fixed checkpoint. '
                            'No weight selection by simulator outcomes or final success estimate.'}
    write_json(folder / 'run.json', record)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        policy, metadata = load_policy('S500', args.checkpoint, buffer_precision=args.buffer_precision)
        metadata['variant'] = label
        record['model'] = metadata
        write_json(folder / 'run.json', record)
        summary = run.rollout(settings, policy, metadata, folder)
        record.update(status='completed', summary=summary)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(folder / 'run.json', record)


if __name__ == '__main__':
    main()
