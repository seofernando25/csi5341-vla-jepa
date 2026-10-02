"""Registered, fixed-checkpoint development cohort; no training or final trials."""

from __future__ import annotations

import argparse
import fcntl
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from evaluation.common import ROOT, environment, file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['B16', 'RGB-parent20k'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path)
    args = parser.parse_args()
    registration_path = ROOT / 'studies/recovery/expanded_development_registration.json'
    registration = read_json(registration_path)
    case = registration['variants'][args.variant]
    if (registration['phase'] != 'development' or registration['tasks'] != list(range(10))
            or registration['episodes_per_task'] != 10):
        raise ValueError('Require the registered100-development-state cohort')
    for name, path in [('loader_sha256', 'evaluation/models.py'), ('evaluator_sha256', 'evaluation/run.py'),
                       ('protocol_sha256', 'evaluation/protocol.json'),
                       ('initial_states_sha256', 'studies/evaluation/initial_states.json')]:
        if file_hash(ROOT / path) != registration[name]:
            raise ValueError('Registered loader/evaluator/protocol/states changed')
    if file_hash(args.checkpoint / 'model.safetensors') != case['checkpoint_sha256']:
        raise ValueError('Registered fixed checkpoint differs')
    if args.variant == 'RGB-parent20k':
        if args.architecture_source is None:
            raise ValueError('Require the corrected-input parent source')
        source = args.architecture_source.resolve()
        manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
        if manifest != case['source_manifest']:
            raise ValueError('Registered parent source differs')
        sys.path.insert(0, str(source / 'src'))
        import lerobot_policy_vla_jepa_smolvlm as plugin
        if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
            raise ValueError('Wrong parent plugin imported')
    elif args.architecture_source is not None:
        raise ValueError('B16 uses only its pinned native implementation')
    import torch
    from evaluation import run
    from evaluation.models import load_policy
    if torch.cuda.get_device_name() != registration['gpu']:
        raise ValueError('Wrong registered local GPU')
    lock_path = ROOT / 'outputs/recovery/expanded-development.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    label = args.variant + '-dev100'
    previous = list((ROOT / 'studies/evaluation/runs').glob('*-' + label + '-rollout/run.json'))
    if previous:
        raise FileExistsError('Existing cohort execution: inspect it before any explicit amendment/retry')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + '-' + label + '-rollout'
    folder = ROOT / 'studies/evaluation/runs' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(variant=label, checkpoint=args.checkpoint,
                               tasks=registration['tasks'], episodes=10, phase='development')
    record = {'run_id': run_id, 'variant': label, 'experiment': 'rollout', 'phase': 'development',
              'status': 'running', 'environment': environment(), 'protocol': run.PROTOCOL,
              'checkpoint_sha256': case['checkpoint_sha256'], 'source_manifest': case['source_manifest'],
              'registration_sha256': file_hash(registration_path), 'wrapper_sha256': file_hash(__file__),
              'loader_sha256': registration['loader_sha256'], 'evaluator_sha256': registration['evaluator_sha256'],
              'initial_states_manifest_sha256': registration['initial_states_sha256'],
              'arguments': {'buffer_precision': case['buffer_precision'], 'episodes': 10,
                            'tasks': registration['tasks'], 'phase': 'development'},
              'numerical_flags': {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                  'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                  'cudnn_deterministic': torch.backends.cudnn.deterministic},
              'limitations': registration['limits'] + ' ' + registration['scope']}
    write_json(folder / 'run.json', record)
    try:
        policy, metadata = load_policy('B16' if args.variant == 'B16' else 'S500', args.checkpoint,
                                      buffer_precision=case['buffer_precision'])
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
    print({'run_id': run_id, 'summary': summary}, flush=True)


if __name__ == '__main__':
    main()
