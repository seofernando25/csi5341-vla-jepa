"""Trace one development episode with the existing evaluation pipeline.

No training, action modification, final-test access or checkpoint selection.
Raw state/command traces and sparse camera frames remain runtime artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def arrays(value):
    if isinstance(value, dict):
        return {key: arrays(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class TraceEnvironment:
    """Observe calls without changing actions, observations or simulator state."""

    def __init__(self, env, folder: Path, frame_interval: int):
        self.env, self.folder, self.frame_interval = env, folder, frame_interval
        self.folder.mkdir(parents=True, exist_ok=False)
        self.stream = (folder / 'commands.jsonl').open('w')
        self.step_index = 0
        self.raw = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def capture(self, raw, force=False):
        if not force and self.step_index % self.frame_interval:
            return
        # Raw simulator images retain their original orientation. The policy
        # continues to use the unchanged environment image processor.
        np.savez_compressed(self.folder / f'frames-{self.step_index:04d}.npz',
                            **raw.get('pixels', {}))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.raw = result[0]
        self.step_index = 0
        self.capture(self.raw)
        return result

    def step(self, action):
        before = arrays(self.raw.get('robot_state', {}))
        command = np.asarray(action).copy()
        result = self.env.step(action)
        raw, reward, terminated, truncated, info = result
        row = {'step': self.step_index, 'action': command.tolist(),
               'state_before': before, 'state_after': arrays(raw.get('robot_state', {})),
               'reward': float(reward), 'terminated': bool(terminated),
               'truncated': bool(truncated), 'success': bool(info.get('is_success', False))}
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()
        self.step_index += 1
        self.raw = raw
        self.capture(raw, force=bool(terminated or truncated or info.get('is_success', False)))
        return result

    def close(self):
        try:
            self.env.close()
        finally:
            self.stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['B16', 'S500'], default='S500')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--architecture-source', type=Path)
    parser.add_argument('--task', type=int, choices=range(10), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-interval', type=int, default=14)
    parser.add_argument('--buffer-precision', choices=['legacy', 'native_rope'], default='legacy')
    args = parser.parse_args()
    if args.frame_interval < 7:
        parser.error('Keep sparse frame capture: interval must be at least seven')
    source = None
    if args.variant == 'S500':
        if args.architecture_source is None or args.checkpoint is None:
            parser.error('S500 requires its checkpoint and architecture snapshot')
        source = args.architecture_source.resolve()
        sys.path.insert(0, str(source / 'src'))
        import lerobot_policy_vla_jepa_smolvlm as plugin
        if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
            raise ValueError('Architecture snapshot was not imported')
    elif args.architecture_source is not None or args.checkpoint is not None:
        parser.error('B16 uses the pinned published checkpoint without overrides')
    import torch
    from evaluation import run
    from evaluation.common import ROOT, environment, file_hash, write_json
    from evaluation.models import artifact, load_policy

    checkpoint = args.checkpoint if args.variant == 'S500' else artifact('baseline')

    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        parser.error('Keep raw traces and camera frames in ignored outputs/')
    output.mkdir(parents=True, exist_ok=False)
    record = {'purpose': 'development_control_diagnostic', 'task_id': args.task,
              'variant': args.variant,
              'trace_implementation_sha256': file_hash(Path(__file__)),
              'checkpoint_sha256': file_hash(checkpoint / 'model.safetensors'),
              'buffer_precision': args.buffer_precision,
              'loader_implementation_sha256': file_hash(ROOT / 'evaluation/models.py'),
              'source_manifest': {str(p.relative_to(source)): file_hash(p)
                                  for p in sorted((source / 'src').rglob('*.py'))} if source else None,
              'initial_states_manifest_sha256': file_hash(ROOT / 'studies/evaluation/initial_states.json'),
              'environment': environment(), 'status': 'running',
              'limitations': 'One development episode; instrumentation adds wall time. Not a latency benchmark or final success estimate.'}
    write_json(output / 'run.json', record)
    original = run.make_environment

    def traced(task_id, phase):
        env, task = original(task_id, phase)
        return TraceEnvironment(env, output / f'task-{task_id}', args.frame_interval), task

    run.make_environment = traced
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        policy, metadata = load_policy(args.variant, args.checkpoint, buffer_precision=args.buffer_precision)
        settings = SimpleNamespace(variant=args.variant + '-trace', checkpoint=checkpoint,
                                   tasks=[args.task], episodes=1, phase='development')
        summary = run.rollout(settings, policy, metadata, output)
        record.update(status='completed', summary=summary, model=metadata)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        run.make_environment = original
        write_json(output / 'run.json', record)


if __name__ == '__main__':
    main()
