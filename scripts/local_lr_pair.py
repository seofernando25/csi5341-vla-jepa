"""Exclusive, restartable local rate pair; no rental, benchmark or rollout actions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import file_hash, read_json, write_json
from evaluation.query_milestone import NATIVE_FILES, verify_counters


def registration_context(source, checkpoint):
    path = ROOT / 'studies/recovery/local_lr_pair_registration.json'
    registration = read_json(path)
    if (registration['supervisor_sha256'] != file_hash(__file__)
            or registration['harness_sha256'] != file_hash(ROOT / 'evaluation/recovery_train.py')
            or registration['counter_verifier_sha256'] != file_hash(ROOT / 'evaluation/query_milestone.py')):
        raise ValueError('Registered implementation changed')
    for name, expected in registration['parent_files'].items():
        if file_hash(checkpoint / name) != expected:
            raise ValueError('Parent checkpoint or processors differ')
    for name, expected in registration['data_manifest_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Data/split/validation manifest differs')
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest']:
        raise ValueError('Source differs')
    for name, expected in registration['recipe_sha256'].items():
        if file_hash(ROOT / f'evaluation/local_lr_{name}_config.json') != expected:
            raise ValueError('Registered recipe changed')
    gate = ROOT / 'studies/recovery/diagnostics/query_local_accumulation_gate.json'
    if file_hash(gate) != registration['native_gate_sha256'] or read_json(gate)['status'] != 'passed':
        raise ValueError('Intended native accumulation gate differs')
    return registration


def native_checkpoints(recipe_hash, registration):
    """Inspect closed-process state; incomplete saves never become resume candidates."""
    from safetensors import safe_open
    found = []
    for record_path in sorted((ROOT / 'studies/recovery/training').glob('*/run.json')):
        record = read_json(record_path)
        if record.get('recipe_sha256') != recipe_hash:
            continue
        if (record['source_manifest'] != registration['source_manifest']
                or record['initial_checkpoint_sha256'] != registration['initial_checkpoint_sha256']):
            raise ValueError('Recorded branch provenance differs')
        base = ROOT / 'outputs/recovery/training' / record['run_id'] / 'train/checkpoints'
        for directory in sorted(base.glob('*')):
            if not directory.name.isdigit() or not all((directory / name).is_file() for name in NATIVE_FILES):
                continue
            step = int(directory.name)
            topology = read_json(directory / 'training_state/training_step.json')
            registered = read_json(directory / 'pretrained_model/recovery_registration.json')
            scheduler = read_json(directory / 'training_state/scheduler_state.json')
            if (topology['step'] != step or topology['batch_size'] != 4
                    or topology['grad_accum_steps'] != 2 or topology['dp_world_size'] != 1
                    or registered['recipe_sha256'] != recipe_hash
                    or registered['source_manifest'] != registration['source_manifest']
                    or scheduler['last_epoch'] != step or step % 2):
                raise ValueError('Native checkpoint topology/registration differs')
            # Validate safetensors headers/length and every populated optimizer counter.
            with safe_open(directory / 'pretrained_model/model.safetensors', framework='pt') as weights:
                if weights.get_slice('model.qwen.query_adapter.delta').get_shape() != [4, 960]:
                    raise ValueError('Native model lost query state')
            verify_counters(directory, step, read_json(record_path.parent / 'trainability.json'),
                            batch_size=4, accumulation_steps=2)
            found.append((step, directory))
    if len({step for step, _ in found}) != len(found):
        raise ValueError('Multiple independently initialized native branch histories; inspect before resuming')
    return sorted(found)


def complete_branch(recipe_hash, registration, available):
    if not available or available[-1][0] != registration['stop_microsteps'][-1]:
        return None
    base = available[-1][1].parent.parent
    journal = read_json(base / 'recovery_checkpoints.json')
    scored = [row for row in journal if 'heldout_arm_mse' in row]
    if sorted(row['step'] for row in scored) != [500, 1000]:
        raise ValueError('Require both registered held-out validations')
    selected = min(scored, key=lambda row: (row['heldout_arm_mse'], row['step']))
    if not selected['retained']:
        raise ValueError('Held-out-selected policy was incorrectly pruned')
    model = base / 'checkpoints' / f"{selected['step']:06d}" / 'pretrained_model/model.safetensors'
    if file_hash(model) != selected['checkpoint_sha256']:
        raise ValueError('Selected policy bytes differ from validation journal')
    latest = available[-1][1]
    files = {name: {'bytes': (latest / name).stat().st_size, 'sha256': file_hash(latest / name)}
             for name in sorted(NATIVE_FILES)}
    return {'recipe_sha256': recipe_hash, 'completed_microsteps': 1000, 'optimizer_updates': 500,
            'selected': selected, 'selected_model': str(model.relative_to(ROOT)),
            'latest_native_checkpoint': str(latest.relative_to(ROOT)), 'latest_native_files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute the registered local pair; default only validates')
    args = parser.parse_args()
    source, checkpoint = args.architecture_source.resolve(), args.checkpoint.resolve()
    registration = registration_context(source, checkpoint)
    if not args.run:
        print('Registered local pair verified; no training launched.')
        return
    folder = ROOT / 'outputs/recovery/local-lr-pair'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {'study': registration['study'], 'registration_sha256': file_hash(
              ROOT / 'studies/recovery/local_lr_pair_registration.json'),
              'status': 'preparing', 'started_at': time.time(), 'branches': {}}
    old = read_json(folder / 'job.json') if (folder / 'job.json').exists() else {}
    if old and old['registration_sha256'] != record['registration_sha256']:
        raise ValueError('Existing local study identity changed')
    record['started_at'] = old.get('started_at', record['started_at'])
    child = None

    def write():
        record['observed_at'] = time.time()
        write_json(folder / 'job.json', record)

    def stop_child():
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)

    def terminate(signum, frame):
        stop_child()
        record['status'] = 'interrupted'
        write()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for branch in registration['execution_order']:
            recipe_hash = registration['recipe_sha256'][branch]
            available = native_checkpoints(recipe_hash, registration)
            completed = complete_branch(recipe_hash, registration, available)
            if completed:
                record['branches'][branch] = completed
                write()
                continue
            # Only the existing local GPU is used. Desktop graphics are not compute clients.
            clients = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                     capture_output=True, text=True, check=True, timeout=20).stdout.strip()
            if clients:
                raise RuntimeError('Another local GPU compute process is active; refusing duplicate training')
            record.update(status='training', active_branch=branch, target_microsteps=1000,
                          resume_microstep=available[-1][0] if available else 0)
            command = [sys.executable, '-u', '-m', 'evaluation.recovery_train', '--recipe',
                       str(ROOT / f'evaluation/local_lr_{branch}_config.json'), '--dataset-root',
                       str(args.dataset_root.resolve()), '--architecture-source', str(source),
                       '--checkpoint', str(checkpoint), '--steps', '1000']
            if available:
                command += ['--resume', str(available[-1][1] / 'pretrained_model')]
            with (folder / f'{branch}.log').open('a') as log:
                os.utime(log.name, None)
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, PYTORCH_ALLOC_CONF='expandable_segments:True',
                                                     MUJOCO_GL='egl'))
                record['child_pid'] = child.pid
                write()
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        activity = Path(log.name).stat().st_mtime
                        for path in (ROOT / 'studies/recovery/training').glob('*/run.json'):
                            candidate = read_json(path)
                            if candidate.get('recipe_sha256') == recipe_hash:
                                metrics = path.parent / 'metrics.jsonl'
                                activity = max(activity, path.stat().st_mtime,
                                               metrics.stat().st_mtime if metrics.exists() else 0)
                                record['training_evidence'] = str(path.parent.relative_to(ROOT))
                        write()
                        if time.time() - activity > registration['maximum_silent_seconds']:
                            stop_child()
                            raise TimeoutError('Native training stopped producing logs/metrics; saved state retained')
            if code:
                raise RuntimeError(f'Native branch {branch} exited {code}; inspect log before retry')
            completed = complete_branch(recipe_hash, registration, native_checkpoints(recipe_hash, registration))
            if not completed:
                raise RuntimeError('Native process ended without registered final state')
            record['branches'][branch] = completed
            write()
        candidates = [(registration['parent_arm_mse'], 0, 'parent')]
        candidates += [(value['selected']['heldout_arm_mse'], value['selected']['optimizer_updates'], name)
                       for name, value in record['branches'].items()]
        record.update(status='completed', active_branch=None, child_pid=None,
                      heldout_selected_branch=min(candidates)[2])
        write()
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        stop_child()
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write()
        raise


if __name__ == '__main__':
    main()
