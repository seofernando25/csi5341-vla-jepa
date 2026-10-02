"""Run a registered recovery study on one already-provisioned GPU host.

No rental actions or provider credentials. The local exporter verifies complete
resume state before deleting the rental. Scientific settings live in the recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True, help='Unix UTC; leaves time for local export')
    args = parser.parse_args()
    os.chdir(ROOT)
    folder = ROOT / 'outputs/recovery/cloud'
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'job.json'
    if state_path.exists():
        raise FileExistsError('Do not overwrite an existing cloud job; inspect and resume explicitly')
    recipe = read_json(ROOT / 'evaluation/recovery_config.json')
    source = args.architecture_source.resolve()
    child_env = dict(os.environ, PYTHONPATH=str(source / 'src') + ':' + str(ROOT))
    record = {'study': recipe['study'], 'recipe_sha256': file_hash(ROOT / 'evaluation/recovery_config.json'),
              'started_at': time.time(), 'deadline': args.deadline, 'status': 'preflight', 'stages': []}
    training_output = None
    telemetry = None
    telemetry_log = None

    def write():
        write_json(state_path, record)

    def run(command, log, max_seconds):
        remaining = min(max_seconds, args.deadline - time.time())
        if remaining < 60:
            raise TimeoutError('Reached the pre-export deadline')
        with (folder / log).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=child_env)
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError('Stage timed out; completed periodic checkpoints remain exportable')
            if process.returncode:
                raise RuntimeError(f'{log} failed with exit code {process.returncode}')

    def checkpoints():
        if training_output is None:
            return []
        base = training_output / 'checkpoints'
        required = ('pretrained_model/model.safetensors', 'pretrained_model/recovery_registration.json',
                    'training_state/optimizer_state.safetensors', 'training_state/rng_state.safetensors',
                    'training_state/scheduler_state.json', 'training_state/optimizer_param_groups.json',
                    'training_state/training_step.json')
        return sorted([p for p in base.iterdir() if p.name.isdigit()
                       and all((p / name).is_file() for name in required)], key=lambda p: int(p.name)) if base.exists() else []

    try:
        write()
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('Native CUDA/BF16 is required')
        probe = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        loss = (probe @ probe.T).float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(probe.grad).all():
            raise RuntimeError('CUDA kernel preflight failed')
        record['gpu'] = {'name': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                         'torch': torch.__version__, 'cuda': torch.version.cuda}
        del probe, loss
        torch.cuda.empty_cache()
        telemetry_log = (folder / 'gpu-telemetry.csv').open('w')
        telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit',
                                      '--format=csv', '-l', '10'], stdout=telemetry_log, stderr=subprocess.STDOUT)
        resume = None
        for stop in recipe['stage_stop_steps']:
            if args.deadline - time.time() < 900:
                break
            record.update(status='training', target_step=stop)
            write()
            before = set((ROOT / 'studies/recovery/training').glob('*'))
            command = [sys.executable, '-m', 'evaluation.recovery_train', '--dataset-root', str(args.dataset_root),
                       '--architecture-source', str(source), '--checkpoint', str(args.checkpoint), '--steps', str(stop)]
            if resume:
                command += ['--resume', str(resume)]
            try:
                run(command, f'train-{stop}.log', args.deadline - time.time())
            finally:
                created = set((ROOT / 'studies/recovery/training').glob('*')) - before
                if len(created) == 1:
                    evidence = created.pop()
                    record['training_evidence'] = str(evidence.relative_to(ROOT))
                    stage_record = read_json(evidence / 'run.json')
                    if training_output is None:
                        training_output = ROOT / 'outputs/recovery/training' / stage_record['run_id'] / 'train'
                    record['checkpoint_root'] = str((training_output / 'checkpoints').relative_to(ROOT))
            available = checkpoints()
            if not available or int(available[-1].name) != stop:
                raise RuntimeError('Training completed without a verified native stop checkpoint')
            resume = available[-1] / 'pretrained_model'
            history = read_json(training_output / 'recovery_checkpoints.json')
            scored = [r for r in history if r['retained'] and 'heldout_arm_mse' in r]
            selected = min(scored, key=lambda r: (r['heldout_arm_mse'], r['step']))
            best = next(p / 'pretrained_model' for p in available if int(p.name) == selected['step'])
            stage = {'completed_step': stop, 'selected_step': selected['step'],
                     'selected_checkpoint_sha256': selected['checkpoint_sha256'],
                     'heldout_arm_mse': selected['heldout_arm_mse'],
                     'heldout_gripper_error': selected['heldout_gripper_error']}
            record['stages'].append(stage)
            # Pin complete stage state with hard links while later native saves
            # prune their own directory. The local exporter can back this up while
            # training continues without racing checkpoint retention.
            backup = ROOT / 'outputs/recovery/training/cloud-backups/train/checkpoints' / f'{stop:06d}'
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(available[-1], backup, copy_function=os.link)
            record['backup_files'] = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                                      for p in backup.rglob('*') if p.is_file()}
            record.update(status='development', selected_checkpoint=str(best.relative_to(ROOT)))
            write()
            label = f'RGB-n0008-{stop}'
            try:
                run([sys.executable, '-m', 'evaluation.run', 'rollout', '--variant', 'S500',
                     '--checkpoint', str(best), '--label', label, '--phase', 'development',
                     '--episodes', str(recipe['development_episodes_per_task'])], f'rollout-{stop}.log', 1500)
                stage['development_status'] = 'completed'
            except RuntimeError as exc:
                stage['development_status'], stage['development_error'] = 'failed', str(exc)
            write()
        if record['stages']:
            record['status'] = 'benchmarking'
            write()
            try:
                run([sys.executable, '-m', 'evaluation.run', 'benchmark', '--variant', 'S500',
                     '--checkpoint', record['selected_checkpoint'], '--label', 'RGB-n0008-selected',
                     '--predictions', '500', '--repetitions', '3'], 'benchmark.log', 1200)
                record['benchmark_status'] = 'completed'
            except (RuntimeError, TimeoutError) as exc:
                record['benchmark_status'], record['benchmark_error'] = 'failed', str(exc)
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        if telemetry:
            telemetry.terminate()
            telemetry.wait(timeout=15)
            telemetry_log.close()
        available = checkpoints()
        if available:
            record['export_files'] = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                                      for checkpoint in available for p in checkpoint.rglob('*') if p.is_file()}
            record['latest_checkpoint'] = str((available[-1] / 'pretrained_model').relative_to(ROOT))
        record['wall_seconds'] = time.time() - record['started_at']
        write()
        print(json.dumps({'status': record['status'], 'stages': record['stages'],
                          'export_files': len(record.get('export_files', {}))}), flush=True)


if __name__ == '__main__':
    main()
