"""Guard one registered parent diagnostic after the existing cloud study ends."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import file_hash, read_json, write_json
from scripts.recovery_export import job_identity


def readiness(job, export, review, rental, now):
    if job.get('status') != 'completed':
        raise ValueError('Query supervisor is not completed; do not interrupt it')
    if export.get('status') != 'verified_completion_review' or now - export.get('at', 0) > 120:
        raise ValueError('Need a fresh verified final-export completion review')
    if review.get('job_identity') != job_identity(job):
        raise ValueError('Completion review belongs to another job')
    if min(review['until'], rental['planned_cleanup_epoch'] - 60) - now < 1200:
        raise ValueError('Insufficient bounded completion-review time')
    if export.get('instance_id') != rental['instance_id']:
        raise ValueError('Exporter and rental identities differ')
    if export['total_spent_usd'] + rental['offer']['dph_total'] / 3 >= 13.25:
        raise ValueError('Insufficient authorized spending reserve')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Default only checks readiness; never starts training')
    parser.add_argument('--key-file', type=Path, help='Existing local Vast key; used only for a fresh spending check')
    args = parser.parse_args()
    folder = ROOT / 'outputs/recovery/cloud'
    registration_path = ROOT / 'studies/recovery/parent5090_native_registration.json'
    registration = read_json(registration_path)
    job, rental = read_json(folder / 'remote-job.json'), read_json(folder / 'rental.json')
    if job.get('recipe_sha256') != registration['query_recipe_sha256']:
        raise ValueError('Cloud job is not the registered query study')
    receipt_path = folder / 'parent-native-receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Inspect the existing diagnostic receipt before any retry')
    try:
        readiness(job, read_json(folder / 'export-status.json'),
                  read_json(folder / 'completion-review.json'), rental, time.time())
    except ValueError as exc:
        if args.run:
            raise
        print('Not ready:', str(exc))
        return
    if not args.run:
        print('Ready for one bounded parent diagnostic; no evaluation launched')
        return
    if args.key_file is None:
        parser.error('--run requires --key-file for a fresh local provider spending check')
    with requests.Session() as session:
        session.headers['Authorization'] = 'Bearer ' + args.key_file.read_text().strip()
        response = session.get('https://console.vast.ai/api/v0/users/current/', timeout=25)
        response.raise_for_status()
        spent = rental['account_credit_at_project_start'] - float(response.json()['credit'])
    if spent + rental['offer']['dph_total'] / 3 >= 13.25:
        raise ValueError('Fresh provider spending leaves insufficient reserve')
    sys.path.insert(0, str(ROOT / 'outputs/cloud'))
    import cloud_ssh
    connection = read_json(folder / 'connection.json')
    if int(connection['id']) != rental['instance_id']:
        raise ValueError('SSH connection belongs to a different rental')
    transport = cloud_ssh.ssh(connection)
    # The wrapper checks checkpoint, architecture, evaluator and protocol hashes.
    # The OS timeout bounds the remote child even if this local process disappears.
    check = subprocess.run(transport + ['nvidia-smi --query-gpu=name --format=csv,noheader'],
                           capture_output=True, text=True, timeout=40, check=True)
    if check.stdout.strip() != registration['gpu']:
        raise ValueError('Actual GPU differs from the registered comparison')
    active = subprocess.run(transport + ['nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
                            capture_output=True, text=True, timeout=40, check=True)
    if active.stdout.strip():
        raise ValueError('GPU still has compute processes; do not overlap evaluations')
    source = ROOT / registration['parent_source_relative']
    checkpoint = ROOT / registration['parent_checkpoint_relative']
    wrapper = ROOT / 'evaluation/inference_precision_eval.py'
    probe = ('import hashlib,pathlib; p=pathlib.Path(' + repr(str(wrapper)) + '); '
             'assert hashlib.sha256(p.read_bytes()).hexdigest() == ' + repr(registration['wrapper_sha256']))
    subprocess.run(transport + [shlex.join([str(ROOT / '.venv/bin/python'), '-c', probe])],
                   timeout=40, check=True)
    subprocess.run(transport + ['cat > ' + shlex.quote(str(registration_path))],
                   input=registration_path.read_text(), text=True, timeout=40, check=True)
    command = ['env', 'PYTHONPATH=' + str(ROOT), 'timeout', '-s', 'INT', '-k', '30s', '900s',
               'flock', '-n', str(folder / 'job.lock'), str(ROOT / '.venv/bin/python'),
               '-m', 'evaluation.inference_precision_eval', '--architecture-source', str(source),
               '--checkpoint', str(checkpoint), '--buffer-precision', 'native_rope',
               '--registration', str(registration_path)]
    current = read_json(folder / 'remote-job.json')
    if job_identity(current) != job_identity(job):
        raise ValueError('Cloud job changed during readiness checks')
    readiness(current, read_json(folder / 'export-status.json'),
              read_json(folder / 'completion-review.json'), rental, time.time())
    receipt = {'registration_sha256': file_hash(registration_path), 'job_identity': job_identity(job),
               'instance_id': rental['instance_id'], 'spent_before_usd': spent,
               'query_selected_checkpoint_sha256': job['stages'][-1]['selected_checkpoint_sha256'],
               'started_at': time.time(), 'status': 'running', 'maximum_seconds': 900}
    write_json(receipt_path, receipt)
    try:
        with (folder / 'parent-native.log').open('w') as log:
            result = subprocess.run(transport + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=970)
        receipt.update(status='completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        if result.returncode:
            raise RuntimeError('Parent diagnostic failed; inspect the receipt and local log before retrying')
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        receipt['finished_at'] = time.time()
        write_json(receipt_path, receipt)
    print('Parent diagnostic completed; exporter collects rollout records')


if __name__ == '__main__':
    main()
