"""Local recovery exporter and spending guard. Credentials never leave this host."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import file_hash, read_json, write_json


def checked_relative(name, parent):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.is_relative_to(parent):
        raise ValueError('Export path escapes the recovery study')
    return path


def transfer_with_updates(command, update=None, interval=60, timeout=7200):
    """Keep metadata fresh during a large immutable checkpoint transfer."""
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                code = process.wait(timeout=min(interval, remaining))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
            except subprocess.TimeoutExpired:
                if update is not None:
                    try:
                        update()
                    except Exception as exc:
                        # A metadata/API failure must not discard an otherwise
                        # healthy checkpoint transfer or expose HTTP headers.
                        print('Transfer metadata retry:', type(exc).__name__, flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--rental', type=Path, required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT / 'outputs/recovery/cloud'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'export.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rental = read_json(args.rental)
    identifier = int(rental['instance_id'])
    status_path = folder / 'export-status.json'
    if status_path.exists() and read_json(status_path).get('status') == 'exported_and_destroyed':
        return
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + args.key_file.read_text().strip()
    base = 'https://console.vast.ai/api/v0/'
    exported = set()
    verified_path = folder / 'verified-checkpoints.json'
    if verified_path.exists():
        exported = set(read_json(verified_path))

    def api(method, path, **kw):
        r = session.request(method, base + path, timeout=25, **kw)
        r.raise_for_status()
        return r.json()

    def status(name, **values):
        write_json(status_path, {'status': name, 'at': time.time(), 'instance_id': identifier, **values})

    def ssh(instance):
        host, port = instance['ssh_host'], int(instance['ssh_port'])
        if not re.fullmatch(r'[A-Za-z0-9.:-]+', host):
            raise ValueError('Unexpected SSH hostname')
        return ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1',
                '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
                '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
                '-o', f'UserKnownHostsFile={args.known_hosts}', '-i', str(args.ssh_key), '-p', str(port), f'root@{host}']

    def sync(instance, relative, destination, filters=(), update=None):
        destination.mkdir(parents=True, exist_ok=True)
        transfer_with_updates(['rsync', '-az', '--partial', '--compress-choice=zstd', *filters,
                        '-e', shlex.join(ssh(instance)[:-1]),
                        f'root@{instance["ssh_host"]}:{ROOT}/{relative}', str(destination)],
                       update=update)

    def metadata(instance):
        sync(instance, 'studies/recovery/training/', ROOT / 'studies/recovery/training')
        sync(instance, 'studies/evaluation/runs/', ROOT / 'studies/evaluation/runs')
        sync(instance, 'outputs/recovery/cloud/', folder / 'remote-logs',
             ['--include=*.log', '--include=*.json', '--include=*.csv', '--exclude=*'])

    def export(instance, files):
        allowed = Path('outputs/recovery/training')
        directories = set()
        for name in files:
            path = checked_relative(name, allowed)
            # .../<run>/train/checkpoints/<step>/{pretrained_model,training_state}/file
            parts = path.relative_to(allowed).parts
            if len(parts) < 6 or parts[1:3] != ('train', 'checkpoints') or not parts[3].isdigit():
                raise ValueError('Unexpected native checkpoint export shape')
            directories.add(allowed.joinpath(*parts[:4]))
        for directory in directories:
            selected = {name: value for name, value in files.items() if Path(name).is_relative_to(directory)}
            model = next((v['sha256'] for n, v in selected.items() if n.endswith('/model.safetensors')), None)
            identity = str(directory) + ':' + str(model)
            if identity in exported and all((ROOT / name).is_file()
                    and (ROOT / name).stat().st_size == expected['bytes']
                    and file_hash(ROOT / name) == expected['sha256'] for name, expected in selected.items()):
                continue
            status('exporting_checkpoint', directory=str(directory))

            def refresh_during_copy():
                remote = subprocess.run(ssh(instance) + [f'cat {shlex.quote(str(ROOT / "outputs/recovery/cloud/job.json"))}'],
                                        capture_output=True, text=True, timeout=40, check=True)
                job = json.loads(remote.stdout)
                write_json(folder / 'remote-job.json', job)
                metadata(instance)
                spent = rental['account_credit_at_project_start'] - float(api('GET', 'users/current/')['credit'])
                if spent >= 13.5:
                    subprocess.run(ssh(instance) + ['pkill -INT -f "python.*evaluation.recovery_train" || true'],
                                   capture_output=True, timeout=40)
                status('exporting_checkpoint', directory=str(directory), job_status=job['status'],
                       target_step=job.get('target_step'), total_spent_usd=spent)

            sync(instance, str(directory) + '/', ROOT / directory, update=refresh_during_copy)
            for name, expected in selected.items():
                p = ROOT / name
                if p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
                    raise RuntimeError('Native export hash mismatch')
            exported.add(identity)
            write_json(verified_path, sorted(exported))
            with (folder / 'checkpoint-backups.jsonl').open('a') as handle:
                handle.write(json.dumps({'at': time.time(), 'directory': str(directory), 'files': selected}) + '\n')

    while True:
        try:
            account = api('GET', 'users/current/')
            spent = rental['account_credit_at_project_start'] - float(account['credit'])
            instances = api('GET', f'instances/{identifier}/').get('instances')
            instance = instances[0] if isinstance(instances, list) and instances else instances
            if not isinstance(instance, dict):
                status('rental_missing', total_spent_usd=spent)
                return
            if instance.get('actual_status') != 'running' or not instance.get('ssh_host'):
                status('waiting_for_host', actual_status=instance.get('actual_status'), total_spent_usd=spent)
                time.sleep(30)
                continue
            transport = ssh(instance)
            if spent >= 13.5:
                # Leave a monetary reserve for exporting the latest completed state.
                subprocess.run(transport + ['pkill -INT -f "python.*evaluation.recovery_train" || true'],
                               capture_output=True, timeout=40)
                status('budget_stop_requested', total_spent_usd=spent)
            remote = subprocess.run(transport + [f'cat {shlex.quote(str(ROOT / "outputs/recovery/cloud/job.json"))}'],
                                    capture_output=True, text=True, timeout=40)
            if remote.returncode:
                status('waiting_for_job', total_spent_usd=spent)
                time.sleep(30)
                continue
            job = json.loads(remote.stdout)
            write_json(folder / 'remote-job.json', job)
            metadata(instance)
            status('monitoring', job_status=job['status'], target_step=job.get('target_step'), total_spent_usd=spent)
            if job.get('backup_files'):
                export(instance, job['backup_files'])
            if job.get('status') in {'completed', 'failed'}:
                files = job.get('export_files', {})
                if not files:
                    status('failed_without_checkpoint', error_type=job.get('error_type'), total_spent_usd=spent)
                    # No new state to lose; the initial warm start remains locally verified.
                else:
                    export(instance, files)
                    # Recheck every final file even if this model hash was backed up earlier.
                    for name, expected in files.items():
                        p = ROOT / checked_relative(name, Path('outputs/recovery/training'))
                        if p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
                            raise RuntimeError('Final re-verification failed')
                metadata(instance)
                if files:
                    # Stage backups have served their purpose; the final native
                    # latest/best checkpoints and compact audit history remain.
                    backups = ROOT / 'outputs/recovery/training/cloud-backups'
                    if backups.exists():
                        shutil.rmtree(backups)
                deleted = api('DELETE', f'instances/{identifier}/', json={})
                if not deleted.get('success'):
                    raise RuntimeError('Provider did not confirm rental deletion')
                schedule = rental.get('cleanup_schedule_id')
                if schedule:
                    api('DELETE', f'commands/schedule_job/{schedule}/', json={})
                status('exported_and_destroyed', verified_files=len(files), job_status=job['status'], total_spent_usd=spent)
                return
        except Exception as exc:
            # Never dump HTTP headers, account data or credentials on failure.
            status('retrying', error_type=type(exc).__name__)
            print('Recovery exporter retry:', type(exc).__name__, flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
