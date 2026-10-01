"""External overnight supervisor; scientific state is changed only through RSI CLI."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
stopping = False


def cli(*args, timeout=60):
    return subprocess.run([str(PYTHON), '-m', 'rsi', *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, check=True)


def gpu_ready():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and 'NVIDIA' in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def latest_progress(state):
    # stdout/stderr record training steps and agent events; journal records transitions.
    files = list(state.rglob('*.log')) + list((state / 'events').glob('*'))
    return max((p.stat().st_mtime for p in files if p.is_file()), default=0)


def terminate(child):
    cli('stop')
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # systemd subsequently kills the whole service cgroup, including isolated jobs.
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    global stopping
    parser = argparse.ArgumentParser()
    parser.add_argument('--until', required=True, help='ISO timestamp with timezone')
    parser.add_argument('--stall-minutes', type=int, default=40)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.until)
    if deadline.tzinfo is None:
        parser.error('--until must include a timezone')
    deadline = deadline.timestamp()
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__('stopping', True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('stopping', True))
    state = ROOT / '.rsi'
    extend = False
    while not stopping and time.time() < deadline:
        if not gpu_ready():
            print('GPU unavailable; waiting for recovery or reboot.', flush=True)
            time.sleep(min(30, max(0, deadline - time.time())))
            continue
        status = json.loads(cli('status').stdout)
        if not status['initialized']:
            cli('init')
        command = [str(PYTHON), '-m', 'rsi', 'run', '--resume']
        if extend:
            command += ['--extend-cycles', '1']
        print('Starting/resuming current study' + (' with one extra cycle.' if extend else '.'), flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        started = time.time()
        failures = 0
        while child.poll() is None and not stopping and time.time() < deadline:
            failures = 0 if gpu_ready() else failures + 1
            progress = max(started, latest_progress(state))
            if failures >= 2 or time.time() - progress > args.stall_minutes * 60:
                print('GPU failure or stalled logs; requesting stop before cgroup restart.', flush=True)
                terminate(child)
                return 75
            time.sleep(min(30, max(0, deadline - time.time())))
        if child.poll() is None:
            terminate(child)
            break
        if child.returncode:
            print(f'RSI exited {child.returncode}; cgroup restart will clean up children.', flush=True)
            return 75
        status = json.loads(cli('status').stdout)
        reason = status.get('paused')
        extend = reason in {'global_attempt_cap', 'reservation_cap', 'global_outer_cap'}
        if not extend:
            print(f'Study stopped: {reason or "finished/operator stop"}.', flush=True)
            break
    cli('stop')
    cli('export', timeout=120)
    print('Overnight window closed; compact results exported.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
