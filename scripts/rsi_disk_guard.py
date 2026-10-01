"""Reclaim completed checkpoints first; preserve a final disk safety reserve."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def low_space():
    return shutil.disk_usage(ROOT).free < 40 * 1024**3 or shutil.disk_usage('/tmp').free < 5 * 1024**3


def main():
    if low_space():
        print('Low disk space: pruning completed evaluation artifacts first.', flush=True)
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluation_cleanup.py'), '--apply'],
                       cwd=ROOT, check=True, timeout=120)
        if not low_space():
            return
        print('No safe reclaimable artifacts remain; preserving disk safety reserve.', flush=True)
        subprocess.run(['systemctl', '--user', 'disable', '--now', 'csi5341-rsi-overnight.service'],
                       check=True, timeout=90)
        subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'rsi', 'export'],
                       cwd=ROOT, check=True, timeout=120)


if __name__ == '__main__':
    main()
